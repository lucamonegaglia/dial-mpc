# Design an online system-identification strategy for a sampling-based MPC with in-rollout domain randomization

You are helping a robotics research group design an algorithm. Take your time and reason carefully: we want a concrete, implementable strategy with an honest account of trade-offs, not a literature survey. Where you are unsure, say so and propose how we would find out.

## 1. The controller we have

We use **DIAL-MPC** (diffusion-inspired annealing for MPPI-style sampling MPC), implemented in **JAX** on top of **MuJoCo MJX** (via Brax's pipeline). At every control step it does the following:

- **Decision variable.** A control sequence over a horizon of **H = 20 control steps at 0.02 s**, i.e. 0.4 s. It is parameterized by **Hnode + 1 = 6 spline nodes per actuator** (quadratic spline). With 8 actuators this gives a 48-dimensional decision per plan.
- **Sampling.**
  - Draw **N = 2048** perturbed node sequences around the current mean plan, using Gaussian noise with a per-node sigma schedule.
  - The first node is pinned to the previous plan, so **every sample shares the same first action u_t**, which is the action that will actually be executed.
  - Clip to [-1, 1] and convert nodes to per-step actions.
- **Evaluation.** Roll every sample out in MJX (a batched, vmapped `lax.scan`). The score is the mean per-step reward over the horizon.
- **Update.** An MPPI softmax:
  - Returns are z-scored using a trimmed mean and std; samples that diverge or are non-finite are masked.
  - Weights are `softmax(z / 0.05)`.
  - The new mean plan is the weighted average of the samples.
- **Annealing.** This repeats Ndiffuse times with shrinking noise: **1 iteration per control step** in the configuration we care about, and 10 at the very first step.
- **Receding horizon.** Execute the first action, shift the plan by one step (warm start), and replan from the newly observed state.
- **Reward.** A weighted sum of velocity tracking (target vx = 1 m/s), yaw-rate tracking, upright, base height, yaw, joint-pose regularization, energy (small), and alive = 1 − done. Termination happens when joints leave a band around the home pose or the robot falls.

## 2. The robot and simulator

- **Robot.** LimX **Tron1 wheeled-foot biped**.
  - Floating base: nq = 15, nv = 14.
  - 8 actuators, per side: abad, hip, knee, wheel.
  - Leg actions are joint position targets. Torque is computed **outside MuJoCo** by an explicit PD law, `tau = kp (q* − q) − kd q̇`, with kp = 120 and kd = 5 on the legs.
  - Wheel actions are velocity targets tracked with gain kd = 0.8 (wheel kp = 0).
- **Simulation settings.** The only model is MuJoCo; there is no analytical or reduced model.
  - Physics timestep **= control dt = 0.02 s**, i.e. one semi-implicit Euler step per control step.
  - `eulerdamp` is disabled and the solver runs 1 iteration and 1 line-search iteration.
  - Contacts: cylinder wheels (condim 3) on a plane.
  - Because the PD law is explicit, the integration's stability margin depends on kp, kd and armature. In a previous study on a humanoid, an armature below about 0.65× nominal made the 20 ms step numerically unstable (a period-2 mode). So the parameters change not only the physics but also the numerical behaviour of the planner's model.
- **Gradients.**
  - JAX can in principle differentiate through MJX, but we have never used gradients. With 20 ms steps, 1 solver iteration and rolling/stick-slip contact, we expect them to be noisy or discontinuous.
  - Hessians are impractical.
  - Treat the simulator as a batched black box. A method may optionally exploit gradients, but must not depend on them.
- **Hardware and compute.**
  - A single RTX 4090 (24 GB).
  - One annealing iteration (2048 rollouts × 20 steps) takes **~12 ms**. This was measured on a humanoid configuration; Tron1 is of similar or lower cost, to be re-measured.
  - The control period is **20 ms (50 Hz)**.
  - Passing the uncertain parameters as traced arguments instead of constants costs about 1.5%. Batching over parameter sets with `vmap` is therefore cheap per rollout: **cost scales roughly linearly in (#sequences × #parameter sets × H)**.
  - The real-time budget is thus **about 2000 horizon-20 rollouts per control step in total**, to be split however the algorithm likes, plus a small amount of non-rollout compute (a few hundred microseconds to a millisecond).
  - In our sim-to-sim harness the loop is synchronous: the plant waits for the planner, so the budget is soft there. On hardware it is hard, because the planner runs asynchronously in a separate process.
- **JAX constraints.**
  - All array shapes must be static: m, k, H and the parameter dimension are fixed at compile time.
  - Nothing may recompile mid-run.
  - Per-step logic must be jit-compatible (`lax.scan` / `lax.cond`, no Python branching on traced values).
  - Anything that needs a variable number of samples, or adaptive loop counts, must be reformulated with masks.

## 3. The uncertain parameters (θ)

In our sim-to-sim evaluation a "plant" is MuJoCo with θ drawn once per episode from the priors below. The planner's model is MuJoCo with some other θ: currently the nominal value, or the true θ as an oracle baseline. There are **53 scalars**, with elements independent per joint:

| Parameter | Dim | Prior |
|---|---|---|
| joint armature | 8 | scale × U_log[0.5, 2.0] of nominal 0.01 |
| joint viscous damping | 8 | absolute U[0, 0.3] (nominal 0.01) |
| joint Coulomb friction (frictionloss) | 8 | absolute U[0, 0.5] N·m (nominal 0 on legs, 0.01 on wheels) |
| PD kp, legs only | 6 | scale × U[0.7, 1.3] |
| PD kd, legs + wheel velocity gain | 8 | scale × U[0.7, 1.3] |
| link masses | 8 | scale × U[0.9, 1.1] |
| base mass (nominal 9.6 kg) | 1 | scale × U[0.8, 1.2] |
| base CoM offset | 3 | add U[−0.02, 0.02] m per axis |
| ground–robot sliding friction μ | 1 | absolute U[0.6, 1.4] (nominal 1.0) |
| wheel radius (nominal 0.127 m) | 2 | scale × U[0.98, 1.02] |

**Evidence so far.** These numbers come from 200 paired episodes of 400 steps (8 s), with the same plant and MPC seed in both arms. In that sweep each parameter group shared one scalar across joints; the per-joint version has not been run yet.

- **Nominal-θ planner:** survives on average 56.5 steps; 0/200 episodes complete.
- **True-θ (oracle) planner:** survives 255.5 steps; 106/200 complete.
- The mean paired gap in total return is −193 (SD 145).
- Binning trials by each parameter:
  - the gap is driven mostly by **leg kp** (the gap shrinks from −317 to −36 as the kp scale goes from 0.75 to 1.25), **armature** (the gap grows with armature) and **kd**;
  - masses, CoM, μ, damping, wheel radius and wheel friction show nearly flat effects over these ranges.

So the control-relevant subspace is probably much smaller than 53. We do not yet know which directions are **identifiable** from short closed-loop data. Relevant to control and identifiable from data are different questions.

Two further facts:
- Episode outcomes are **chaotic**: re-running an identical plant and seed cell gives a return SD of 50–80. Any evaluation needs many seeds and paired comparisons.
- Even the oracle fails about half the time at this budget. So online identification can at best close the gap to the oracle, although risk-aversion might also help on its own.

## 4. The idea we want to develop

**In-rollout domain randomization with a risk-averse score, plus an online belief update:**

1. Keep a belief b_t(θ). At each control step, draw **k parameter sets θ_1..θ_k** from it and use them for **all m action sequences** (common random numbers), so m·k ≈ the rollout budget.
2. Roll out every (sequence, θ_j) pair. Score each sequence by a **risk-averse aggregate** of its k returns: CVaR_α, a mean–variance penalty, worst-case, entropic risk, or similar. Feed that score into the MPPI update.
3. Execute the first action and observe the next state x_{t+1}.
4. Use the observed transition to update b_t → b_{t+1}, which shapes the k environments sampled at the next step.

A useful structural fact: because the first action is pinned and shared by all samples, the rollouts already contain the **one-step prediction x̂_{t+1}(θ_j) for the action that is actually executed, for every θ_j**. So a transition likelihood p(x_{t+1} | x_t, u_t, θ_j) for each of the k parameter sets costs essentially nothing extra. Multi-step likelihoods over a window of past transitions would need extra rollouts. Those rollouts are cheap if they are few, but they come out of the same budget.

**Candidates we have considered:**
- **Bayesian optimization** over θ.
- A **dual-layer MPPI**: an outer MPPI over θ, weighted by transition likelihood, around the inner MPPI over actions.
- **Kalman filtering**: EKF, UKF or ensemble KF on θ, possibly jointly with the state.

We are open to anything better, for example:
- particle filters / SMC with resampling and jitter;
- cross-entropy or adaptive importance sampling on θ;
- Stein variational particles;
- moving-horizon estimation;
- likelihood-free posteriors (BayesSim/DROPO/SimOpt-style);
- a differentiable-sim estimator.

## 5. Observations available per step

- **In simulation (the first target):** the full plant state (qpos, qvel) after every control step, noise-free unless we add noise, together with the applied action and the actual joint torques.
- **On hardware (later):** joint encoders (q, q̇), IMU (gyro, orientation) and motor torques/currents. Base linear velocity and position come only from a state estimator, so they are noisy or biased.
- **Noise model.** The simulated plant is deterministic, so all prediction error in sim2sim comes from θ mismatch. A likelihood still needs a noise or structural-error model, to avoid degenerate collapse in sim and to cope with unmodelled dynamics on hardware.

## 6. What we want from you

1. **A recommended strategy**, justified against the constraints above. Rank BO, dual-layer MPPI and Kalman filtering (and any alternative you propose) for *this* setting, giving concrete reasons: dimensionality, budget, black-box simulator, static shapes, non-Gaussian and possibly multimodal posteriors, non-identifiable directions, contact discontinuities.
2. **The coupling between the two roles of the k samples.** They serve both **robustness**, i.e. tail coverage for the risk measure, and **estimation**, i.e. posterior representation. How do we prevent:
   - posterior collapse and overconfidence from destroying robustness;
   - robustness-driven spread from slowing identification?
   Should the planning set of θ differ from the estimation set, e.g. posterior samples plus a few tail or prior samples, or tempered likelihoods? How should the belief handle directions the data cannot identify: stay at the prior, or be inflated by process noise? Should θ be modelled as static or slowly drifting (μ may change with terrain)?
3. **Budget split.** How to choose m versus k within about 2000 rollouts per step, and whether k, α or the temperature should adapt online as the belief sharpens (under static shapes). Include how many samples a CVaR_α estimate needs to be meaningful, and how the risk aggregation interacts with the z-scored softmax.
4. **The estimator in detail.** Pseudocode for one control step that fits the JAX constraints (fixed shapes, jit, vmap), with:
   - the per-step rollout count;
   - the likelihood or noise model;
   - which state components to use and how to weight them (q vs q̇, base vs joints; units differ);
   - one-step versus windowed likelihood;
   - resampling or regularization;
   - initialization from the prior.
5. **Identifiability first.** A cheap experiment to run before building anything that measures which θ directions are identifiable from a few control steps of closed-loop data. For example: a finite-difference Fisher information of one-step or multi-step predictions around typical states, and a comparison with the control-relevance ranking above. Say how its result should change the design, e.g. estimate a reduced set and marginalize the rest by keeping them randomized.
6. **Active excitation / dual control.** Is it worth adding an information-gain term to the action score, and if so the cheapest version that fits the budget. Or should we rely on the natural excitation of walking?
7. **An evaluation plan** using our paired sim-to-sim harness, which runs arms on the same plant and seed.
   - Arms: nominal, oracle, risk-averse DR with a fixed prior, and DR plus online update. Suggest ablations.
   - Metrics: return, survival, belief error ‖θ̂ − θ‖ restricted to the identifiable directions, calibration of the belief, and wall-clock time per step.
   - How many trials the chaos noted above requires.
8. **Failure modes and diagnostics** to log, e.g. likelihood degeneracy, effective sample size of the θ particles, belief-to-truth coverage, and the numerical-instability regime at low armature or high gains, where some θ_j rollouts diverge.

Prefer something a researcher can implement in a few hundred lines of JAX and get working. State your assumptions explicitly, and separate what you are confident about from speculation.
