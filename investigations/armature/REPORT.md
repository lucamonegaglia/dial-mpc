# Why does armature randomisation make the nominal planner look better than the true planner?

H1 loco, DIAL-MPC sim2sim harness, **armature-only** randomisation (all other θ axes nominal).
Δ = nominal-planner arm − matched (true-θ) planner arm on the same plant with the same MPC seed;
Δ > 0 means the nominal planner did better. Scripts and CSV/JSON results live in this directory.

## TL;DR

* **In no armature regime does the nominal planner beat the true planner on total return.**
  With 40 seeds per plant scale (0.5×–2.0×), mean Δreturn is ≤ 0 at every scale. With 32
  seeds and a numerically stable integrator it is significantly negative on *both* sides
  (−21 at 0.5×, −27 at 2.0×). In the cross grid (16 seeds per cell), return is best at or next
  to the matched planner armature for every plant scale.
* The apparent "armature ↑ ⇒ nominal relatively better" is a **regression slope created by the
  low-armature side**. Below ≈0.65× the plant's 20 ms Euler step is **numerically unstable**
  (ρ = 1.60 at 0.5×, a period-2 flip-flop mode). The 1.0× planner model cannot see this mode,
  excites it and loses ~70–80 steps of survival at 0.5–0.7×. The matched model sees the samples diverge and
  avoids them. Regressing Δ on armature therefore gives a positive slope (+27 return per
  ln-unit, p = 1e-4) even though Δ ≤ 0 everywhere. Fixing the discretisation (two 10 ms
  substeps, torque held) removes the slope entirely (−0.35 per ln-unit, p = 0.96) and turns
  Δ(armature) into a symmetric V.
* Two measurement effects amplify it:
  1. **The metric.** Per-step reward falls over an episode (0.82 → 0.38), so any per-step mean
     (`return_mean`) rewards the arm that dies first. Here that is the nominal arm: `d_rmean`
     is +0.37 at 0.5× and +0.08–0.09 at 1.4×/2.0× (p = 0.002), while survival is shorter.
  2. **The noise floor.** Termination on the narrow joint band makes episode length
     chaotic: rerunning the *identical* cell gives a Δreturn SD of 52–78, as large as the
     nominal-vs-true spread. With physical-limit termination every trial at ≥0.7× survives
     400 steps and the matched planner wins by a small, clean margin.
* Model error does behave as expected. The matched model ranks samples at the chaos ceiling
  (Spearman 0.91–0.98 vs 0.84–0.88 for the 1.0× model on mismatched plants, 0.59 at 0.5×). Its one-step plan scores
  better on the plant at every scale (significant at 0.5× and 2.0×).

## 1. Numerical instability at low armature (the dominant effect)

H1 loco integrates one semi-implicit Euler step of 20 ms per control step (`timestep == dt`).
The XML sets `eulerdamp="disable"`, and the PD torque `kp(q*−q) − kd q̇` is computed outside
MuJoCo. So stiffness **and all damping** (kd + dof_damping) are explicit, and armature adds
directly to the inertia that sets the discrete stability margin.

The table gives the spectral radius ρ (and the most negative real eigenvalue) of the exact
Jacobian of the MJX one-step map: PD plus physics, floating base, gravity and contact off
(`stability.py`, `stability_dt.py`).

| armature | as shipped (1×20 ms) | 2×10 ms, torque held | 2×10 ms, PD each substep | 1×20 ms, kd implicit (eulerdamp) |
|---|---|---|---|---|
| 0.35× | **2.40** (−2.40) | **1.46** (−1.46) | 0.93 | 0.93 |
| 0.5×  | **1.60** (−1.60) | 0.94 (−0.91) | 0.93 | 0.93 |
| 0.6×  | **1.22** (−1.22) | 0.94 (−0.63) | 0.93 | 0.93 |
| 0.7×  | 0.93 (−0.92) | 0.94 (−0.40) | 0.93 | 0.94 |
| 1.0×  | 0.93 (−0.28) | 0.94 (+0.25) | 0.93 | 0.94 |
| 2.0×  | 0.93 (+0.51) | 0.94 (+0.57) | 0.94 | 0.94 |

* As shipped, the joint loop is **unstable below ≈0.65×**. About 19 % of log-uniform[0.5, 2]
  draws land there. At 0.7× it is barely stable, with a lightly damped mode near Nyquist.
  A per-joint diagonal analysis predicts stability, so the unstable mode comes from joint
  coupling.
* The instability is a discretisation artefact: substepping or implicit damping removes it.
* Open-loop hold with a 1e-3 qvel kick (`standing.py`): flip-flop amplitude grows from 0.04
  to 28 at 0.4–0.5×. Joint-velocity lag-1 autocorrelation is negative at ≤0.65× and
  positive at ≥0.7×.
* Closed-loop signature (40 seeds): joint-velocity lag-1 autocorrelation at 0.5× is 0.18 for
  the nominal arm vs 0.42 for the matched arm. The high-frequency energy ratio is 1.70 vs
  1.06. The matched arm also chatters, but less: its rollouts show the divergence, so the
  MPPI softmax down-weights plans that excite the mode.
* The implicit-damping variant (kd folded into `dof_damping`, eulerdamp on) is **not** a
  usable fix in closed loop. Episodes end after ~10 steps with joint velocities up to 17.
  Likely cause: MJX computes contact forces without the implicit damping term and applies
  the correction afterwards, which is inconsistent when h·D/I ≈ 0.4–1.4. (`paired_impl.csv`; kept for reference only.)

## 2. Paired closed-loop results (as shipped, 40 seeds per scale)

| plant | Δreturn (± SE) | p | Δsteps | p | d_rmean | Δ per-step reward while both alive |
|---|---|---|---|---|---|---|
| 0.5× | −25.0 ± 6.2 | <1e-3 | −77 | <1e-3 | +0.365 | −0.003 |
| 0.6× | −48.2 ± 6.4 | <1e-3 | −79 | <1e-3 | +0.094 | −0.014 |
| 0.7× | −46.7 ± 9.8 | <1e-3 | −69 | <1e-3 | +0.048 | −0.003 |
| 0.8× | −31.5 ± 9.5 | 0.003 | −44 | 0.001 | +0.016 | −0.006 |
| 0.9× | −16.0 ± 10.8 | 0.20 | −13 | 0.48 | −0.021 | +0.006 |
| 1.0× (identical arms) | −2.9 ± 8.1 | 0.59 | −5 | 0.65 | −0.002 | – |
| 1.2× | −4.0 ± 12.1 | 0.84 | −17 | 0.58 | +0.059 | +0.044 (p=0.08) |
| 1.4× | −2.1 ± 10.6 | 0.92 | −19 | 0.22 | +0.092 | +0.066 (p=0.03) |
| 1.7× | −12.5 ± 9.6 | 0.20 | −19 | 0.13 | −0.017 | −0.028 |
| 2.0× | −9.0 ± 7.4 | 0.34 | −27 | 0.011 | +0.075 | +0.044 (p=0.015) |

The only place the nominal planner "wins" is per-step reward at 1.2/1.4/2.0×.
* It is almost entirely velocity tracking. The planner that underestimates inertia commands
  ~1.2–1.6 N·m less RMS torque and moves slightly slower during the slow velocity ramp.
* The sign is not consistent (1.7× goes the other way).
* It is worth ~+5 return per trial against ~−20 from earlier termination.
* It disappears with substepping (§4).

The highest-precision run (`--precision highest`, 16 seeds) matches the TF32 run: Δ at 0.5×
is −14 (p = 0.002), at 0.7× −33 (p = 0.02), and at 2.0× −24 (Δsteps −35, p = 0.04). TF32 is
not a factor.

## 3. The noise floor

Rerunning the identical (plant, planner, seed) cell twice (`noise.csv`, 24 seeds):

| plant | SD(Δreturn) between repeats | SD(Δsteps) | identical outcomes | SD of return across seeds |
|---|---|---|---|---|
| 0.7× | 76 | 99 | 0 % | 35 |
| 1.0× | 52 | 75 | 4 % | 44 |
| 1.4× | 78 | 107 | 0 % | 53 |

GPU f32 nondeterminism plus closed-loop chaos makes repeats **effectively independent draws**:
the SD of the repeat difference is roughly √2 × the across-seed SD. So pairing by seed buys no
variance reduction. The spread of the paired nominal-vs-true Δ at ≥0.8× (47–76) is no larger than this
floor. The narrow joint-sampling band used for termination is the amplifier (e.g. knee
[0, 1.5], ankle [−0.6, 0.4], hip yaw/roll ±0.2). With `terminate_on_physical_limits: true`
(`paired_phys.csv`; the torso checks stay active), every trial at 0.7–2.0× in both arms
survives all 400 steps. So at those scales the band, not falling, ends every episode. Δreturn there is small but clean and always
favours the matched planner: −0.6 at 0.7× (p = 0.013) and −0.95 at 2.0× (p = 0.009, vel_err
0.091 vs 0.077). At 0.5× the nominal planner dies within ~20 steps while the matched planner
survives 62 % of trials, which is the instability again.

In an 8-axis sweep, a per-axis standardized regression at this noise level will produce
spurious slopes, and the one real armature slope is the §1/§4 artefact.

## 4. How "armature ↑ ⇒ nominal relatively better" arises, and the causal test

Slope of Δ on ln(armature), 1.0× excluded:

| run | range | Δreturn slope per ln-unit | p | mean Δreturn |
|---|---|---|---|---|
| as shipped, 40 seeds | 0.5–2.0× | **+27.1** | 1e-4 | −21.7 |
| | ≥0.8× | +19.0 | 0.13 | −12.5 |
| | <0.8× | −66 | 0.04 | −40.0 |
| 2×10 ms substeps, 32 seeds | 0.5–2.0× | **−0.35** | 0.96 | −16.1 |
| | ≥0.8× (1.4, 2.0) | −64 | 0.035 | −15.1 |
| | <0.8× | +26 | 0.47 | −16.8 |

The positive slope exists only because the low side is catastrophically bad for the nominal
planner, not because the high side is good for it. With substeps, the slope on each side has
the V-shape sign: mismatch hurts more the further armature moves from 1.0×.

With substeps (`paired_sub2.csv`), the matched planner's return is flat across armature
(90–111). The nominal planner's return falls off in both directions: 80 / 94 / 92 / 90 / 91 / 69
at 0.5 / 0.6 / 0.7 / 1.0 / 1.4 / 2.0×. Δreturn is −21 (p = 0.02), −17 (p = 0.01), −12, +4, −4
and −27 (p = 0.005); Δsteps at 2.0× is −44 (p = 0.002). The high-armature per-step "advantage"
vanishes: Δ per-step reward while both arms are alive is within ±0.007 at ≥0.7×, and −0.036
(p ≤ 0.006) at 0.5–0.6×, i.e. in favour of the matched planner. Joint-velocity lag-1 at 0.5×
is 0.45 vs 0.50, so the chatter is gone. `d_rmean` at 2.0× is still +0.054, again purely from
earlier termination.

Cross grid (as shipped, 16 seeds, mean return; bold = best per row):

| plant \ planner | 0.7 | 0.85 | 1.0 | 1.2 | 1.4 | 2.0 |
|---|---|---|---|---|---|---|
| 0.7 | **87** | 60 | 37 | 28 | 42 | 25 |
| 1.0 | **110** | 101 | **110** | 89 | 77 | 54 |
| 1.4 | 70 | 85 | 87 | **105** | 104 | 91 |
| 2.0 | 81 | 76 | 66 | 85 | 88 | **91** |

Seed-centred return is highest at log(planner/plant) = 0 (+21 ± 5). Overestimating armature
by ≥ 0.5 in log costs 19–22. At a 0.7× plant every planner other than 0.7× is poor. At 1.4×
the 1.2 and 1.4 planners tie.

## 5. Planner-level evidence (sample-ranking landscape, 4 seeds, ~430 states)

`landscape.py` works on states taken from the matched closed loop:
* It draws the exact sample batch that `reverse_once` would draw.
* It scores the batch under each planner model and under the plant. A second plant scoring
  with qvel perturbed by ε = 1e-5 gives the chaos ceiling.
* It evaluates each model's one-step `reverse_once` plan on the plant.

| plant | Spearman: matched model | Spearman: 1.0× model | chaos ceiling | J(matched plan) − J(1.0× plan) | p |
|---|---|---|---|---|---|
| 0.5× | 0.91 | 0.59 | 0.90 | +0.212 | <1e-3 |
| 0.7× | 0.96 | 0.85 | 0.96 | +0.025 | 0.10 |
| 1.4× | 0.97 | 0.88 | 0.97 | +0.015 | 0.13 |
| 1.7× | 0.97 | 0.86 | 0.97 | +0.019 | 0.23 |
| 2.0× | 0.98 | 0.84 | 0.98 | +0.018 | 0.03 |

* The matched model reaches the chaos ceiling everywhere.
* ESS is 1.3–1.8, so each MPPI update is effectively an argmax. That keeps the per-step value
  gain of a better ranking small (+0.02 per step) relative to closed-loop noise.
* At 0.5× the matched model's sample weights are much flatter (ESS ≈ 15), yet its plan is
  still far better on the plant (+0.21). The 1.0× model's plan at 0.5× is worse than the
  incumbent (−0.18, p = 0.001).
* At 1.4× and 1.7×, the matched plan beats the incumbent (+0.02, p < 0.001) and the 1.0× plan
  does not (≈ 0).

## 6. Other checks

* **Stale derived constants.** `apply_theta` rescales `dof_armature` but not
  `body_invweight0` / `dof_invweight0` (set by mj_setConst, used for constraint
  regularisation). At 0.5× the true/stale ratio is 1.13 for the feet and 1.1–1.9 per dof; at
  2× it is 0.87 and 0.5–0.85 (`stale_consts.py`). Both arms carry the same stale values, so
  this is a model-consistency caveat, not a plant/planner mismatch.
* **Torque saturation** is ≤ 1 % in both arms at every scale, so it is not a factor.
* **Metric (already fixed upstream).** Commit e24e9c8 moved the headline metric from
  `return_mean` to `return_sum` for the reason in §2. Any per-step mean over the survived
  horizon carries the same survival-length bias.

## Recommendations

1. Make the plant numerically stable over the randomisation range. Use `timestep: 0.01`
   (2 substeps, torque held; stable down to 0.5×), or restrict armature to ≥ 0.7×. Otherwise
   the sweep partly measures integrator stability, not model error.
2. Report effects per θ bin, or with a V-shaped/|log θ| regressor, rather than a linear
   slope on θ. Mismatch hurts in both directions.
3. Consider physical-limit termination (or a wider band) for sim2sim comparisons. The narrow
   joint band turns ulp-level differences into ±100-step survival swings.
4. Keep `return_sum` as the headline, and treat per-step means as survival-biased.

## Reproduce

```
export PYTHONPATH=<worktree root>
python3 stability.py; python3 stability_dt.py; python3 standing.py; python3 stale_consts.py
python3 grid.py --tag paired --plant 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.7 2.0 --planner match 1.0 --seeds $(seq 0 39) --save-rollouts
python3 grid.py --tag paired_sub2 --timestep 0.01 --plant 1.0 0.5 0.6 0.7 1.4 2.0 --planner match 1.0 --seeds $(seq 0 31) --save-rollouts
python3 grid.py --tag noise --plant 1.0 1.4 0.7 --planner match --repeat 2 --seeds $(seq 0 23) --save-rollouts
python3 grid.py --tag cross --plant 1.0 1.4 2.0 0.7 --planner 0.7 0.85 1.0 1.2 1.4 2.0 --seeds $(seq 0 15)
python3 grid.py --tag paired_hi --precision highest --plant 0.5 0.7 1.0 1.4 2.0 --planner match 1.0 --seeds $(seq 0 15) --save-rollouts
python3 grid.py --tag paired_phys --phys-term --plant 0.5 0.7 1.0 1.4 2.0 --planner match 1.0 --seeds $(seq 0 15) --save-rollouts
python3 landscape.py --plants 0.5 0.7 1.0 1.4 1.7 2.0 --models 0.7 1.0 1.4 2.0 --n-states 20 --every 4 --seed S --out landscape_sS
python3 analyze_paired.py --tag <tag>; python3 analyze_grid.py; python3 analyze_landscape.py
```
