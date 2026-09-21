"""The plant/planner split and the closed-loop trial runner.

`dial_core.main()` uses one env object for both roles: the same `env.step` is both the
"real world" applied each iteration and what `MBDPI` rolls out in imagination, so model
error is identically zero. Here we build **two** env instances from the same `env_name`:

  - `planner_env` — nominal parameters, handed to `MBDPI`. Compiled once; never touched
    again, so the expensive 2048-sample vmapped rollout never recompiles.
  - `plant_env`   — the "real world". Its `sys` (and controller gains) are swapped in
    *inside* a jitted step function via a contextmanager, so `sys` becomes a traced
    argument rather than a Python closure constant. This is the same pattern brax's own
    `DomainRandomizationVmapWrapper` uses. Verified empirically: this traces exactly
    once no matter how many distinct parameter draws are stepped through it.

`dial_core.main()` is not imported from here for the rollout loop (it's entangled with
argparse/Flask/plotting); `MBDPI` is reused directly, and the ~25-line loop is
reimplemented with the two things the original lacks: termination handling
(`state.done` is computed by every env but never consulted) and domain-shift metrics.
"""

from __future__ import annotations

import contextlib
import dataclasses
from dataclasses import dataclass
from typing import Callable, ContextManager, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple, cast

import jax
import jax.numpy as jnp
import numpy as np

import brax.envs as brax_envs
from brax.base import System
from brax.envs.base import State
from brax.mjx.base import State as PipelineState

from dial_mpc.config.base_env_config import BaseEnvConfig
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.core.dial_core import MBDPI
from dial_mpc.envs.base_env import BaseEnv
from dial_mpc.utils.function_utils import global_to_body_velocity

class Model(NamedTuple):
    """A complete set of dynamics parameters: the MJX model plus the controller gains.

    A NamedTuple is a pytree, so it travels through `jax.jit` as a traced argument. Keeping
    the three in one node makes the plant and planner models structurally identical by
    construction, which is what lets both experiment arms share a single compilation.
    """

    sys: System
    kp: jax.Array
    kd: jax.Array


class PlannerModel(NamedTuple):
    """What the planner is told about the plant.

    Deliberately *not* a full `Model`. Passing an entire `System` as a traced argument
    costs a measured **1.61x** in steady-state MPC time, because XLA can no longer
    constant-fold the mass/inertia arrays into the 2048-sample MJX kernel. Only 5 of the
    System's 342 fields are ever randomized, so this carries just those, in a fixed field
    order, and the planner reconstructs the System inside the trace from a constant
    nominal. Measured cost of that version: **1.015x**. See `PlannerStepper`.

    `values` holds the randomized `sys` fields in `PlannerStepper.fields` order; the field
    *names* stay a Python constant on the stepper and must never enter this pytree.
    """

    values: Tuple[jax.Array, ...]
    kp: jax.Array
    kd: jax.Array


# `jax.jit` returns an opaque wrapper whose return type is Any, which erases `State` at
# every call site; these aliases are what let a type checker follow `state` through the loop.
StepFn = Callable[[Model, State, jax.Array], State]
ResetFn = Callable[[Model, jax.Array], State]
PlannerStepFn = Callable[[PlannerModel, State, jax.Array], State]
DiffuseFn = Callable[
    [PlannerModel, jax.Array, jax.Array, State],
    Tuple[jax.Array, jax.Array, Dict[str, jax.Array]],
]


def make_model_swap(env: BaseEnv) -> Callable[..., ContextManager[BaseEnv]]:
    """Build a contextmanager that temporarily installs a `Model` onto `env`.

    Entered *during tracing*, so `sys`/`kp`/`kd` become traced inputs of the enclosing jit
    rather than closure constants -- the whole reason the harness compiles once instead of
    once per parameter draw. Shared by the plant and the planner so the two can never
    drift apart.
    """
    # If joint_range was never overridden by the env subclass (i.e. it's just an alias for
    # physical_joint_range, as set in BaseEnv.__init__), it's safe to keep it in sync with
    # a randomized jnt_range. Envs that hand-tune joint_range to a different (usually
    # tighter) band -- e.g. UnitreeH1LocoEnv -- must NOT have it overwritten.
    refresh_joint_range = bool(
        np.array_equal(np.asarray(env.joint_range), np.asarray(env.physical_joint_range))
    )
    # The swap rewrites physical_joint_range wholesale from sys.jnt_range, so an env that
    # reshapes it after BaseEnv.__init__ (UnitreeH1PushCrateEnv drops its last row) would be
    # silently un-truncated -- and `termination_joint_range` would then mismatch
    # `joint_range`'s row count. Refuse up front rather than fail deep inside a trace.
    if len(env.physical_joint_range) != len(env.sys.jnt_range[1:]):
        raise ValueError(
            f"{type(env).__name__} reshapes physical_joint_range after BaseEnv.__init__ "
            f"({len(env.physical_joint_range)} rows vs {len(env.sys.jnt_range[1:])} in "
            "sys.jnt_range[1:]); the model swap cannot reconstruct it."
        )

    @contextlib.contextmanager
    def _swap(sys: System, kp: jax.Array, kd: jax.Array):
        o_sys = env.sys
        o_cfg = env._config
        o_pjr = env.physical_joint_range
        o_jr = env.joint_range
        o_jtr = env.joint_torque_range
        try:
            env.sys = sys
            env._config = dataclasses.replace(o_cfg, kp=kp, kd=kd)
            # BaseEnv.__init__ snapshots these from sys; refresh unconditionally so
            # act2joint/act2tau/termination see the swapped model rather than the
            # jnt_range/actuator_ctrlrange baked in at construction.
            env.physical_joint_range = sys.jnt_range[1:]
            env.joint_torque_range = sys.actuator_ctrlrange
            if refresh_joint_range:
                env.joint_range = sys.jnt_range[1:]
            yield env
        finally:
            env.sys, env._config = o_sys, o_cfg
            env.physical_joint_range, env.joint_range = o_pjr, o_jr
            env.joint_torque_range = o_jtr

    return _swap


def build_envs(env_name: str, env_config: BaseEnvConfig) -> Tuple[BaseEnv, BaseEnv]:
    """Two independent env instances, so the plant's attribute swap can never leak into
    the planner env that `MBDPI` was built against."""
    # get_environment is typed as brax's generic `Env`; every dial_mpc env derives BaseEnv.
    planner_env = cast(BaseEnv, brax_envs.get_environment(env_name, config=env_config))
    plant_env = cast(BaseEnv, brax_envs.get_environment(env_name, config=env_config))
    return planner_env, plant_env


class PlantStepper:
    """Wraps `plant_env` so its physics model and controller gains are swappable
    per-call without ever rebuilding the env object (which would force a recompile).
    """

    plant_env: BaseEnv
    nominal: Model
    torso_idx: int
    step_jit: StepFn
    reset_jit: ResetFn

    def __init__(self, plant_env: BaseEnv):
        env = plant_env
        self.plant_env = env
        self.nominal = Model(
            env.sys, jnp.asarray(env._config.kp), jnp.asarray(env._config.kd)
        )

        # `_torso_idx` is set by the concrete env subclass, not by BaseEnv, so resolve it
        # once here with a clear failure instead of reaching through `env` at every
        # metric call site (where a missing attribute would surface mid-rollout).
        torso_idx = getattr(env, "_torso_idx", None)
        if torso_idx is None:
            raise AttributeError(
                f"{type(env).__name__} has no `_torso_idx`; the sim2sim metrics need a "
                "torso body index to compute body-frame velocity errors."
            )
        self.torso_idx = int(torso_idx)

        self._swap = make_model_swap(env)

        def _step(model: Model, state: State, action: jax.Array) -> State:
            with self._swap(*model) as e:
                return e.step(state, action)

        # reset() calls pipeline_init(), which reads self.sys at *trace* time. Jitting
        # env.reset directly would bake in the nominal model, so every trial's initial
        # pipeline_state (contacts, derived dynamics quantities) would come from the
        # planner's model rather than the plant's -- the one place domain shift could
        # silently leak out of the plant. Route it through the same swap as step, which
        # also keeps it at a single trace across all parameter draws.
        def _reset(model: Model, rng: jax.Array) -> State:
            with self._swap(*model) as e:
                return e.reset(rng)

        self.step_jit = cast(StepFn, jax.jit(_step))
        self.reset_jit = cast(ResetFn, jax.jit(_reset))


class PlannerStepper:
    """The planner-side twin of `PlantStepper`.

    Exposes `step_fn(model, state, action)` whose dynamics parameters are traced
    arguments, so `MBDPI` can roll out its 2048-sample imagination under an arbitrary
    parameter set without rebuilding anything. That is what makes "what if the planner
    knew the true parameters?" a per-trial *argument* rather than a per-trial recompile.

    Only the randomized fields travel as tracers; everything else in the System is closed
    over as a Python constant so XLA can still fold it into the kernel. Measured at
    Ndiffuse=1 on an RTX 4090, median over 10 distinct draws:

        baked-in constant model (today's MBDPI)   11.57 ms   1.000x
        full System passed as a traced argument   18.58 ms   1.606x
        this class (5 randomized fields traced)   11.75 ms   1.015x

    `nominal_sys` must be the **plant's** System object, not `planner_env.sys`. The two
    envs are built separately and therefore hold distinct `mj_model` objects, and
    `mj_model` is a *static* pytree field -- so models built from different envs have
    different treedefs and would compile twice, one per experiment arm, silently making
    the arms incomparable.

    There is deliberately no `reset` (the planner always starts from the plant's current
    pipeline_state), and `step_fn` is deliberately not jitted: it is only ever called
    inside `DiffuseStepper`'s jit, where it is traced once as part of the scan body.
    """

    planner_env: BaseEnv
    fields: Tuple[str, ...]
    nominal: PlannerModel
    step_fn: PlannerStepFn

    def __init__(self, planner_env: BaseEnv, nominal_sys: System,
                 fields: Sequence[str], nominal_kp: jax.Array, nominal_kd: jax.Array):
        env = planner_env
        self.planner_env = env
        self.fields = tuple(fields)
        self._nominal_sys = nominal_sys
        self.nominal = self.model_from(nominal_sys, nominal_kp, nominal_kd)
        self._swap = make_model_swap(env)

        def step(model: PlannerModel, state: State, action: jax.Array) -> State:
            sys = self._nominal_sys.tree_replace(dict(zip(self.fields, model.values)))
            with self._swap(sys, model.kp, model.kd) as e:
                return e.step(state, action)

        self.step_fn = step

    def model_from(self, sys: System, kp: jax.Array, kd: jax.Array) -> PlannerModel:
        """Project a full `Model`/System down to just the randomized fields."""
        return PlannerModel(
            tuple(cast(jax.Array, getattr(sys, f)) for f in self.fields), kp, kd
        )


class DiffuseStepper:
    """Wraps `mbdpi.reverse_once` into two persistent jitted functions -- one for
    `Ndiffuse_init` steps (used at t=0), one for `Ndiffuse` steps (used at t>0) -- built
    exactly once and reused across every MPC step of every trial.

    This exists because `jax.lax.scan(reverse_scan, ...)` called directly (not wrapped in
    an enclosing `jax.jit`) retraces its body from scratch on every call if `reverse_scan`
    is a fresh Python closure each time. `dial_core.main()` never hits this because it
    only builds `reverse_scan` once for its single run; a multi-trial harness that
    rebuilt it inside `run_trial()` would retrace -- and therefore recompile the whole
    diffusion scan -- on every single trial. Building it here once, outside the trial
    loop, is what makes the "compiles once" performance design (see the plan's cost
    section) actually hold across a sweep, not just within one trial.
    """

    mbdpi: MBDPI
    dial_config: DialConfig
    diffuse_init_jit: DiffuseFn
    diffuse_jit: DiffuseFn

    def __init__(self, mbdpi: MBDPI, dial_config: DialConfig):
        self.mbdpi = mbdpi
        self.dial_config = dial_config

        def make_diffuse(n_diffuse: int) -> DiffuseFn:
            factors = mbdpi.sigma_control * dial_config.traj_diffuse_factor ** (jnp.arange(n_diffuse))[:, None]

            def diffuse(model: PlannerModel, rng: jax.Array, Y0: jax.Array, state: State):
                # `reverse_scan` lives in here so it can close over `model`. That closure
                # is rebuilt per call, but harmlessly: the jit cache is keyed on `diffuse`,
                # which is created exactly once below. (Rebuilding a scan body per call
                # only causes retracing when the scan runs *outside* a jit -- the failure
                # this class was written to prevent.) Passing the model through the closure
                # rather than the scan carry also keeps ~300 model leaves out of the carry.
                def reverse_scan(carry, factor):
                    rng_, Y0_, state_ = carry
                    rng_, Y0_, info_ = mbdpi.reverse_once(state_, rng_, Y0_, factor, model)
                    return (rng_, Y0_, state_), info_

                (rng, Y0, _), info = jax.lax.scan(reverse_scan, (rng, Y0, state), factors)
                return rng, Y0, info

            return cast(DiffuseFn, jax.jit(diffuse))

        self.diffuse_init_jit = make_diffuse(dial_config.Ndiffuse_init)
        self.diffuse_jit = make_diffuse(dial_config.Ndiffuse)


@dataclass
class TrialResult:
    theta: Dict[str, np.ndarray]
    return_sum: float
    return_mean: float
    steps_survived: int
    survived: bool
    # Tracked separately from `survived`: a NaN reward makes `done` NaN too, and
    # `bool(nan > 0.5)` is False, so a diverged trial would otherwise look like a perfect one.
    diverged: bool
    # Mean fraction of the planner's sampled rollouts rejected as non-finite, per MPC step.
    frac_diverged: float
    plan_return_mean: float
    optimism_gap: float
    pred_err_1step: float
    vel_err: float
    yaw_rate_err: float
    torque_rms: float
    # Always populated; the sweep decides which trials are worth writing to disk.
    rollout: Optional[Dict[str, np.ndarray]] = None


def _body_vel(pipeline_state: PipelineState, torso_idx: int) -> Tuple[jax.Array, jax.Array]:
    """Body-frame linear and angular velocity of the torso.

    NOTE: this deliberately differs from the envs' own reward code, which writes
    `xd.ang[...] * jnp.pi / 180.0` (unitree_h1_env.py:272,333,510,792,860 and the go2
    equivalents). `xd.ang` is already rad/s in brax, so that factor is an upstream
    unit bug that shrinks the angular term by ~57.3x. Reproducing it here would make
    `yaw_rate_err` report a number ~57x smaller than the actual rad/s error. The reward
    is left exactly as the repo wrote it (changing it would change the task); this
    metric reports true rad/s.

    For the default `unitree_h1_loco` config this is a pure rescaling of the reported
    metric -- `default_vyaw: 0.0`, so `ang_vel_tar[2]` stays 0 and only the magnitude,
    not the comparison, was affected.
    """
    x, xd = pipeline_state.x, pipeline_state.xd
    vb = global_to_body_velocity(xd.vel[torso_idx - 1], x.rot[torso_idx - 1])
    ab = global_to_body_velocity(xd.ang[torso_idx - 1], x.rot[torso_idx - 1])
    return vb, ab


def run_trial(
    dial_config: DialConfig,
    mbdpi: MBDPI,
    stepper: PlantStepper,
    diffuse: DiffuseStepper,
    plant_model: Model,
    planner_model: PlannerModel,
    theta: Mapping[str, jax.Array],
    rng: jax.Array,
) -> TrialResult:
    """Run one closed-loop trial of `plant_model`, controlled by MPC that plans with
    `planner_model`.

    The two arms of the experiment differ *only* in `planner_model`: pass the nominal
    parameters for the mismatched arm, or `plant_model` itself for the matched arm in
    which the planner is told the plant's true parameters. Both models have identical
    pytree structure, so both arms hit the same compiled code.

    Mirrors `dial_core.main()`'s loop (apply -> shift -> replan), plus termination handling
    and domain-shift metrics that the original lacks. `diffuse` must be built once (outside
    any per-trial loop) and reused across trials -- see `DiffuseStepper`'s docstring for why
    that matters for performance.
    """
    n_steps = dial_config.n_steps

    def pipeline_of(st: State) -> PipelineState:
        """Narrow brax's Optional, backend-agnostic pipeline state to the mjx one, whose
        `ctrl`/`qpos`/`qvel` the metrics below need."""
        ps = st.pipeline_state
        if ps is None:
            raise RuntimeError("env returned a State with no pipeline_state")
        return cast(PipelineState, ps)

    rng, rng_reset = jax.random.split(rng)
    state: State = stepper.reset_jit(plant_model, rng_reset)
    Y0 = jnp.zeros([dial_config.Hnode + 1, mbdpi.nu])

    return_sum = 0.0
    plan_returns: List[float] = []
    pred_errs: List[float] = []
    vel_errs: List[float] = []
    yaw_errs: List[float] = []
    torque_sq_sum = 0.0
    steps_survived = n_steps
    survived = True
    diverged = False
    frac_div: List[float] = []
    prev_qbar1: Optional[jax.Array] = None

    # Recorded every control step (env.dt = 0.02 s -> 50 Hz for the H1 configs), which
    # is the rate the MPC actually commands at; there is no sub-step logging because
    # nothing in the closed loop changes faster than this.
    log: Dict[str, List[np.ndarray]] = {
        "qpos": [], "qvel": [], "action": [], "ctrl": [],
        "reward": [], "done": [], "torso_pos": [], "torso_quat": [],
        "vel_body": [], "vel_tar": [], "ang_vel_tar": [],
    }

    for t in range(n_steps):
        action = Y0[0]
        state = stepper.step_jit(plant_model, state, action)

        ps = pipeline_of(state)
        r = float(state.reward)
        qpos, qvel = np.asarray(ps.qpos), np.asarray(ps.qvel)
        if not (np.isfinite(r) and np.isfinite(qpos).all() and np.isfinite(qvel).all()):
            # The plant received a non-finite action, or integrated to a non-finite state.
            # Stop here and record it as a divergence rather than a survival.
            diverged = True
            steps_survived = t
            survived = False
            break
        return_sum += r
        torque_sq_sum += float(jnp.mean(jnp.square(ps.ctrl)))

        vb, ab = _body_vel(ps, stepper.torso_idx)
        vel_errs.append(float(jnp.linalg.norm(vb[:2] - state.info["vel_tar"][:2])))
        yaw_errs.append(float(jnp.abs(ab[-1] - state.info["ang_vel_tar"][-1])))

        if prev_qbar1 is not None:
            pred_errs.append(float(jnp.linalg.norm(ps.q - prev_qbar1)))

        log["qpos"].append(qpos)
        log["qvel"].append(qvel)
        log["action"].append(np.asarray(action))
        log["ctrl"].append(np.asarray(ps.ctrl))
        log["reward"].append(np.asarray(r, dtype=np.float32))
        log["done"].append(np.asarray(state.done))
        log["torso_pos"].append(np.asarray(ps.x.pos[stepper.torso_idx - 1]))
        log["torso_quat"].append(np.asarray(ps.x.rot[stepper.torso_idx - 1]))
        log["vel_body"].append(np.asarray(vb))
        log["vel_tar"].append(np.asarray(state.info["vel_tar"]))
        log["ang_vel_tar"].append(np.asarray(state.info["ang_vel_tar"]))

        done = bool(state.done > 0.5)
        if done:
            steps_survived = t + 1
            survived = False
            break

        Y0 = mbdpi.shift(Y0)
        diffuse_fn = diffuse.diffuse_init_jit if t == 0 else diffuse.diffuse_jit
        rng, Y0, info = diffuse_fn(planner_model, rng, Y0, state)
        plan_returns.append(float(info["rew_plan"][-1]))
        frac_div.append(float(np.asarray(info["frac_diverged"]).mean()))
        # The planner's 1-step prediction of the configuration the plant will reach next,
        # under whichever PLANNER model this trial was given: qbar[-1] is the last diffusion
        # iterate's weighted-mean predicted trajectory from the state just reached above, and
        # its index 0 is the configuration after applying us[0] == Y0[0]
        prev_qbar1 = info["qbar"][-1][0]

    plan_return_mean = float(np.mean(plan_returns)) if plan_returns else 0.0
    return_mean = return_sum / steps_survived if steps_survived > 0 else 0.0
    if diverged and steps_survived == 0:
        return_mean = float("nan")

    rollout: Dict[str, np.ndarray] = {k: np.stack(v) for k, v in log.items() if v}
    rollout["time"] = np.arange(len(log["reward"]), dtype=np.float64) * float(stepper.plant_env.dt)

    return TrialResult(
        theta={k: np.asarray(v) for k, v in theta.items()},
        return_sum=return_sum,
        return_mean=return_mean,
        steps_survived=steps_survived,
        survived=survived,
        diverged=diverged,
        frac_diverged=float(np.mean(frac_div)) if frac_div else 0.0,
        plan_return_mean=plan_return_mean,
        optimism_gap=plan_return_mean - return_mean,
        pred_err_1step=float(np.mean(pred_errs)) if pred_errs else float("nan"),
        vel_err=float(np.mean(vel_errs)) if vel_errs else float("nan"),
        yaw_rate_err=float(np.mean(yaw_errs)) if yaw_errs else float("nan"),
        torque_rms=float(np.sqrt(torque_sq_sum / max(steps_survived, 1))),
        rollout=rollout,
    )
