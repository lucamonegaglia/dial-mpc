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
from typing import Callable, Dict, List, Mapping, Optional, Tuple, cast

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

# Signatures of the jitted closures below. `jax.jit` returns an opaque `Wrapped` whose
# return type is Any, which silently erases `State` at every call site; naming the
# signatures here is what lets a type checker follow `state` through the rollout loop.
StepFn = Callable[[System, jax.Array, jax.Array, State, jax.Array], State]
ResetFn = Callable[[System, jax.Array, jax.Array, jax.Array], State]
DiffuseFn = Callable[
    [jax.Array, jax.Array, State], Tuple[jax.Array, jax.Array, Dict[str, jax.Array]]
]


def build_envs(env_name: str, env_config: BaseEnvConfig) -> Tuple[BaseEnv, BaseEnv]:
    """Build two independent env instances from the same registered env/config.

    Two instances (rather than one env reused) so that mutating `plant_env.sys` inside
    the swap contextmanager can never leak into the planner env that `MBDPI` was built
    against.
    """
    # get_environment is typed as returning brax's generic `Env`; every env registered
    # by dial_mpc.envs derives from BaseEnv, which is the surface this harness uses.
    planner_env = cast(BaseEnv, brax_envs.get_environment(env_name, config=env_config))
    plant_env = cast(BaseEnv, brax_envs.get_environment(env_name, config=env_config))
    return planner_env, plant_env


class PlantStepper:
    """Wraps `plant_env` so its physics model and controller gains are swappable
    per-call without ever rebuilding the env object (which would force a recompile).
    """

    plant_env: BaseEnv
    nominal_sys: System
    nominal_kp: jax.Array
    nominal_kd: jax.Array
    torso_idx: int
    step_jit: StepFn
    reset_jit: ResetFn

    def __init__(self, plant_env: BaseEnv):
        env = plant_env
        self.plant_env = env
        self.nominal_sys = env.sys
        self.nominal_kp = jnp.asarray(env._config.kp)
        self.nominal_kd = jnp.asarray(env._config.kd)
        self._nominal_config = env._config

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

        # If joint_range was never overridden by the env subclass (i.e. it's just an
        # alias for physical_joint_range, as set in BaseEnv.__init__), it's safe to
        # keep it in sync with a randomized jnt_range. Envs that hand-tune joint_range
        # to a different (usually tighter) safety band -- e.g. UnitreeH1LocoEnv -- must
        # NOT have it overwritten by a physical-range refresh.
        self._refresh_joint_range = bool(
            np.array_equal(np.asarray(env.joint_range), np.asarray(env.physical_joint_range))
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
                # BaseEnv.__init__ snapshots these three from sys at construction time
                # (base_env.py:23-25); if the randomized fields feed them, refresh so
                # act2joint/act2tau/termination see the randomized model, not the
                # nominal one baked in at __init__.
                env.physical_joint_range = sys.jnt_range[1:]
                env.joint_torque_range = sys.actuator_ctrlrange
                if self._refresh_joint_range:
                    env.joint_range = sys.jnt_range[1:]
                yield env
            finally:
                env.sys, env._config = o_sys, o_cfg
                env.physical_joint_range, env.joint_range = o_pjr, o_jr
                env.joint_torque_range = o_jtr

        self._swap = _swap

        def _step(sys: System, kp: jax.Array, kd: jax.Array, state: State,
                  action: jax.Array) -> State:
            with self._swap(sys, kp, kd) as e:
                return e.step(state, action)

        # reset() calls pipeline_init(), which reads self.sys at *trace* time. Jitting
        # env.reset directly would bake in the nominal model, so every trial's initial
        # pipeline_state (contacts, derived dynamics quantities) would come from the
        # planner's model rather than the plant's -- the one place domain shift could
        # silently leak out of the plant. Route it through the same swap as step, which
        # also keeps it at a single trace across all parameter draws.
        def _reset(sys: System, kp: jax.Array, kd: jax.Array, rng: jax.Array) -> State:
            with self._swap(sys, kp, kd) as e:
                return e.reset(rng)

        self.step_jit = cast(StepFn, jax.jit(_step))
        self.reset_jit = cast(ResetFn, jax.jit(_reset))


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

        def reverse_scan(carry, factor):
            rng_, Y0_, state_ = carry
            rng_, Y0_, info_ = mbdpi.reverse_once(state_, rng_, Y0_, factor)
            return (rng_, Y0_, state_), info_

        def make_diffuse(n_diffuse: int) -> DiffuseFn:
            factors = mbdpi.sigma_control * dial_config.traj_diffuse_factor ** (jnp.arange(n_diffuse))[:, None]

            def diffuse(rng: jax.Array, Y0: jax.Array, state: State):
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
    plan_return_mean: float
    optimism_gap: float
    pred_err_1step: float
    vel_err: float
    yaw_rate_err: float
    torque_rms: float
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
    sys: System,
    kp: jax.Array,
    kd: jax.Array,
    theta: Mapping[str, jax.Array],
    rng: jax.Array,
    save_rollout: bool = False,
) -> TrialResult:
    """Run one closed-loop trial: nominal planner (inside `mbdpi`), plant driven by
    `(sys, kp, kd)`. Mirrors `dial_core.main()`'s loop (apply -> shift -> replan), plus
    termination handling and domain-shift metrics that the original lacks.

    `diffuse` must be built once (outside any per-trial loop) and reused across trials --
    see `DiffuseStepper`'s docstring for why that matters for performance.
    """
    n_steps = dial_config.n_steps

    def pipeline_of(st: State) -> PipelineState:
        """`State.pipeline_state` is Optional[...] on brax's generic env State, and is
        typed as the backend-agnostic `brax.base.State`. Every dial_mpc env runs the mjx
        backend, whose pipeline state also carries the `mjx.Data` fields (`ctrl`, `qpos`,
        `qacc`); narrowing here once keeps the metric code below both checked and
        navigable instead of silently Any."""
        ps = st.pipeline_state
        if ps is None:
            raise RuntimeError("env returned a State with no pipeline_state")
        return cast(PipelineState, ps)

    rng, rng_reset = jax.random.split(rng)
    state: State = stepper.reset_jit(sys, kp, kd, rng_reset)
    Y0 = jnp.zeros([dial_config.Hnode + 1, mbdpi.nu])

    return_sum = 0.0
    plan_returns: List[float] = []
    pred_errs: List[float] = []
    vel_errs: List[float] = []
    yaw_errs: List[float] = []
    torque_sq_sum = 0.0
    steps_survived = n_steps
    survived = True
    prev_xbar1: Optional[jax.Array] = None

    rollout_states: List[np.ndarray] = []
    rollout_actions: List[np.ndarray] = []

    for t in range(n_steps):
        action = Y0[0]
        state = stepper.step_jit(sys, kp, kd, state, action)

        ps = pipeline_of(state)
        r = float(state.reward)
        return_sum += r
        torque_sq_sum += float(jnp.sum(jnp.square(ps.ctrl)))

        vb, ab = _body_vel(ps, stepper.torso_idx)
        vel_errs.append(float(jnp.linalg.norm(vb[:2] - state.info["vel_tar"][:2])))
        yaw_errs.append(float(jnp.abs(ab[-1] - state.info["ang_vel_tar"][-1])))

        if prev_xbar1 is not None:
            pred_errs.append(float(jnp.linalg.norm(ps.x.pos - prev_xbar1)))

        if save_rollout:
            rollout_states.append(np.asarray(ps.q))
            rollout_actions.append(np.asarray(action))

        done = bool(state.done > 0.5)
        if done:
            steps_survived = t + 1
            survived = False
            break

        Y0 = mbdpi.shift(Y0)
        diffuse_fn = diffuse.diffuse_init_jit if t == 0 else diffuse.diffuse_jit
        rng, Y0, info = diffuse_fn(rng, Y0, state)
        plan_returns.append(float(info["rews"][-1].mean()))
        # xbar[-1] is the last diffusion iterate's weighted-mean predicted trajectory,
        # under the (nominal) PLANNER model, starting from the state just reached above.
        # xbar[-1][1] is its 1-step-ahead prediction -- compared against the plant's
        # actual next state at the top of the following iteration.
        prev_xbar1 = info["xbar"][-1][1]

    plan_return_mean = float(np.mean(plan_returns)) if plan_returns else 0.0
    return_mean = return_sum / steps_survived if steps_survived > 0 else 0.0

    rollout = None
    if save_rollout:
        rollout = {"q": np.stack(rollout_states), "action": np.stack(rollout_actions)}

    return TrialResult(
        theta={k: np.asarray(v) for k, v in theta.items()},
        return_sum=return_sum,
        return_mean=return_mean,
        steps_survived=steps_survived,
        survived=survived,
        plan_return_mean=plan_return_mean,
        optimism_gap=plan_return_mean - return_mean,
        pred_err_1step=float(np.mean(pred_errs)) if pred_errs else float("nan"),
        vel_err=float(np.mean(vel_errs)) if vel_errs else float("nan"),
        yaw_rate_err=float(np.mean(yaw_errs)) if yaw_errs else float("nan"),
        torque_rms=float(np.sqrt(torque_sq_sum / max(steps_survived, 1))),
        rollout=rollout,
    )
