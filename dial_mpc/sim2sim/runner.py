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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

import brax.envs as brax_envs

from dial_mpc.core.dial_config import DialConfig
from dial_mpc.core.dial_core import MBDPI
from dial_mpc.utils.function_utils import global_to_body_velocity


def build_envs(env_name: str, env_config) -> tuple[Any, Any]:
    """Build two independent env instances from the same registered env/config.

    Two instances (rather than one env reused) so that mutating `plant_env.sys` inside
    the swap contextmanager can never leak into the planner env that `MBDPI` was built
    against.
    """
    planner_env = brax_envs.get_environment(env_name, config=env_config)
    plant_env = brax_envs.get_environment(env_name, config=env_config)
    return planner_env, plant_env


@dataclass
class PlantStepper:
    """Wraps `plant_env` so its physics model and controller gains are swappable
    per-call without ever rebuilding the env object (which would force a recompile).
    """

    plant_env: Any
    nominal_sys: Any = field(init=False)
    nominal_kp: jax.Array = field(init=False)
    nominal_kd: jax.Array = field(init=False)
    _nominal_config: Any = field(init=False, repr=False)
    _refresh_joint_range: bool = field(init=False)
    step_jit: Any = field(init=False, repr=False)
    reset_jit: Any = field(init=False, repr=False)

    def __post_init__(self):
        env = self.plant_env
        self.nominal_sys = env.sys
        self.nominal_kp = jnp.asarray(env._config.kp)
        self.nominal_kd = jnp.asarray(env._config.kd)
        self._nominal_config = env._config
        # If joint_range was never overridden by the env subclass (i.e. it's just an
        # alias for physical_joint_range, as set in BaseEnv.__init__), it's safe to
        # keep it in sync with a randomized jnt_range. Envs that hand-tune joint_range
        # to a different (usually tighter) safety band -- e.g. UnitreeH1LocoEnv -- must
        # NOT have it overwritten by a physical-range refresh.
        self._refresh_joint_range = bool(
            np.array_equal(np.asarray(env.joint_range), np.asarray(env.physical_joint_range))
        )
        @contextlib.contextmanager
        def _swap(sys, kp, kd):
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

        def _step(sys, kp, kd, state, action):
            with self._swap(sys, kp, kd) as e:
                return e.step(state, action)

        # reset() calls pipeline_init(), which reads self.sys at *trace* time. Jitting
        # env.reset directly would bake in the nominal model, so every trial's initial
        # pipeline_state (contacts, derived dynamics quantities) would come from the
        # planner's model rather than the plant's -- the one place domain shift could
        # silently leak out of the plant. Route it through the same swap as step, which
        # also keeps it at a single trace across all parameter draws.
        def _reset(sys, kp, kd, rng):
            with self._swap(sys, kp, kd) as e:
                return e.reset(rng)

        self.step_jit = jax.jit(_step)
        self.reset_jit = jax.jit(_reset)


@dataclass
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
    diffuse_init_jit: Any = field(init=False, repr=False)
    diffuse_jit: Any = field(init=False, repr=False)

    def __post_init__(self):
        mbdpi = self.mbdpi
        dial_config = self.dial_config

        def reverse_scan(carry, factor):
            rng_, Y0_, state_ = carry
            rng_, Y0_, info_ = mbdpi.reverse_once(state_, rng_, Y0_, factor)
            return (rng_, Y0_, state_), info_

        def make_diffuse(n_diffuse: int):
            factors = mbdpi.sigma_control * dial_config.traj_diffuse_factor ** (jnp.arange(n_diffuse))[:, None]

            def diffuse(rng, Y0, state):
                (rng, Y0, _), info = jax.lax.scan(reverse_scan, (rng, Y0, state), factors)
                return rng, Y0, info

            return jax.jit(diffuse)

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


def _body_vel(env, pipeline_state):
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
    vb = global_to_body_velocity(xd.vel[env._torso_idx - 1], x.rot[env._torso_idx - 1])
    ab = global_to_body_velocity(xd.ang[env._torso_idx - 1], x.rot[env._torso_idx - 1])
    return vb, ab


def run_trial(
    dial_config: DialConfig,
    mbdpi: MBDPI,
    stepper: PlantStepper,
    diffuse: DiffuseStepper,
    sys,
    kp,
    kd,
    theta: Dict[str, np.ndarray],
    rng: jax.Array,
    save_rollout: bool = False,
) -> TrialResult:
    """Run one closed-loop trial: nominal planner (inside `mbdpi`), plant driven by
    `(sys, kp, kd)`. Mirrors `dial_core.main()`'s loop (apply -> shift -> replan), plus
    termination handling and domain-shift metrics that the original lacks.

    `diffuse` must be built once (outside any per-trial loop) and reused across trials --
    see `DiffuseStepper`'s docstring for why that matters for performance.
    """
    env = stepper.plant_env
    n_steps = dial_config.n_steps

    rng, rng_reset = jax.random.split(rng)
    state = stepper.reset_jit(sys, kp, kd, rng_reset)
    Y0 = jnp.zeros([dial_config.Hnode + 1, mbdpi.nu])

    return_sum = 0.0
    plan_returns: List[float] = []
    pred_errs: List[float] = []
    vel_errs: List[float] = []
    yaw_errs: List[float] = []
    torque_sq_sum = 0.0
    steps_survived = n_steps
    survived = True
    prev_xbar1 = None

    rollout_states = [] if save_rollout else None
    rollout_actions = [] if save_rollout else None

    for t in range(n_steps):
        action = Y0[0]
        state = stepper.step_jit(sys, kp, kd, state, action)

        r = float(state.reward)
        return_sum += r
        torque_sq_sum += float(jnp.sum(jnp.square(state.pipeline_state.ctrl)))

        vb, ab = _body_vel(env, state.pipeline_state)
        vel_errs.append(float(jnp.linalg.norm(vb[:2] - state.info["vel_tar"][:2])))
        yaw_errs.append(float(jnp.abs(ab[-1] - state.info["ang_vel_tar"][-1])))

        if prev_xbar1 is not None:
            pred_errs.append(float(jnp.linalg.norm(state.pipeline_state.x.pos - prev_xbar1)))

        if save_rollout:
            rollout_states.append(np.asarray(state.pipeline_state.q))
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
