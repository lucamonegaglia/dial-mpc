from dataclasses import dataclass, field
from typing import Any, List, Sequence, Union

import numpy as np

import jax
import jax.numpy as jnp
from functools import partial

from brax import math
import brax.base as base
from brax.base import System
from brax import envs as brax_envs
from brax.envs.base import State
from brax.io import mjcf

import mujoco

from dial_mpc.envs.base_env import BaseEnv, BaseEnvConfig
from dial_mpc.utils.function_utils import global_to_body_velocity
from dial_mpc.utils.io_utils import get_model_path


@dataclass
class LimxTron1WFEnvConfig(BaseEnvConfig):
    # Per-actuator gains (abad_L, hip_L, knee_L, wheel_L, abad_R, hip_R, knee_R, wheel_R) of one
    # PD law, tau = kp * (q_tar - q) + kd * (qd_tar - qd). Legs hold a position (qd_tar = 0);
    # wheels have no position target (kp = 0) and track a velocity, so their kd is the wheel
    # velocity gain. Replaces the scalar kp/kd + `wheel_kd` of 40f9164: the sim2sim harness
    # randomizes kp/kd as vectors, and folding the wheel gain into kd lets that reach the wheels.
    kp: Union[float, jax.Array] = field(default_factory=lambda: jnp.array(
        [120.0, 120.0, 120.0, 0.0, 120.0, 120.0, 120.0, 0.0]
    ))
    kd: Union[float, jax.Array] = field(default_factory=lambda: jnp.array(
        [5.0, 5.0, 5.0, 0.8, 5.0, 5.0, 5.0, 0.8]
    ))
    max_wheel_vel: float = 20.0
    default_vx: float = 1.0
    default_vy: float = 0.0
    default_vyaw: float = 0.0
    ramp_up_time: float = 1.0
    default_height: float = 0.72


class LimxTron1WFEnv(BaseEnv):
    """LIMX Tron1 wheeled-foot biped (WF_TRON1A), velocity-tracking task.

    The 8 actuators split into 6 bounded leg joints driven as position targets and 2 continuous
    wheel joints driven as velocity targets, so ``act2joint``/``act2tau`` both diverge from
    ``BaseEnv``, whose mapping assumes every actuator has a bounded range.
    """

    def __init__(self, config: LimxTron1WFEnvConfig):
        super().__init__(config)

        if config.leg_control != "torque":
            raise ValueError(
                "LimxTron1WFEnv only supports leg_control='torque': the wheel joints are "
                "continuous and have no position target to send as a position command."
            )

        self._wheel_radius = 0.127
        # actuator order: abad_L, hip_L, knee_L, wheel_L, abad_R, hip_R, knee_R, wheel_R
        self._leg_idx = jnp.array([0, 1, 2, 4, 5, 6])
        self._wheel_idx = jnp.array([3, 7])

        self._torso_idx = mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "base_Link"
        )

        self._init_q = jnp.array(self.sys.mj_model.keyframe("home").qpos)
        self._default_pose = jnp.array(self.sys.mj_model.keyframe("home").qpos[7:])
        self._default_leg_pose = self._default_pose[self._leg_idx]

        # Terminate on the physical leg limits by default: the action band below is tight
        # enough that ordinary tracking error leaves it. `_leg_phys_range` is a property, not
        # a snapshot, because the sim2sim model swap rewrites physical_joint_range.
        self.terminate_on_physical_limits = True
        # Action sampling range: home pose +/- a per-joint half width (abad, hip, knee),
        # clipped to the physical limits. It must stay centred on the home pose so that a
        # zero action holds the nominal stance instead of yanking the legs off it.
        half_range = jnp.array([0.30, 0.50, 0.44, 0.30, 0.50, 0.44])
        self.joint_range = jnp.stack(
            [
                jnp.maximum(
                    self._default_leg_pose - half_range, self._leg_phys_range[:, 0]
                ),
                jnp.minimum(
                    self._default_leg_pose + half_range, self._leg_phys_range[:, 1]
                ),
            ],
            axis=-1,
        )

        wheel_site_id = [
            mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, s)
            for s in ("wheel_L", "wheel_R")
        ]
        assert not any(id_ == -1 for id_ in wheel_site_id), "Site not found."
        self._wheel_site_id = jnp.array(wheel_site_id)

    @property
    def _leg_phys_range(self) -> jax.Array:
        """Physical limits of the 6 leg joints; the wheel rows are meaningless (continuous)."""
        return self.physical_joint_range[self._leg_idx]

    @property
    def termination_joint_range(self) -> jax.Array:
        """Leg-only, since `joint_range` is leg-only and the wheels rotate without bound."""
        if self.terminate_on_physical_limits:
            return self._leg_phys_range
        return self.joint_range

    def make_system(self, config: LimxTron1WFEnvConfig) -> System:
        model_path = get_model_path("limx_tron1_wf", "mjx_scene_tron1_wf.xml")
        sys = mjcf.load(model_path)
        sys = sys.tree_replace({"opt.timestep": config.timestep})
        return sys

    @partial(jax.jit, static_argnums=(0,))
    def act2joint(self, act: jax.Array) -> jax.Array:
        """Map the 6 leg entries of the action to leg joint position targets."""
        act_normalized = (
            act[self._leg_idx] * self._config.action_scale + 1.0
        ) / 2.0  # normalize to [0, 1]
        joint_targets = self.joint_range[:, 0] + act_normalized * (
            self.joint_range[:, 1] - self.joint_range[:, 0]
        )
        joint_targets = jnp.clip(
            joint_targets, self._leg_phys_range[:, 0], self._leg_phys_range[:, 1]
        )
        return joint_targets

    @partial(jax.jit, static_argnums=(0,))
    def act2wheelvel(self, act: jax.Array) -> jax.Array:
        """Map the 2 wheel entries of the action to wheel angular velocity targets (rad/s)."""
        return (
            act[self._wheel_idx]
            * self._config.action_scale
            * self._config.max_wheel_vel
        )

    @partial(jax.jit, static_argnums=(0,))
    def act2tau(self, act: jax.Array, pipline_state) -> jax.Array:
        q = pipline_state.qpos[7:]
        qd = pipline_state.qvel[6:]

        # wheels: q_tar = q (no position error), legs: qd_tar = 0
        q_tar = q.at[self._leg_idx].set(self.act2joint(act))
        qd_tar = jnp.zeros_like(qd).at[self._wheel_idx].set(self.act2wheelvel(act))
        tau = self._config.kp * (q_tar - q) + self._config.kd * (qd_tar - qd)
        tau = jnp.clip(tau, self.joint_torque_range[:, 0], self.joint_torque_range[:, 1])
        return tau

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self._init_q, jnp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "pos_tar": jnp.array([0.0, 0.0, self._config.default_height]),
            "vel_tar": jnp.zeros(3),
            "ang_vel_tar": jnp.zeros(3),
            "yaw_tar": 0.0,
            "step": 0,
            "randomize_target": self._config.randomize_tasks,
        }

        obs = self._get_obs(pipeline_state, state_info)
        reward, done = jnp.zeros(2)
        metrics = {}
        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        rng, cmd_rng = jax.random.split(state.info["rng"], 2)

        # physics step
        ctrl = self.act2tau(action, state.pipeline_state)
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
        x, xd = pipeline_state.x, pipeline_state.xd

        # observation data
        obs = self._get_obs(pipeline_state, state.info)

        # switch to new target if randomize_target is True
        def dont_randomize():
            return (
                jnp.array([self._config.default_vx, self._config.default_vy, 0.0]),
                jnp.array([0.0, 0.0, self._config.default_vyaw]),
            )

        def randomize():
            return self.sample_command(cmd_rng)

        vel_tar, ang_vel_tar = jax.lax.cond(
            (state.info["randomize_target"]) & (state.info["step"] % 500 == 0),
            randomize,
            dont_randomize,
        )
        state.info["vel_tar"] = jnp.minimum(
            vel_tar * state.info["step"] * self.dt / self._config.ramp_up_time, vel_tar
        )
        state.info["ang_vel_tar"] = jnp.minimum(
            ang_vel_tar * state.info["step"] * self.dt / self._config.ramp_up_time,
            ang_vel_tar,
        )

        torso_idx = self._torso_idx - 1
        vb = global_to_body_velocity(xd.vel[torso_idx], x.rot[torso_idx])
        ab = global_to_body_velocity(
            xd.ang[torso_idx] * jnp.pi / 180.0, x.rot[torso_idx]
        )

        # velocity tracking. No gait reward: the wheels roll continuously, there is no swing phase.
        reward_vel = -jnp.sum((vb[:2] - state.info["vel_tar"][:2]) ** 2)
        reward_ang_vel = -jnp.sum((ab[2] - state.info["ang_vel_tar"][2]) ** 2)
        # stay upright
        vec_tar = jnp.array([0.0, 0.0, 1.0])
        vec = math.rotate(vec_tar, x.rot[torso_idx])
        reward_upright = -jnp.sum(jnp.square(vec - vec_tar))
        # hold ride height
        reward_height = -jnp.sum((x.pos[torso_idx, 2] - state.info["pos_tar"][2]) ** 2)
        # yaw orientation
        yaw_tar = (
            state.info["yaw_tar"]
            + state.info["ang_vel_tar"][2] * self.dt * state.info["step"]
        )
        yaw = math.quat_to_euler(x.rot[torso_idx])[2]
        d_yaw = yaw - yaw_tar
        reward_yaw = -jnp.square(jnp.atan2(jnp.sin(d_yaw), jnp.cos(d_yaw)))
        # keep the legs near the nominal crouch instead of folding up
        joint_angles = pipeline_state.q[7:]
        reward_pose = -jnp.sum(
            jnp.square(joint_angles[self._leg_idx] - self._default_leg_pose)
        )
        reward_alive = 1.0 - state.done
        # energy
        reward_energy = -jnp.sum(
            jnp.maximum(ctrl * pipeline_state.qvel[6:] / 160.0, 0.0) ** 2
        )

        reward = (
            reward_vel * 1.0
            + reward_ang_vel * 1.0
            + reward_upright * 0.5
            + reward_height * 1.0
            + reward_yaw * 0.3
            + reward_pose * 0.2
            + reward_energy * 0.01
            + reward_alive * 1.0
        )

        # done. Wheel DOFs are excluded from the range check: they rotate without bound.
        up = jnp.array([0.0, 0.0, 1.0])
        leg_angles = joint_angles[self._leg_idx]
        done = jnp.dot(math.rotate(up, x.rot[torso_idx]), up) < 0
        done |= jnp.any(leg_angles < self.termination_joint_range[:, 0])
        done |= jnp.any(leg_angles > self.termination_joint_range[:, 1])
        done |= x.pos[torso_idx, 2] < 0.35
        done = done.astype(jnp.float32)

        # state management
        state.info["step"] += 1
        state.info["rng"] = rng

        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
        return state

    def _get_obs(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
    ) -> jax.Array:
        x, xd = pipeline_state.x, pipeline_state.xd
        torso_idx = self._torso_idx - 1
        vb = global_to_body_velocity(xd.vel[torso_idx], x.rot[torso_idx])
        ab = global_to_body_velocity(
            xd.ang[torso_idx] * jnp.pi / 180.0, x.rot[torso_idx]
        )
        joint_angles = pipeline_state.qpos[7:]
        wheel_angles = joint_angles[self._wheel_idx]
        obs = jnp.concatenate(
            [
                state_info["vel_tar"],
                state_info["ang_vel_tar"],
                pipeline_state.ctrl,
                pipeline_state.qpos[:7],
                joint_angles[self._leg_idx],
                # wheel angle is unbounded, so it enters the observation as sin/cos
                jnp.sin(wheel_angles),
                jnp.cos(wheel_angles),
                vb,
                ab,
                pipeline_state.qvel[6:],
            ]
        )
        return obs

    def render(
        self,
        trajectory: List[base.State],
        camera: str | None = None,
        width: int = 240,
        height: int = 320,
    ) -> Sequence[np.ndarray]:
        camera = camera or "track"
        return super().render(trajectory, camera=camera, width=width, height=height)

    def sample_command(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        lin_vel_x = [-1.5, 1.5]  # min max [m/s]
        lin_vel_y = [-0.3, 0.3]  # min max [m/s]
        ang_vel_yaw = [-1.5, 1.5]  # min max [rad/s]

        _, key1, key2, key3 = jax.random.split(rng, 4)
        lin_vel_x = jax.random.uniform(
            key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1]
        )
        new_lin_vel_cmd = jnp.array([lin_vel_x[0], lin_vel_y[0], 0.0])
        new_ang_vel_cmd = jnp.array([0.0, 0.0, ang_vel_yaw[0]])
        return new_lin_vel_cmd, new_ang_vel_cmd


brax_envs.register_environment("limx_tron1_wf_walk", LimxTron1WFEnv)
