"""Shared setup for the armature-only investigation: one env build + one compile, then
closed-loop trials with independent plant and planner armature scales."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import mujoco
import yaml

import dial_mpc.envs as dial_envs
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.core.dial_core import MBDPI
from dial_mpc.sim2sim.randomize import ParamSpec, apply_theta, resolve_specs
from dial_mpc.sim2sim.runner import (
    DiffuseStepper, Model, PlannerStepper, PlantStepper, TrialResult, build_envs, run_trial,
)
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

ARMATURE_SPEC = {
    "armature": dict(target="sys", field="dof_armature", select="actuated", mode="scale",
                     range=[0.5, 2.0], dist="log_uniform"),
}


def load_configs(overrides: Optional[Dict[str, Any]] = None):
    cfg = yaml.safe_load(open(get_example_path("unitree_h1_loco_sim2sim.yaml")))
    cfg.update(overrides or {})
    dial_config = load_dataclass_from_dict(DialConfig, cfg)
    env_config = load_dataclass_from_dict(dial_envs.get_config(dial_config.env_name), cfg,
                                          convert_list_to_array=True)
    return cfg, dial_config, env_config


@dataclass
class Harness:
    dial_config: DialConfig
    stepper: PlantStepper
    planner: PlannerStepper
    mbdpi: MBDPI
    diffuse: DiffuseStepper
    specs: list[ParamSpec]

    def model(self, arm_scale: float) -> Model:
        theta = {"armature": jnp.array([arm_scale], dtype=jnp.float32)}
        n = self.stepper.nominal
        sys, kp, kd = apply_theta(n.sys, n.kp, n.kd, theta, self.specs)
        return Model(sys, kp, kd)

    def trial(self, plant_scale: float, planner_scale: float, seed: int) -> TrialResult:
        plant = self.model(plant_scale)
        planner = self.planner.model_from(*self.model(planner_scale))
        theta = {"armature": jnp.array([plant_scale])}
        return run_trial(self.dial_config, self.mbdpi, self.stepper, self.diffuse,
                         plant, planner, theta, jax.random.PRNGKey(seed))


def _implicit_damping(env) -> None:
    """Move the external PD's kd into dof_damping and enable eulerdamp, so all joint damping
    is integrated implicitly (the H1 loco XML disables eulerdamp and kd is applied outside
    MuJoCo, which makes the 20 ms Euler step unstable below ~0.65x armature)."""
    kd = jnp.asarray(env._config.kd)
    env.sys = env.sys.tree_replace({
        "dof_damping": env.sys.dof_damping.at[6:].add(kd),
        "opt.disableflags": env.sys.opt.disableflags & ~int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP),
    })
    env._config = dataclasses.replace(env._config, kd=jnp.zeros_like(kd))


def build(overrides: Optional[Dict[str, Any]] = None, phys_term: bool = False,
          implicit_damping: bool = False) -> Harness:
    _, dial_config, env_config = load_configs(overrides)
    planner_env, plant_env = build_envs(dial_config.env_name, env_config)
    planner_env.terminate_on_physical_limits = phys_term
    plant_env.terminate_on_physical_limits = phys_term
    if implicit_damping:
        _implicit_damping(planner_env)
        _implicit_damping(plant_env)
    stepper = PlantStepper(plant_env)
    n = stepper.nominal
    specs = resolve_specs(ARMATURE_SPEC, n.sys, {"kp": n.kp, "kd": n.kd})
    fields = sorted({s.field for s in specs if s.target == "sys"})
    planner = PlannerStepper(planner_env, n.sys, fields, n.kp, n.kd)
    mbdpi = MBDPI(dial_config, planner_env, model_step_fn=planner.step_fn)
    diffuse = DiffuseStepper(mbdpi, dial_config)
    return Harness(dial_config, stepper, planner, mbdpi, diffuse, specs)


RESULT_KEYS = ["return_sum", "return_mean", "steps_survived", "survived", "diverged",
               "frac_diverged", "plan_return_mean", "optimism_gap", "pred_err_1step",
               "vel_err", "yaw_rate_err", "torque_rms"]


def result_row(r: TrialResult) -> Dict[str, float]:
    return {k: float(getattr(r, k)) for k in RESULT_KEYS}
