"""dial-mpc-sim2sim-reproduce: replay specific historical sim2sim trials.

`dial-mpc-sim2sim-eval` only writes full 50 Hz state logs for the (small) subset of
trials `_select_interesting` picked; every other trial's exact rollout is gone once the
sweep ends. It is nonetheless fully reproducible: both arms of a trial are a pure
function of `(config_used.yaml, theta, seed)`, and `trials.csv` records the seed plus
every sampled theta component as flattened columns (see `sweep._flatten_theta`). This
script rebuilds theta from that row (or from `interesting/index.json`, if the trial
happens to already be saved there), re-runs both arms with `runner.run_trial`, and
writes the state logs into `<run_dir>/reproduced/trial_NNNN/` -- the same per-trial-
directory shape `sweep._write_interesting` uses under `interesting/`, so a reproduction
can never overwrite, or be mistaken for, the sweep's own selection, and
`view.generate_trial_outputs` can turn either one into a compare.png / html the same way.

`--trial` takes one or more indices; the env build and JIT compilation (the expensive
part) happen once and are reused across all of them, since they depend only on
`config_used.yaml`, not on any particular trial's theta/seed.

Usage:
    dial-mpc-sim2sim-reproduce --run <run_dir> --trial 168
    dial-mpc-sim2sim-reproduce --run <run_dir> --trial 168 --html
    dial-mpc-sim2sim-reproduce --run <run_dir> --trial 168 --html --force
    dial-mpc-sim2sim-reproduce --run <run_dir> --trial 168 248 118 204
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import yaml

import dial_mpc.envs as dial_envs
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.core.dial_core import MBDPI
from dial_mpc.utils.io_utils import load_dataclass_from_dict

# `dial_core` sets `plt.style.use("science")` on import, which turns on `text.usetex` --
# fine for `dial-mpc-sim2sim-eval` (never plots in-process) but this script also calls
# `generate_trial_outputs` in-process, and a LaTeX install is not a repo dependency.
# Revert just that rcParam rather than the whole style, since the rest of it is harmless.
plt.rcParams["text.usetex"] = False

from dial_mpc.sim2sim import determinism
from dial_mpc.sim2sim.groups import GROUP_NOMINAL_PLANNER, GROUP_TRUE_PLANNER
from dial_mpc.sim2sim.randomize import ParamSpec, apply_theta, load_domain_rand_config, resolve_specs
from dial_mpc.sim2sim.runner import (
    DiffuseStepper, Model, PlannerStepper, PlantStepper, build_envs, run_trial,
)
from dial_mpc.sim2sim.view import generate_trial_outputs


def _unflatten_theta(row: Dict[str, str], specs: List[ParamSpec]) -> Dict[str, np.ndarray]:
    """Inverse of `sweep._flatten_theta`: rebuild {name: array} from one trials.csv row."""
    theta: Dict[str, np.ndarray] = {}
    for spec in specs:
        shape = spec.elem_shape if spec.per_element else (1,)
        size = int(np.prod(shape)) if shape else 1
        if size == 1:
            # `_flatten_theta` writes a bare `name` column for any 1-element array,
            # per_element or not -- there is no `_0` suffix to look for.
            theta[spec.name] = np.asarray([float(row[spec.name])])
        else:
            arr = np.zeros(shape, dtype=np.float64)
            for idx in np.ndindex(shape):
                suffix = "_".join(str(i) for i in idx)
                arr[idx] = float(row[f"{spec.name}_{suffix}"])
            theta[spec.name] = arr
    return theta


def _load_seed_and_theta(run_dir: str, trial: int, specs: List[ParamSpec]) -> tuple:
    """Prefer `interesting/index.json` (theta already parsed) if this trial is in it;
    otherwise reconstruct theta from `trials.csv` via `_unflatten_theta`."""
    index_path = os.path.join(run_dir, "interesting", "index.json")
    if os.path.exists(index_path):
        manifest = json.load(open(index_path))
        entry = next((e for e in manifest["entries"] if e["trial"] == trial), None)
        if entry is not None:
            theta = {k: np.asarray(v) for k, v in entry["theta"].items()}
            return int(entry["seed"]), theta

    csv_path = os.path.join(run_dir, "trials.csv")
    rows = [r for r in csv.DictReader(open(csv_path)) if int(r["trial"]) == trial]
    if not rows:
        raise SystemExit(f"Trial {trial} not found in {csv_path}")
    return int(rows[0]["seed"]), _unflatten_theta(rows[0], specs)


def _already_present(trial_dir: str) -> bool:
    return all(
        os.path.exists(os.path.join(trial_dir, f"{group}.npz"))
        for group in (GROUP_NOMINAL_PLANNER, GROUP_TRUE_PLANNER)
    )


@dataclass
class ReproContext:
    """Everything a reproduction needs that depends only on `config_used.yaml`, not on
    any particular trial -- built once and reused across a `--trial` batch."""

    run_dir: str
    dial_config: DialConfig
    mbdpi: MBDPI
    stepper: PlantStepper
    diffuse: DiffuseStepper
    planner: PlannerStepper
    specs: List[ParamSpec]


def build_context(run_dir: str) -> ReproContext:
    config_path = os.path.join(run_dir, "config_used.yaml")
    if not os.path.exists(config_path):
        raise SystemExit(f"Need {config_path} to rebuild the env this run used.")
    config_dict = yaml.safe_load(open(config_path))
    dial_config = load_dataclass_from_dict(DialConfig, config_dict)
    env_config_type = dial_envs.get_config(dial_config.env_name)
    env_config = load_dataclass_from_dict(env_config_type, config_dict, convert_list_to_array=True)
    drc = load_domain_rand_config(config_dict)

    print(f"Building envs for '{dial_config.env_name}' (same as the original sweep)...")
    planner_env, plant_env = build_envs(dial_config.env_name, env_config)
    stepper = PlantStepper(plant_env)
    nominal = stepper.nominal
    specs = resolve_specs(drc.params, nominal.sys, config_arrays={"kp": nominal.kp, "kd": nominal.kd})
    sys_fields = sorted({sp.field for sp in specs if sp.target == "sys"})
    planner = PlannerStepper(planner_env, nominal.sys, sys_fields, nominal.kp, nominal.kd)
    mbdpi = MBDPI(dial_config, planner_env, model_step_fn=planner.step_fn)
    diffuse = DiffuseStepper(mbdpi, dial_config)
    if drc.terminate_on_physical_limits:
        planner_env.terminate_on_physical_limits = True
        plant_env.terminate_on_physical_limits = True
        print("Termination: physical joint limits (not the narrower action-scaling band)")
    else:
        print("Termination: env default (the hand-tuned action-scaling band)")

    return ReproContext(run_dir, dial_config, mbdpi, stepper, diffuse, planner, specs)


def reproduce_trial(ctx: ReproContext, trial: int, html: bool = False, force: bool = False):
    trial_dir = os.path.join(ctx.run_dir, "reproduced", f"trial_{trial:04d}")
    if not force and _already_present(trial_dir):
        print(f"Trial {trial} is already reproduced in {trial_dir}/ (pass --force to "
              f"redo it). Inspect with:\n  dial-mpc-sim2sim-view --traj-dir {trial_dir}")
        return None

    seed, theta_np = _load_seed_and_theta(ctx.run_dir, trial, ctx.specs)
    theta = {k: jnp.asarray(v) for k, v in theta_np.items()}
    theta_rounded = {k: np.asarray(v).round(4).tolist() for k, v in theta_np.items()}
    print(f"Trial {trial}: seed={seed}")
    print(f"theta: {theta_rounded}")

    nominal = ctx.stepper.nominal
    sys, kp, kd = apply_theta(nominal.sys, nominal.kp, nominal.kd, theta, ctx.specs)
    plant_model = Model(sys, kp, kd)
    run_rng = jax.random.PRNGKey(seed)

    print("Running nominal_planner arm (planner believes nominal parameters)...")
    result_nom = run_trial(
        ctx.dial_config, ctx.mbdpi, ctx.stepper, ctx.diffuse, plant_model,
        ctx.planner.nominal, theta, run_rng,
    )
    print("Running true_planner arm (planner told the true parameters)...")
    result_true = run_trial(
        ctx.dial_config, ctx.mbdpi, ctx.stepper, ctx.diffuse, plant_model,
        ctx.planner.model_from(sys, kp, kd), theta, run_rng,
    )
    results = {GROUP_NOMINAL_PLANNER: result_nom, GROUP_TRUE_PLANNER: result_true}

    print(f"\n{'group':<16}{'steps':>7}{'survived':>10}{'return_mean':>14}")
    for group, r in results.items():
        print(f"{group:<16}{r.steps_survived:>7}{str(r.survived):>10}{r.return_mean:>14.4f}")

    os.makedirs(trial_dir, exist_ok=True)
    theta_json = json.dumps({k: np.asarray(v).tolist() for k, v in theta_np.items()})
    for group, r in results.items():
        assert r.rollout is not None
        path = os.path.join(trial_dir, f"{group}.npz")
        np.savez_compressed(
            path,
            env_name=np.asarray(ctx.dial_config.env_name),
            trial=np.asarray(trial),
            group=np.asarray(group),
            seed=np.asarray(seed),
            theta_json=np.asarray(theta_json),
            **r.rollout,
        )
        print(f"Wrote {path}")

    delta_return = result_nom.return_mean - result_true.return_mean
    delta_steps = result_nom.steps_survived - result_true.steps_survived
    title = f"Trial {trial} (reproduced) — Δreturn {delta_return:+.4f}, Δsteps {delta_steps:+d}"
    generate_trial_outputs(os.path.abspath(ctx.run_dir), trial_dir, title, html=html)
    return seed, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True, help="a sim2sim run directory")
    parser.add_argument("--trial", type=int, required=True, nargs="+",
                         help="one or more trial indices to reproduce")
    parser.add_argument("--html", action="store_true", help="also write brax 3D playbacks")
    parser.add_argument("--force", action="store_true", help="regenerate even if already saved")
    parser.add_argument("--determinism", choices=determinism.MODES, default="exact",
                        help="replay defaults to 'exact' (CPU, bitwise reproducible) so a "
                             "trial replays identically; 'fast' uses the GPU but will not "
                             "match the original step-for-step")
    parser.add_argument("--matmul-precision", choices=determinism.PRECISIONS,
                        default="default",
                        help="'highest' disables TF32 f32 matmuls, matching CPU physics to "
                             "~1e-5 relative at ~12%% cost; the TF32 default is off by ~600x "
                             "the run-to-run spread")
    args = parser.parse_args()

    determinism.configure(args.determinism, args.matmul_precision)

    ctx = build_context(args.run)
    for trial in args.trial:
        reproduce_trial(ctx, trial, html=args.html, force=args.force)


if __name__ == "__main__":
    main()
