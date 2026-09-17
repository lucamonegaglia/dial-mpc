"""dial-mpc-sim2sim-eval: sample domain-randomized plant parameters and run paired
closed-loop DIAL-MPC trials, measuring how a fixed-nominal-parameter planner degrades
under plant/planner mismatch.

Usage:
    dial-mpc-sim2sim-eval --example unitree_h1_loco_sim2sim
    dial-mpc-sim2sim-eval --config path/to/my_config.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List

import art
import jax
import numpy as np
import yaml
from tqdm import tqdm

import dial_mpc.envs as dial_envs
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.core.dial_core import MBDPI
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict

from dial_mpc.sim2sim.randomize import load_domain_rand_config, resolve_specs, sample_theta, apply_theta
from dial_mpc.sim2sim.runner import build_envs, PlantStepper, DiffuseStepper, run_trial, TrialResult


def _flatten_theta(theta: Dict[str, np.ndarray]) -> Dict[str, float]:
    row = {}
    for name, arr in theta.items():
        arr = np.asarray(arr)
        if arr.size == 1:
            row[name] = float(arr.reshape(()))
        else:
            for idx in np.ndindex(arr.shape):
                suffix = "_".join(str(i) for i in idx)
                row[f"{name}_{suffix}"] = float(arr[idx])
    return row


def _trial_row(idx: int, seed: int, result: TrialResult) -> Dict[str, Any]:
    row = {"trial": idx, "seed": seed}
    row.update(_flatten_theta(result.theta))
    d = asdict(result)
    del d["theta"], d["rollout"]
    d["survived"] = int(d["survived"])  # CSV-friendly; bool "True"/"False" isn't numeric
    row.update(d)
    return row


def main():
    art.tprint("DIAL-MPC\nsim2sim eval", font="small", chr_ignore=True)
    parser = argparse.ArgumentParser()
    config_or_example = parser.add_mutually_exclusive_group(required=True)
    config_or_example.add_argument("--config", type=str, default=None)
    config_or_example.add_argument("--example", type=str, default=None)
    parser.add_argument("--custom-env", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=None, help="override domain_randomization.n_trials")
    parser.add_argument("--n-steps", type=int, default=None, help="override n_steps (for smoke tests)")
    args = parser.parse_args()

    if args.custom_env is not None:
        import importlib
        import sys as _sys

        _sys.path.append(os.getcwd())
        importlib.import_module(args.custom_env)

    config_path = get_example_path(args.example + ".yaml") if args.example else args.config
    config_dict = yaml.safe_load(open(config_path))

    dial_config = load_dataclass_from_dict(DialConfig, config_dict)
    if args.n_steps is not None:
        dial_config.n_steps = args.n_steps

    env_config_type = dial_envs.get_config(dial_config.env_name)
    env_config = load_dataclass_from_dict(env_config_type, config_dict, convert_list_to_array=True)

    drc = load_domain_rand_config(config_dict)
    if args.n_trials is not None:
        drc.n_trials = args.n_trials

    print(f"Building envs for '{dial_config.env_name}' (planner: nominal, plant: randomized)")
    planner_env, plant_env = build_envs(dial_config.env_name, env_config)
    mbdpi = MBDPI(dial_config, planner_env)
    stepper = PlantStepper(plant_env)
    diffuse = DiffuseStepper(mbdpi, dial_config)

    specs = resolve_specs(
        drc.params,
        stepper.nominal_sys,
        config_arrays={"kp": stepper.nominal_kp, "kd": stepper.nominal_kd},
    )
    print(f"Domain-randomization axes ({len(specs)}): " + ", ".join(s.name for s in specs))

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(dial_config.output_dir, f"{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    if drc.save_rollouts:
        os.makedirs(os.path.join(run_dir, "rollouts"), exist_ok=True)
    with open(os.path.join(run_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(config_dict, f)

    rng = jax.random.PRNGKey(drc.seed)
    rows: List[Dict[str, Any]] = []

    with tqdm(range(drc.n_trials), desc="Sim2sim trials") as pbar:
        for i in pbar:
            rng, theta_rng, run_rng = jax.random.split(rng, 3)
            trial_seed = int(jax.random.randint(theta_rng, (), 0, 2**31 - 1))
            theta = sample_theta(theta_rng, specs)
            sys, kp, kd = apply_theta(stepper.nominal_sys, stepper.nominal_kp, stepper.nominal_kd, theta, specs)

            result = run_trial(
                dial_config, mbdpi, stepper, diffuse, sys, kp, kd, theta, run_rng,
                save_rollout=drc.save_rollouts,
            )
            row = _trial_row(i, trial_seed, result)
            row["group"] = "randomized"
            rows.append(row)

            if drc.save_rollouts and result.rollout is not None:
                np.savez(os.path.join(run_dir, "rollouts", f"trial_{i:04d}_randomized.npz"), **result.rollout)

            if drc.paired:
                nominal_result = run_trial(
                    dial_config, mbdpi, stepper, diffuse,
                    stepper.nominal_sys, stepper.nominal_kp, stepper.nominal_kd,
                    theta, run_rng, save_rollout=drc.save_rollouts,
                )
                nrow = _trial_row(i, trial_seed, nominal_result)
                nrow["group"] = "nominal"
                rows.append(nrow)
                if drc.save_rollouts and nominal_result.rollout is not None:
                    np.savez(
                        os.path.join(run_dir, "rollouts", f"trial_{i:04d}_nominal.npz"),
                        **nominal_result.rollout,
                    )
                delta = result.return_mean - nominal_result.return_mean
                pbar.set_postfix({"ret": f"{result.return_mean:.2e}", "d_ret": f"{delta:.2e}"})
            else:
                pbar.set_postfix({"ret": f"{result.return_mean:.2e}"})

    fieldnames = sorted({k for row in rows for k in row})
    fieldnames = ["trial", "group", "seed"] + [f for f in fieldnames if f not in ("trial", "group", "seed")]
    csv_path = os.path.join(run_dir, "trials.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    rand_rows = [r for r in rows if r["group"] == "randomized"]
    summary = {
        "n_trials": drc.n_trials,
        "paired": drc.paired,
        "env_name": dial_config.env_name,
        "n_steps": dial_config.n_steps,
        "return_mean": float(np.mean([r["return_mean"] for r in rand_rows])),
        "return_std": float(np.std([r["return_mean"] for r in rand_rows])),
        "survival_rate": float(np.mean([r["survived"] for r in rand_rows])),
        "optimism_gap_mean": float(np.mean([r["optimism_gap"] for r in rand_rows])),
        "pred_err_1step_mean": float(np.nanmean([r["pred_err_1step"] for r in rand_rows])),
    }
    if drc.paired:
        nom_rows = [r for r in rows if r["group"] == "nominal"]
        deltas = [rr["return_mean"] - nr["return_mean"] for rr, nr in zip(rand_rows, nom_rows)]
        summary["nominal_return_mean"] = float(np.mean([r["return_mean"] for r in nom_rows]))
        summary["paired_delta_return_mean"] = float(np.mean(deltas))
        summary["paired_delta_return_std"] = float(np.std(deltas))
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {len(rows)} trial rows to {csv_path}")
    print(json.dumps(summary, indent=2))
    print(f"\nRun: dial-mpc-sim2sim-report --run {run_dir}")


if __name__ == "__main__":
    main()
