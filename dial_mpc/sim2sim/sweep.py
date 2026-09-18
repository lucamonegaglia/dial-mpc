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
import subprocess
import time
from dataclasses import asdict
from typing import Any, Dict, List, cast

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
from dial_mpc.sim2sim.groups import GROUP_NOMINAL_PLANNER, GROUP_TRUE_PLANNER, GROUPS
from dial_mpc.sim2sim.runner import (
    build_envs, PlantStepper, PlannerStepper, DiffuseStepper, Model, run_trial, TrialResult,
)


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


# A trial where the true-parameter planner's return_mean is this much lower (more
# negative) than the nominal-parameter planner's is always kept, even if every other
# slot is already spoken for -- see `_select_interesting`.
TRUE_PLANNER_MUCH_WORSE_FRAC = 0.5


def _select_interesting(
    pairs: List[Dict[str, Any]], n_keep: int,
    much_worse_frac: float = TRUE_PLANNER_MUCH_WORSE_FRAC,
) -> List[Dict[str, Any]]:
    """Pick the trial pairs worth keeping a full state log for.

    Four qualitatively different kinds of divergence, taken together rather than just
    ranking by |delta return|, because they answer different questions:

      * `true_planner_much_worse` -- the true-parameter planner's return_mean is more than
        `much_worse_frac` lower than the nominal-parameter planner's, despite having the
        correct model. Reserved a slot unconditionally: this is the one
        category a caller explicitly wants guaranteed, so it is not subject to the
        round-robin below.
      * `survival_flip` -- one arm fell and the other did not. Same plant, same seed, so
        this is purely the planner's model error deciding the episode. The clearest
        possible domain-shift failure, and invisible in a mean.
      * `nominal_planner_worse` / `nominal_planner_better` -- the extreme tails of paired
        delta return. The `better` side matters: the nominal-parameter planner beating the
        true-parameter planner means model error was not the binding constraint for that
        draw (sampling noise, or a mismatch that happened to help).
      * `steps_gap` -- largest |delta steps survived|, which catches pairs that both
        eventually fall but at very different times.

    Selection is two-phase: `true_planner_much_worse` trials are reserved first (up to
    `n_keep`), then the remaining slots are filled round-robin across the other four
    categories. A straight priority sort (survival flips always first) previously let one
    crowded category consume every slot -- concretely, a 256-trial run had >=12 distinct
    `survival_flip` pairs, so they filled all 12 slots and silently dropped the single
    largest |delta_return| pair in the whole run (tagged only `nominal_planner_better`).
    Round-robin instead gives every category a turn.
    """
    tagged: Dict[int, Dict[str, Any]] = {}

    def tag(pair: Dict[str, Any], reason: str):
        entry = tagged.setdefault(pair["trial"], {"pair": pair, "reasons": []})
        if reason not in entry["reasons"]:
            entry["reasons"].append(reason)

    for p in pairs:
        if bool(p["survived_nominal_planner"]) != bool(p["survived_true_planner"]):
            tag(p, "survival_flip")
        nom_r, true_r = p["return_nominal_planner"], p["return_true_planner"]
        if true_r <= nom_r - much_worse_frac * abs(nom_r):
            tag(p, "true_planner_much_worse")

    # Each bucket is sign-filtered so a tag always means what it says. Without that, a
    # sweep with fewer pairs than 2 * per_bucket tags the same pair both `worse` and
    # `better`, and tags `steps_gap` on pairs whose arms survived equally long.
    by_delta = sorted(pairs, key=lambda p: p["delta_return"])
    per_bucket = max(1, n_keep // 4)
    for p in [q for q in by_delta if q["delta_return"] < 0][:per_bucket]:
        tag(p, "nominal_planner_worse")
    for p in [q for q in reversed(by_delta) if q["delta_return"] > 0][:per_bucket]:
        tag(p, "nominal_planner_better")
    for p in [q for q in sorted(pairs, key=lambda q: -abs(q["delta_steps"]))
              if q["delta_steps"] != 0][:per_bucket]:
        tag(p, "steps_gap")

    by_extremity = lambda e: -abs(e["pair"]["delta_return"])

    selected: Dict[int, Dict[str, Any]] = {}
    for e in sorted((e for e in tagged.values() if "true_planner_much_worse" in e["reasons"]),
                     key=by_extremity):
        if len(selected) >= n_keep:
            break
        selected[e["pair"]["trial"]] = e

    round_robin_categories = ["survival_flip", "nominal_planner_worse",
                              "nominal_planner_better", "steps_gap"]
    by_category = {
        c: sorted((e for e in tagged.values() if c in e["reasons"]), key=by_extremity)
        for c in round_robin_categories
    }
    cursor = {c: 0 for c in round_robin_categories}
    progressed = True
    while len(selected) < n_keep and progressed:
        progressed = False
        for c in round_robin_categories:
            if len(selected) >= n_keep:
                break
            lst = by_category[c]
            while cursor[c] < len(lst) and lst[cursor[c]]["pair"]["trial"] in selected:
                cursor[c] += 1
            if cursor[c] < len(lst):
                e = lst[cursor[c]]
                selected[e["pair"]["trial"]] = e
                cursor[c] += 1
                progressed = True

    return list(selected.values())


DIFF_FILENAME = "git_diff.patch"


def _git_provenance(run_dir: str) -> Dict[str, Any]:
    """Commit for the code that produced a run, so any result traces back to exact source.
    Any working-tree diff is written to `<run_dir>/git_diff.patch` and only referenced from
    the returned dict, which keeps it out of config_used.yaml."""
    def run(*cmd: str) -> str:
        try:
            return subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)),
                                  capture_output=True, text=True, timeout=30,
                                  check=True).stdout.strip()
        except Exception as exc:
            return f"<unavailable: {exc}>"

    diff = run("git", "diff", "HEAD")
    if diff:
        with open(os.path.join(run_dir, DIFF_FILENAME), "w") as f:
            f.write(diff + "\n")
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(diff),
        # Untracked files are invisible to `git diff`, so name them explicitly.
        "untracked": [f for f in run("git", "ls-files", "--others",
                                     "--exclude-standard").split("\n") if f],
        "diff": DIFF_FILENAME if diff else None,
    }


def _cache_size(fn: Any) -> int:
    """How many times a jitted function has been compiled. `_cache_size` is private API on
    jax's wrapper, so fall back to -1 ("unknown") rather than crashing a finished sweep."""
    probe = getattr(fn, "_cache_size", None)
    try:
        return int(cast(Any, probe())) if callable(probe) else -1
    except Exception:
        return -1


def _trial_row(idx: int, seed: int, result: TrialResult) -> Dict[str, Any]:
    row: Dict[str, Any] = {"trial": idx, "seed": seed}
    row.update(_flatten_theta(result.theta))
    d = asdict(result)
    del d["theta"], d["rollout"]
    d["survived"] = int(d["survived"])  # CSV-friendly; bool "True"/"False" isn't numeric
    d["diverged"] = int(d["diverged"])
    row.update(d)
    return row


def _write_interesting(
    run_dir: str,
    pairs: List[Dict[str, Any]],
    logs: Dict[int, Dict[str, Dict[str, np.ndarray]]],
    n_keep: int,
    dial_config: DialConfig,
    csv_path: str,
) -> None:
    """Write 50 Hz state logs for the most divergent trial pairs, plus an index that
    ties each one back to the exact parameters that produced it.

    Every entry is reproducible from the files it names: `config` is the verbatim config
    the sweep ran (including the seed and the randomization ranges), `trials_csv` holds
    the full metric row, and `theta` is the realised parameter draw, also duplicated
    inside each .npz so a trajectory file is self-describing if it gets moved.

    Each trial gets its own subdirectory (`trial_0168/nominal_planner.npz`, ...) instead
    of flat `trial_0168_nominal_planner.npz` files, so `dial_mpc.sim2sim.reproduce` can
    write a reproduced trial in the identical shape under `reproduced/` and both are
    handled by the same `view.generate_trial_outputs`. Breaks index.json compatibility
    with runs written before this change -- there is no reader for the old flat layout.
    """
    selected = _select_interesting(pairs, n_keep)
    if not selected:
        return
    out_dir = os.path.join(run_dir, "interesting")
    os.makedirs(out_dir, exist_ok=True)

    index: List[Dict[str, Any]] = []
    for entry in selected:
        pair = entry["pair"]
        i = pair["trial"]
        trial_dir_name = f"trial_{i:04d}"
        trial_dir = os.path.join(out_dir, trial_dir_name)
        os.makedirs(trial_dir, exist_ok=True)
        files = {}
        for group in GROUPS:
            log = logs.get(i, {}).get(group)
            if log is None:
                continue
            fname = f"{group}.npz"
            np.savez_compressed(
                os.path.join(trial_dir, fname),
                env_name=np.asarray(dial_config.env_name),
                trial=np.asarray(i),
                group=np.asarray(group),
                seed=np.asarray(pair["seed"]),
                theta_json=np.asarray(json.dumps(pair["theta"])),
                **log,
            )
            files[group] = fname
        index.append({
            "trial": i,
            "dir": trial_dir_name,
            "seed": pair["seed"],
            "reasons": entry["reasons"],
            "delta_return": pair["delta_return"],
            "delta_steps": pair["delta_steps"],
            GROUP_NOMINAL_PLANNER: {"return_mean": pair["return_nominal_planner"],
                                    "steps_survived": pair["steps_nominal_planner"],
                                    "survived": pair["survived_nominal_planner"]},
            GROUP_TRUE_PLANNER: {"return_mean": pair["return_true_planner"],
                                 "steps_survived": pair["steps_true_planner"],
                                 "survived": pair["survived_true_planner"]},
            "files": files,
            "theta": pair["theta"],
        })

    manifest = {
        "run_dir": os.path.abspath(run_dir),
        "config": "../config_used.yaml",
        "trials_csv": "../" + os.path.basename(csv_path),
        "env_name": dial_config.env_name,
        "log_rate_hz": 50.0,
        "note": (
            "Each trial's two .npz files live under `<dir>/` (e.g. trial_0168/) and are "
            "the SAME theta-perturbed plant driven at the SAME MPC seed; they differ only "
            "in what the planner was told. "
            f"'{GROUP_NOMINAL_PLANNER}' planned with the nominal parameters, "
            f"'{GROUP_TRUE_PLANNER}' planned with the plant's true parameters, so the "
            "difference between them is the planner's model error alone. `theta` is the "
            "realised draw and applies to BOTH files. Reproduce a single trial exactly "
            "with `dial-mpc-sim2sim-reproduce --run <run_dir> --trial <n>` (uses the "
            "seed and theta from `trials_csv`/this index); trial index and seed are "
            "recorded per entry."
        ),
        "entries": index,
    }
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {len(index)} divergent trial pairs to {out_dir}/ (see index.json)")


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

    print(f"Building envs for '{dial_config.env_name}' (plant is theta-perturbed in BOTH arms)")
    planner_env, plant_env = build_envs(dial_config.env_name, env_config)
    stepper = PlantStepper(plant_env)
    nominal = stepper.nominal

    specs = resolve_specs(
        drc.params,
        nominal.sys,
        config_arrays={"kp": nominal.kp, "kd": nominal.kd},
    )
    print(f"Domain-randomization axes ({len(specs)}): " + ", ".join(s.name for s in specs))

    # Which sys fields theta actually touches -- the planner traces only these and keeps
    # the other ~337 as constants (see PlannerModel for the measured cost of not doing so).
    sys_fields = sorted({sp.field for sp in specs if sp.target == "sys"})
    planner = PlannerStepper(
        planner_env, nominal.sys, sys_fields, nominal.kp, nominal.kd
    )
    # MBDPI must be built with the planner's step fn, and after it.
    mbdpi = MBDPI(dial_config, planner_env, model_step_fn=planner.step_fn)
    diffuse = DiffuseStepper(mbdpi, dial_config)

    if drc.terminate_on_physical_limits:
        # This decides when the PLANT's episode ends, which is the whole point. Set on the
        # planner env too only for consistency: `rollout_us` never truncates on `done`, and
        # UnitreeH1LocoEnv weights reward_alive at 0.0, so planner-side it is a no-op today.
        planner_env.terminate_on_physical_limits = True
        plant_env.terminate_on_physical_limits = True
        print("Termination: physical joint limits (not the narrower action-scaling band)")
    else:
        print("Termination: env default (the hand-tuned action-scaling band)")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(dial_config.output_dir, f"{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    if drc.save_rollouts:
        os.makedirs(os.path.join(run_dir, "rollouts"), exist_ok=True)
    config_dict = dict(config_dict, git=_git_provenance(run_dir))
    with open(os.path.join(run_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False)

    rng = jax.random.PRNGKey(drc.seed)
    rows: List[Dict[str, Any]] = []
    pairs: List[Dict[str, Any]] = []
    logs: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}
    theta_columns: List[str] = []

    with tqdm(range(drc.n_trials), desc="Sim2sim trials") as pbar:
        for i in pbar:
            rng, theta_rng = jax.random.split(rng)
            theta = sample_theta(theta_rng, specs)
            # The MPC key is derived from an explicit integer that goes into the CSV, so one
            # trial can be replayed without replaying the whole sweep's RNG stream. Both arms
            # share it -- that identity is what makes the pair a controlled comparison.
            trial_seed = (drc.seed * 1_000_003 + i) % (2**31 - 1)
            run_rng = jax.random.PRNGKey(trial_seed)
            sys, kp, kd = apply_theta(nominal.sys, nominal.kp, nominal.kd, theta, specs)

            plant_model = Model(sys, kp, kd)
            if i == 0:
                # Both arms must share one compiled kernel, or they are not comparable.
                assert jax.tree_util.tree_structure(planner.nominal) == jax.tree_util.tree_structure(
                    planner.model_from(sys, kp, kd)
                ), "planner model treedef differs between arms -> two compilations"

            # Arm A: the planner believes the nominal parameters. This is the domain shift.
            result = run_trial(
                dial_config, mbdpi, stepper, diffuse,
                plant_model, planner.nominal, theta, run_rng,
            )
            row = _trial_row(i, trial_seed, result)
            row["group"] = GROUP_NOMINAL_PLANNER
            rows.append(row)
            if not theta_columns:
                theta_columns = sorted(_flatten_theta(result.theta))

            if result.rollout is not None:
                logs.setdefault(i, {})[GROUP_NOMINAL_PLANNER] = result.rollout
            if drc.save_rollouts and result.rollout is not None:
                np.savez(os.path.join(run_dir, "rollouts",
                                      f"trial_{i:04d}_{GROUP_NOMINAL_PLANNER}.npz"), **result.rollout)

            if drc.paired:
                # Arm B: identical plant, identical MPC seed -- the planner is simply told
                # the truth. The difference between the arms is the planner's model error
                # and nothing else.
                nominal_result = run_trial(
                    dial_config, mbdpi, stepper, diffuse,
                    plant_model, planner.model_from(sys, kp, kd), theta, run_rng,
                )
                nrow = _trial_row(i, trial_seed, nominal_result)
                nrow["group"] = GROUP_TRUE_PLANNER
                rows.append(nrow)
                if nominal_result.rollout is not None:
                    logs.setdefault(i, {})[GROUP_TRUE_PLANNER] = nominal_result.rollout
                if drc.save_rollouts and nominal_result.rollout is not None:
                    np.savez(
                        os.path.join(run_dir, "rollouts",
                                     f"trial_{i:04d}_{GROUP_TRUE_PLANNER}.npz"),
                        **nominal_result.rollout,
                    )
                delta = result.return_mean - nominal_result.return_mean
                pairs.append({
                    "trial": i,
                    "seed": trial_seed,
                    "delta_return": delta,
                    "delta_steps": result.steps_survived - nominal_result.steps_survived,
                    "survived_nominal_planner": result.survived,
                    "survived_true_planner": nominal_result.survived,
                    "return_nominal_planner": result.return_mean,
                    "return_true_planner": nominal_result.return_mean,
                    "steps_nominal_planner": result.steps_survived,
                    "steps_true_planner": nominal_result.steps_survived,
                    "theta": {k: np.asarray(v).tolist() for k, v in theta.items()},
                })
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

    rand_rows = [r for r in rows if r["group"] == GROUP_NOMINAL_PLANNER]
    summary = {
        "n_trials": drc.n_trials,
        "paired": drc.paired,
        # So `analyze.py` knows which CSV columns are randomization axes without a denylist.
        "theta_columns": theta_columns,
        "env_name": dial_config.env_name,
        "n_steps": dial_config.n_steps,
        "return_mean": float(np.nanmean([r["return_mean"] for r in rand_rows])),
        "return_std": float(np.nanstd([r["return_mean"] for r in rand_rows])),
        "survival_rate": float(np.mean([r["survived"] for r in rand_rows])),
        "diverged_trials": int(sum(r["diverged"] for r in rows)),
        "planner_frac_diverged_mean": float(np.nanmean([r["frac_diverged"] for r in rows])),
        "optimism_gap_mean": float(np.nanmean([r["optimism_gap"] for r in rand_rows])),
        "pred_err_1step_mean": float(np.nanmean([r["pred_err_1step"] for r in rand_rows])),
    }
    if drc.paired:
        nom_rows = [r for r in rows if r["group"] == GROUP_TRUE_PLANNER]
        deltas = np.array(
            [rr["return_mean"] - nr["return_mean"] for rr, nr in zip(rand_rows, nom_rows)],
            dtype=float,
        )
        summary["true_planner_return_mean"] = float(np.nanmean([r["return_mean"] for r in nom_rows]))
        summary["paired_delta_return_mean"] = float(np.nanmean(deltas))
        summary["paired_delta_return_std"] = float(np.nanstd(deltas))
        summary["paired_trials_usable"] = int(np.isfinite(deltas).sum())
    # Compile guards: fixed costs that must NOT grow with the number of trials. A draw that
    # forced a retrace would collapse runtime and, worse, could leave the two arms running
    # different compiled code. `plant_step` settles at 2 (the episode's first action comes
    # from a freshly-allocated zeros array, later ones out of the diffusion); that is benign.
    caches = {
        "diffuse_init": _cache_size(diffuse.diffuse_init_jit),
        "diffuse": _cache_size(diffuse.diffuse_jit),
        "plant_step": _cache_size(stepper.step_jit),
        "plant_reset": _cache_size(stepper.reset_jit),
    }
    expected = {"diffuse_init": 1, "diffuse": 1, "plant_step": 2, "plant_reset": 1}
    summary["jit_cache_sizes"] = caches
    over = {k: v for k, v in caches.items() if v > expected[k]}
    if over:
        print(f"\n*** WARNING: more compilations than expected {over}; "
              f"expected {expected}. Something retraced per-trial. ***\n")

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if pairs and drc.n_interesting > 0:
        _write_interesting(run_dir, pairs, logs, drc.n_interesting, dial_config, csv_path)

    print(f"\nWrote {len(rows)} trial rows to {csv_path}")
    print(json.dumps(summary, indent=2))
    print(f"\nRun: dial-mpc-sim2sim-report --run {run_dir}")


if __name__ == "__main__":
    main()
