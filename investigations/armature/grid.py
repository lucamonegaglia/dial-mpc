"""Closed-loop trials over a (plant armature scale x planner armature scale x seed) grid.

Appends to `results/<tag>.csv` and skips cells already present, so it can be resumed and
extended. Rollouts (50 Hz state logs) are kept as .npz for every trial under
`results/<tag>_rollouts/` when --save-rollouts is passed.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from dial_mpc.sim2sim import determinism


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--plant", type=float, nargs="+", required=True)
    p.add_argument("--planner", type=str, nargs="+", required=True,
                   help="planner scales; 'match' means planner == plant")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--precision", choices=determinism.PRECISIONS, default="default")
    p.add_argument("--mode", choices=determinism.MODES, default="fast")
    p.add_argument("--save-rollouts", action="store_true")
    p.add_argument("--phys-term", action="store_true",
                   help="terminate on physical joint limits instead of the action-scaling band")
    p.add_argument("--implicit-damping", action="store_true",
                   help="fold kd into dof_damping and enable eulerdamp (stable 20 ms Euler at low armature)")
    p.add_argument("--repeat", type=int, default=1,
                   help="run each cell this many times (measures run-to-run GPU nondeterminism)")
    args = p.parse_args()

    determinism.configure(args.mode, args.precision)
    from common import RESULTS, RESULT_KEYS, build, result_row  # after jax config

    os.makedirs(RESULTS, exist_ok=True)
    csv_path = os.path.join(RESULTS, f"{args.tag}.csv")
    roll_dir = os.path.join(RESULTS, f"{args.tag}_rollouts")
    if args.save_rollouts:
        os.makedirs(roll_dir, exist_ok=True)
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                done.add((float(r["plant"]), r["planner_arg"], int(r["seed"]), int(r["rep"])))

    overrides = {"n_steps": args.n_steps} if args.n_steps else None
    h = build(overrides, phys_term=args.phys_term, implicit_damping=args.implicit_damping)
    fields = ["plant", "planner", "planner_arg", "seed", "rep", "wall_s"] + RESULT_KEYS
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        for seed in args.seeds:
            for plant in args.plant:
                for parg in args.planner:
                    planner = plant if parg == "match" else float(parg)
                    for rep in range(args.repeat):
                        if (plant, parg, seed, rep) in done:
                            continue
                        t0 = time.time()
                        r = h.trial(plant, planner, seed)
                        row = {"plant": plant, "planner": planner, "planner_arg": parg,
                               "seed": seed, "rep": rep, "wall_s": time.time() - t0}
                        row.update(result_row(r))
                        w.writerow(row)
                        f.flush()
                        if args.save_rollouts and r.rollout is not None:
                            np.savez_compressed(
                                os.path.join(roll_dir, f"p{plant:.3f}_q{planner:.3f}_{parg}_s{seed}_r{rep}.npz"),
                                **r.rollout)
                        print(f"plant={plant:.3f} planner={planner:.3f} seed={seed} rep={rep} "
                              f"ret={r.return_sum:8.2f} steps={r.steps_survived:3d} "
                              f"fdiv={r.frac_diverged:.3f} ({row['wall_s']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
