"""Paired analysis of a grid CSV + rollouts: delta statistics and rollout diagnostics.

delta = nominal-planner arm - matched arm (positive: the nominal planner did better).
`prefix_rew` compares per-step reward over the first min(T_nominal, T_matched) steps of the
pair, i.e. only while both arms were alive, which removes the time-to-termination noise.
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
# H1 loco actuator_ctrlrange upper bounds (hips/knee 200/300, ankle 40, torso 200)
TAU_MAX = np.array([200, 200, 200, 300, 40, 200, 200, 200, 300, 40, 200], dtype=float)


def load_rollouts(tag: str):
    out = {}
    for f in glob.glob(os.path.join(HERE, "results", f"{tag}_rollouts", "*.npz")):
        m = re.search(r"p([\d.]+)_q([\d.]+)_(?:([a-z0-9.]+)_)?s(\d+)_r(\d+)", os.path.basename(f))
        if m:
            parg = m.group(3) or ("match" if m.group(1) == m.group(2) else str(float(m.group(2))))
            out[(float(m.group(1)), parg, int(m.group(4)), int(m.group(5)))] = dict(np.load(f))
    return out


def diagnostics(r):
    qv = r["qvel"][:, 6:17]
    ctrl = r["ctrl"]
    lag1 = [np.corrcoef(qv[:-1, j], qv[1:, j])[0, 1] for j in range(qv.shape[1])] if len(qv) > 3 else [np.nan]
    return {
        "qvel_lag1": float(np.nanmean(lag1)),
        "qvel_lag1_min": float(np.nanmin(lag1)),
        "hf_ratio": float(np.mean(np.diff(qv, axis=0) ** 2) / (np.mean(qv ** 2) + 1e-9)) if len(qv) > 1 else np.nan,
        "sat_frac": float(np.mean(np.abs(ctrl) >= TAU_MAX * 0.999)),
        "act_abs": float(np.mean(np.abs(r["action"]))),
        "act_edge_frac": float(np.mean(np.abs(r["action"]) > 0.95)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="paired")
    ap.add_argument("--ref", default="1.0", help="planner_arg of the nominal arm")
    args = ap.parse_args()
    d = pd.read_csv(os.path.join(HERE, "results", f"{args.tag}.csv"))
    rolls = load_rollouts(args.tag)

    rows = []
    for (plant, seed), g in d.groupby(["plant", "seed"]):
        a = g[g.planner_arg.astype(str) == args.ref]
        b = g[g.planner_arg.astype(str) == "match"]
        if a.empty or b.empty:
            continue
        a, b = a.iloc[0], b.iloc[0]
        ra = rolls.get((plant, args.ref, seed, 0))
        rb = rolls.get((plant, "match", seed, 0))
        if plant == float(args.ref) and ra is rb:
            ra = None  # legacy filenames: both arms collided on one file
        row = {"plant": plant, "seed": seed,
               "d_ret": a.return_sum - b.return_sum,
               "d_steps": a.steps_survived - b.steps_survived,
               "d_rmean": a.return_mean - b.return_mean,
               "d_vel_err": a.vel_err - b.vel_err}
        if ra is not None and rb is not None:
            T = min(len(ra["reward"]), len(rb["reward"]))
            row["prefix_T"] = T
            row["d_prefix_rew"] = float(ra["reward"][:T].mean() - rb["reward"][:T].mean())
            da, db = diagnostics(ra), diagnostics(rb)
            for k in da:
                row[f"nom_{k}"], row[f"true_{k}"] = da[k], db[k]
        rows.append(row)
    p = pd.DataFrame(rows)

    def summ(x):
        x = x.dropna()
        if len(x) < 3:
            return pd.Series({"n": len(x), "mean": x.mean()})
        w = wilcoxon(x).pvalue if (x != 0).any() else 1.0
        return pd.Series({"n": len(x), "mean": x.mean(), "se": x.std(ddof=1) / np.sqrt(len(x)),
                          "median": x.median(), "P(>0)": (x > 0).mean(), "wilcoxon_p": w})

    pd.set_option("display.width", 200)
    for col in ["d_ret", "d_steps", "d_rmean", "d_prefix_rew"]:
        if col in p:
            print(f"\n== {col} (nominal - matched) ==")
            print(p.groupby("plant")[col].apply(summ).unstack().round(3).to_string())
    diag = [c for c in p.columns if c.startswith("nom_") or c.startswith("true_")]
    if diag:
        print("\n== rollout diagnostics (means) ==")
        print(p.groupby("plant")[diag].mean().round(3).T.to_string())
    p.to_csv(os.path.join(HERE, "results", f"{args.tag}_pairs.csv"), index=False)


if __name__ == "__main__":
    main()
