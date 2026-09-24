"""Noise floor (noise.csv: identical plant/planner/seed repeated) and cross grid (cross.csv:
plant x planner armature). Compares the repeat-to-repeat spread with the paired nominal-vs-true
spread so that a paired delta can be read against the noise it would show with zero model error."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x))


def noise():
    f = os.path.join(R, "noise.csv")
    if not os.path.exists(f):
        return
    d = pd.read_csv(f)
    w = d.pivot_table(index=["plant", "seed"], columns="rep", values=["return_sum", "steps_survived"])
    w = w.dropna()
    out = []
    for plant, g in w.groupby(level=0):
        dr = g[("return_sum", 0)] - g[("return_sum", 1)]
        ds = g[("steps_survived", 0)] - g[("steps_survived", 1)]
        out.append({"plant": plant, "n": len(g), "sd_dret": dr.std(ddof=1), "mean|dret|": dr.abs().mean(),
                    "sd_dsteps": ds.std(ddof=1), "frac_identical": float((ds == 0).mean()),
                    "sd_ret_between_seeds": g[("return_sum", 0)].std(ddof=1)})
    print("== noise floor: rep0 - rep1 at identical plant/planner/seed ==")
    print(pd.DataFrame(out).round(2).to_string(index=False))
    pp = os.path.join(R, "paired_pairs.csv")
    if os.path.exists(pp):
        p = pd.read_csv(pp)
        print("\n   vs paired nominal-true spread (sd_dret, sd_dsteps):")
        print(p.groupby("plant")[["d_ret", "d_steps"]].std().round(2).to_string())


def cross():
    f = os.path.join(R, "cross.csv")
    if not os.path.exists(f):
        return
    d = pd.read_csv(f)
    for col in ["return_sum", "steps_survived"]:
        m = d.pivot_table(index="plant", columns="planner", values=col, aggfunc="mean")
        s = d.pivot_table(index="plant", columns="planner", values=col, aggfunc=se)
        print(f"\n== cross grid: mean {col} (rows plant armature, cols planner armature), +- SE ==")
        print((m.round(1).astype(str) + " +-" + s.round(1).astype(str)).to_string())
    # within each plant, seed-paired deviation from the plant's own mean across planners
    d["ret_c"] = d.return_sum - d.groupby(["plant", "seed"]).return_sum.transform("mean")
    print("\n== seed-centred return by planner armature (pooled over plants), mean +- SE ==")
    g = d.groupby("planner").ret_c
    print(pd.DataFrame({"mean": g.mean(), "se": g.apply(se), "n": g.size()}).round(2).to_string())
    d["log_ratio"] = np.round(np.log(d.planner / d.plant), 2)
    print("\n== seed-centred return by log(planner/plant) ==")
    g = d.groupby("log_ratio").ret_c
    print(pd.DataFrame({"mean": g.mean(), "se": g.apply(se), "n": g.size()}).round(2).to_string())


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    noise()
    cross()
