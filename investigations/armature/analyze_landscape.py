"""Aggregate landscape_s*.json: ranking fidelity and one-step plan value per (plant, model).

J_* are plant-evaluated returns of the plan that one reverse_once produces under each model
(same sample batch). dJ_true_nom > 0 means the matched model's plan scored better on the plant.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "results", "landscape_s[0-9]*.json"))):
        d = json.load(open(f))
        seed = d["args"]["seed"]
        for r in d["records"]:
            base = {"seed": seed, "plant": r["plant"], "t": r["t"], "chaos": r["chaos_spearman"],
                    "incumbent_J": r["incumbent_J"], "oracle_J": r.get("oracle_J", np.nan)}
            for m, v in r["models"].items():
                rows.append({**base, "model": float(m), "spearman": v["spearman"], "J": v["J_ref_plan"],
                             "ess": v["ess"], "frac_gated": v["frac_gated"]})
    df = pd.DataFrame(rows)
    df["role"] = np.where(np.isclose(df.model, df.plant), "true", "other")
    pd.set_option("display.width", 200)

    print(f"states: {df.groupby(['seed', 'plant', 't']).ngroups}  seeds: {sorted(df.seed.unique())}")
    print("\n== mean Spearman(model ranking, plant ranking) by plant x model  [chaos = plant vs perturbed plant] ==")
    tab = df.pivot_table(index="plant", columns="model", values="spearman", aggfunc="mean")
    tab["chaos"] = df.groupby("plant").chaos.mean()
    print(tab.round(3).to_string())

    print("\n== mean plant-evaluated J of the one-step plan, by plant x model ==")
    tab = df.pivot_table(index="plant", columns="model", values="J", aggfunc="mean")
    st = df.drop_duplicates(["seed", "plant", "t"]).groupby("plant")
    tab["incumbent"], tab["oracle"] = st.incumbent_J.mean(), st.oracle_J.mean()
    print(tab.round(3).to_string())

    print("\n== ESS / frac_gated by plant x model ==")
    print(df.pivot_table(index="plant", columns="model", values=["ess", "frac_gated"], aggfunc="mean").round(3).to_string())

    wide = df.pivot_table(index=["seed", "plant", "t", "incumbent_J", "oracle_J"], columns="model",
                          values="J").reset_index()
    out = []
    for plant, g in wide.groupby("plant"):
        if plant not in g.columns or 1.0 not in g.columns:
            continue
        for lbl, x in [("J_true - J_nom", g[plant] - g[1.0]),
                       ("J_true - incumbent", g[plant] - g.incumbent_J),
                       ("J_nom - incumbent", g[1.0] - g.incumbent_J),
                       ("oracle - incumbent", g.oracle_J - g.incumbent_J)]:
            x = x.dropna()
            p = wilcoxon(x).pvalue if len(x) > 5 and (x != 0).any() else np.nan
            out.append({"plant": plant, "stat": lbl, "n": len(x), "mean": x.mean(),
                        "se": x.std(ddof=1) / np.sqrt(len(x)), "P(>0)": (x > 0).mean(), "wilcoxon_p": p})
    print("\n== paired per-state differences ==")
    print(pd.DataFrame(out).round(3).to_string(index=False))
    df.to_csv(os.path.join(HERE, "results", "landscape_all.csv"), index=False)


if __name__ == "__main__":
    main()
