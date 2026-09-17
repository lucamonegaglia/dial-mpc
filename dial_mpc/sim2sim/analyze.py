"""dial-mpc-sim2sim-report: aggregate stats and figures for a sim2sim sweep.

Standalone and re-runnable on an existing `trials.csv` -- does not touch JAX/MJX, so
it's cheap to iterate on figures without re-running trials.

Usage:
    dial-mpc-sim2sim-report --run <output_dir>/sim2sim_<timestamp>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, cast, Sequence, Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated default palette (dial_mpc.sim2sim.analyze does not touch brand color --
# these are the skill's pre-validated reference values, used as-is).
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRID = "#e1e0d9"
_BASELINE = "#c3c2b7"
_BLUE = "#2a78d6"  # categorical slot 1 / diverging positive pole
_ORANGE = "#eb6834"  # categorical slot 2
_RED = "#e34948"  # diverging negative pole (categorical slot 8, reused for polarity)
_GRAY_MID = "#f0efec"  # diverging neutral midpoint


def _style_axes(ax):
    ax.set_facecolor(_SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_BASELINE)
        ax.spines[spine].set_linewidth(1.0)
    ax.tick_params(colors=_INK_MUTED, labelsize=9)
    ax.xaxis.label.set_color(_INK_SECONDARY)
    ax.yaxis.label.set_color(_INK_SECONDARY)
    ax.grid(True, color=_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def load_trials(run_dir: str) -> List[Dict]:
    path = os.path.join(run_dir, "trials.csv")
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            parsed = {}
            for k, v in r.items():
                if v == "" or v is None:
                    parsed[k] = None
                    continue
                if k in ("group",):
                    parsed[k] = v
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            rows.append(parsed)
    return rows


def _theta_columns(rows: List[Dict]) -> List[str]:
    known = {
        "trial", "group", "seed", "return_sum", "return_mean", "steps_survived",
        "survived", "plan_return_mean", "optimism_gap", "pred_err_1step",
        "vel_err", "yaw_rate_err", "torque_rms",
    }
    if not rows:
        return []
    return sorted(k for k in rows[0] if k not in known)


def fig_return_ecdf(rows: List[Dict], out_path: str):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    for group, color, label in (("randomized", _BLUE, "Randomized plant"), ("nominal", _ORANGE, "Nominal plant")):
        vals = sorted(r["return_mean"] for r in rows if r["group"] == group)
        if not vals:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where="post", color=color, linewidth=2.0, solid_capstyle="round", label=label)
    _style_axes(ax)
    ax.set_xlabel("Mean per-step reward")
    ax.set_ylabel("Cumulative fraction of trials")
    ax.set_title("Return distribution: domain shift vs nominal", color=_INK, fontsize=11, loc="left")
    leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for text in leg.get_texts():
        text.set_color(_INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=_SURFACE)
    plt.close(fig)


def fig_paired_delta(rows: List[Dict], out_path: str):
    by_trial = {}
    for r in rows:
        by_trial.setdefault(int(r["trial"]), {})[r["group"]] = r
    deltas = [
        d["randomized"]["return_mean"] - d["nominal"]["return_mean"]
        for d in by_trial.values()
        if "randomized" in d and "nominal" in d
    ]
    if not deltas:
        return
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    counts, edges, patches = ax.hist(deltas, bins=min(24, max(6, len(deltas) // 3)), zorder=2)
    patches = cast(Sequence[Any], patches)  # single dataset -> one BarContainer, not a list of them
    for patch, left, right in zip(patches, edges[:-1], edges[1:]):
        center = 0.5 * (left + right)
        patch.set_facecolor(_BLUE if center >= 0 else _RED)
        patch.set_edgecolor(_SURFACE)
        patch.set_linewidth(1.0)
    ax.axvline(0.0, color=_INK_MUTED, linewidth=1.2, linestyle=(0, (3, 2)), zorder=3)
    _style_axes(ax)
    ax.set_xlabel("Δ mean reward  (randomized − nominal, same seed)")
    ax.set_ylabel("Trial count")
    ax.set_title("Paired domain-shift effect on return", color=_INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=_SURFACE)
    plt.close(fig)


def fig_survival(rows: List[Dict], out_path: str):
    fig, ax = plt.subplots(figsize=(4.5, 4), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    groups = [g for g in ("randomized", "nominal") if any(r["group"] == g for r in rows)]
    colors = {"randomized": _BLUE, "nominal": _ORANGE}
    labels = {"randomized": "Randomized", "nominal": "Nominal"}
    rates = [np.mean([r["survived"] for r in rows if r["group"] == g]) for g in groups]
    bars = ax.bar(
        [labels[g] for g in groups], rates,
        color=[colors[g] for g in groups], width=0.55, zorder=2,
    )
    for b, rate in zip(bars, rates):
        ax.annotate(
            f"{rate:.0%}", (float(b.get_x() + b.get_width() / 2), float(rate)),
            xytext=(0, 4), textcoords="offset points", ha="center",
            fontsize=9, color=_INK,
        )
    _style_axes(ax)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Survival rate (never fell)")
    ax.set_title("Full-episode survival", color=_INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=_SURFACE)
    plt.close(fig)


def fig_sensitivity(rows: List[Dict], out_path: str):
    by_trial = {}
    for r in rows:
        by_trial.setdefault(int(r["trial"]), {})[r["group"]] = r
    paired = [d for d in by_trial.values() if "randomized" in d and "nominal" in d]
    if not paired:
        return
    theta_cols = _theta_columns(rows)
    theta_cols = [c for c in theta_cols if all(d["randomized"].get(c) is not None for d in paired)]
    if not theta_cols:
        return

    n = len(theta_cols)
    ncols = min(3, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), dpi=150, squeeze=False)
    fig.patch.set_facecolor(_SURFACE)

    def spearman(x, y):
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        if np.std(rx) == 0 or np.std(ry) == 0:
            return float("nan")
        return float(np.corrcoef(rx, ry)[0, 1])

    for i, col in enumerate(theta_cols):
        ax = axes[i // ncols][i % ncols]
        x = np.array([d["randomized"][col] for d in paired])
        y = np.array([d["randomized"]["return_mean"] - d["nominal"]["return_mean"] for d in paired])
        ax.scatter(x, y, s=14, color=_BLUE, alpha=0.75, edgecolors="none", zorder=2)
        ax.axhline(0.0, color=_INK_MUTED, linewidth=1.0, linestyle=(0, (3, 2)), zorder=1)
        rho = spearman(x, y)
        _style_axes(ax)
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel("Δ return", fontsize=8)
        ax.set_title(f"ρ = {rho:.2f}", fontsize=9, color=_INK, loc="left")

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("Per-parameter sensitivity of domain shift", color=_INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, facecolor=_SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True, help="path to a sim2sim_<timestamp> run directory")
    args = parser.parse_args()

    rows = load_trials(args.run)
    if not rows:
        raise SystemExit(f"No trials found in {args.run}/trials.csv")

    fig_dir = os.path.join(args.run, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    fig_return_ecdf(rows, os.path.join(fig_dir, "return_ecdf.png"))
    has_paired = any(r["group"] == "nominal" for r in rows)
    if has_paired:
        fig_paired_delta(rows, os.path.join(fig_dir, "paired_delta_return.png"))
        fig_sensitivity(rows, os.path.join(fig_dir, "sensitivity.png"))
    fig_survival(rows, os.path.join(fig_dir, "survival.png"))

    rand_rows = [r for r in rows if r["group"] == "randomized"]
    summary_path = os.path.join(args.run, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {
            "n_trials": len(rand_rows),
            "return_mean": float(np.mean([r["return_mean"] for r in rand_rows])),
            "return_std": float(np.std([r["return_mean"] for r in rand_rows])),
            "survival_rate": float(np.mean([r["survived"] for r in rand_rows])),
        }
    print(f"Figures written to {fig_dir}/")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
