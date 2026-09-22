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
import textwrap
from typing import Any, Dict, List, Optional, Sequence, cast

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dial_mpc.sim2sim.groups import (
    DISPLAY, GROUPS, GROUP_NOMINAL_PLANNER, GROUP_TRUE_PLANNER,
)

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
        for r in csv.DictReader(f):
            parsed = {}
            for k, v in r.items():
                if v == "" or v is None:
                    parsed[k] = None
                elif k == "group":
                    parsed[k] = v
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            rows.append(parsed)
    return rows


def _theta_columns(rows: List[Dict], declared: Optional[Sequence[str]]) -> List[str]:
    """Which CSV columns are randomized parameters, as recorded by the sweep."""
    if not rows:
        return []
    if not declared:
        raise SystemExit(
            "summary.json has no `theta_columns`; re-run the sweep to produce it."
        )
    return sorted(c for c in declared if c in rows[0])


def fig_return_ecdf(rows: List[Dict], out_path: str):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    for group, color, label in ((GROUP_NOMINAL_PLANNER, _BLUE, DISPLAY[GROUP_NOMINAL_PLANNER]),
                                (GROUP_TRUE_PLANNER, _ORANGE, DISPLAY[GROUP_TRUE_PLANNER])):
        # A trial that diverged before its first step has a NaN return (runner.py) and
        # would otherwise sort to the end of the ECDF as if it were the best trial.
        vals = sorted(v for v in (r["return_sum"] for r in rows if r["group"] == group)
                      if np.isfinite(v))
        if not vals:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where="post", color=color, linewidth=2.0, solid_capstyle="round", label=label)
    _style_axes(ax)
    ax.set_xlabel("Total trajectory reward")
    ax.set_ylabel("Cumulative fraction of trials")
    ax.set_title("Return distribution: domain shift vs nominal", color=_INK, fontsize=11, loc="left")
    # Both curves rise steeply into the lower-right corner, so a legend there sits on top
    # of the lines; anchor at upper-left instead, where the curves are flat near y=0.
    leg = ax.legend(frameon=True, fontsize=9, loc="upper left",
                     facecolor=_SURFACE, edgecolor=_BASELINE, framealpha=0.95)
    for text in leg.get_texts():
        text.set_color(_INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_paired_delta(rows: List[Dict], out_path: str):
    by_trial = {}
    for r in rows:
        by_trial.setdefault(int(r["trial"]), {})[r["group"]] = r
    deltas = [
        d[GROUP_NOMINAL_PLANNER]["return_sum"] - d[GROUP_TRUE_PLANNER]["return_sum"]
        for d in by_trial.values()
        if GROUP_NOMINAL_PLANNER in d and GROUP_TRUE_PLANNER in d
    ]
    deltas = [d for d in deltas if np.isfinite(d)]
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
    # A handful of trials sit far out in either tail; a linear y-axis makes the near-zero
    # bulk unreadable once they're in frame, so log-scale keeps both visible without
    # changing the bins or hiding the outliers.
    ax.set_yscale("symlog", linthresh=1)
    _style_axes(ax)
    ax.set_xlabel("\n".join(textwrap.wrap(
        "Δ total reward (nominal-parameter − true-parameter planner, same plant & seed).",
        width=60)))
    ax.set_ylabel("Trial count (log)")
    ax.set_title("Cost of planning with the wrong model", color=_INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_survival(rows: List[Dict], out_path: str):
    fig, ax = plt.subplots(figsize=(4.5, 4), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    groups = [g for g in GROUPS if any(r["group"] == g for r in rows)]
    colors = {GROUP_NOMINAL_PLANNER: _BLUE, GROUP_TRUE_PLANNER: _ORANGE}
    labels = {g: DISPLAY[g] for g in GROUPS}
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
    fig.savefig(out_path, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_sensitivity(rows: List[Dict], theta_cols: List[str], out_path: str):
    by_trial = {}
    for r in rows:
        by_trial.setdefault(int(r["trial"]), {})[r["group"]] = r
    paired = [d for d in by_trial.values() if GROUP_NOMINAL_PLANNER in d and GROUP_TRUE_PLANNER in d]
    if not paired:
        return
    theta_cols = [c for c in theta_cols if all(d[GROUP_NOMINAL_PLANNER].get(c) is not None for d in paired)]
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
        x = np.array([d[GROUP_NOMINAL_PLANNER][col] for d in paired])
        y = np.array([d[GROUP_NOMINAL_PLANNER]["return_sum"] - d[GROUP_TRUE_PLANNER]["return_sum"] for d in paired])
        ax.scatter(x, y, s=14, color=_BLUE, alpha=0.75, edgecolors="none", zorder=2)
        ax.axhline(0.0, color=_INK_MUTED, linewidth=1.0, linestyle=(0, (3, 2)), zorder=1)
        rho = spearman(x, y)
        _style_axes(ax)
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel("Δ return", fontsize=8)
        ax.set_title(f"ρ = {rho:.2f}", fontsize=9, color=_INK, loc="left")

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("Per-parameter sensitivity of planner model error",
                 color=_INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (3.2 * nrows + 0.5)))
    fig.savefig(out_path, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)


def _spec_groups(theta_cols: List[str]) -> Dict[str, List[str]]:
    """Group per-element theta columns back under the spec that drew them.

    `sweep.py` flattens a per-element draw into `<name>_<i>` (or `<name>_<i>_<j>`), so
    `limb_mass_3` and `limb_mass_7` are two elements of one randomization axis. For a
    summary chart the axis is the unit of interest, not its individual elements.
    """
    groups: Dict[str, List[str]] = {}
    for col in theta_cols:
        base = col
        while True:
            head, sep, tail = base.rpartition("_")
            if sep and tail.isdigit():
                base = head
            else:
                break
        groups.setdefault(base, []).append(col)
    return groups


def _group_value(row: Dict, cols: List[str]) -> float:
    """Scalar summary of one randomization axis for one trial.

    For a single-element axis this is just the value. For a per-element axis it is the
    mean across elements, which is the physically meaningful aggregate: the mean of 11
    iid limb-mass scale factors is (proportional to) total limb mass, and the mean of a
    per-axis CoM offset is its net displacement. Individual-element effects are left to
    the detailed scatter grid.
    """
    vals = [row[c] for c in cols if row.get(c) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def _standardized_effect(x: np.ndarray, y: np.ndarray, n_boot: int = 2000,
                         seed: int = 0) -> Dict[str, float]:
    """OLS slope of y on x, expressed per 1 SD of x, with a bootstrap 95% interval.

    Standardizing matters because the axes are not in comparable units: `friction` and
    `limb_mass` are dimensionless scale factors, `com_offset` is metres, `damping` is
    log-uniform. A raw slope would make the metre-scale axis look negligible purely
    because its numbers are small. "Effect per 1 SD of the sampled range" puts every
    axis on the same footing: how much does the metric move across the spread this
    sweep actually explored?
    """
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 8 or np.std(x) == 0:
        return {"beta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "spearman": float("nan"), "n": float(len(x))}
    sd = float(np.std(x))
    slope = float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    n = len(x)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        boot[b] = np.polyfit(xb, yb, 1)[0] * np.std(xb) if np.std(xb) > 0 else np.nan
    boot = boot[np.isfinite(boot)]
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rho = float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > 0 and np.std(ry) > 0 else float("nan")
    return {
        "beta": slope * sd,
        "lo": float(np.percentile(boot, 2.5)) if boot.size else float("nan"),
        "hi": float(np.percentile(boot, 97.5)) if boot.size else float("nan"),
        "spearman": rho,
        "n": float(n),
    }


def fig_sensitivity_summary(rows: List[Dict], theta_cols: List[str], out_path: str,
                            csv_path: str):
    """One bar per randomization axis, ranked by how much it moves the paired delta.

    Two responses side by side because they decompose the same failure differently: total
    reward folds in both how well the plant tracked and how long it stayed up (the env
    weights `reward_alive` at 1.0), while steps-survived isolates the survival half. A
    parameter that shows up in the first but not the second moved tracking quality, not
    uptime.
    """
    by_trial: Dict[int, Dict[str, Dict]] = {}
    for r in rows:
        by_trial.setdefault(int(r["trial"]), {})[cast(str, r["group"])] = r
    paired = [d for d in by_trial.values() if GROUP_NOMINAL_PLANNER in d and GROUP_TRUE_PLANNER in d]
    if len(paired) < 8:
        return

    groups = _spec_groups([c for c in theta_cols
                           if all(d[GROUP_NOMINAL_PLANNER].get(c) is not None for d in paired)])
    if not groups:
        return

    d_ret = np.array([d[GROUP_NOMINAL_PLANNER]["return_sum"] - d[GROUP_TRUE_PLANNER]["return_sum"] for d in paired])
    d_steps = np.array([d[GROUP_NOMINAL_PLANNER]["steps_survived"] - d[GROUP_TRUE_PLANNER]["steps_survived"] for d in paired])

    stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, cols in groups.items():
        x = np.array([_group_value(d[GROUP_NOMINAL_PLANNER], cols) for d in paired])
        stats[name] = {
            "return": _standardized_effect(x, d_ret),
            "steps": _standardized_effect(x, d_steps),
        }

    order = sorted(stats, key=lambda k: -abs(stats[k]["return"]["beta"]))
    ypos = np.arange(len(order))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(order) + 2.9), dpi=150, sharey=True)
    fig.patch.set_facecolor(_SURFACE)

    panels = [("return", "Δ total reward", axes[0]), ("steps", "Δ steps survived", axes[1])]
    for key, label, ax in panels:
        betas = np.array([stats[k][key]["beta"] for k in order])
        los = np.array([stats[k][key]["lo"] for k in order])
        his = np.array([stats[k][key]["hi"] for k in order])
        colors = [_RED if b < 0 else _BLUE for b in betas]
        ax.barh(ypos, betas, height=0.6, color=colors, zorder=3)
        # A bar whose interval crosses zero is not distinguishable from no effect.
        for yp, b, lo, hi in zip(ypos, betas, los, his):
            ax.plot([lo, hi], [yp, yp], color=_INK_SECONDARY, linewidth=1.4, zorder=4,
                    solid_capstyle="butt")
        ax.axvline(0.0, color=_BASELINE, linewidth=1.2, zorder=2)
        _style_axes(ax)
        ax.set_yticks(ypos)
        ax.set_yticklabels(order, fontsize=9)
        ax.set_xlabel(f"{label}   per 1 SD of parameter", fontsize=9)

    subtitle = textwrap.fill(
        "Same plant and MPC seed in both arms; only the planner's model differs. Bars: OLS effect "
        "per 1 SD of the sampled range. Lines: bootstrap 95% CI -- crossing 0 means no detected "
        "effect.",
        width=118)
    n_subtitle_lines = subtitle.count("\n") + 1
    fig.suptitle("Which model errors hurt the planner most?",
                 color=_INK, fontsize=12.5, x=0.010, y=0.985, ha="left")
    fig.text(0.010, 0.945, subtitle, fontsize=8.5, color=_INK_SECONDARY, ha="left", va="top")
    top_margin = 0.915 - 0.02 * max(0, n_subtitle_lines - 1)
    fig.tight_layout(rect=(0, 0, 1, top_margin))
    fig.savefig(out_path, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "response", "effect_per_1sd", "ci_lo", "ci_hi",
                    "spearman_rho", "n_trials"])
        for k in order:
            for key in ("return", "steps"):
                st = stats[k][key]
                w.writerow([k, key, f"{st['beta']:.6g}", f"{st['lo']:.6g}",
                            f"{st['hi']:.6g}", f"{st['spearman']:.4f}", int(st["n"])])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True, help="path to a sim2sim_<timestamp> run directory")
    args = parser.parse_args()

    rows = load_trials(args.run)
    if not rows:
        raise SystemExit(f"No trials found in {args.run}/trials.csv")

    summary_path = os.path.join(args.run, "summary.json")
    summary: Dict[str, Any] = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    fig_dir = os.path.join(args.run, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    fig_return_ecdf(rows, os.path.join(fig_dir, "return_ecdf.png"))
    has_paired = any(r["group"] == GROUP_TRUE_PLANNER for r in rows)
    if has_paired:
        theta_cols = _theta_columns(rows, summary.get("theta_columns"))
        fig_paired_delta(rows, os.path.join(fig_dir, "paired_delta_return.png"))
        fig_sensitivity(rows, theta_cols, os.path.join(fig_dir, "sensitivity.png"))
        fig_sensitivity_summary(
            rows, theta_cols,
            os.path.join(fig_dir, "sensitivity_summary.png"),
            os.path.join(args.run, "sensitivity.csv"),
        )
    fig_survival(rows, os.path.join(fig_dir, "survival.png"))

    if not summary:
        rand_rows = [r for r in rows if r["group"] == GROUP_NOMINAL_PLANNER]
        summary = {
            "n_trials": len(rand_rows),
            "return_sum_mean": float(np.nanmean([r["return_sum"] for r in rand_rows])),
            "return_sum_std": float(np.nanstd([r["return_sum"] for r in rand_rows])),
            "survival_rate": float(np.mean([r["survived"] for r in rand_rows])),
        }
    print(f"Figures written to {fig_dir}/")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
