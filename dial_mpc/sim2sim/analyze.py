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
from matplotlib.ticker import MaxNLocator

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
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # counts, so no half-trial ticks
    _style_axes(ax)
    ax.set_xlabel("\n".join(textwrap.wrap(
        "Δ total reward (nominal-parameter − true-parameter planner, same plant & seed).",
        width=60)))
    ax.set_ylabel("Trial count")
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


def _binned_delta(x: np.ndarray, y: np.ndarray, n_bins: int = 5, n_boot: int = 2000,
                  seed: int = 0) -> List[Dict[str, float]]:
    """Mean of y within equal-count quantile bins of x, with a bootstrap 95% CI per bin.

    Reports the raw paired delta where it happens rather than a linear slope: a slope
    hides the level (a large constant gap gives slope 0) and folds a V-shaped response
    (mismatch hurting on both sides of nominal) into a misleading single sign.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2 * n_bins or np.ptp(x) == 0:
        return []
    order = np.argsort(x, kind="stable")
    rng = np.random.default_rng(seed)
    bins = []
    for idx in np.array_split(order, n_bins):
        xb, yb = x[idx], y[idx]
        boot = rng.choice(yb, size=(n_boot, len(yb)), replace=True).mean(axis=1)
        bins.append({
            "x_lo": float(xb.min()), "x_hi": float(xb.max()), "x_med": float(np.median(xb)),
            "n": float(len(yb)), "mean": float(yb.mean()),
            "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)),
        })
    return bins


def _plot_binned_panels(stats: Dict[str, List[Dict[str, float]]], overall: float, label: str,
                        out_path: str):
    order = sorted(stats, key=lambda k: -np.ptp([b["mean"] for b in stats[k]]))
    n = len(order)
    ncols = min(5, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.5 * nrows + 1.0), dpi=150,
                             sharey=True, squeeze=False)
    fig.patch.set_facecolor(_SURFACE)
    for i, name in enumerate(order):
        ax = axes[i // ncols][i % ncols]
        bins = stats[name]
        xm = np.array([b["x_med"] for b in bins])
        mean = np.array([b["mean"] for b in bins])
        lo = np.array([b["lo"] for b in bins])
        hi = np.array([b["hi"] for b in bins])
        for k, b in enumerate(bins):
            if k % 2 == 0:
                ax.axvspan(b["x_lo"], b["x_hi"], color=_GRID, alpha=0.6, linewidth=0, zorder=0)
        ax.axhline(0.0, color=_BASELINE, linewidth=1.2, zorder=1)
        ax.axhline(overall, color=_INK_MUTED, linewidth=1.0, linestyle=(0, (3, 2)), zorder=1)
        ax.vlines(xm, lo, hi, color=_INK_SECONDARY, linewidth=1.4, zorder=2)
        ax.plot(xm, mean, color=_BLUE, linewidth=2.0, zorder=3)
        ax.scatter(xm, mean, s=40, color=_BLUE, edgecolors=_SURFACE, linewidths=1.5, zorder=4)
        _style_axes(ax)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_xlabel(name, fontsize=9)
        ax.set_title(f"spread {np.ptp(mean):.0f}", fontsize=8.5, color=_INK_SECONDARY, loc="left")
        if i % ncols == 0:
            ax.set_ylabel(label, fontsize=9)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    subtitle = textwrap.fill(
        f"Bins: per panel, "
        f"trials are sorted by the plant's sampled value of that parameter (mean over elements for "
        f"per-element axes) and split into {len(stats[order[0]])} equal-count bins of "
        f"~{int(stats[order[0]][0]['n'])} trials; shaded bands alternate to show each bin's value "
        f"range. Points: mean Δ in the bin at the bin's median value; vertical lines: bootstrap 95% "
        f"CI. Solid line: Δ = 0; dashed: mean over all trials ({overall:.1f}). Below 0 = the "
        f"true-parameter planner did better. Panels ranked by spread (max − min of the bin means). "
        f"Same plant and MPC seed in both arms; only the planner's model differs.",
        width=int(26 * ncols))
    n_lines = subtitle.count("\n") + 1
    fig_h = fig.get_figheight()
    fig.suptitle(f"How the planner's model error varies with each parameter   "
                 f"{label} = nominal-planner − true-planner"
                 f"{n} paired trials)",
                 color=_INK, fontsize=12, x=0.010, y=1 - 0.12 / fig_h, ha="left", va="top")
    fig.text(0.010, 1 - 0.45 / fig_h, subtitle, fontsize=8.5, color=_INK_SECONDARY,
             ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.5 + 0.15 * n_lines) / fig_h))
    fig.savefig(out_path, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_sensitivity_summary(rows: List[Dict], theta_cols: List[str], out_dir: str,
                            csv_path: str):
    """Raw paired delta by parameter quintile, one small multiple per randomization axis.

    Written for both total reward and steps survived: total reward folds in tracking
    quality and uptime (the env weights `reward_alive` at 1.0), steps-survived isolates
    uptime. Replaces the earlier per-1-SD OLS slope summary (commit 6cabdd6), whose
    sign was routinely misread as the sign of the delta itself.
    """
    by_trial: Dict[int, Dict[str, Dict]] = {}
    for r in rows:
        by_trial.setdefault(int(r["trial"]), {})[cast(str, r["group"])] = r
    paired = [d for d in by_trial.values() if GROUP_NOMINAL_PLANNER in d and GROUP_TRUE_PLANNER in d]
    if len(paired) < 10:
        return

    groups = _spec_groups([c for c in theta_cols
                           if all(d[GROUP_NOMINAL_PLANNER].get(c) is not None for d in paired)])
    if not groups:
        return

    responses = {
        "return": ("Δ total reward", "return_sum"),
        "steps": ("Δ steps survived", "steps_survived"),
    }
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "response", "bin", "x_lo", "x_hi", "x_median", "n_trials",
                    "mean_delta", "ci_lo", "ci_hi"])
        for key, (label, field) in responses.items():
            delta = np.array([d[GROUP_NOMINAL_PLANNER][field] - d[GROUP_TRUE_PLANNER][field]
                              for d in paired])
            stats = {}
            for name, cols in groups.items():
                x = np.array([_group_value(d[GROUP_NOMINAL_PLANNER], cols) for d in paired])
                bins = _binned_delta(x, delta)
                if bins:
                    stats[name] = bins
            if not stats:
                continue
            _plot_binned_panels(stats, float(np.nanmean(delta)), label,
                                os.path.join(out_dir, f"sensitivity_summary_{key}.png"))
            for name, bins in stats.items():
                for i, b in enumerate(bins):
                    w.writerow([name, key, i, f"{b['x_lo']:.6g}", f"{b['x_hi']:.6g}",
                                f"{b['x_med']:.6g}", int(b["n"]), f"{b['mean']:.6g}",
                                f"{b['lo']:.6g}", f"{b['hi']:.6g}"])


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
        fig_sensitivity_summary(rows, theta_cols, fig_dir, os.path.join(args.run, "sensitivity.csv"))
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
