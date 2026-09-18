"""dial-mpc-sim2sim-view: inspect one saved sim2sim trajectory.

Reads the 50 Hz state logs written one subdirectory per trial (`<run>/interesting/
trial_0007/{group}.npz`) by `dial-mpc-sim2sim-eval`, or the identically-shaped
`<run>/reproduced/trial_0007/` written by `dial_mpc.sim2sim.reproduce`, and renders them
two ways:

  * a side-by-side time-series comparison of the two arms of the same trial -- same
    theta-perturbed plant and MPC seed, differing only in the planner's model (default);
    cheap, no JAX, answers "what went wrong and when";
  * a brax 3D playback (`--html`) -- rebuilds pipeline states from the logged qpos/qvel
    so the gait itself can be watched.

Usage:
    dial-mpc-sim2sim-view --run <run_dir>                 # list what was saved
    dial-mpc-sim2sim-view --run <run_dir> --trial 7       # compare both arms of trial 7
    dial-mpc-sim2sim-view --run <run_dir> --trial 7 --html
    dial-mpc-sim2sim-view --traj-dir <run>/reproduced/trial_0168 --html
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional, cast

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dial_mpc.sim2sim.groups import DISPLAY, GROUP_NOMINAL_PLANNER, GROUP_TRUE_PLANNER
from dial_mpc.sim2sim.analyze import (
    _BLUE,
    _INK,
    _INK_MUTED,
    _INK_SECONDARY,
    _ORANGE,
    _RED,
    _SURFACE,
    _style_axes,
)


def load_traj(path: str) -> Dict[str, Any]:
    """Load one .npz state log. `theta_json` is stored as a 0-d string array."""
    with np.load(path, allow_pickle=False) as z:
        out: Dict[str, Any] = {k: z[k] for k in z.files}
    for key in ("env_name", "group", "theta_json"):
        if key in out:
            out[key] = str(out[key])
    if "theta_json" in out:
        out["theta"] = json.loads(out["theta_json"])
    for key in ("trial", "seed"):
        if key in out:
            out[key] = int(out[key])
    return out


def load_traj_dir(trial_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load every group's .npz in a per-trial directory (e.g. `interesting/trial_0168/`
    or `reproduced/trial_0168/`), keyed by group name (the filename stem)."""
    trajs = {}
    for fn in sorted(os.listdir(trial_dir)):
        if fn.endswith(".npz"):
            trajs[fn[:-4]] = load_traj(os.path.join(trial_dir, fn))
    return trajs


def load_index(run_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(run_dir, "interesting", "index.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def print_index(manifest: Dict[str, Any]) -> None:
    print(f"Run: {manifest['run_dir']}")
    print(f"Config: {manifest['config']}   Metrics: {manifest['trials_csv']}")
    print(f"{len(manifest['entries'])} divergent trial pairs saved at "
          f"{manifest['log_rate_hz']:.0f} Hz\n")
    print(f"{'trial':>5}  {'d_return':>9}  {'d_steps':>7}  {'nom.planner':>12}  {'true planner':>12}  reasons")
    for e in manifest["entries"]:
        r, n = e[GROUP_NOMINAL_PLANNER], e[GROUP_TRUE_PLANNER]
        print("%5d  %9.4f  %7d  %5d st %s  %5d st %s  %s" % (
            e["trial"], e["delta_return"], e["delta_steps"],
            r["steps_survived"], "ok  " if r["survived"] else "FELL",
            n["steps_survived"], "ok  " if n["survived"] else "FELL",
            ",".join(e["reasons"]),
        ))
    print("\nCompare one with: dial-mpc-sim2sim-view --run <run> --trial <n>")


def _fall_marker(ax, traj: Dict[str, Any], color: str) -> None:
    """Mark the step where the episode terminated, if it did."""
    done = np.asarray(traj["done"]).reshape(-1)
    idx = np.nonzero(done > 0.5)[0]
    if idx.size:
        t = float(traj["time"][idx[0]])
        ax.axvline(t, color=color, linewidth=1.2, linestyle=(0, (2, 2)), zorder=5)


def fig_compare(trajs: Dict[str, Dict[str, Any]], out_path: str, title: str) -> None:
    """Time series of the quantities that explain a divergence: how high the torso is,
    whether it is tracking the commanded speed, and what the controller is paying."""
    colors = {GROUP_NOMINAL_PLANNER: _ORANGE, GROUP_TRUE_PLANNER: _BLUE}

    fig, axes = plt.subplots(4, 1, figsize=(9, 9), dpi=150, sharex=True)
    fig.patch.set_facecolor(_SURFACE)

    for group, traj in trajs.items():
        c = colors.get(group, _RED)
        t = np.asarray(traj["time"])

        label = DISPLAY.get(group, group)
        axes[0].plot(t, np.asarray(traj["torso_pos"])[:, 2], color=c, linewidth=2.0,
                     label=label, zorder=3)
        axes[1].plot(t, np.asarray(traj["vel_body"])[:, 0], color=c, linewidth=2.0,
                     label=label, zorder=3)
        axes[2].plot(t, np.asarray(traj["reward"]).reshape(-1), color=c, linewidth=2.0,
                     label=label, zorder=3)
        ctrl = np.asarray(traj["ctrl"])
        axes[3].plot(t, np.sqrt(np.mean(np.square(ctrl), axis=1)), color=c,
                     linewidth=2.0, label=label, zorder=3)
        for ax in axes:
            _fall_marker(ax, traj, c)

    # Commanded forward speed is identical for both arms; draw it once as a reference.
    any_traj = next(iter(trajs.values()))
    axes[1].plot(np.asarray(any_traj["time"]), np.asarray(any_traj["vel_tar"])[:, 0],
                 color=_INK_MUTED, linewidth=1.4, linestyle=(0, (4, 2)),
                 label="commanded", zorder=2)

    for ax, label in zip(axes, ["Torso height (m)", "Forward speed, body frame (m/s)",
                                "Reward per step", "Torque RMS"]):
        _style_axes(ax)
        ax.set_ylabel(label, fontsize=9)
    axes[-1].set_xlabel("Time (s)  —  logged at 50 Hz", fontsize=9)
    # One figure-level legend instead of per-axes boxes, which collided with the traces.
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=_INK_SECONDARY,
               ncol=3, loc="upper right", bbox_to_anchor=(0.99, 0.975))

    fig.suptitle(title, color=_INK, fontsize=12, x=0.010, y=0.985, ha="left")
    fig.text(0.010, 0.955, "Dashed vertical line = episode terminated (robot fell).",
             fontsize=8.5, color=_INK_SECONDARY, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    fig.savefig(out_path, facecolor=_SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


def render_html(traj: Dict[str, Any], run_dir: str, out_path: str) -> None:
    """Rebuild brax pipeline states from the logged qpos/qvel and write a 3D playback.

    The log stores only qpos/qvel (all that is needed to reconstruct a pose), so this
    re-runs forward kinematics through the env's own pipeline rather than storing full
    pipeline states, which would be far larger on disk.
    """
    import jax
    import jax.numpy as jnp
    import yaml
    from brax.base import System
    from brax.io import html

    import dial_mpc.envs as dial_envs
    from dial_mpc.core.dial_config import DialConfig
    from dial_mpc.utils.io_utils import load_dataclass_from_dict
    from dial_mpc.sim2sim.runner import build_envs

    config_path = os.path.join(run_dir, "config_used.yaml")
    if not os.path.exists(config_path):
        raise SystemExit(f"Need {config_path} to rebuild the env for rendering.")
    config_dict = yaml.safe_load(open(config_path))
    dial_config = load_dataclass_from_dict(DialConfig, config_dict)
    env_config = load_dataclass_from_dict(
        dial_envs.get_config(dial_config.env_name), config_dict, convert_list_to_array=True
    )
    _, env = build_envs(dial_config.env_name, env_config)

    qpos = np.asarray(traj["qpos"])
    qvel = np.asarray(traj["qvel"])
    # Rendering only needs geometry + pose; the randomized masses do not change how the
    # model looks, so the nominal sys is used for both arms.
    states = jax.vmap(env.pipeline_init)(jnp.asarray(qpos), jnp.asarray(qvel))
    frames = [jax.tree.map(lambda x, i=i: x[i], states) for i in range(qpos.shape[0])]
    with open(out_path, "w") as f:
        render_sys = cast(System, env.sys.tree_replace({"opt.timestep": env.dt}))
        f.write(html.render(render_sys, frames))
    print(f"Wrote {out_path}  (open in a browser)")


def generate_trial_outputs(run_dir: str, trial_dir: str, title: str, html: bool = False) -> None:
    """Given a directory holding one trial's per-group .npz logs (`<group>.npz`), write
    `compare.png` there, and optionally `<group>.html` for each group.

    The one place that turns saved .npz logs into figures/playbacks, shared by this
    module's own `--trial`/`--traj-dir` modes and by `dial_mpc.sim2sim.reproduce`, which
    writes its output in the identical per-trial-directory shape under `reproduced/`.
    """
    trajs = load_traj_dir(trial_dir)
    fig_compare(trajs, os.path.join(trial_dir, "compare.png"), title)
    if html:
        for group, traj in trajs.items():
            render_html(traj, run_dir, os.path.join(trial_dir, f"{group}.html"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None, help="a sim2sim run directory")
    parser.add_argument("--trial", type=int, default=None,
                        help="trial index to compare (needs --run)")
    parser.add_argument("--traj-dir", type=str, default=None,
                        help="a per-trial directory of .npz logs, "
                             "e.g. <run>/interesting/trial_0168 or <run>/reproduced/trial_0168")
    parser.add_argument("--html", action="store_true", help="also write brax 3D playbacks")
    args = parser.parse_args()

    if args.traj_dir:
        trial_dir = os.path.abspath(args.traj_dir)
        run_dir = os.path.dirname(os.path.dirname(trial_dir))
        generate_trial_outputs(run_dir, trial_dir, os.path.basename(trial_dir), html=args.html)
        return

    if not args.run:
        raise SystemExit("Pass --run <run_dir> (optionally with --trial N) or --traj-dir <dir>")

    manifest = load_index(args.run)
    if manifest is None:
        raise SystemExit(
            f"No {args.run}/interesting/index.json -- re-run the sweep with "
            "domain_randomization.n_interesting > 0."
        )
    if args.trial is None:
        print_index(manifest)
        return

    entry = next((e for e in manifest["entries"] if e["trial"] == args.trial), None)
    if entry is None:
        have = ", ".join(str(e["trial"]) for e in manifest["entries"])
        raise SystemExit(f"Trial {args.trial} was not saved. Available: {have}")

    trial_dir = os.path.join(args.run, "interesting", entry["dir"])
    title = (f"Trial {entry['trial']} — Δreturn {entry['delta_return']:+.4f}, "
             f"Δsteps {entry['delta_steps']:+d}  ({', '.join(entry['reasons'])})")
    generate_trial_outputs(os.path.abspath(args.run), trial_dir, title, html=args.html)

    print("\nParameters for this trial (theta):")
    for k, v in entry["theta"].items():
        arr = np.asarray(v)
        if arr.size == 1:
            print(f"  {k:<14} {float(arr.reshape(())):+.4f}")
        else:
            print(f"  {k:<14} mean {float(arr.mean()):+.4f}  "
                  f"[{float(arr.min()):+.4f}, {float(arr.max()):+.4f}]  ({arr.size} elements)")
    print(f"\nFull metric row: {manifest['trials_csv']} (trial={entry['trial']}), "
          f"config: {manifest['config']}")


if __name__ == "__main__":
    main()
