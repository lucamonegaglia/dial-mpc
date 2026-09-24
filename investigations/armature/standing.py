"""Open-loop standing test on CPU: hold the action that targets the home pose, kick qvel by
1e-3, and measure whether the period-2 (flip-flop) joint mode grows or decays with the
robot in ground contact. Complements stability.py, which linearises in the air."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json

import jax
import jax.numpy as jnp
import numpy as np

from common import RESULTS, build

SCALES = [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0, 1.4, 2.0]
T = 150


def main():
    jax.config.update("jax_default_matmul_precision", "highest")
    h = build()
    env = h.stepper.plant_env
    jr = np.asarray(env.joint_range)
    home = np.asarray(env._default_pose)
    a_home = jnp.asarray(2 * (home - jr[:, 0]) / (jr[:, 1] - jr[:, 0]) - 1)
    out = {}
    for s in SCALES:
        model = h.model(s)
        st = h.stepper.reset_jit(model, jax.random.PRNGKey(0))
        for _ in range(25):  # settle onto the ground
            st = h.stepper.step_jit(model, st, a_home)
        ps = st.pipeline_state
        kick = 1e-3 * jax.random.normal(jax.random.PRNGKey(1), ps.qvel.shape)
        st = st.replace(pipeline_state=ps.replace(qvel=ps.qvel + kick))
        qv, done_at = [], None
        for t in range(T):
            st = h.stepper.step_jit(model, st, a_home)
            v = np.asarray(st.pipeline_state.qvel[6:])
            if not np.isfinite(v).all():
                done_at = t
                break
            qv.append(v)
        qv = np.array(qv)
        # period-2 component: (-1)^t projection of joint velocities
        alt = qv * ((-1.0) ** np.arange(len(qv)))[:, None]
        amp_early = float(np.abs(alt[:10].mean(0)).max()) if len(qv) >= 10 else float("nan")
        amp_late = float(np.abs(alt[-10:].mean(0)).max()) if len(qv) >= 20 else float("nan")
        lag1 = float(np.nanmean([np.corrcoef(qv[:-1, j], qv[1:, j])[0, 1] for j in range(qv.shape[1])]))
        rms_late = float(np.sqrt(np.mean(qv[-20:] ** 2)))
        out[s] = {"flipflop_amp_first10": amp_early, "flipflop_amp_last10": amp_late,
                  "qvel_rms_last20": rms_late, "qvel_lag1": lag1, "nonfinite_at": done_at,
                  "torso_z_end": float(st.pipeline_state.x.pos[h.stepper.torso_idx - 1, 2])}
        print(f"armature x{s:4.2f}: flip-flop amp {amp_early:.2e} -> {amp_late:.2e}, "
              f"qvel rms(last20) {rms_late:.3e}, lag1 {lag1:+.2f}, z_end {out[s]['torso_z_end']:.2f}"
              + (f", NONFINITE at t={done_at}" if done_at is not None else ""))
    with open(os.path.join(RESULTS, "standing.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
