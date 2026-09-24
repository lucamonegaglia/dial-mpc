"""Is the low-armature divergence a discretisation artefact? Spectral radius of the joint-level
PD loop (floating base, gravity/contact off, as in stability.py) under integration variants:

  base      : 1 Euler step of 20 ms, PD outside MuJoCo, eulerdamp disabled (the H1 loco setup)
  zoh2      : torque computed once per 20 ms, held over 2 substeps of 10 ms
  pd2       : PD recomputed every 10 ms substep (2 substeps)
  implicitD : 1 step of 20 ms, kd moved into dof_damping with eulerdamp enabled (implicit damping)
"""

from __future__ import annotations

import json
import os

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from common import RESULTS, build

SCALES = [0.35, 0.5, 0.6, 0.7, 0.85, 1.0, 1.4, 2.0]
VARIANTS = ["base", "zoh2", "pd2", "implicitD"]


def main():
    h = build()
    env = h.stepper.plant_env
    dt = float(env.dt)
    kp = jnp.asarray(h.stepper.nominal.kp)
    kd = jnp.asarray(h.stepper.nominal.kd)
    q0 = jnp.asarray(env.sys.mj_model.keyframe("home").qpos)
    nv, nu = env.sys.nv, len(kp)
    target = q0[7:]
    out = {}
    for s in SCALES:
        sys = h.model(s).sys
        flags = sys.opt.disableflags | mujoco.mjtDisableBit.mjDSBL_CONTACT
        sys = sys.tree_replace({"opt.gravity": jnp.zeros(3), "opt.disableflags": flags})
        half = sys.tree_replace({"opt.timestep": jnp.asarray(dt / 2)})
        impl = sys.tree_replace({
            "opt.disableflags": flags & ~int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP),
            "dof_damping": sys.dof_damping.at[6:].add(kd),
        })

        def make_map(variant):
            def step_map(x):
                qpos = q0.at[7:].set(x[:nu] + target)
                qvel = jnp.zeros(nv).at[6:].set(x[nu:])
                m = {"base": sys, "zoh2": half, "pd2": half, "implicitD": impl}[variant]
                d = mjx.forward(m, mjx.make_data(m).replace(qpos=qpos, qvel=qvel))
                n_sub = 2 if variant in ("zoh2", "pd2") else 1
                kd_ext = 0.0 if variant == "implicitD" else kd
                tau = kp * (target - d.qpos[7:]) - kd_ext * d.qvel[6:]
                for _ in range(n_sub):
                    if variant == "pd2":
                        tau = kp * (target - d.qpos[7:]) - kd * d.qvel[6:]
                    d = mjx.step(m, d.replace(ctrl=tau))
                return jnp.concatenate([d.qpos[7:] - target, d.qvel[6:]])
            return step_map

        row = {}
        for v in VARIANTS:
            with jax.default_matmul_precision("highest"):
                J = np.asarray(jax.jacfwd(make_map(v))(jnp.zeros(2 * nu)))
            ev = np.linalg.eigvals(J)
            row[v] = {"rho": float(np.abs(ev).max()), "min_re": float(ev.real.min())}
        out[s] = row
        print(f"armature x{s:4.2f}: " + "  ".join(f"{v}: rho={row[v]['rho']:.3f} (minRe {row[v]['min_re']:+.2f})"
                                                   for v in VARIANTS))
    with open(os.path.join(RESULTS, "stability_dt.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
