"""Discrete-time stability of the plant's joint-level PD loop vs armature scale.

The H1 loco model integrates with one explicit Euler step of 20 ms (timestep == dt),
`eulerdamp="disable"`, and the PD torque kp*(q*-q) - kd*qd is computed outside MuJoCo, so
stiffness and *all* damping (kd and dof_damping) are explicit. Semi-implicit Euler on
I*qdd = -k q - c qd has det = 1 - c dt/I and trace = 2 - k dt^2/I - c dt/I, so it is
stable iff c dt/I < 2 and k dt^2/I + 2 c dt/I < 4. Armature adds directly to I.

Two estimates:
  * analytic: per actuated joint, with I = 1 / (M^-1)_ii at the home pose;
  * exact: jacobian of the full MJX one-step map (PD + physics, floating base, gravity and
    contacts off so the linearisation is of the joint loop only) -> spectral radius.
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

SCALES = [0.25, 0.35, 0.5, 0.6, 0.7, 0.85, 1.0, 1.2, 1.4, 1.7, 2.0, 3.0]


def main():
    os.makedirs(RESULTS, exist_ok=True)
    h = build()
    env = h.stepper.plant_env
    dt = float(env.dt)
    kp = np.asarray(h.stepper.nominal.kp)
    kd = np.asarray(h.stepper.nominal.kd)
    names = [mujoco.mj_id2name(env.sys.mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
             for j in range(1, env.sys.mj_model.njnt)]
    out = {"dt": dt, "joints": names, "scales": SCALES, "analytic": {}, "exact": {}}

    q0 = jnp.asarray(env.sys.mj_model.keyframe("home").qpos)
    nv = env.sys.nv
    nu = len(kp)

    for s in SCALES:
        model = h.model(s)
        sys = model.sys
        # CPU mass matrix at home pose with the scaled armature
        mjm = env.sys.mj_model
        arm = np.asarray(sys.dof_armature)
        mjm_s = mjm.__copy__()
        mjm_s.dof_armature[:] = arm
        d = mujoco.MjData(mjm_s)
        d.qpos[:] = np.asarray(q0)
        mujoco.mj_forward(mjm_s, d)
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(mjm_s, d, M)
        Minv = np.linalg.inv(M)
        I_app = 1.0 / np.diag(Minv)[6:6 + nu]
        c = kd + np.asarray(sys.dof_damping)[6:6 + nu]
        a = kp * dt * dt / I_app
        b = c * dt / I_app
        rho = []
        for ai, bi in zip(a, b):
            A = np.array([[1 - ai, 1 - bi], [-ai, 1 - bi]])
            rho.append(float(np.max(np.abs(np.linalg.eigvals(A)))))
        out["analytic"][str(s)] = {
            "I_app": I_app.tolist(), "armature": arm[6:6 + nu].tolist(),
            "kp_dt2_over_I": a.tolist(), "c_dt_over_I": b.tolist(), "rho": rho,
        }

        # exact jacobian of the MJX step under PD torque, no gravity / no contact
        sys_nc = sys.tree_replace({
            "opt.gravity": jnp.zeros(3),
            "opt.disableflags": sys.opt.disableflags | mujoco.mjtDisableBit.mjDSBL_CONTACT,
        })
        target = q0[7:]

        def step_map(x):
            qpos = q0.at[7:].set(x[:nu] + target)
            qvel = jnp.zeros(nv).at[6:].set(x[nu:])
            dmj = mjx.make_data(sys_nc).replace(qpos=qpos, qvel=qvel)
            dmj = mjx.forward(sys_nc, dmj)
            tau = kp * (target - dmj.qpos[7:]) - kd * dmj.qvel[6:]
            dmj = dmj.replace(ctrl=tau)
            dmj = mjx.step(sys_nc, dmj)
            return jnp.concatenate([dmj.qpos[7:] - target, dmj.qvel[6:]])

        with jax.default_matmul_precision("highest"):
            J = np.asarray(jax.jacfwd(step_map)(jnp.zeros(2 * nu)))
        ev = np.linalg.eigvals(J)
        out["exact"][str(s)] = {
            "rho": float(np.max(np.abs(ev))),
            "eig_abs_sorted": sorted(np.abs(ev).tolist(), reverse=True)[:6],
            "most_negative_real": float(np.min(ev.real)),
        }
        print(f"scale {s:4.2f}: exact rho={out['exact'][str(s)]['rho']:.3f} "
              f"min Re={out['exact'][str(s)]['most_negative_real']:+.3f} | analytic rho per joint "
              + " ".join(f"{r:.2f}" for r in rho))

    with open(os.path.join(RESULTS, "stability.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
