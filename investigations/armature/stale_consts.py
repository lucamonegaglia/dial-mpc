"""How stale are the mj_setConst-derived constants when only dof_armature is rescaled?

`apply_theta` rewrites `dof_armature` but not `body_invweight0` / `dof_invweight0`, which
MJX reads to set constraint regularisation (contact / joint-limit R). This compares the
values the harness uses (computed at nominal armature) with a recomputed mj_setConst.
Symmetric between arms (plant and planner both carry the stale values), so it is a
model-consistency caveat rather than a source of plant/planner mismatch.
"""

import mujoco
import numpy as np

from dial_mpc.utils.io_utils import get_model_path

m = mujoco.MjModel.from_xml_path(str(get_model_path("unitree_h1", "mjx_scene_h1_loco.xml")))
d = mujoco.MjData(m)
b0, d0 = m.body_invweight0.copy(), m.dof_invweight0.copy()
act = np.arange(6, m.nv)
arm0 = m.dof_armature.copy()
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
feet = [n for n in names if n and "ankle" in n]
for s in [0.5, 2.0]:
    m.dof_armature[act] = arm0[act] * s
    mujoco.mj_setConst(m, d)
    rb = m.body_invweight0[:, 0] / np.maximum(b0[:, 0], 1e-12)
    rd = m.dof_invweight0[act] / d0[act]
    print(f"armature x{s}: body_invweight0 (translational) true/stale:",
          {n: round(float(r), 3) for n, r in zip(names, rb) if n in feet + ["pelvis", "torso_link"]})
    print(f"   dof_invweight0 true/stale (actuated): {np.round(rd, 3)}")
m.dof_armature[:] = arm0
mujoco.mj_setConst(m, d)
colliding = sorted({names[m.geom_bodyid[g]] for g in range(m.ngeom) if m.geom_contype[g]})
print("colliding bodies:", colliding)
print("opt: iterations", m.opt.iterations, "ls_iterations", m.opt.ls_iterations,
      "integrator", m.opt.integrator, "cone", m.opt.cone, "solver", m.opt.solver,
      "disableflags", m.opt.disableflags)
