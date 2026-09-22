# LIMX Tron1 (WF_TRON1A) — wheeled-foot biped

Source: [LIMX Dynamics `pointfoot-mujoco-sim`](https://github.com/limxdynamics/pointfoot-mujoco-sim),
`robot-description/pointfoot/WF_TRON1A`. Meshes are copied verbatim; the MJCF is re-derived from
`WF_TRON1A/xml/robot.xml` with MJX-oriented changes.

## Changes from the upstream MJCF

- Floor, skybox and lights moved into `mjx_scene_tron1_wf.xml`; `meshdir` now points inside this package.
- Solver tuned for MJX (`iterations=2`, `ls_iterations=5`, `eulerdamp` disabled), matching `unitree_go2`.
- Wheel collision geom is the upstream **cylinder** (r=0.127 m, half-height 0.005 m, `quat` putting
  its local z on the hinge axis). MJX supports `PLANE x CYLINDER`. A sphere of the same radius was
  tried first and is cheaper (1 contact per wheel instead of 2), but the cylinder keeps the true
  disc edge and a real contact patch.
- Leg-link collision geoms (abad/hip/knee) are non-colliding. Only the base box and the two wheels
  touch the floor — 4 contacts when standing (2 per wheel).
- `wheel_L` / `wheel_R` sites added at the wheel centres.
- Wheel joints are `limited="false"` rather than carrying upstream's `range="-1e6 1e6"` sentinel, so
  that continuous rotation is expressed directly in the model.
- Added a `home` keyframe (absent upstream): a balanced crouch at base height 0.72 m with
  `hip=±0.40397`, `knee=±0.91652`, solved so the wheel-ground contact sits under the whole-body COM.

Actuator torque limits are upstream's: ±80 N·m on abad/hip/knee, ±40 N·m on the wheels.
