"""The two experiment arms, named in one place.

Both arms run a **theta-perturbed plant**. The only thing that differs is what model the
planner is given, which is why neither is called "randomized":

    nominal_planner   plant = theta,  planner = nominal parameters   (domain shift)
    true_planner      plant = theta,  planner = theta                (planner is right)

`nominal_planner - true_planner` is therefore exactly the cost of the planner's model being
wrong, with the plant held identical and the MPC seed shared. **Negative means the domain
shift hurt.**

Kept dependency-free on purpose: `analyze.py` and `view.py` import it and must stay
importable without JAX.
"""

from __future__ import annotations

GROUP_NOMINAL_PLANNER = "nominal_planner"
GROUP_TRUE_PLANNER = "true_planner"

GROUPS = (GROUP_NOMINAL_PLANNER, GROUP_TRUE_PLANNER)

DISPLAY = {
    GROUP_NOMINAL_PLANNER: "Nominal-parameter planner",
    GROUP_TRUE_PLANNER: "True-parameter planner",
}

SHORT = {
    GROUP_NOMINAL_PLANNER: "nominal planner",
    GROUP_TRUE_PLANNER: "true planner",
}

# Runs produced before the control arm was redefined used these labels. The mapping lets
# old CSVs still be read, but it is NOT a rename: the old "nominal" arm ran a *nominal
# plant*, which is a different experiment, so its deltas are not comparable with new ones.
# `analyze.load_trials` applies this and warns.
LEGACY_GROUPS = {
    "randomized": GROUP_NOMINAL_PLANNER,
    "nominal": GROUP_TRUE_PLANNER,
}

LEGACY_WARNING = (
    "This run uses the pre-{ts} group labels. Its control arm was a NOMINAL PLANT with a "
    "nominal planner, not a theta-perturbed plant with a true-parameter planner, so its "
    "paired deltas measure something different (planner error + intrinsic plant "
    "difficulty, rather than planner error alone). Figures are labelled with the new "
    "names for convenience -- do not compare these numbers against a new run."
).format(ts="2026-09-18")
