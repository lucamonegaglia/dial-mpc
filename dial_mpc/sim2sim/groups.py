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
