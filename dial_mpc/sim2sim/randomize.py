"""Domain-randomization parameter specs: parsing, sampling, and application.

A `DomainRandConfig` describes a set of named parameter perturbations to apply to a
brax/MJX `System` (and, for controller gains, to the env's `BaseEnvConfig`). Each entry
names a field on `sys` (or `config`), a subset of elements to perturb, a sampling
distribution, and how the sampled value combines with the nominal value.

The whole thing is designed to be resolved once (index arrays, nominal values) and then
sampled/applied many times inside jit — `sample_theta` and `apply_theta` are pure JAX and
safe to call from a jitted function.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, List, Optional, Sequence, Union, cast

import jax
import jax.numpy as jnp
import mujoco  # NOTE: ships no py.typed/.pyi, so every `mujoco.*` access below is
# unresolvable to a type checker and carries an explicit ignore. This is a gap in the
# mujoco package itself (it affects dial_mpc's envs identically), not something this
# module can fix; the ignores are kept narrow so real errors still surface.
import numpy as np

from brax.base import System

# Fields that must never be randomized because they change what `dt`/n_frames mean
# (n_frames = dt / timestep is fixed at env construction).
_FORBIDDEN_SYS_FIELDS = {"opt.timestep"}


@dataclass
class ParamSpec:
    """One randomized parameter."""

    name: str
    target: str  # "sys" or "config"
    field: str  # attribute name on sys or on the env config dataclass
    mode: str = "scale"  # "scale" | "add" | "abs"
    range: Sequence[float] = (1.0, 1.0)
    dist: str = "uniform"  # "uniform" | "log_uniform"
    per_element: bool = False
    select: Union[str, Sequence[str], Sequence[int]] = "all"
    column: Optional[int] = None  # for 2D fields (e.g. geom_friction), which column
    scale_inertia: bool = True  # only used when field == "body_mass"

    # resolved at `resolve_specs` time
    indices: np.ndarray = dc_field(default_factory=lambda: np.array([], dtype=np.int64), repr=False)
    n_draws: int = dc_field(default=1, repr=False)
    # shape of the selected sub-array (len(indices) [+ trailing dims if `column` is None
    # and the field is >1D, e.g. body_ipos's (n_idx, 3)]). `per_element=True` draws one
    # iid value per scalar in this shape; `per_element=False` draws a single shared scalar.
    elem_shape: tuple = dc_field(default=(), repr=False)

    def __post_init__(self):
        if self.target not in ("sys", "config"):
            raise ValueError(f"param '{self.name}': target must be 'sys' or 'config', got {self.target!r}")
        if self.mode not in ("scale", "add", "abs"):
            raise ValueError(f"param '{self.name}': mode must be 'scale'/'add'/'abs', got {self.mode!r}")
        if self.dist not in ("uniform", "log_uniform"):
            raise ValueError(f"param '{self.name}': dist must be 'uniform'/'log_uniform', got {self.dist!r}")
        if len(self.range) != 2:
            raise ValueError(f"param '{self.name}': range must be [lo, hi], got {self.range!r}")
        if self.field in _FORBIDDEN_SYS_FIELDS and self.target == "sys":
            raise ValueError(
                f"param '{self.name}': field '{self.field}' cannot be randomized "
                "(it changes env.dt semantics; n_frames is fixed at construction)."
            )
        if self.dist == "log_uniform" and (self.range[0] <= 0 or self.range[1] <= 0):
            raise ValueError(f"param '{self.name}': log_uniform range must be strictly positive, got {self.range!r}")


@dataclass
class DomainRandConfig:
    n_trials: int = 64
    seed: int = 12345
    paired: bool = True
    save_rollouts: bool = False
    # how many of the most divergent paired trials get a full 50 Hz state log written
    n_interesting: int = 12
    # Judge `done` against the model's real joint limits instead of the env's hand-tuned
    # action-scaling band. The band is far tighter than physical limits, so leaving this
    # off ends episodes on ordinary tracking error rather than on a fall.
    terminate_on_physical_limits: bool = False
    params: Dict[str, Dict[str, Any]] = dc_field(default_factory=dict)


def load_domain_rand_config(config_dict: Dict[str, Any]) -> DomainRandConfig:
    """Pull the `domain_randomization:` block out of the flat YAML dict."""
    block = config_dict.get("domain_randomization", {})
    kwargs = {k: v for k, v in block.items() if k != "params"}
    drc = DomainRandConfig(**kwargs)
    drc.params = dict(block.get("params", {}))
    return drc


def _actuated_dof_mask(mj_model: "mujoco.MjModel") -> np.ndarray:  # type: ignore[name-defined]
    """Boolean mask over dofs (length nv) that are actuated joints, i.e. excludes the
    6 free-base dofs (or 0 dofs if the model has no free joint)."""
    mask = np.zeros(mj_model.nv, dtype=bool)
    for j in range(mj_model.njnt):
        jtype = mj_model.jnt_type[j]
        dof_adr = mj_model.jnt_dofadr[j]
        if jtype == mujoco.mjtJoint.mjJNT_FREE:  # type: ignore[attr-defined]
            continue  # 6 dofs, left unmasked (False)
        elif jtype == mujoco.mjtJoint.mjJNT_BALL:  # type: ignore[attr-defined]
            ndof = 3
        else:  # hinge or slide
            ndof = 1
        mask[dof_adr : dof_adr + ndof] = True
    return mask


def _resolve_indices(spec: ParamSpec, sys: System, n_field: int) -> np.ndarray:
    """Turn `select` into a concrete index array over the leading dimension of `field`."""
    mj = sys.mj_model
    select = spec.select

    if isinstance(select, str) and select == "all":
        return np.arange(n_field, dtype=np.int64)

    if isinstance(select, str) and select == "actuated":
        if spec.field in ("dof_damping", "dof_armature", "dof_frictionloss"):
            mask = _actuated_dof_mask(mj)
            return np.nonzero(mask)[0].astype(np.int64)
        raise ValueError(
            f"param '{spec.name}': select='actuated' is only defined for dof_* fields, "
            f"got field='{spec.field}'"
        )

    if isinstance(select, (list, tuple)) and len(select) > 0 and isinstance(select[0], str):
        # names -> indices, resolved against the field's natural object type
        if spec.field.startswith("body_"):
            objtype = mujoco.mjtObj.mjOBJ_BODY  # type: ignore[attr-defined]
        elif spec.field.startswith("geom_"):
            objtype = mujoco.mjtObj.mjOBJ_GEOM  # type: ignore[attr-defined]
        elif spec.field.startswith("dof_"):
            raise ValueError(
                f"param '{spec.name}': dof_* fields must use select='all'/'actuated'/int indices, "
                "not joint names (dof index != joint index for multi-dof joints)."
            )
        elif spec.field.startswith("actuator_"):
            objtype = mujoco.mjtObj.mjOBJ_ACTUATOR  # type: ignore[attr-defined]
        elif spec.field.startswith("jnt_"):
            objtype = mujoco.mjtObj.mjOBJ_JOINT  # type: ignore[attr-defined]
        else:
            raise ValueError(f"param '{spec.name}': cannot resolve names for field '{spec.field}'")
        idx = []
        for nm in select:
            i = mujoco.mj_name2id(mj, objtype, nm)  # type: ignore[attr-defined]
            if i < 0:
                raise ValueError(f"param '{spec.name}': name '{nm}' not found for field '{spec.field}'")
            idx.append(i)
        return np.array(idx, dtype=np.int64)

    # explicit integer indices
    return np.array(select, dtype=np.int64)


def resolve_specs(
    params: Dict[str, Dict[str, Any]],
    sys: System,
    config_arrays: Optional[Dict[str, Any]] = None,
) -> List[ParamSpec]:
    """Parse the raw `params` dict into `ParamSpec`s with resolved index arrays.

    `config_arrays` maps env-config field name (e.g. "kp", "kd") to its nominal array,
    needed to size `n_draws` for `target: config` specs with `per_element: true`.

    Call once per plant model (e.g. at harness startup); the result is reused across
    all trials.
    """
    config_arrays = config_arrays or {}
    specs = []
    for name, raw in params.items():
        spec = ParamSpec(name=name, **raw)
        if spec.target == "sys":
            arr = getattr(sys, spec.field)
            n_field = arr.shape[0]
            spec.indices = _resolve_indices(spec, sys, n_field)
            if spec.column is not None or arr.ndim == 1:
                spec.elem_shape = (len(spec.indices),)
            else:
                spec.elem_shape = (len(spec.indices),) + tuple(arr.shape[1:])
        else:  # config
            if spec.field not in config_arrays:
                raise ValueError(
                    f"param '{spec.name}': target='config' field '{spec.field}' not found "
                    f"in config_arrays {sorted(config_arrays)}"
                )
            n_field = np.asarray(config_arrays[spec.field]).shape[0]
            spec.indices = np.arange(n_field, dtype=np.int64)
            spec.elem_shape = (n_field,)
        spec.n_draws = int(np.prod(spec.elem_shape)) if spec.per_element else 1
        specs.append(spec)
    return specs


def sample_theta(rng: jax.Array, specs: Sequence[ParamSpec]) -> Dict[str, jax.Array]:
    """Draw one perturbation per spec.

    Returns {name: array}, shaped `spec.elem_shape` when `per_element=True` (matching the
    selected sub-array exactly, e.g. (n_idx, 3) for a per-axis body_ipos offset), else
    shape (1,) (one shared scalar).
    """
    theta = {}
    for spec in specs:
        rng, key = jax.random.split(rng)
        lo, hi = spec.range
        if spec.dist == "uniform":
            draw = jax.random.uniform(key, (spec.n_draws,), minval=lo, maxval=hi)
        else:  # log_uniform
            log_draw = jax.random.uniform(key, (spec.n_draws,), minval=jnp.log(lo), maxval=jnp.log(hi))
            draw = jnp.exp(log_draw)
        if spec.per_element:
            draw = draw.reshape(spec.elem_shape)
        theta[spec.name] = draw
    return theta


def _apply_one(nominal: jax.Array, indices: np.ndarray, draw: jax.Array, mode: str,
                per_element: bool, column: Optional[int]) -> jax.Array:
    idx = jnp.asarray(indices)
    if column is not None:
        sel = nominal[idx, column]
    else:
        sel = nominal[idx]
    d = draw if per_element else jnp.broadcast_to(draw[0], sel.shape)
    if mode == "scale":
        new = sel * d
    elif mode == "add":
        new = sel + d
    else:  # abs
        new = jnp.broadcast_to(d, sel.shape)
    if column is not None:
        return nominal.at[idx, column].set(new)
    return nominal.at[idx].set(new)


def apply_theta(
    nominal_sys: System,
    nominal_kp: jax.Array,
    nominal_kd: jax.Array,
    theta: Dict[str, jax.Array],
    specs: Sequence[ParamSpec],
) -> tuple[System, jax.Array, jax.Array]:
    """Apply a sampled theta to (sys, kp, kd). Pure; safe under jit/vmap."""
    sys_updates: Dict[str, jax.Array] = {}
    kp, kd = nominal_kp, nominal_kd

    for spec in specs:
        draw = theta[spec.name]
        if spec.target == "sys":
            field_name = spec.field
            nominal_field = sys_updates.get(field_name, getattr(nominal_sys, field_name))
            if nominal_field is None:
                raise ValueError(
                    f"param '{spec.name}': sys.{field_name} is None on this model, "
                    "so there is nothing to randomize."
                )
            current = cast(jax.Array, nominal_field)
            new_field = _apply_one(current, spec.indices, draw, spec.mode, spec.per_element, spec.column)
            sys_updates[field_name] = new_field
            if field_name == "body_mass" and spec.scale_inertia and spec.mode == "scale":
                ratio = jnp.ones_like(nominal_sys.body_inertia[:, 0]).at[jnp.asarray(spec.indices)].set(
                    draw if spec.per_element else jnp.broadcast_to(draw[0], (len(spec.indices),))
                )
                cur_inertia = sys_updates.get("body_inertia", nominal_sys.body_inertia)
                sys_updates["body_inertia"] = cur_inertia * ratio[:, None]
        else:  # config: kp/kd are arrays on the env config
            base = kp if spec.field == "kp" else kd
            base = jnp.asarray(base)
            if spec.per_element:
                d = draw
            else:
                d = jnp.broadcast_to(draw[0], base.shape)
            if spec.mode == "scale":
                new = base * d
            elif spec.mode == "add":
                new = base + d
            else:
                new = jnp.broadcast_to(d, base.shape)
            if spec.field == "kp":
                kp = new
            else:
                kd = new

    # tree_replace is annotated as taking a Mapping of Optional values and returning the
    # base PyTreeNode; it preserves the concrete type at runtime.
    sys = cast(System, nominal_sys.tree_replace(dict(sys_updates))) if sys_updates else nominal_sys
    return sys, kp, kd
