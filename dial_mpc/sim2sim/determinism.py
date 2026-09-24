"""Numerical determinism and precision controls for sim2sim runs.

Measured on this stack (jax 0.6.2 / mujoco-mjx 3.13.0, CUDA, RTX 4090):

* A single vmapped `mjx.step` is not bitwise reproducible on GPU. Repeated calls to the
  same compiled function with the same input buffers differ at f32-ulp level, both
  in-process and across processes, with and without contacts, and with autotuning
  disabled (`--xla_gpu_autotune_level=0`). So it is nondeterministic reduction order
  inside the kernels, not per-process kernel selection.
* `--xla_gpu_deterministic_ops=true` and `--xla_gpu_exclude_nondeterministic_ops=true`
  both produce all-NaN rollouts here, so GPU bitwise determinism is not reachable by
  flag. The CPU backend *is* bitwise deterministic, at ~23x the cost of a 2048-sample
  rollout batch (0.53 s vs 0.023 s) -- affordable for replaying single trials, not for a
  256-trial sweep. That is what `mode="exact"` buys.
* Independently: XLA's default f32 matmul precision on GPU is TF32, which perturbs the
  physics far more than the nondeterminism does. On a 2048x20 rollout batch the
  mean-reward sum is -1017.6 at default precision, -1020.19 at `highest`, and -1020.20 on
  CPU -- i.e. `highest` agrees with CPU to ~1e-5 relative while the TF32 default is off by
  ~2.6, roughly 600x the run-to-run spread. `highest` costs ~12% (23.3 -> 26.2 ms/batch).

`precision` defaults to "default" so that enabling this module does not silently change
the physics of existing GPU runs; pass "highest" to opt into CPU-matching accuracy. The
choice is always recorded in `config_used.yaml` under `determinism`.

Consequence for reading `fast`-mode results: a per-trial `steps_survived` is not
reproducible when the episode ends on a threshold crossing, because ulp noise moves the
crossing time. Statistics that saturate (survival at the step cap) are far more stable
than ones recording when a boundary was first touched.
"""

from __future__ import annotations

import os
from typing import Any, Dict

MODES = ("fast", "exact")
PRECISIONS = ("default", "highest")


def configure(mode: str = "fast", precision: str = "default") -> Dict[str, Any]:
    """Apply precision/determinism settings. Must run before any jax computation, since
    the backend choice only takes effect before the backend is initialized."""
    if mode not in MODES:
        raise ValueError(f"determinism mode must be one of {MODES}, got {mode!r}")
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")

    import jax

    if mode == "exact":
        # jax reads JAX_PLATFORMS into its own config at import time, so mutating
        # os.environ here alone would be ignored; update the config directly. The env var
        # is set too so that any subprocess inherits the choice.
        os.environ["JAX_PLATFORMS"] = "cpu"
        jax.config.update("jax_platforms", "cpu")
        # On CPU the f32 matmul is exact anyway, and `highest` keeps GPU comparisons
        # meaningful, so exact mode always pins it.
        precision = "highest"

    if precision == "highest":
        jax.config.update("jax_default_matmul_precision", "highest")

    backend = jax.default_backend()
    if mode == "exact" and backend != "cpu":
        raise RuntimeError(
            f"determinism mode 'exact' needs the CPU backend for bitwise reproducibility, "
            f"but the jax backend was already initialized as {backend!r}. configure() must "
            f"be called before any jax computation."
        )

    info = {
        "mode": mode,
        "backend": backend,
        "matmul_precision": precision,
        "bitwise_reproducible": mode == "exact",
    }
    if mode == "exact":
        print(f"Determinism: exact (CPU backend, {precision} matmuls) -- bitwise "
              f"reproducible, ~23x slower than GPU")
    else:
        print(f"Determinism: fast ({backend} backend, {precision} matmuls) -- per-step "
              f"results vary at f32-ulp level between runs")
    return info
