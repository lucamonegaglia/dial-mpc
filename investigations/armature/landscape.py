"""Open-loop planning quality of the true vs nominal armature model, at fixed plant states.

States come from a closed-loop run of the plant under the matched planner. At each
checkpoint we draw one MPPI sample batch exactly as `reverse_once` does (Ndiffuse=1 noise
scale) and evaluate it under several planner models. The reference ("what the plant will
actually do") is the plant model started from a copy of the state whose qvel carries a
tiny perturbation -- the same order as the f32 nondeterminism that separates two GPU runs.

Per (plant scale, state, planner model) we record:
  * spearman / pearson between the model's sample returns and the reference returns;
  * J_ref(plan_m): the reference return of the MPPI-weighted plan built from model m;
  * self-consistency of the reference under two independent perturbations (chaos floor);
  * incumbent-plan joint-angle divergence vs horizon step.
"""

from __future__ import annotations

import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import pearsonr, spearmanr

from dial_mpc.sim2sim import determinism


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plants", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.85, 1.0, 1.4, 2.0])
    ap.add_argument("--models", type=float, nargs="+", default=[0.5, 0.7, 1.0, 1.4, 2.0])
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--n-states", type=int, default=12)
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precision", choices=determinism.PRECISIONS, default="default")
    ap.add_argument("--out", default="landscape")
    args = ap.parse_args()
    determinism.configure("fast", args.precision)
    from common import RESULTS, build

    h = build()
    mb, dc = h.mbdpi, h.dial_config
    sigma = mb.sigma_control  # Ndiffuse=1 -> factor traj_diffuse_factor**0
    nu = mb.nu

    @jax.jit
    def eval_batch(pm, state, us):
        rews, ps = mb.rollout_us_model_vmap(pm, state, us)
        return rews.mean(-1), ps.q[..., 7:]

    def perturb(state, key):
        ps = state.pipeline_state
        qv = ps.qvel + args.eps * jax.random.normal(key, ps.qvel.shape)
        return state.replace(pipeline_state=ps.replace(qvel=qv))

    def gate(R):
        """Same acceptance rule as reverse_once."""
        return np.isfinite(R) & (R >= -1e5) & (R <= 1e5)

    out = []
    rng = jax.random.PRNGKey(args.seed)
    for p in args.plants:
        plant = h.model(p)
        pm = {m: h.planner.model_from(*h.model(m)) for m in set(args.models) | {p}}
        # collect states by running the closed loop with the matched planner
        rng, rr = jax.random.split(rng)
        state = h.stepper.reset_jit(plant, rr)
        Y0 = jnp.zeros([dc.Hnode + 1, nu])
        states = []
        for t in range(args.every * args.n_states + 10):
            state = h.stepper.step_jit(plant, state, Y0[0])
            if bool(state.done > 0.5) or not np.isfinite(float(state.reward)):
                break
            Y0 = mb.shift(Y0)
            fn = h.diffuse.diffuse_init_jit if t == 0 else h.diffuse.diffuse_jit
            rng, Y0, _ = fn(pm[p], rng, Y0, state)
            if t >= 10 and (t - 10) % args.every == 0:
                states.append((t, state, Y0))
        print(f"plant {p}: {len(states)} states (loop ended at t={t})", flush=True)

        for (t, st, Ybar) in states:
            rng, k1, k2, k3, k4 = jax.random.split(rng, 5)
            # identical sample batch to reverse_once(st, k1, Ybar, sigma, .)
            _, ys_rng = jax.random.split(k1)
            eps_Y = jax.random.normal(ys_rng, (dc.Nsample, dc.Hnode + 1, nu))
            Y0s = eps_Y * sigma[None, :, None] + Ybar
            Y0s = Y0s.at[:, 0].set(Ybar[0])
            Y0s = jnp.clip(jnp.concatenate([Y0s, Ybar[None]], 0), -1.0, 1.0)
            us = mb.node2u_vvmap(Y0s)

            ref_a, q_ref = eval_batch(pm[p], perturb(st, k2), us)
            ref_b, _ = eval_batch(pm[p], perturb(st, k3), us)
            ref_a, ref_b, q_ref = np.asarray(ref_a), np.asarray(ref_b), np.asarray(q_ref)
            ga, gb = gate(ref_a), gate(ref_b)
            fin = ga & gb
            rec = {"plant": p, "t": t,
                   "chaos_spearman": float(spearmanr(ref_a[fin], ref_b[fin])[0]),
                   "ref_frac_gated": float(1 - ga.mean()),
                   "ref_frac_nonfinite": float(1 - np.isfinite(ref_a).mean()),
                   "incumbent_J": float(ref_a[-1]), "models": {}}

            def J_of(u_plan, key):
                return [float(v) for v in
                        jax.vmap(lambda kk: eval_batch(pm[p], perturb(st, kk), u_plan[None])[0][0])(
                            jax.random.split(key, 8))]

            for m, pmod in pm.items():
                R, q = eval_batch(pmod, st, us)
                R, q = np.asarray(R), np.asarray(q)
                g = gate(R)
                ok = fin & g
                _, Yplan, info = mb.reverse_once(st, k1, Ybar, sigma, pmod)
                J = J_of(mb.node2u_vmap(Yplan), k4)
                qdiv = np.sqrt(np.mean((q[-1] - q_ref[-1]) ** 2, axis=-1))
                # sign agreement on the "explodes" label: does the model know which samples blow up?
                rec["models"][str(m)] = {
                    "spearman": float(spearmanr(R[ok], ref_a[ok])[0]),
                    "pearson": float(pearsonr(R[ok], ref_a[ok])[0]),
                    "J_ref_plan": float(np.mean(J)),
                    "J_ref_plan_frac_gated": float(np.mean(~gate(np.asarray(J)))),
                    "ess": float(info["ess"]),
                    "frac_gated": float(1 - g.mean()),
                    "gated_overlap_with_ref": float((~g & ~ga).sum() / max((~ga).sum(), 1)),
                    "plan_first_node_delta": float(jnp.linalg.norm(Yplan[1] - Ybar[1])),
                    "incumbent_qdiv": qdiv.tolist(),
                }
            _, Yo, _ = mb.reverse_once(perturb(st, k3), k1, Ybar, sigma, pm[p])
            rec["oracle_J"] = float(np.mean(J_of(mb.node2u_vmap(Yo), k4)))
            out.append(rec)
            ms = rec["models"]
            print(f"  t={t:3d} chaos={rec['chaos_spearman']:.2f} refgate={rec['ref_frac_gated']:.2f} "
                  f"inc={rec['incumbent_J']:.3f} orc={rec['oracle_J']:.3f} | "
                  + " ".join(f"m{m}:rho={v['spearman']:.2f},J={v['J_ref_plan']:.3f},g={v['frac_gated']:.2f}"
                             for m, v in sorted(ms.items())), flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"{args.out}.json"), "w") as f:
        json.dump({"args": vars(args), "records": out}, f)


if __name__ == "__main__":
    main()
