"""Permutation validity: calibration of the null distribution (§6.2 / Fig. 3a /
Table S2).

**Question**: under the spatio-temporal weight ``w = w_s · w_t``, the statement "type
is independent of location" admits two null hypotheses that are not equivalent:

* ``type`` (**global label permutation**): tests type ⊥ (space, time);
* ``type_period`` (**block permutation within period**): permutes independently
  inside each period, preserving the type composition of that period, and tests
  type ⊥ space | temporal structure.

**Construction**: ``gen_spatiotemporal`` lets the type composition of the periods
vary with ``delta`` (``delta=0``: every period has the same composition;
``delta→1``: each period almost contains a single type), while **type and location
stay independent within a period**, that is, **H0 holds** for the statement
"type ⊥ space". Any rejection is therefore a false positive, and the type-I error
rate reads off directly.

**Expected**: with ``w_t = 1`` both null hypotheses are calibrated; once ``w_t`` is
switched on, the global label permutation mistakes "type depends on time" for "type
depends on space", so the type-I error rate inflates with ``delta`` and with the
strength of the temporal weight, whereas the block permutation stays calibrated.
A **moment-matched normal approximation** (analytic p-value from the mean and the
variance of the permutation distribution) is added as well, to test whether the
"analytic shortcut" inherits the same bias.

    PYTHONPATH=src python results/ga/exp_calibration.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from stclq import STCLQ  # noqa: E402

OUT = HERE / "calibration"
ALPHA = 0.05
PAIR = (1, 1)          # pre-specified test pair: within-type co-location CLQ(1,1)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def gen_spatiotemporal(n, K, delta, rng, n_periods=6, side=1000.0,
                       spatial_cluster=False, sep=45.0, target=(1, 2)):
    """Spatio-temporal point pattern: type and location are independent within a
    period, and the type composition of the periods spreads with ``delta``.

    With ``spatial_cluster=True`` the ``target`` types are drawn towards a per-period
    centre inside each period, which creates the alternative to "type ⊥ space" (used
    for the power curve).
    """
    coords = rng.uniform(0.0, side, size=(n, 2))
    periods = rng.integers(1, n_periods + 1, size=n)
    types = np.empty(n, dtype=np.int64)
    base = np.full(K, 1.0 / K)
    for p in range(1, n_periods + 1):
        idx = np.where(periods == p)[0]
        peak = np.zeros(K)
        peak[(p - 1) % K] = 1.0
        probs = (1.0 - delta) * base + delta * peak
        types[idx] = rng.choice(np.arange(1, K + 1), size=idx.size, p=probs)
    if spatial_cluster:
        for p in range(1, n_periods + 1):
            c = rng.uniform(0.2 * side, 0.8 * side, size=2)
            m = (periods == p) & np.isin(types, target)
            if m.any():
                coords[m] = c + rng.normal(0.0, sep, size=(int(m.sum()), 2))
    return coords, types, periods


def normal_pvalue(sims, observed):
    """Moment-matched normal approximation: a one-sided (attraction) p-value built
    from the mean and variance of the permutation distribution."""
    mu = sims.mean(axis=0)
    sd = sims.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, (observed - mu) / sd, 0.0)
    from scipy.stats import norm
    return 1.0 - norm.cdf(z)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--n-periods", type=int, default=6)
    ap.add_argument("--n-perm", type=int, default=199)
    ap.add_argument("--reps", type=int, default=150)
    ap.add_argument("--bw", type=float, default=95.0)
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.0, 0.3, 0.6, 0.9])
    ap.add_argument("--temporals", nargs="+", default=["none", "exponential"])
    ap.add_argument("--scenarios", nargs="+", default=["H0", "H1"],
                    choices=["H0", "H1"])
    ap.add_argument("--sep", type=float, default=45.0,
                    help="clustering scale (metres) of the target type pair in H1; "
                         "larger means lower power")
    ap.add_argument("--tag", default="", help="suffix for the output file names (e.g. _power)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    from stclq.inference import permutation_test

    rows = []
    t_start = time.perf_counter()
    for scenario, cluster in (("H0", False), ("H1", True)):
        if scenario not in args.scenarios:
            continue
        for temporal in args.temporals:
            for delta in args.deltas:
                recs = []
                obs_vals = []
                for i in range(args.reps):
                    seed = 100_000 + i + int(delta * 1000) * 7 + (
                        0 if temporal == "none" else 50_000) + (
                        0 if scenario == "H0" else 500_000)
                    model = STCLQ(temporal_weight=temporal, temporal_bandwidth=1.0,
                                  distance_metric="projected", diagnostics=False,
                                  n_jobs=1)
                    rng = np.random.default_rng(seed)
                    coords, types, periods = gen_spatiotemporal(
                        args.n, args.K, delta, rng, n_periods=args.n_periods,
                        spatial_cluster=cluster, sep=args.sep)
                    eng = model._make_engine(coords, types, args.bw, None, periods)
                    obs = eng.global_clq()
                    a, b = PAIR
                    rec = {"observed_pair": float(obs[a - 1, b - 1])}
                    for null in ("type", "type_period"):
                        res = permutation_test(eng, obs, level="global", null=null,
                                               n_perm=args.n_perm,
                                               seed=seed, correction=False, n_jobs=1,
                                               return_sims=True)
                        rec[f"p_perm[{null}]"] = float(res["p_attract"][a - 1, b - 1])
                        rec[f"p_perm_repel[{null}]"] = float(res["p_repel"][a - 1, b - 1])
                        # moment-matched normal approximation (reuses the simulated
                        # distribution of the same permutation run)
                        rec[f"p_norm[{null}]"] = float(
                            normal_pvalue(res["sims"], obs)[a - 1, b - 1])
                    recs.append(rec)
                    obs_vals.append(rec["observed_pair"])
                rec = {"scenario": scenario, "temporal": temporal,
                       "delta": delta, "n_reps": args.reps,
                       "mean_observed_pair": float(np.mean(obs_vals))}
                for null in ("type", "type_period"):
                    for tag in ("perm", "norm"):
                        key = f"p_perm[{null}]" if tag == "perm" else f"p_norm[{null}]"
                        rej = float(np.mean([r[key] < ALPHA for r in recs]))
                        rec[f"type1_error[{tag}:{null}]"] = rej
                        rec[f"se[{tag}:{null}]"] = float(
                            np.sqrt(max(rej * (1 - rej), 1e-12) / args.reps))
                rows.append(rec)
                print(f"{scenario:>2} temporal={temporal:>11} delta={delta:.1f}  "
                      f"obs={rec['mean_observed_pair']:.3f}  "
                      f"type={rec['type1_error[perm:type]']:.3f}"
                      f"±{rec['se[perm:type]']:.3f}  "
                      f"block={rec['type1_error[perm:type_period]']:.3f}"
                      f"±{rec['se[perm:type_period]']:.3f}  "
                      f"norm(type)={rec['type1_error[norm:type]']:.3f}   "
                      f"[{time.perf_counter()-t_start:.0f}s]", flush=True)

    with open(OUT / f"calibration{args.tag}.json", "w", encoding="utf-8") as fh:
        json.dump({"settings": vars(args), "alpha": ALPHA, "pair": PAIR,
                   "rows": rows}, fh, ensure_ascii=False, indent=2)
    with open(OUT / f"calibration{args.tag}.csv", "w", encoding="utf-8") as fh:
        keys = list(rows[0].keys())
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"\nresults -> {OUT}")


if __name__ == "__main__":
    main()
