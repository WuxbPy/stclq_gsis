"""Correctness check (§6.1 / Table S1).

Compares the sparse engine with the ``O(N²)`` reference implementation element by
element, over the **full factor grid**
(4 spatial kernels × 3 temporal weights × 3 type counts × 2 levels = 72 cases):

* spatial kernel: gaussian / box / epanechnikov / inverse_distance
* temporal weight: none / inverse_dt / exponential
* number of types K: 2 / 3 / 5
* level: global / local
* invariance self-check: under a label permutation both the CSR and ``w_s·w_t`` stay
  bit-for-bit unchanged

Criterion: ``max|Δ| ≤ atol = 1e-10``.

    PYTHONPATH=src python results/ga/exp_correctness.py
"""

from __future__ import annotations

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
from _ref import reference_global_clq, reference_local_clq  # noqa: E402

OUT = HERE / "correctness"
ATOL = 1e-10


def make_points(n=1200, K=3, seed=0, side=1000.0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, side, size=(K, 2))
    per = n // K
    pts, tys = [], []
    for k in range(K):
        m = per if k < K - 1 else n - len(pts)
        pts.append(rng.normal(centers[k], side * 0.06, size=(m, 2)))
        tys.append(np.full(m, k + 1, dtype=np.int64))
    coords = np.vstack(pts)
    types = np.concatenate(tys)
    o = rng.permutation(coords.shape[0])
    periods = rng.integers(1, 5, size=coords.shape[0])
    return coords[o], types[o], periods[o]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    print(f"{'level':>7} {'kernel':>17} {'K':>2} {'temporal':>13} "
          f"{'n':>6} {'max|Δ|':>11} {'pass':>5} {'engine_ms':>10} {'ref_ms':>9}")

    for K in (2, 3, 5):
        for kernel in ("gaussian", "box", "epanechnikov", "inverse_distance"):
            for temporal in ("none", "inverse_dt", "exponential"):
                coords, types, periods = make_points(n=1200, K=K, seed=K * 10 + 1)
                b = 90.0
                tp = periods if temporal != "none" else None

                model = STCLQ(distance_metric="projected", spatial_kernel=kernel,
                              temporal_weight=temporal, temporal_bandwidth=2.0,
                              diagnostics=False, n_jobs=1)

                t0 = time.perf_counter()
                eng = model._make_engine(coords, types, b, None, tp)
                g_eng = eng.global_clq()
                t_eng = time.perf_counter() - t0

                t0 = time.perf_counter()
                g_ref = reference_global_clq(coords, types, b, K, kernel=kernel,
                                             periods=tp, temporal=temporal, tau=2.0)
                t_ref = time.perf_counter() - t0

                d_g = float(np.max(np.abs(g_eng - g_ref)))
                rows.append({"level": "global", "kernel": kernel, "K": K,
                             "temporal": temporal, "n": coords.shape[0],
                             "max_abs_diff": d_g, "pass": bool(d_g <= ATOL),
                             "engine_s": t_eng, "reference_s": t_ref})
                print(f"{'global':>7} {kernel:>17} {K:>2} {temporal:>13} "
                      f"{coords.shape[0]:>6} {d_g:>11.2e} "
                      f"{str(d_g <= ATOL):>5} {t_eng*1e3:>10.2f} {t_ref*1e3:>9.1f}")

                # Local level: the **same full factor grid** as the global level
                # (4 kernels × 3 temporal weights × 3 values of K). n stays small so
                # that the O(N²) reference cost is bounded. Earlier runs sampled only
                # two kernels at K=2, so 42 of the cases could not be written as the
                # "4×3×3×2" direct product.
                cs, ts, ps = make_points(n=600, K=K, seed=K * 100 + 99)
                tps = ps if temporal != "none" else None
                m2 = STCLQ(distance_metric="projected", spatial_kernel=kernel,
                           temporal_weight=temporal, temporal_bandwidth=2.0,
                           diagnostics=False, n_jobs=1)
                t0 = time.perf_counter()
                e2 = m2._make_engine(cs, ts, 120.0, None, tps)
                l_eng = e2.local_clq()
                t_leng = time.perf_counter() - t0
                t0 = time.perf_counter()
                l_ref = reference_local_clq(cs, ts, 120.0, K, kernel=kernel,
                                            periods=tps, temporal=temporal, tau=2.0)
                t_lref = time.perf_counter() - t0
                d_l = float(np.max(np.abs(l_eng - l_ref)))
                rows.append({"level": "local", "kernel": kernel, "K": K,
                             "temporal": temporal, "n": cs.shape[0],
                             "max_abs_diff": d_l, "pass": bool(d_l <= ATOL),
                             "engine_s": t_leng, "reference_s": t_lref})
                print(f"{'local':>7} {kernel:>17} {K:>2} {temporal:>13} "
                      f"{cs.shape[0]:>6} {d_l:>11.2e} {str(d_l <= ATOL):>5} "
                      f"{t_leng*1e3:>10.2f} {t_lref*1e3:>9.1f}")

    # --- Invariance self-check ---------------------------------------------
    coords, types, periods = make_points(n=3000, K=3, seed=7)
    eng = STCLQ(distance_metric="projected", temporal_weight="inverse_dt",
                temporal_bandwidth=2.0, diagnostics=False, n_jobs=1
                )._make_engine(coords, types, 80.0, None, periods)
    inv = eng.invariance_check(n_draws=16, seed=3)
    print(f"\ninvariance self-check: unchanged={inv['unchanged']}  "
          f"max|Δdata|={inv['max_data_delta']:.1e}")

    summary = {
        "atol": ATOL,
        "n_cases": len(rows),
        "n_pass": int(sum(r["pass"] for r in rows)),
        "max_abs_diff": float(np.nanmax([r["max_abs_diff"] for r in rows])),
        "invariance": inv,
        "worst_case": max(rows, key=lambda r: r["max_abs_diff"]),
    }
    with open(OUT / "correctness.json", "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "cases": rows}, fh,
                  ensure_ascii=False, indent=2)
    with open(OUT / "correctness.csv", "w", encoding="utf-8") as fh:
        keys = ["level", "kernel", "K", "temporal", "n", "max_abs_diff", "pass"]
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")

    print(f"\n{summary['n_pass']}/{summary['n_cases']} cases pass "
          f"(max|Δ| = {summary['max_abs_diff']:.2e} ≤ atol={ATOL:.0e})")
    print(f"results -> {OUT}")


if __name__ == "__main__":
    main()
