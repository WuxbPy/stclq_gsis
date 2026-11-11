"""Large-scale benchmark (§6.3–6.4 / Fig. 4–6 / Table 3).

Measurements (each ``N`` is measured in its **own subprocess**, so that the peak
resident set reported by ``ru_maxrss`` is clean):

* build time (KD-Tree to CSR, once);
* time of a single global STCLQ evaluation;
* time of a single permutation (serial and ``--n-jobs`` parallel);
* time of a single permutation of the reference ``O(N²)`` implementation, and the
  speed-up that follows;
* peak memory (CSR as measured vs the dense ``O(N²)`` estimate);
* ``nnz``, ``k̄``, ``ρ = nnz/N²``.

Control variable: for each ``N`` a bisection picks the bandwidth that gives
``k̄ ≈ --target-knn``, which separates the "algorithmic gain" from "more data"; the
bandwidth convergence law at constant ``k̄`` is reported as well.

    PYTHONPATH=src python results/ga/exp_benchmark.py
    PYTHONPATH=src python results/ga/exp_benchmark.py --ns 1000 10000 --n-jobs 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import resource
import subprocess
import sys
import time

# Benchmark control variable: with process-level parallelism the underlying BLAS
# threads are capped at 1, so that p × nproc does not oversubscribe the machine.
# This must be set before numpy is imported (OpenBLAS reads it when the library
# initialises).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

OUT = HERE / "benchmark"


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def make_data(n, K=8, side=20000.0, seed=0):
    """Clustered points inside a ``side × side`` metre square (projected coordinates).

    Note that the type labels must number **exactly** `n`: the earlier form
    ``np.repeat(np.arange(1, K + 1), n // K)[:n]`` yields only
    `K * (n // K) < n` labels when `n % K != 0` (a slice can truncate but cannot
    pad), so it does not match the length of `coords`. Here the labels are assigned
    from equal-count boundaries, which gives `n` labels for any `n`; when `n` is
    divisible by `K` the result is **identical** to the old form, so existing results
    are unaffected.
    """
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, side, size=(K, 2))
    edges = np.linspace(0, n, K + 1).astype(int)
    pts = []
    for k in range(K):
        m = int(edges[k + 1] - edges[k])
        pts.append(rng.normal(centers[k], side * 0.03, size=(m, 2)))
    coords = np.vstack(pts)
    types = np.empty(n, dtype=int)
    for k in range(K):
        types[edges[k]:edges[k + 1]] = k + 1
    o = rng.permutation(n)
    return coords[o], types[o]


def bandwidth_for_knn(model, coords, n, target_knn, lo=1.0, hi=20000.0, iters=22,
                      sample=200000, cutoff_factor=3.0, seed=42):
    """Bisect for the bandwidth that gives ``k̄ ≈ target_knn`` (the KD-Tree is built
    once)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    idx = (np.random.default_rng(seed).choice(n, sample, replace=False)
           if n > sample else np.arange(n))
    pts = coords[idx]
    target_rho = target_knn / n
    lo_, hi_, best = lo, hi, lo
    for _ in range(iters):
        mid = 0.5 * (lo_ + hi_)
        cnt = tree.query_ball_point(pts, mid * cutoff_factor, return_length=True)
        if float(np.mean(cnt)) / n > target_rho:
            hi_ = mid
        else:
            lo_ = mid
            best = mid
    return best


def reference_perm_time(coords, types, bandwidth, cutoff, K, n_perm=3, chunk=256):
    """Reference ``O(N²)``: every permutation recomputes everything from scratch (no
    neighbourhood structure is cached)."""
    n = coords.shape[0]
    t0 = time.perf_counter()
    for _ in range(n_perm):
        t = np.random.permutation(types)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            d = np.sqrt(((coords[s:e, None, :] - coords[None, :, :]) ** 2).sum(2))
            w = np.exp(-0.5 * (d / bandwidth) ** 2)
            w[d > cutoff] = 0.0
            w[np.arange(e - s), s + np.arange(e - s)] = 0.0
            for b in range(K):
                np.bincount(t[s:e] - 1, weights=w[:, t == b + 1].sum(axis=1),
                            minlength=K)
    return (time.perf_counter() - t0) / n_perm


# ---------------------------------------------------------------------------
# Single-point measurement (runs in a subprocess)
# ---------------------------------------------------------------------------
def measure_one(n, target_knn=150.0, n_perm=100, n_jobs=1, max_ref_n=30000,
                reps=3, K=8, ref_perm=3):
    from stclq import STCLQ
    from stclq.engine import design_matrix

    coords, types = make_data(n, K=K)
    model = STCLQ(distance_metric="projected", diagnostics=False, n_jobs=1)

    bw = bandwidth_for_knn(model, coords, n, target_knn)
    rho = model.rho(coords, bw)
    kbar = rho * n

    t0 = time.perf_counter()
    eng = model._make_engine(coords, types, bw, None, None)
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    eng.global_clq()
    global_s = time.perf_counter() - t0

    # --- Cost breakdown (each entry is the median of 3 runs) ---
    Y = design_matrix(eng.types, eng.K)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        eng.spmm(Y)                    # pure SpMM (fixed Y, no permutation, no reduction)
        ts.append(time.perf_counter() - t0)
    spmm_s = float(np.median(ts))

    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        p = np.random.default_rng(0).permutation(n)
        Yp = Y[p]                      # label permutation plus design-matrix reordering
        ts.append(time.perf_counter() - t0)
    permgen_s = float(np.median(ts))

    ts = []
    for _ in range(reps):
        Z = eng.spmm(Y)
        t0 = time.perf_counter()
        eng.reduce_clq(Z, eng.types - 1)  # bincount reduction (BLAS-free)
        ts.append(time.perf_counter() - t0)
    reduce_s = float(np.median(ts))

    rng = np.random.default_rng(0)
    plan = [rng.permutation(eng.types) for _ in range(reps)]
    ts = []
    for p in plan:
        t0 = time.perf_counter()
        eng.global_clq(p)              # end-to-end single permutation
        ts.append(time.perf_counter() - t0)
    perm_s = float(np.median(ts))

    # --- Serial vs parallel (same batch of permutation seeds, so the two compare) ---
    from stclq.inference import permutation_test
    plan_seeds = np.arange(1000, 1000 + n_perm, dtype=np.int64)
    obs = eng.global_clq()
    t0 = time.perf_counter()
    permutation_test(eng, obs, level="global", null="type", n_perm=n_perm,
                     plan=plan_seeds, correction=False, n_jobs=1)
    ser_total = time.perf_counter() - t0
    if ser_total >= 1.0:                       # parallel is timed only above this size
        t0 = time.perf_counter()
        permutation_test(eng, obs, level="global", null="type", n_perm=n_perm,
                         plan=plan_seeds, correction=False, n_jobs=n_jobs)
        par_total = time.perf_counter() - t0
        perm_par_s = par_total / n_perm
    else:
        par_total = float("nan")
        perm_par_s = float("nan")

    ref_s = float("nan")
    if n <= max_ref_n:
        ref_s = reference_perm_time(coords, eng.types, bw, bw * 3.0, K,
                                    n_perm=max(2, ref_perm))

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "n": n, "K": K, "kbar": kbar, "rho": rho, "bandwidth_m": bw,
        "nnz": int(eng.W.nnz),
        "build_s": build_s, "global_s": global_s,
        "spmm_s": spmm_s, "permgen_s": permgen_s, "reduce_s": reduce_s,
        "perm_s": perm_s, "perm_parallel_s": perm_par_s,
        "serial_total_s": ser_total, "parallel_total_s": par_total,
        "cmp_n_perm": n_perm,
        "ref_perm_s": ref_s,
        "speedup_vs_ref": (ref_s / perm_s) if np.isfinite(ref_s) and perm_s else float("nan"),
        "speedup_parallel": (ser_total / par_total) if par_total else float("nan"),
        "rss_mb": rss_mb,
        "csr_mb_est": 16.0 * eng.W.nnz / 2 ** 20,
        "dense_mb_est": 8.0 * n * n / 2 ** 20,
        "time_999_perms_s": perm_s * 999,
        "n_jobs": n_jobs,
    }


# ---------------------------------------------------------------------------
def slope_fit(xs, ys):
    """Least-squares slope of ``log y = a + s log x``."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
    if m.sum() < 2:
        return float("nan"), float("nan")
    lx, ly = np.log(xs[m]), np.log(ys[m])
    s, a = np.polyfit(lx, ly, 1)
    pred = a + s * lx
    r2 = 1.0 - ((ly - pred) ** 2).sum() / max(((ly - ly.mean()) ** 2).sum(), 1e-30)
    return float(s), float(r2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+",
                    default=[1000, 3000, 10000, 30000, 100000, 300000, 1000000])
    ap.add_argument("--target-knn", type=float, default=150.0)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=20)
    ap.add_argument("--max-ref-n", type=int, default=30000)
    ap.add_argument("--ref-perm", type=int, default=3,
                    help="number of timed permutations for the reference O(N²) run")
    ap.add_argument("--repeat", type=int, default=1,
                    help="independent repeats per N (the median is taken)")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore single-point results already in rows/ and remeasure")
    ap.add_argument("--worker", type=int, default=None, help="internal use")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ROWS = OUT / "rows"
    ROWS.mkdir(parents=True, exist_ok=True)

    def dump(rows):
        """Write the finished points to the final artefacts, so that work completed
        so far survives a kill at any moment."""
        s_perm, r2_perm = slope_fit([r["n"] for r in rows], [r["perm_s"] for r in rows])
        s_spmm, r2_spmm = slope_fit([r["n"] for r in rows], [r["spmm_s"] for r in rows])
        ref_rows = [r for r in rows if np.isfinite(r["ref_perm_s"])]
        s_ref, r2_ref = slope_fit([r["n"] for r in ref_rows],
                                  [r["ref_perm_s"] for r in ref_rows])
        s_build, r2_build = slope_fit([r["n"] for r in rows], [r["build_s"] for r in rows])
        s_rss, r2_rss = slope_fit([r["n"] for r in rows], [r["rss_mb"] for r in rows])
        fits = {
            "perm_slope": s_perm, "perm_r2": r2_perm,
            "spmm_slope": s_spmm, "spmm_r2": r2_spmm,
            "ref_slope": s_ref, "ref_r2": r2_ref,
            "build_slope": s_build, "build_r2": r2_build,
            "rss_slope": s_rss, "rss_r2": r2_rss,
        }
        with open(OUT / "benchmark.json", "w", encoding="utf-8") as fh:
            json.dump({"settings": vars(args), "rows": rows, "fits": fits}, fh,
                      ensure_ascii=False, indent=2)
        with open(OUT / "benchmark.csv", "w", encoding="utf-8") as fh:
            keys = list(rows[0].keys())
            fh.write(",".join(keys) + "\n")
            for r in rows:
                fh.write(",".join(str(r[k]) for k in keys) + "\n")
        return fits

    if args.worker is not None:
        res = measure_one(args.worker, args.target_knn, args.n_perm,
                          args.n_jobs, args.max_ref_n, ref_perm=args.ref_perm)
        print("@@RESULT@@" + json.dumps(res))
        return

    rows = []
    print(f"{'N':>9} {'kbar':>6} {'rho':>9} {'build_s':>8} {'nnz':>11} "
          f"{'spmm_ms':>8} {'perm_ms':>8} {'ref_ms':>10} {'speedup':>8} "
          f"{'par_x':>6} {'rss_MB':>8} {'999_perms':>10}")
    for n in args.ns:
        cache = ROWS / f"n{n:09d}.json"
        if cache.exists() and not args.no_resume:
            row = json.loads(cache.read_text(encoding="utf-8"))
            rows.append(row)
            print(f"{row['n']:>9.0f} {row['kbar']:>6.0f} {row['rho']:>9.5f} "
                  f"{row['build_s']:>8.2f} {row['nnz']:>11.0f} "
                  f"{row['spmm_s']*1e3:>8.2f} {row['perm_s']*1e3:>8.2f} "
                  f"{(row['ref_perm_s']*1e3):>10.2f} "
                  f"{row['speedup_vs_ref']:>8.1f} {row['speedup_parallel']:>6.1f} "
                  f"{row['rss_mb']:>8.0f} {row['time_999_perms_s']:>10.1f}  [cached]")
            continue
        got = []
        for _ in range(args.repeat):
            cmd = [sys.executable, str(pathlib.Path(__file__).resolve()),
                   "--worker", str(n), "--target-knn", str(args.target_knn),
                   "--n-perm", str(args.n_perm), "--n-jobs", str(args.n_jobs),
                   "--max-ref-n", str(args.max_ref_n), "--ref-perm", str(args.ref_perm)]
            p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
            line = [l for l in p.stdout.splitlines() if l.startswith("@@RESULT@@")]
            if not line:
                print(f"  [warn] N={n} subprocess failed: {p.stderr.strip()[-300:]}",
                      flush=True)
                break
            got.append(json.loads(line[0][len("@@RESULT@@"):]))
        if not got:
            continue
        row = {k: float(np.median([g[k] for g in got]))
               if k != "n" and k != "nnz" and k != "K" and k != "n_jobs"
               else got[0][k] for k in got[0]}
        rows.append(row)
        cache.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
        print(f"{row['n']:>9.0f} {row['kbar']:>6.0f} {row['rho']:>9.5f} "
              f"{row['build_s']:>8.2f} {row['nnz']:>11.0f} "
              f"{row['spmm_s']*1e3:>8.2f} {row['perm_s']*1e3:>8.2f} "
              f"{(row['ref_perm_s']*1e3):>10.2f} "
              f"{row['speedup_vs_ref']:>8.1f} {row['speedup_parallel']:>6.1f} "
              f"{row['rss_mb']:>8.0f} {row['time_999_perms_s']:>10.1f}", flush=True)
        dump(rows)                            # write to disk after every completed point

    if not rows:
        print("no rows")
        return
    fits = dump(rows)

    print("\nlog-log slope fits:")
    print(f"  pure SpMM (W@Y) : slope = {fits['spmm_slope']:.3f} (R²={fits['spmm_r2']:.5f})  [theory O(nnz) -> 1]")
    print(f"  single perm end-to-end : slope = {fits['perm_slope']:.3f} (R²={fits['perm_r2']:.5f})")
    print(f"  reference O(N²) : slope = {fits['ref_slope']:.3f} (R²={fits['ref_r2']:.5f})  [theory 2]")
    print(f"  neighbourhood build : slope = {fits['build_slope']:.3f} (R²={fits['build_r2']:.5f})")
    print(f"  peak memory     : slope = {fits['rss_slope']:.3f} (R²={fits['rss_r2']:.5f})")
    print(f"\nresults -> {OUT}")


if __name__ == "__main__":
    main()
