"""Standalone scaling probe for the large-scale benchmark (backup, produces no numbers
for the main text).

Measures:
  * build time (KD-Tree → CSR, once)
  * time of a single global STCLQ evaluation
  * time of a single permutation (label permutation; neighbourhood cache reused)
  * speed-up relative to the O(N²) reference implementation
  * peak memory (ru_maxrss)

Control variable: for each N a bisection picks the bandwidth that makes the mean
neighbourhood size k̄ ≈ target_knn (which separates the "algorithmic gain" from "more
data"), and ρ = k̄/N is reported.

    PYTHONPATH=src python results/ga/benchmark_scaling.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import resource
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stclq import STCLQ  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "benchmark"


def make_data(n, K=8, side=20000.0, seed=0):
    """Generate clustered synthetic points in a side×side metre square (projected
    coordinates, distance_metric='projected')."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, side, size=(K, 2))
    per = n // K
    pts = []
    for k in range(K):
        m = per if k < K - 1 else n - len(pts)
        pts.append(rng.normal(centers[k], side * 0.03, size=(m, 2)))
    coords = np.vstack(pts)
    types = np.repeat(np.arange(1, K + 1), per)[:n]
    order = rng.permutation(n)
    return coords[order], types[order]


def bandwidth_for_knn(model, coords, n, target_knn, lo=1.0, hi=5000.0, iters=16):
    target_rho = target_knn / n
    best = lo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if model.rho(coords, mid) > target_rho:
            hi = mid
        else:
            lo = mid
            best = mid
    return best


def reference_perm_time(coords, types, bandwidth, cutoff, n_perm=5, K=8, chunk=512):
    """O(N²) reference implementation: every permutation recomputes the global CLQ
    from scratch (the main path of MCCLQ/asw_stclq)."""
    n = len(coords)
    t0 = time.perf_counter()
    for _ in range(n_perm):
        t = np.random.permutation(types)
        pnw = np.zeros((K, K))
        den = np.zeros(K)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            d = coords[s:e, None, :] - coords[None, :, :]
            dist = np.sqrt((d * d).sum(axis=2))
            w = np.exp(-0.5 * (dist / bandwidth) ** 2)
            w[dist > cutoff] = 0.0
            w[np.arange(e - s), s + np.arange(e - s)] = 0.0
            src = t[s:e] - 1
            for b in range(K):
                pnw[:, b] += np.bincount(src, weights=w[:, t == b + 1].sum(axis=1), minlength=K)
            den += np.bincount(src, weights=w.sum(axis=1), minlength=K)
    return (time.perf_counter() - t0) / n_perm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[1000, 3000, 10000, 30000, 100000])
    ap.add_argument("--target-knn", type=float, default=150.0)
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--max-ref-n", type=int, default=10000)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'N':>9} {'kbar':>6} {'rho':>9} {'build_s':>8} {'global_s':>9} "
          f"{'perm_ms':>8} {'ref_ms':>9} {'speedup':>8} {'rss_MB':>8}")

    for n in args.ns:
        coords, types = make_data(n)
        model = STCLQ(distance_metric="projected", diagnostics=False)

        bw = bandwidth_for_knn(model, coords, n, args.target_knn)
        rho = model.rho(coords, bw)
        kbar = rho * n

        t0 = time.perf_counter()
        engine = model._make_engine(coords, types, bw, None, None)
        build_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        g = engine.global_clq()
        global_s = time.perf_counter() - t0

        # single permutation (labels)
        rng = np.random.default_rng(0)
        t0 = time.perf_counter()
        for _ in range(args.n_perm):
            engine.global_clq(rng.permutation(engine.types))
        perm_s = (time.perf_counter() - t0) / args.n_perm

        ref_s = np.nan
        if n <= args.max_ref_n:
            ref_s = reference_perm_time(coords, engine.types, bw, bw * 3.0,
                                        n_perm=max(3, args.n_perm // 4), K=engine.K)
        speedup = (ref_s / perm_s) if np.isfinite(ref_s) and perm_s > 0 else np.nan

        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        rows.append({"n": n, "kbar": kbar, "rho": rho, "bandwidth_m": bw,
                     "build_s": build_s, "global_s": global_s,
                     "perm_s": perm_s, "ref_perm_s": ref_s, "speedup": speedup,
                     "rss_mb": rss_mb, "nnz": int(engine.W.nnz)})
        print(f"{n:>9} {kbar:>6.0f} {rho:>9.5f} {build_s:>8.2f} {global_s:>9.3f} "
              f"{perm_s*1000:>8.2f} {ref_s*1000:>9.2f} {speedup:>8.1f} {rss_mb:>8.0f}")

    with open(OUT / "benchmark.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
    with open(OUT / "benchmark.csv", "w", encoding="utf-8") as fh:
        keys = list(rows[0].keys())
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")

    # optional figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ns = [r["n"] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].loglog(ns, [r["perm_s"] for r in rows], "o-", label="STCLQ (O(nnz))")
        refn = [r["n"] for r in rows if np.isfinite(r["ref_perm_s"])]
        refv = [r["ref_perm_s"] for r in rows if np.isfinite(r["ref_perm_s"])]
        if refn:
            ax[0].loglog(refn, refv, "s--", label="reference (O(N²))")
        ax[0].set_xlabel("N"); ax[0].set_ylabel("per-permutation time (s)")
        ax[0].legend(); ax[0].grid(True, which="both", ls=":")
        sp = [(r["n"], r["speedup"]) for r in rows if np.isfinite(r["speedup"])]
        if sp:
            ax[1].semilogx([x for x, _ in sp], [y for _, y in sp], "o-")
        ax[1].set_xlabel("N"); ax[1].set_ylabel("speedup vs O(N²)")
        ax[1].grid(True, which="both", ls=":")
        fig.tight_layout()
        fig.savefig(OUT / "fig_benchmark.png", dpi=150)
        print(f"figure -> {OUT / 'fig_benchmark.png'}")
    except Exception as exc:  # pragma: no cover
        print(f"(skip figure: {exc})")

    print(f"\nresults -> {OUT}")


if __name__ == "__main__":
    main()
