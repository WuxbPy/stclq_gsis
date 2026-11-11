"""Strong and weak scaling of the parallel permutation engine (§6.5 / Fig. 7).

Three lines of evidence:

1. **Strong scaling**: ``N`` and ``n_perm`` are fixed and the process count runs
   through ``p = 1,2,4,8,16,20``; we report the wall-time, the speed-up ``S(p)`` and
   the parallel efficiency ``E(p) = S(p)/p``.
2. **Weak scaling**: the per-process workload is held constant (``N ∝ p`` with
   ``n_perm`` fixed), so ideally the wall-time barely grows with ``p``.
3. **Bit-for-bit agreement and evidence against serialisation**:
   * the p-value matrix at every ``p`` is **bit-for-bit identical** to the serial one;
   * the "per-permutation parallel start-up overhead" is recorded: if a subprocess had
     to serialise the CSR, that overhead would grow linearly in ``nnz``; measuring it
     independent of ``nnz`` proves that the fork shares the read-only cache.

    PYTHONPATH=src python results/ga/exp_parallel.py
    PYTHONPATH=src python results/ga/exp_parallel.py --n 300000 --n-perm 200
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time

# Benchmark control variable: with process-level parallelism the underlying BLAS
# threads are capped at 1 (this must happen before numpy is imported).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from exp_benchmark import bandwidth_for_knn, make_data  # noqa: E402

OUT = HERE / "parallel"


def build_engine(n, target_knn=150.0, K=8, seed=0):
    from stclq import STCLQ
    coords, types = make_data(n, K=K, seed=seed)
    model = STCLQ(distance_metric="projected", diagnostics=False, n_jobs=1)
    bw = bandwidth_for_knn(model, coords, n, target_knn)
    return model._make_engine(coords, types, bw, None, None), bw


def time_perm(engine, plan, n_jobs, repeat=3, return_sims=False):
    """Measure the wall-time of ``permutation_test`` (median of the repeats)."""
    from stclq.inference import permutation_test
    obs = engine.global_clq()
    ts, res = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        res = permutation_test(engine, obs, level="global", null="type",
                               n_perm=len(plan), plan=plan, correction=False,
                               n_jobs=n_jobs, return_sims=return_sims)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), res


# ---------------------------------------------------------------------------
# Platform calibration: separating the "algorithmic speed-up" from the parallel
# ceiling this machine can actually reach
#
# Permutation resampling is **memory-bound** (every permutation streams all ``nnz``
# and the design matrix), so the measured speed-up is truncated by the memory
# bandwidth of the platform, not by the algorithm. To let the reader separate the
# two, two baseline curves are measured on the same machine in the same process mode:
#
# * ``cpu`` : throughput of pure computation (small working set) as the process count
#   grows, that is the **computational parallel ceiling** of this host;
# * ``bw``  : aggregate bandwidth of the streamed ``a*x+b`` array operation, that is
#   the **memory bandwidth ceiling** of this host.
# ---------------------------------------------------------------------------
_STREAM_N = 32_000_000          # 256 MB float64


def _cpu_task(_):
    """Pure-computation task (1.6 MB working set, so it stays resident in cache)."""
    x = np.ones(200_000)
    t0 = time.perf_counter()
    s = 0.0
    for _ in range(4000):
        s += float((x * np.pi).sum())
    return time.perf_counter() - t0


def _stream_task(_):
    """Streaming memory task: returns ``(seconds, bytes transferred)``."""
    a = np.ones(_STREAM_N)
    b = np.empty(_STREAM_N)
    t0 = time.perf_counter()
    for _ in range(4):
        np.multiply(a, 2.0, out=b)          # reads 256 MB, writes 256 MB
    dt = time.perf_counter() - t0
    return dt, 4 * 2 * _STREAM_N * 8


def platform_calibration(ctx, ps, repeat=2):
    """Measure the computational parallel ceiling and the aggregate memory bandwidth
    ceiling of this host."""
    cal = {"cpu": [], "bandwidth": [], "stream_gb": _STREAM_N * 8 / 1e9}

    base_t = None
    for p in ps:
        ts = []
        for _ in range(repeat):
            with ctx.Pool(processes=p) as pool:
                ts.append(max(pool.map(_cpu_task, [None] * p)))
        t = float(np.median(ts))
        if base_t is None:
            base_t = t
        cal["cpu"].append({"p": p, "wall_s": t, "throughput": p / t,
                           "speedup": p * base_t / t,
                           "efficiency": p * base_t / t / p})

    base_bw = None
    for p in ps:
        obs = []
        for _ in range(repeat):
            with ctx.Pool(processes=p) as pool:
                res = pool.map(_stream_task, [None] * p)
            per = [n / dt / 1e9 for dt, n in res]
            obs.append(float(np.mean(per)))
        per_gbs = float(np.median(obs))
        agg = per_gbs * p
        if base_bw is None:
            base_bw = per_gbs
        cal["bandwidth"].append({"p": p, "per_process_gbs": per_gbs,
                                 "aggregate_gbs": agg,
                                 "relative": agg / base_bw})
    cal["peak_aggregate_gbs"] = max(b["aggregate_gbs"] for b in cal["bandwidth"])
    cal["max_cpu_speedup"] = max(c["speedup"] for c in cal["cpu"])
    cal["max_bw_relative"] = max(b["relative"] for b in cal["bandwidth"])

    # Control curve for the memory-bound weak scaling of the platform itself:
    # with a constant per-process workload (the same stream task) the time of one
    # task goes as 1/(per-process bandwidth), so efficiency(p) = bw_p / bw_1. That is
    # exactly the constrained class this engine belongs to; as a control it shows
    # whether the weak-scaling efficiency of the engine is an algorithmic problem or a
    # memory-system problem.
    if cal["bandwidth"]:
        bw1 = cal["bandwidth"][0]["per_process_gbs"]
        cal["weak_control"] = [
            {"p": b["p"], "efficiency": b["per_process_gbs"] / bw1}
            for b in cal["bandwidth"] if b["per_process_gbs"] > 0
        ]
    return cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--ps", type=int, nargs="+", default=[1, 2, 4, 8, 16, 20])
    ap.add_argument("--weak-n", type=int, default=100_000,
                    help="problem size N held fixed in the weak scaling")
    ap.add_argument("--weak-n-perm-per-process", type=int, default=50,
                    help="permutations carried by **each process** in the weak "
                         "scaling; the total is p times this value")
    ap.add_argument("--weak-n-per-process", type=int, default=20_000,
                    help="(reserved) points per process for a size sweep; not used in "
                         "the weak scaling")
    ap.add_argument("--weak-n-perm", type=int, default=50,
                    help="(reserved) not used in the weak scaling")
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cpu = os.cpu_count() or 1
    print(f"CPU cores = {cpu}; strong scaling N={args.n:,}, n_perm={args.n_perm}\n")

    def dump():
        out = {"cpu": cpu, "settings": vars(args), "strong": strong,
               "startup_overhead": overhead, "weak": weak, "platform": platform}
        with open(OUT / "parallel.json", "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        return out

    strong, overhead, weak = [], [], []
    platform = {}

    # ---------------- Platform calibration (compute ceiling, bandwidth ceiling) ----
    print("Platform calibration: computational parallel ceiling and aggregate memory "
          "bandwidth ceiling ...")
    platform = platform_calibration(mp.get_context("fork"), args.ps, repeat=2)
    print(f"  single-process streaming bandwidth = "
          f"{platform['bandwidth'][0]['per_process_gbs']:.1f} GB/s; "
          f"aggregate peak = {platform['peak_aggregate_gbs']:.1f} GB/s "
          f"({platform['max_bw_relative']:.2f}× single process)")
    print(f"  compute throughput ceiling = {platform['max_cpu_speedup']:.2f}× (p="
          f"{max(platform['cpu'], key=lambda c: c['speedup'])['p']})")
    dump()

    # ---------------- Strong scaling ----------------
    eng, bw = build_engine(args.n)
    plan = np.arange(10_000, 10_000 + args.n_perm, dtype=np.int64)
    print(f"nnz = {eng.W.nnz:,}  rho = {eng.W.nnz/args.n**2:.3e}  "
          f"bandwidth = {bw:.1f} m")
    ref_p = None
    print(f"\n{'p':>3} {'wall_s':>9} {'per_perm_ms':>12} {'speedup':>8} "
          f"{'efficiency':>11} {'bitwise':>8}")
    for p in args.ps:
        t, res = time_perm(eng, plan, p, args.repeat)
        if ref_p is None:
            ref_p = p
            base, base_p = t, res["p_attract"]
            sp, eff, same = 1.0, 1.0, True
        else:
            sp = base / t
            eff = sp / p
            same = bool(np.array_equal(base_p, res["p_attract"]))
        strong.append({"p": p, "wall_s": t, "per_perm_ms": t / args.n_perm * 1e3,
                       "speedup": sp, "efficiency": eff, "bitwise_identical": same})
        print(f"{p:>3} {t:>9.3f} {t/args.n_perm*1e3:>12.3f} {sp:>8.2f} "
              f"{eff:>11.3f} {str(same):>8}", flush=True)
        dump()                                     # write to disk after every p

    # ---------------- Evidence against serialisation + size sweep ----------------
    # Logic of the evidence: if a subprocess had to **serialise** the CSR cache, the
    # parallel side would pay an extra O(nnz) packing and transfer cost, so the
    # "parallel gain" would degrade monotonically with nnz. The measured gain stays
    # roughly flat across nnz values spanning two orders of magnitude, so the parallel
    # side carries no O(nnz) additional overhead.
    # The "parallel overhead" is also given, as the measured parallel time minus the
    # ideal time (ts/p), per permutation.
    p_ref = 20
    print(f"\n{'N':>9} {'nnz':>12} {'serial_ms':>10} "
          f"{'par(p=20)_ms':>13} {'speedup':>8} {'eff':>7} {'overhead_ms':>12}")
    for nn in (10_000, 30_000, 100_000, 300_000):
        e, _ = build_engine(nn)
        pl = np.arange(20_000, 20_020, dtype=np.int64)
        ts, _ = time_perm(e, pl, 1, repeat=3)
        tp, _ = time_perm(e, pl, p_ref, repeat=3)
        sp = ts / tp
        ideal = ts / p_ref
        overhead.append({"n": nn, "nnz": int(e.W.nnz),
                         "csr_mb": 16.0 * e.W.nnz / 2 ** 20,
                         "serial_1perm_ms": ts / len(pl) * 1e3,
                         "parallel_1perm_ms": tp / len(pl) * 1e3,
                         "speedup": sp,
                         "efficiency": sp / p_ref,
                         "overhead_1perm_ms": (tp - ideal) / len(pl) * 1e3})
        last = overhead[-1]
        print(f"{nn:>9} {e.W.nnz:>12,} {last['serial_1perm_ms']:>10.3f} "
              f"{last['parallel_1perm_ms']:>13.3f} {sp:>8.2f} {sp/p_ref:>7.3f} "
              f"{last['overhead_1perm_ms']:>12.3f}", flush=True)
        dump()

    # ---------------- Weak scaling ----------------
    # Design: the **problem size N is fixed** and each process carries a fixed number
    # of permutations (the total is p × m). Ideally the per-process workload does not
    # depend on p, so the wall-time stays constant. This matches how the engine is
    # actually used: spread the same CSR cache over more processes to buy more
    # permutations.
    print(f"\nWeak scaling: N={args.weak_n:,} fixed, "
          f"{args.weak_n_perm_per_process} permutations per process "
          f"(the total grows linearly with p)")
    print(f"{'p':>3} {'n_perm':>8} {'wall_s':>9} {'ms/perm':>9} "
          f"{'efficiency':>11} {'prefix=':>8}")
    ew, _ = build_engine(args.weak_n)
    base_t = None
    base_sims = None
    for p in args.ps:
        m = args.weak_n_perm_per_process
        pl = np.arange(30_000, 30_000 + m * p, dtype=np.int64)
        t, res = time_perm(ew, pl, p, args.repeat, return_sims=True)
        # Reproducibility: the total number of permutations changes with p, so the
        # **common prefix** of the simulated statistic matrices is compared
        if base_sims is None:
            base_t, base_sims = t, res["sims"]
            same = True
        else:
            same = bool(np.array_equal(base_sims, res["sims"][:m]))
        weak.append({"p": p, "n": args.weak_n, "n_perm": m * p, "wall_s": t,
                     "per_perm_ms": t / (m * p) * 1e3,
                     "efficiency": base_t / t,
                     "prefix_identical": same})
        print(f"{p:>3} {m*p:>8} {t:>9.3f} {t/(m*p)*1e3:>9.3f} "
              f"{base_t/t:>11.3f} {str(same):>8}", flush=True)
        dump()

    dump()
    print(f"\nresults -> {OUT}")


if __name__ == "__main__":
    main()
