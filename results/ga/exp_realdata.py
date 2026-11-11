"""Real-data experiments (§6.6 / Fig. 1b / Table S3).

Data: UK Police open data (preprocessed, with ``period_id`` / ``type_id``)

* **Avon and Somerset Constabulary**: 139,001 points × 12 periods × 4 types (the
  large-scale benchmark);
* **City of London Police**: 15,776 points × 12 periods × 4 types (the small-sample
  control).

Pipeline: bisect for the bandwidth that meets the ``ρ`` target, build the CSR once,
evaluate the global GTWCLQ (``inverse_dt``), run 999 permutations (serial and with 20
processes, checking bit-for-bit agreement), test with both null hypotheses at once
(``type`` global label permutation and ``type_period`` block permutation within
period) and compare them, then export the significant co-locating pairs.

    PYTHONPATH=src python results/ga/exp_realdata.py
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import resource
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stclq import STCLQ  # noqa: E402
from stclq.io import global_to_dataframe  # noqa: E402

OUT = HERE / "realdata"
DATASETS = {
    "avon_and_somerset": ROOT / "data" / "avon_and_somerset_2023.csv",
    "city_of_london": ROOT / "data" / "city_of_london_2023.csv",
}


def load_uk_police(path, max_periods=None):
    """Read ``data/*_2023.csv`` into coords (lon/lat) / types / periods / type names."""
    lon, lat, ty, per, names = [], [], [], [], {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            try:
                lo, la = float(r["longitude"]), float(r["latitude"])
            except (TypeError, ValueError):
                continue
            t = int(r["type_id"])
            names.setdefault(t, r.get("type_name") or f"Type_{t}")
            lon.append(lo)
            lat.append(la)
            ty.append(t)
            per.append(int(r["period_id"]))
    coords = np.column_stack([np.asarray(lon), np.asarray(lat)])
    types = np.asarray(ty, dtype=np.int64)
    periods = np.asarray(per, dtype=np.int64)
    if max_periods is not None:
        m = periods <= int(max_periods)
        coords, types, periods = coords[m], types[m], periods[m]
    return coords, types, periods, names


def bandwidth_for_rho(model, coords, target_rho, lo=2.0, hi=8000.0, iters=24):
    """Bisect for the spatial bandwidth with ρ≈target_rho (ρ increases monotonically
    with the bandwidth)."""
    best = lo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if model.rho(coords, mid) > target_rho:
            hi = mid
        else:
            lo = mid
            best = mid
    return best


def run_dataset(key, path, target_rho=0.001, n_perm=999, n_jobs=20,
                temporal="inverse_dt", tau=1.0, seed=42, rounds=2):
    print("=" * 74)
    print(f"Dataset: {key}   ({path.name})")
    print("=" * 74)
    coords, types, periods, names = load_uk_police(path)
    n = coords.shape[0]
    K = int(np.max(types))
    print(f"points={n:,}  types={K}  periods={len(np.unique(periods))}  "
          f"type names={sorted(names.values())}")

    model = STCLQ(temporal_weight=temporal, temporal_bandwidth=tau,
                  distance_metric="auto", diagnostics=False, n_jobs=1)

    t0 = time.perf_counter()
    bw = bandwidth_for_rho(model, coords, target_rho)
    bw_s = time.perf_counter() - t0
    rho = model.rho(coords, bw)

    t0 = time.perf_counter()
    eng = model._make_engine(coords, types, bw, None, periods)
    build_s = time.perf_counter() - t0
    nnz = int(eng.W.nnz)
    print(f"bandwidth={bw:.1f} m (selected in {bw_s:.1f}s)  ρ={rho:.5f}  "
          f"k̄={rho*n:.0f}  nnz={nnz:,}  build={build_s:.2f}s")

    t0 = time.perf_counter()
    obs = eng.global_clq()
    global_s = time.perf_counter() - t0
    print(f"global GTWCLQ: {global_s:.3f}s")

    # --- Bit-for-bit agreement: serial vs parallel (small n_perm) ---
    from stclq.inference import permutation_test
    plan_small = np.arange(7_000, 7_000 + 100, dtype=np.int64)
    r_ser = permutation_test(eng, obs, level="global", null="type",
                             n_perm=100, plan=plan_small, correction=False,
                             n_jobs=1)
    r_par = permutation_test(eng, obs, level="global", null="type",
                             n_perm=100, plan=plan_small, correction=False,
                             n_jobs=n_jobs)
    bitwise = bool(np.array_equal(r_ser["p_attract"], r_par["p_attract"]))
    print(f"bit-for-bit agreement (serial vs {n_jobs} processes, 100 permutations): "
          f"{bitwise}")

    # --- 999 permutations: **interleaved** timing of the serial and parallel runs ---
    # Measuring each of them once lets host load drift leak into the speed-up: if the
    # serial run happens to hit a busy stretch and the parallel run a quiet one, the
    # speed-up is systematically inflated. Several A/B/A/B rounds therefore make both
    # quantities experience the same load, and the median of each is taken.
    ser_ts, par_ts = [], []
    test_par = None
    for _ in range(rounds):
        t0 = time.perf_counter()
        test_ser = permutation_test(eng, obs, level="global", null="type",
                                    n_perm=n_perm, seed=seed, n_jobs=1)
        ser_ts.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        test_par = permutation_test(eng, obs, level="global", null="type",
                                    n_perm=n_perm, seed=seed, n_jobs=n_jobs)
        par_ts.append(time.perf_counter() - t0)
    ser_s = float(np.median(ser_ts))
    par_s = float(np.median(par_ts))
    print(f"{n_perm} permutations (null=type, median of {rounds} interleaved rounds): "
          f"serial {ser_s:.1f}s {ser_ts} → {n_jobs} processes {par_s:.1f}s {par_ts}"
          f" (speed-up {ser_s/par_s:.1f}×)")

    # --- Block permutation within period (conditional null hypothesis) ---
    t0 = time.perf_counter()
    test_block = permutation_test(eng, obs, level="global", null="type_period",
                                  n_perm=n_perm, seed=seed, n_jobs=n_jobs)
    block_s = time.perf_counter() - t0

    labels = [names.get(t, f"Type_{t}") for t in range(1, K + 1)]

    # Significant "type pairs" count **off-diagonal** entries only. The K diagonal
    # entries of the CLQ matrix are the co-location quotient of a type with itself
    # (always strong attraction), which is not the "type pair" of the paper; counting
    # them would turn K trivial self-pairs into findings and inflate the count.
    off = ~np.eye(K, dtype=bool)
    p_global_off = np.asarray(test_par["p_attract_fdr"])[off]
    p_block_off = np.asarray(test_block["p_attract_fdr"])[off]
    sig_global = p_global_off < 0.05
    sig_block = p_block_off < 0.05

    summary = {
        "dataset": key, "n_points": n, "n_types": K,
        "n_periods": int(len(np.unique(periods))),
        "type_names": labels, "temporal_weight": temporal,
        "target_rho": target_rho, "bandwidth_m": float(bw), "rho": float(rho),
        "kbar": float(rho * n), "nnz": nnz,
        "build_s": build_s, "global_s": global_s,
        "n_perm": n_perm,
        "serial_seconds": ser_s, "parallel_seconds": par_s,
        "parallel_speedup": ser_s / par_s, "n_jobs": n_jobs,
        "timing_rounds": int(rounds),
        "serial_rounds_s": [round(float(x), 3) for x in ser_ts],
        "parallel_rounds_s": [round(float(x), 3) for x in par_ts],
        "bitwise_identical": bitwise,
        "block_null_seconds": block_s,
        "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "n_pairs_offdiag": int(off.sum()),
        "n_significant_fdr[type]": int(sig_global.sum()),
        "n_significant_fdr[type_period]": int(sig_block.sum()),
        # how far the two null hypotheses agree on this dataset
        "n_decision_flip": int((sig_global != sig_block).sum()),
        "max_abs_p_diff": float(np.max(np.abs(p_global_off - p_block_off))),
        "n_p_increase": int((p_block_off > p_global_off + 1e-12).sum()),
        "n_p_decrease": int((p_block_off < p_global_off - 1e-12).sum()),
        "mean_p_global": float(p_global_off.mean()),
        "mean_p_block": float(p_block_off.mean()),
        "mean_abs_log_clq": float(np.mean(np.abs(np.log(np.clip(obs, 1e-9, None))))),
    }

    df_type = global_to_dataframe(obs, test_par["p_attract_fdr"], type_names=labels)
    df_type["null"] = "type"
    df_block = global_to_dataframe(obs, test_block["p_attract_fdr"],
                                   type_names=labels)
    df_block["null"] = "type_period"

    OUT.mkdir(parents=True, exist_ok=True)
    df_type.to_csv(OUT / f"{key}_pairs_global_null.csv", index=False)
    df_block.to_csv(OUT / f"{key}_pairs_block_null.csv", index=False)
    np.save(OUT / f"{key}_global_matrix.npy", obs)
    with open(OUT / f"{key}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"\nSignificant type pairs (FDR<0.05, {summary['n_pairs_offdiag']} "
          f"off-diagonal pairs; global label permutation / block permutation): "
          f"{summary['n_significant_fdr[type]']} / "
          f"{summary['n_significant_fdr[type_period]']} "
          f"(decision flips {summary['n_decision_flip']}, "
          f"max |Δp| = {summary['max_abs_p_diff']:.4f})")
    for tag, df in (("type", df_type), ("type_period", df_block)):
        sig = df[df["significant"]].sort_values("stclq", ascending=False).head(6)
        print(f"  [{tag}]")
        if len(sig):
            print(sig[["type_a", "type_b", "stclq", "p_value"]]
                  .to_string(index=False).replace("\n", "\n    "))
        else:
            print("    (none)")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["avon_and_somerset", "city_of_london"])
    ap.add_argument("--target-rho", type=float, default=0.001)
    ap.add_argument("--n-perm", type=int, default=999)
    ap.add_argument("--n-jobs", type=int, default=20)
    ap.add_argument("--temporal", default="inverse_dt",
                    choices=["none", "inverse_dt", "exponential"])
    ap.add_argument("--rounds", type=int, default=2,
                    help="number of interleaved serial/parallel timing rounds (one "
                         "measurement per round, median taken)")
    args = ap.parse_args()

    summaries = []
    for key in args.datasets:
        path = DATASETS[key]
        if not path.exists():
            print(f"[skip] not found: {path}")
            continue
        summaries.append(run_dataset(key, path, args.target_rho, args.n_perm,
                                     args.n_jobs, args.temporal, rounds=args.rounds))
        # dump the summary after each dataset, so an interrupted long run keeps the
        # datasets already completed
        with open(OUT / "summary_all.json", "w", encoding="utf-8") as fh:
            json.dump(summaries, fh, ensure_ascii=False, indent=2)
        print()
    with open(OUT / "summary_all.json", "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, ensure_ascii=False, indent=2)
    print(f"results -> {OUT}")


if __name__ == "__main__":
    main()
