"""Month-by-month STCLQ stability check (temporal-stability evidence for §6.6).

Background
----------
``exp_realdata.py`` reports FDR-significant co-locating type pairs on the **pooled
period only** and never answers whether "these attraction findings hold over the 12
months". This script fills that gap: for every dataset and every ordered type pair
(a -> b) that the pooled analysis calls significant, it recomputes STCLQ for each of
the 12 periods at **the same spatial bandwidth**.

Temporal weight convention
--------------------------
All points inside a single period share the same period_id, so ``|dt| = 0`` and
``w_t = 1/(1+|dt|) = 1``: the per-month STCLQ therefore reduces to "the spatial CLQ
of that month at that bandwidth". Cross-month comparisons are thus made under **the
same definition and the same spatial scale**, not with a different statistic.

    PYTHONPATH=src python results/ga/exp_stability.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from stclq import STCLQ  # noqa: E402
from exp_realdata import DATASETS, load_uk_police  # noqa: E402

OUT = HERE / "realdata"


def significant_pairs(key: str) -> list[tuple[int, int, str, str, float]]:
    """Read the FDR<0.05 ordered type pairs (with their pooled STCLQ) from the pooled
    CSV."""
    csv = OUT / f"{key}_pairs_global_null.csv"
    rows = csv.read_text(encoding="utf-8").strip().split("\n")[1:]
    out = []
    for r in rows:
        f = r.split(",")
        # type_a_id,type_b_id,type_a,type_b,stclq,p_value,significant,category,null
        if f[6].strip().lower() == "true":
            out.append((int(f[0]), int(f[1]), f[2], f[3], float(f[4])))
    return out


def stability(key: str, path, bandwidth: float, temporal="inverse_dt", tau=1.0):
    coords, types, periods, names = load_uk_police(path)
    K = int(np.max(types))
    labels = [names.get(t, f"Type_{t}") for t in range(1, K + 1)]
    pids = sorted(int(p) for p in np.unique(periods))
    pairs = significant_pairs(key)
    print(f"--- {key}: {coords.shape[0]:,} points, {len(pids)} periods, "
          f"bandwidth {bandwidth:.2f} m")
    print(f"    significant ordered pairs {len(pairs)}: "
          + ", ".join(f"{a}->{b}({v:.3f})" for _, _, a, b, v in pairs))

    model = STCLQ(temporal_weight=temporal, temporal_bandwidth=tau,
                  distance_metric="auto", diagnostics=False, n_jobs=1)

    # pooled over all periods (must match stclq in the CSV; serves as a self-check)
    eng_all = model._make_engine(coords, types, bandwidth, None, periods)
    pooled = np.asarray(eng_all.global_clq())

    monthly = {pid: np.full((K, K), np.nan) for pid in pids}
    n_by_period = {}
    for pid in pids:
        m = periods == pid
        n_by_period[pid] = int(m.sum())
        sub_c, sub_t = coords[m], types[m]
        # dt=0 within a period, hence w_t=1; the period id is still passed so that the
        # construction path stays the same as in the main experiment
        eng = model._make_engine(sub_c, sub_t, bandwidth, None, np.full(m.sum(), pid))
        monthly[pid] = np.asarray(eng.global_clq())

    recs = []
    for a, b, na, nb, pooled_v in pairs:
        vals = np.array([monthly[pid][a - 1, b - 1] for pid in pids], dtype=float)
        recs.append({
            "type_a": na, "type_b": nb, "a_id": a, "b_id": b,
            "pooled_stclq": round(float(pooled[a - 1, b - 1]), 4),
            "pooled_stclq_csv": round(pooled_v, 4),
            "monthly_stclq": [round(float(v), 4) for v in vals],
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
            "median": round(float(np.median(vals)), 4),
            "n_months_attract": int((vals > 1.0).sum()),
            "n_months": len(pids),
            "sign_consistent": bool((vals > 1.0).all() or (vals < 1.0).all()),
        })

    out = {
        "dataset": key,
        "bandwidth_m": float(bandwidth),
        "n_periods": len(pids),
        "period_ids": pids,
        "n_points_by_period": {str(k): v for k, v in n_by_period.items()},
        "temporal_weight": temporal,
        "note": ("Within a single period all time stamps coincide, so w_t = 1 and the "
                 "per-period STCLQ is the spatial co-location quotient at the same "
                 "bandwidth."),
        "pairs": recs,
        "n_pairs_stable": int(sum(r["sign_consistent"] for r in recs)),
        "n_pairs": len(recs),
        "min_months_attract": int(min(r["n_months_attract"] for r in recs)) if recs else 0,
        "max_months_attract": int(max(r["n_months_attract"] for r in recs)) if recs else 0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{key}_period_stability.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"    -> {key}_period_stability.json  "
          f"(pairs with a consistent sign {out['n_pairs_stable']}/{out['n_pairs']})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    a = ap.parse_args()
    for key in a.datasets:
        summ = json.loads((OUT / f"{key}_summary.json").read_text(encoding="utf-8"))
        stability(key, DATASETS[key], summ["bandwidth_m"])


if __name__ == "__main__":
    main()
