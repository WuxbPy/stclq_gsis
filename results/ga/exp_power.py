"""Power curve of the permutation test (§6.2 / Fig. 3b / Table S5).

**Motivation**: the main calibration experiment (``exp_calibration.py``) already
saturates at a power of 1.000 under the strongly clustered alternative with
``sep = 45 m``, so it cannot tell the two null hypotheses apart. A power curve has to
scan **moderate** alternatives before one can see whether the block permutation that
stays calibrated pays for it in power.

**Design**: scan the **clustering scale** ``sep`` (metres, smaller means more
clustered) and the **imbalance** ``delta`` of the per-period type composition. At
``delta = 0`` the global label permutation is already calibrated under H0, so the
rejection rate reads cleanly as power; at ``delta = 0.6`` the global label
permutation is itself anti-conservative (see ``exp_calibration.py``), so its
"rejections" no longer stand for power, which is exactly why **power must be measured
under a calibrated null hypothesis**.

Each grid point repeats ``--reps`` independent alternative datasets and records the
rejection rate of the two null hypotheses at ``alpha``:

* ``power[perm:type]``        : power of the global label permutation;
* ``power[perm:type_period]`` : power of the block permutation within period;
* ``power[norm:type]``        : power of the moment-matched normal approximation
  (the analytic shortcut, as a control).

The data generator draws the ``target`` type pair towards a per-period centre inside
each period (the strength is set by ``sep``), which violates "type ⊥ space | temporal
structure", so the rejection rate here is **power**.

    PYTHONPATH=src python results/ga/exp_power.py
    PYTHONPATH=src python results/ga/exp_power.py --deltas 0.0 0.6 --seps 25 60 120 220 400
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
from exp_calibration import PAIR, gen_spatiotemporal, normal_pvalue  # noqa: E402

OUT = HERE / "calibration"
ALPHA = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--n-periods", type=int, default=6)
    ap.add_argument("--n-perm", type=int, default=199)
    ap.add_argument("--reps", type=int, default=150)
    ap.add_argument("--bw", type=float, default=95.0)
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.0, 0.6],
                    help="imbalance of the per-period type composition; at 0 the "
                         "global null is calibrated, so the rejection rate is power")
    ap.add_argument("--seps", type=float, nargs="+",
                    default=[25.0, 60.0, 120.0, 220.0, 400.0],
                    help="clustering scale (metres); smaller means more clustered, "
                         "hence higher power")
    ap.add_argument("--temporals", nargs="+", default=["exponential"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore the cached grid points and recompute from scratch "
                         "(resume is the default)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    from stclq.inference import permutation_test

    out_json = OUT / f"calibration_power{args.tag}.json"

    # --- Resume: reuse finished grid points keyed by (temporal, delta, sep_m) ---
    # Long runs get killed when the sandbox restarts, and each grid point takes
    # minutes; caching per grid point means a rerun only fills in what is missing.
    done: dict[tuple, dict] = {}
    if out_json.exists() and not args.no_resume:
        try:
            prev = json.loads(out_json.read_text(encoding="utf-8"))
            for r in prev.get("rows", []):
                done[(r["temporal"], float(r["delta"]), float(r["sep_m"]))] = r
            if done:
                print(f"[resume] reusing {len(done)} finished grid points: "
                      f"{out_json.name}", flush=True)
        except (OSError, ValueError, KeyError):
            done = {}

    rows: list[dict] = []
    t_start = time.perf_counter()
    for temporal in args.temporals:
        for delta in args.deltas:
            for sep in args.seps:
                key = (temporal, float(delta), float(sep))
                if key in done:
                    rows.append(done[key])
                    r0 = done[key]
                    print(f"[resume] temporal={temporal:>11} delta={delta:.1f} "
                          f"sep={sep:>6.1f} m  "
                          f"type={r0['power[perm:type]']:.3f} "
                          f"block={r0['power[perm:type_period]']:.3f}  (cached)",
                          flush=True)
                    continue
                recs = []
                obs_vals = []
                for i in range(args.reps):
                    seed = (700_000 + i + int(sep) * 13 + int(delta * 1000) * 7
                            + (0 if temporal == "none" else 50_000))
                    model = STCLQ(temporal_weight=temporal, temporal_bandwidth=1.0,
                                  distance_metric="projected", diagnostics=False,
                                  n_jobs=1)
                    rng = np.random.default_rng(seed)
                    coords, types, periods = gen_spatiotemporal(
                        args.n, args.K, delta, rng, n_periods=args.n_periods,
                        spatial_cluster=True, sep=sep)
                    eng = model._make_engine(coords, types, args.bw, None, periods)
                    obs = eng.global_clq()
                    a, b = PAIR
                    rec = {"observed_pair": float(obs[a - 1, b - 1])}
                    for null in ("type", "type_period"):
                        res = permutation_test(eng, obs, level="global", null=null,
                                               n_perm=args.n_perm, seed=seed,
                                               correction=False, n_jobs=1,
                                               return_sims=True)
                        rec[f"p_perm[{null}]"] = float(res["p_attract"][a - 1, b - 1])
                        rec[f"p_norm[{null}]"] = float(
                            normal_pvalue(res["sims"], obs)[a - 1, b - 1])
                    recs.append(rec)
                    obs_vals.append(rec["observed_pair"])
                rec = {"temporal": temporal, "delta": delta, "sep_m": sep,
                       "n_reps": args.reps,
                       "mean_observed_pair": float(np.mean(obs_vals))}
                for null in ("type", "type_period"):
                    for tag in ("perm", "norm"):
                        key = f"p_perm[{null}]" if tag == "perm" else f"p_norm[{null}]"
                        rej = float(np.mean([r[key] < ALPHA for r in recs]))
                        rec[f"power[{tag}:{null}]"] = rej
                        rec[f"se[{tag}:{null}]"] = float(
                            np.sqrt(max(rej * (1 - rej), 1e-12) / args.reps))
                rows.append(rec)
                print(f"temporal={temporal:>11} delta={delta:.1f} sep={sep:>6.1f} m  "
                      f"obs={rec['mean_observed_pair']:.3f}  "
                      f"type={rec['power[perm:type]']:.3f}"
                      f"±{rec['se[perm:type]']:.3f}  "
                      f"block={rec['power[perm:type_period]']:.3f}"
                      f"±{rec['se[perm:type_period]']:.3f}  "
                      f"norm(type)={rec['power[norm:type]']:.3f}  "
                      f"[{time.perf_counter()-t_start:.0f}s]", flush=True)
                with open(out_json, "w", encoding="utf-8") as fh:
                    json.dump({"settings": vars(args), "alpha": ALPHA, "pair": PAIR,
                               "rows": rows}, fh, ensure_ascii=False, indent=2)

    out = {"settings": vars(args), "alpha": ALPHA, "pair": PAIR, "rows": rows}
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    with open(OUT / f"calibration_power{args.tag}.csv", "w", encoding="utf-8") as fh:
        keys = list(rows[0].keys())
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"\nresults -> {OUT}")


if __name__ == "__main__":
    main()
