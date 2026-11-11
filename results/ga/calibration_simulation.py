"""Permutation calibration simulation (imported by exp_calibration.py).

Scenarios:
  A) Spatial: type-I error rate (H0) and power (H1) of the label permutation;
  B) Attribute weighting: type-I error rate and power of three null hypotheses
     (``type`` / ``attribute`` (global) / ``attribute_within_type`` (within type,
     conditional));
  C) Multi-period: ``type`` (global label permutation) vs ``type_period`` (block
     permutation within period);
  D) **Multi-construction sweep**: over the grid (number of types K × attribute-type
     coupling strength coupling × attribute bandwidth attr_bw), compare the type-I
     error rate (i.i.d. attributes) and the power (spatially smooth attributes) of
     the three null hypotheses.

Meaning of the three attribute-related null hypotheses:
  * ``type``                  : label permutation, tests "type ⊥ location";
  * ``attribute``             : **global** attribute permutation, tests
    "attribute ⊥ location" (ignoring type);
  * ``attribute_within_type`` : **within-type** attribute permutation (conditional
    null hypothesis), tests "attribute ⊥ location | type".

    PYTHONPATH=src python results/ga/calibration_simulation.py
    PYTHONPATH=src python results/ga/calibration_simulation.py --scenarios D --sweep-reps 50 --n-perm 99
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stclq import STCLQ, AttributeSpec, AttributeType  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "calibration"
ALPHA = 0.05
TARGET_PAIR = (1, 2)
WITHIN_PAIR = (1, 1)
NULLS = ["type", "attribute", "attribute_within_type"]


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def _uniform(rng, n, side=1000.0):
    return rng.uniform(0.0, side, size=(n, 2))


def _smooth_attribute(coords, rng, side=1000.0):
    """Spatially smooth attribute field (two Gaussian centres plus noise)."""
    c1, c2 = rng.uniform(0, side, 2), rng.uniform(0, side, 2)
    z = (np.exp(-((coords - c1) ** 2).sum(1) / (2 * (0.15 * side) ** 2))
         - np.exp(-((coords - c2) ** 2).sum(1) / (2 * (0.15 * side) ** 2)))
    return z + rng.normal(0, 0.05, coords.shape[0])


def gen_null(n, K, rng, side=1000.0):
    """H0: type is independent of location."""
    coords = _uniform(rng, n, side)
    types = rng.integers(1, K + 1, size=n)
    return coords, types


def gen_attract(n, K, rng, side=1000.0, sep=120.0):
    """H1: the target type pair (1,2) cluster together (attraction)."""
    coords, types = gen_null(n, K, rng, side)
    c = rng.uniform(0.2 * side, 0.8 * side, size=2)
    for a in TARGET_PAIR:
        idx = np.where(types == a)[0]
        coords[idx] = c + rng.normal(0, sep, size=(idx.size, 2))
    return coords, types


def gen_coupled(n, K, coupling, rng, side=1000.0, spatial_attr=False):
    """Generate "attribute → type" data at a given coupling strength.

    ``coupling`` controls how strongly type and attribute are related (0 = unrelated,
    1 = type fully determined by the attribute): the score is
    ``score = c·z + sqrt(1-c²)·eps``, which is then discretised into K types by
    quantile.

    * ``spatial_attr=False``: attributes are i.i.d., so types are spatially random
      and H0 (type ⊥ location, attribute ⊥ location | type) **holds**; the type-I
      error rate can be measured;
    * ``spatial_attr=True`` : attributes are spatially smooth, so types cluster in
      space, H0 does not hold, and power can be measured.
    """
    coords = _uniform(rng, n, side)
    eps = rng.normal(0.0, 1.0, n)
    if spatial_attr:
        z = _smooth_attribute(coords, rng, side)
        z = (z - z.mean()) / (z.std() + 1e-12)
    else:
        z = rng.normal(0.0, 1.0, n)
    c = float(np.clip(coupling, 0.0, 1.0))
    score = c * z + math.sqrt(max(0.0, 1.0 - c * c)) * eps
    qs = np.quantile(score, np.linspace(0.0, 1.0, K + 1)[1:-1])
    types = (np.digitize(score, qs) + 1).astype(np.int64)
    return coords, types, z[:, None]


def gen_period_case(n, K, rng, n_periods=4, side=1000.0):
    """Multi-period case: the periods differ in type composition, yet given a period
    type and location are independent."""
    coords = _uniform(rng, n, side)
    periods = rng.integers(1, n_periods + 1, size=n)
    types = np.empty(n, dtype=np.int64)
    for p in range(1, n_periods + 1):
        idx = np.where(periods == p)[0]
        probs = np.linspace(0.2, 0.8, K)
        probs = probs / probs.sum()
        types[idx] = rng.choice(np.arange(1, K + 1), size=idx.size, p=probs)
    return coords, types, periods


# ---------------------------------------------------------------------------
# Single evaluation
# ---------------------------------------------------------------------------
def reject(model, coords, types, bw, null, n_perm, seed,
           attributes=None, time_periods=None, pair=TARGET_PAIR):
    g = model.global_clq(coords, types, bw, attributes=attributes,
                         time_periods=time_periods)["gclq"]
    res = model.permutation_test(coords, types, g, bw, attributes=attributes,
                                 time_periods=time_periods, level="global",
                                 null=null, n_perm=n_perm, seed=seed,
                                 correction=False)
    a, b = pair
    return bool(res["p_attract"][a - 1, b - 1] < ALPHA)


# ---------------------------------------------------------------------------
# Scenarios A / B / C
# ---------------------------------------------------------------------------
def scenario_A(reps, n_perm, n, K, seed0=100, bw=100.0):
    model = STCLQ(distance_metric="projected", diagnostics=False)
    h0 = sum(reject(model, *gen_null(n, K, np.random.default_rng(seed0 + i)), bw,
                    "type", n_perm, seed0 + i) for i in range(reps))
    h1 = sum(reject(model, *gen_attract(n, K, np.random.default_rng(10000 + i)), bw,
                    "type", n_perm, 10000 + i) for i in range(reps))
    return {"type1_error": h0 / reps, "power": h1 / reps}


def scenario_B(reps, n_perm, n, K=2, coupling=0.5, attr_bw=0.3, seed0=200, bw=100.0):
    """Three null hypotheses under attribute weighting: type-I error rate on H0
    (i.i.d. attributes) plus power on H1 (spatially smooth attributes)."""
    specs = [AttributeSpec("z", AttributeType.CONTINUOUS, bandwidth=attr_bw)]
    model = STCLQ(attribute_specs=specs, use_attribute_weights=True,
                  distance_metric="projected", diagnostics=False)
    h0 = {k: 0 for k in NULLS}
    h1 = {k: 0 for k in NULLS}
    for i in range(reps):
        coords, types, attrs = gen_coupled(n, K, coupling, np.random.default_rng(seed0 + i))
        for k in NULLS:
            h0[k] += reject(model, coords, types, bw, k, n_perm, seed0 + i,
                            attributes=attrs, pair=WITHIN_PAIR)
        coords, types, attrs = gen_coupled(n, K, coupling, np.random.default_rng(5000 + i),
                                           spatial_attr=True)
        for k in NULLS:
            h1[k] += reject(model, coords, types, bw, k, n_perm, 5000 + i,
                            attributes=attrs, pair=WITHIN_PAIR)
    out = {f"type1_error[{k}]": h0[k] / reps for k in NULLS}
    out.update({f"power[{k}]": h1[k] / reps for k in NULLS})
    return out


def scenario_C(reps, n_perm, n, K, seed0=300, bw=100.0, n_periods=4):
    model = STCLQ(distance_metric="projected", diagnostics=False)
    err_global = err_block = 0
    for i in range(reps):
        coords, types, periods = gen_period_case(n, K, np.random.default_rng(seed0 + i),
                                                 n_periods)
        err_global += reject(model, coords, types, bw, "type", n_perm, seed0 + i,
                             time_periods=periods)
        err_block += reject(model, coords, types, bw, "type_period", n_perm,
                            seed0 + i, time_periods=periods)
    return {"type1_error_global": err_global / reps,
            "type1_error_block": err_block / reps}


# ---------------------------------------------------------------------------
# Scenario D: multi-construction sweep
# ---------------------------------------------------------------------------
def sweep_coupling(reps, n_perm, n, Ks, couplings, attr_bws, bw=100.0,
                   seed0=700, spatial_attr=False):
    """Sweep the rejection rate of the three null hypotheses over the
    (K × coupling × attr_bw) grid."""
    rows = []
    for K in Ks:
        for coupling in couplings:
            for abw in attr_bws:
                specs = [AttributeSpec("z", AttributeType.CONTINUOUS, bandwidth=abw)]
                model = STCLQ(attribute_specs=specs, use_attribute_weights=True,
                              distance_metric="projected", diagnostics=False)
                cnt = {k: 0 for k in NULLS}
                for i in range(reps):
                    coords, types, attrs = gen_coupled(
                        n, K, coupling, np.random.default_rng(seed0 + i),
                        spatial_attr=spatial_attr)
                    for k in NULLS:
                        cnt[k] += reject(model, coords, types, bw, k, n_perm,
                                         seed0 + i, attributes=attrs, pair=WITHIN_PAIR)
                for k in NULLS:
                    rows.append({"K": K, "coupling": coupling, "attr_bw": abw,
                                 "spatial_attr": spatial_attr, "null": k,
                                 "reject_rate": cnt[k] / reps})
    return rows


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n-perm", type=int, default=199)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--bw", type=float, default=100.0)
    ap.add_argument("--scenarios", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--Ks", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--couplings", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--attr-bws", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    ap.add_argument("--sweep-reps", type=int, default=50)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    results = {"settings": vars(args), "alpha": ALPHA,
               "target_pair": TARGET_PAIR, "within_pair": WITHIN_PAIR}

    if "A" in args.scenarios:
        results["A_spatial"] = scenario_A(args.reps, args.n_perm, args.n, args.K, bw=args.bw)
        print("A spatial      :", results["A_spatial"])
    if "B" in args.scenarios:
        results["B_attribute"] = scenario_B(args.reps, args.n_perm, args.n, bw=args.bw)
        print("B attribute    :", results["B_attribute"])
    if "C" in args.scenarios:
        results["C_multiperiod"] = scenario_C(args.reps, args.n_perm, args.n, args.K, bw=args.bw)
        print("C multi-period :", results["C_multiperiod"])
    if "D" in args.scenarios:
        print("D multi-construction sweep: running ...")
        sweep = (sweep_coupling(args.sweep_reps, args.n_perm, args.n, args.Ks,
                                args.couplings, args.attr_bws, bw=args.bw,
                                spatial_attr=False)
                 + sweep_coupling(args.sweep_reps, args.n_perm, args.n, args.Ks,
                                  args.couplings, args.attr_bws, bw=args.bw,
                                  spatial_attr=True))
        results["D_sweep"] = sweep
        _write_sweep(sweep)
        _print_sweep(sweep)

    with open(OUT / "calibration.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    with open(OUT / "calibration.csv", "w", encoding="utf-8") as fh:
        fh.write("scenario,metric,value\n")
        for scen, block in results.items():
            if not isinstance(block, dict) or scen == "settings":
                continue
            for k, v in block.items():
                fh.write(f"{scen},{k},{v}\n")

    _maybe_figure(results, args)
    print(f"\nresults -> {OUT}")


def _write_sweep(sweep):
    path = OUT / "sweep_coupling.csv"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("K,coupling,attr_bw,spatial_attr,null,reject_rate\n")
        for r in sweep:
            fh.write(f"{r['K']},{r['coupling']},{r['attr_bw']},"
                     f"{r['spatial_attr']},{r['null']},{r['reject_rate']}\n")


def _print_sweep(sweep):
    print(f"\n{'spatial_attr':>13} {'K':>2} {'coupl':>6} {'attr_bw':>7} "
          f"{'type':>7} {'attribute':>10} {'within_type':>12}")
    keys = ("K", "coupling", "attr_bw", "spatial_attr")
    seen = {}
    for r in sweep:
        seen.setdefault(tuple(r[k] for k in keys), {})[r["null"]] = r["reject_rate"]
    for key, d in seen.items():
        sa, K, cp, ab = key
        print(f"{str(sa):>13} {K:>2} {cp:>6.2f} {ab:>7.2f} "
              f"{d.get('type', float('nan')):>7.3f} "
              f"{d.get('attribute', float('nan')):>10.3f} "
              f"{d.get('attribute_within_type', float('nan')):>12.3f}")


def _maybe_figure(results, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"(skip figure: {exc})")
        return
    if "D_sweep" not in results:
        return
    sweep = results["D_sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, sa, title in zip(axes, (False, True),
                             ("H0 (i.i.d. attributes): type-I error",
                              "H1 (spatial attributes): power")):
        for null, style in zip(NULLS, ("o-", "s--", "^:")):
            xs, ys = [], []
            for K in args.Ks:
                for ab in args.attr_bws:
                    for cp in args.couplings:
                        for r in sweep:
                            if (r["K"] == K and abs(r["attr_bw"] - ab) < 1e-9
                                    and abs(r["coupling"] - cp) < 1e-9
                                    and r["spatial_attr"] == sa and r["null"] == null):
                                xs.append(cp)
                                ys.append(r["reject_rate"])
            if xs:
                order = np.argsort(xs)
                ax.plot(np.array(xs)[order], np.array(ys)[order], style, alpha=0.5,
                        label=null)
        ax.axhline(ALPHA, color="r", ls="--")
        ax.set_xlabel("coupling"); ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7); ax.grid(True, ls=":")
    fig.tight_layout()
    fig.savefig(OUT / "fig_sweep_coupling.png", dpi=150)
    print(f"figure -> {OUT / 'fig_sweep_coupling.png'}")


if __name__ == "__main__":
    main()
