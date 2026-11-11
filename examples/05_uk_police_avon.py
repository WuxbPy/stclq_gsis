"""Example 05: run STCLQ with a permutation test on the bundled UK Police extract.

Source: UK Home Office, data.police.uk (OGL v3.0), see ``data/README.md``.
By default this takes month 1 of Avon and Somerset (2023-01), about 10,000 events.

    python examples/05_uk_police_avon.py
    python examples/05_uk_police_avon.py --dataset city_of_london --period 3
    python examples/05_uk_police_avon.py --bandwidth 100 --n-perm 499

Note: Home Office releases coordinates anonymised to "on or near", so many events share
one point and a bandwidth of a few tens of metres already pulls in a large neighbourhood
(the output reports rho and the mean neighbour count).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from stclq import STCLQ, load_csv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATASETS = {
    "avon_and_somerset": DATA / "avon_and_somerset_2023.csv",
    "city_of_london": DATA / "city_of_london_2023.csv",
}


def type_names() -> dict[int, str]:
    with open(DATA / "type_mapping.csv", newline="", encoding="utf-8") as fh:
        return {int(r["type_id"]): r["type_name"] for r in csv.DictReader(fh)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="avon_and_somerset")
    ap.add_argument("--period", type=int, default=1, help="month index 1-12 (default: 1)")
    ap.add_argument("--bandwidth", type=float, default=50.0, help="spatial bandwidth, metres")
    ap.add_argument("--n-perm", type=int, default=199)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    names = type_names()
    coords, types, periods, _ = load_csv(DATASETS[args.dataset], period=[args.period])
    print(f"{args.dataset}, period {args.period}: N={len(coords):,}, "
          f"unique coordinates={len(np.unique(coords, axis=0)):,}")

    model = STCLQ()
    out = model.run(coords, types, bandwidth=args.bandwidth, time_periods=periods,
                    level="global", null="type", n_perm=args.n_perm, seed=args.seed)

    print(f"bandwidth={out['bandwidth']:.0f} m, rho={out['rho']:.4f}")
    g = out["gclq"]
    p = out["global_test"]["p_attract_fdr"]
    labels = [names.get(i + 1, f"Type_{i + 1}") for i in range(g.shape[0])]

    header = " " * 15 + "".join(f"{lab:>14s}" for lab in labels)
    print("\nSTCLQ (row = origin type):")
    print(header)
    for i, lab in enumerate(labels):
        cells = "".join(f"{g[i, j]:14.3f}" for j in range(g.shape[1]))
        print(f"{lab:>14s}{cells}")

    print("\nFDR-adjusted attraction p-values:")
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i != j and p[i, j] < 0.05:
                print(f"  {a} -> {b}: STCLQ={g[i, j]:.3f}, p={p[i, j]:.4f}, "
                      f"{model.classify(g[i, j], p[i, j])['category']}")


if __name__ == "__main__":
    main()
