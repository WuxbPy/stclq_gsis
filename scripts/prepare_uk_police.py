#!/usr/bin/env python3
"""Rebuild the released UK Police extracts from the raw Home Office archives.

The raw files are the monthly ``YYYY-MM.zip`` archives published at
https://data.police.uk/data/archive/.  Unzip a few months and point this script at the
directory that holds the ``*-street.csv`` files (sub-directories are searched):

    python scripts/prepare_uk_police.py --raw-dir raw/ --out-dir data
    python scripts/prepare_uk_police.py --raw-dir raw/ --verify data

The script keeps the four offence classes used in the paper, snaps nothing (coordinates
are already anonymised by the publisher), and drops the identifiers listed in
``data/README.md``.  ``--verify`` rebuilds in memory and compares against the shipped
files as a multiset of rows, so it can be run without touching the release.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

# Raw `Crime type` -> (type_id, type_name).  This is the mapping documented in
# data/type_mapping.csv; the four classes are the ones with enough events in each
# monthly slice to support a 4 x 4 quotient matrix at the monthly scale.
TYPE_MAP = {
    "Burglary": (1, "Burglary"),
    "Robbery": (2, "Robbery"),
    "Other theft": (3, "Theft"),
    "Shoplifting": (3, "Theft"),
    "Theft from the person": (3, "Theft"),
    "Bicycle theft": (4, "Vehicle Crime"),
    "Vehicle crime": (4, "Vehicle Crime"),
}

# Force name as it appears in the `Reported by` / `Falls within` columns -> output file.
FORCES = {
    "Avon and Somerset Constabulary": "avon_and_somerset_2023.csv",
    "City of London Police": "city_of_london_2023.csv",
}

HEADER = ["longitude", "latitude", "type_id", "type_name", "period_id", "month"]


def _period(month: str) -> int:
    """``2023-07`` -> 7.  Guards against a stray year sneaking into the archive."""
    year, mon = month.split("-")[:2]
    if year != "2023":
        raise ValueError(f"unexpected month outside 2023: {month!r}")
    return int(mon)


def build(raw_dir: Path) -> dict[str, list[list[str]]]:
    """Collect rows per force from every ``*-street.csv`` under ``raw_dir``."""
    files = sorted(raw_dir.rglob("*-street.csv"))
    if not files:
        sys.exit(f"no *-street.csv found under {raw_dir}")

    out: dict[str, list[list[str]]] = {name: [] for name in FORCES.values()}
    for path in files:
        # Each monthly file holds every force; the force is identified per row.
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            force_col = "Reported by" if "Reported by" in (reader.fieldnames or []) else "Falls within"
            for r in reader:
                target = FORCES.get((r.get(force_col) or "").strip())
                if target is None:
                    continue
                raw_type = (r.get("Crime type") or "").strip()
                if raw_type not in TYPE_MAP:
                    continue
                try:
                    lon, lat = float(r["Longitude"]), float(r["Latitude"])
                except (TypeError, ValueError, KeyError):
                    continue
                tid, tname = TYPE_MAP[raw_type]
                month = (r.get("Month") or "").strip()
                out[target].append([f"{lon:.6f}", f"{lat:.6f}", str(tid), tname,
                                    str(_period(month)), f"{month}-01"])
    return out


def write(rows: dict[str, list[list[str]]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, body in rows.items():
        # Row order follows the archive order (month, then the publisher's order);
        # nothing downstream depends on it, and preserving it keeps the rebuild
        # comparable line by line with a fresh download.
        with open(out_dir / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(HEADER)
            w.writerows(body)
        print(f"{name}: {len(body):,} rows -> {out_dir / name}")
    write_type_mapping(out_dir / "type_mapping.csv")


def write_type_mapping(path: Path) -> None:
    by_class: dict[tuple[int, str], list[str]] = {}
    for raw, key in TYPE_MAP.items():
        by_class.setdefault(key, []).append(raw)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["type_id", "type_name", "raw_crime_types"])
        for (tid, tname), raws in sorted(by_class.items()):
            w.writerow([tid, tname, ";".join(sorted(raws))])
    print(f"type_mapping.csv -> {path}")


def verify(rows: dict[str, list[list[str]]], data_dir: Path) -> int:
    """Compare the rebuild against the shipped files as multisets of rows."""
    bad = 0
    for name, body in rows.items():
        target = data_dir / name
        if not target.exists():
            print(f"[MISS] {target}")
            bad += 1
            continue
        with open(target, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != HEADER:
                print(f"[FAIL] {name}: header {reader.fieldnames} != {HEADER}")
                bad += 1
                continue
            shipped = Counter(tuple(r[h] for h in HEADER) for r in reader)
        rebuilt = Counter(tuple(r) for r in body)
        if shipped == rebuilt:
            print(f"[ OK ] {name}: {sum(shipped.values()):,} rows identical")
            continue
        bad += 1
        only_ship = shipped - rebuilt
        only_build = rebuilt - shipped
        print(f"[FAIL] {name}: shipped {sum(shipped.values()):,}, rebuilt {sum(rebuilt.values()):,}, "
              f"{sum(only_ship.values())} only in shipped, {sum(only_build.values())} only in rebuild")
        for r, n in list(only_ship.items())[:3]:
            print(f"        shipped only: {r} x{n}")
        for r, n in list(only_build.items())[:3]:
            print(f"        rebuild only: {r} x{n}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True, type=Path,
                    help="directory containing the unzipped *-street.csv extracts")
    ap.add_argument("--out-dir", type=Path, default=Path("data"),
                    help="where to write the released files (default: data/)")
    ap.add_argument("--verify", nargs="?", const="data", default=None, metavar="DATA_DIR",
                    help="compare against the shipped files instead of writing")
    args = ap.parse_args()

    rows = build(args.raw_dir)
    if args.verify is not None:
        sys.exit(1 if verify(rows, Path(args.verify)) else 0)
    write(rows, args.out_dir)


if __name__ == "__main__":
    main()
