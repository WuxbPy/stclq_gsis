# Data

Two preprocessed extracts of the UK Home Office open police-recorded crime data are
released here. They are the real-data inputs of §6.6 (Table S3, Table S8, Fig. 1b).

| File | Force | Events | Offence categories | Monthly periods |
|---|---|---|---|---|
| `avon_and_somerset_2023.csv` | Avon and Somerset Constabulary | 139,001 | 4 | 12 (2023-01 … 2023-12) |
| `city_of_london_2023.csv` | City of London Police | 15,776 | 4 | 12 (2023-01 … 2023-12) |
| `type_mapping.csv` | n/a | n/a | 4 | n/a |

## Source and licence

Source: UK Home Office, *Police recorded crime data*, <https://data.police.uk>, monthly
street-level extracts for 2023, published under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
The licence permits redistribution and reuse, including commercially, provided the
source is attributed. Attribution: "Contains public sector information licensed under
the Open Government Licence v3.0. Source: UK Home Office, data.police.uk."

The raw monthly archives were downloaded from <https://data.police.uk/data/archive/>
(one `YYYY-MM.zip` per month, each containing one `*-street.csv` per force). They are
not deposited here: they are large, they are re-downloadable, and they carry record
identifiers that this release does not need.

## Fields

| Column | Type | Meaning |
|---|---|---|
| `longitude` | float, 6 d.p. | WGS84 longitude as published by the Home Office |
| `latitude` | float, 6 d.p. | WGS84 latitude as published by the Home Office |
| `type_id` | int | offence class, 1–4 (see `type_mapping.csv`) |
| `type_name` | str | offence class label |
| `period_id` | int | month index, 1 = 2023-01 … 12 = 2023-12 |
| `month` | str | `YYYY-MM-01`, the month the offence was recorded |

`type_mapping.csv` maps each `type_id` to the raw `Crime type` strings it aggregates:

| `type_id` | `type_name` | raw `Crime type` values |
|---|---|---|
| 1 | Burglary | Burglary |
| 2 | Robbery | Robbery |
| 3 | Theft | Other theft; Shoplifting; Theft from the person |
| 4 | Vehicle Crime | Bicycle theft; Vehicle crime |

## Coordinate precision and duplicates

The Home Office does not publish exact offence locations. Each record is snapped to an
anonymised point, typically the centre of a street segment or an "on or near" locality,
so many records share a coordinate. In the Avon and Somerset extract 139,001 events
collapse to **17,860 unique coordinates** (87.2 % duplication; the most frequent single
point carries 1,046 events). In the City of London extract 15,776 events collapse to
**391 unique coordinates**.

This is not a defect to be cleaned away: it is a property of the source, it is why a
bandwidth of a few tens of metres already captures a large neighbourhood
(k̄ ≈ 137 at a 31 m bandwidth), and it is why the coincidence term
k_s(0) = 1 for same-point pairs. The manuscript treats same-point pairs explicitly
(§5 and §6.6); do not jitter the coordinates if you want to match the reported numbers.

A further 120 Avon and Somerset records (0.09 %) fall outside the force area after the
police-force boundary is applied; those are retained in the CSV and excluded at the
plotting stage, exactly as in `results/ga/make_figures.py`.

## What was removed before release

The development extract carried the full set of Home Office columns. The following were
dropped because no experiment in this repository reads them and they add no analytical
value:

- `Crime ID`: a persistent per-record identifier. It is a stable key that would allow
  linkage across releases, so it is not redistributed.
- `LSOA code`, `LSOA name`, `Location`, `Context`, `Last outcome category`,
  `Reported by`, `Falls within`, `Region`: administrative and descriptive fields.
- `Crime type` (the raw label; preserved indirectly through `type_mapping.csv`),
  `severity`, `time_of_day`, `area_type`: the three attribute columns serve the
  attribute-weighted variant of the quotient, which lies outside the scope of this
  paper, so they are not part of this release.

Row counts, the per-period counts and the type mapping are unchanged with respect to the
development extract: only columns were dropped, no row was added, removed or reordered.

## Rebuilding from the raw archives

`scripts/prepare_uk_police.py` re-derives both files from a directory of raw
`*-street.csv` extracts:

```bash
python scripts/prepare_uk_police.py --raw-dir /path/to/street_csvs --out-dir data
python scripts/prepare_uk_police.py --raw-dir /path/to/street_csvs --verify data
```

`--verify` recomputes the release and compares it field by field against the shipped
files, so the recipe can be checked without overwriting anything.
