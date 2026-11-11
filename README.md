# stclq — scalable permutation inference for spatio-temporal co-location quotients

Reference implementation and reproducibility artifact for

> Wu, X., and Hu, W. *Two null hypotheses for one statistic: structure-invariant
> permutation inference for spatio-temporal co-location quotients at city scale.*
> Manuscript under review at **Geo-Spatial Information Science**.

Repository: <https://github.com/WuxbPy/stclq_gsis>

The package implements the spatio-temporal co-location quotient (STCLQ) and its
global/local variants together with four permutation nulls. The permutation engine is
sparse: the CSR structure of the spatial neighbourhood is built once and reused, so a
single permutation costs one sparse matrix–matrix product instead of a re-traversal of
all *N*² ordered pairs. On the two UK Police extracts shipped here (139,001 events) that
turns a test that is out of reach into a 43-second serial run.

---

## What is in this repository

| Path | Contents |
|---|---|
| `src/stclq/` | the package: engine, neighbourhood construction, bandwidth selection, permutation inference, parallel driver, diagnostics |
| `tests/` | unit tests (`pytest`, 15 tests) |
| `examples/` | five runnable examples, from synthetic data to the shipped UK Police extract |
| `data/` | the two preprocessed UK Police extracts used in §6.6, plus the type mapping (see `data/README.md`) |
| `results/ga/` | the experiment drivers, the machine-readable results they produced, and the seven figures of the paper |
| `manuscript/` | manuscript sources, reference list, Tables 1–3 and S1–S8, and the submitted PDF |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # numpy + scipy only
pip install -e ".[full]"         # + pandas, matplotlib (needed for the experiments)
```

Python ≥ 3.9. No compiled extensions are required.

## Quick start

```python
import numpy as np
from stclq import STCLQ, load_csv

coords, types, periods, _ = load_csv("data/avon_and_somerset_2023.csv")
model = STCLQ()
out = model.run(coords, types, bandwidth=50.0, time_periods=periods,
                level="global", null="type", n_perm=199, seed=42)
print(np.round(out["gclq"], 3))
print(out["global_test"]["p_attract_fdr"])
```

`bandwidth` is in metres. 50 m is a few times the snapping scale of the published
coordinates; the paper fixes it by bisection on a target capture fraction instead
(31.1 m for ρ ≈ 0.001, see `bandwidth_for_rho` in `results/ga/exp_realdata.py`), which is
why its neighbourhoods are large (k̄ ≈ 137).

`examples/01_quickstart.py` … `examples/05_uk_police_avon.py` run standalone:

```bash
python examples/01_quickstart.py
python examples/05_uk_police_avon.py --dataset avon_and_somerset --period 1 --n-perm 199
```

## Reproducing the paper

Everything below the line in the manuscript is produced from this tree:

```bash
./run_all.sh          # the same commands, wrapped, with PYTHONPATH set
```

or, step by step:

```bash
for s in exp_correctness exp_benchmark exp_calibration exp_power \
         exp_parallel exp_regime exp_realdata exp_stability; do
    PYTHONPATH=src python results/ga/$s.py
done
PYTHONPATH=src python results/ga/make_tables.py    # Tables 1-3, S1-S8  -> manuscript/tables/
PYTHONPATH=src python results/ga/make_figures.py   # Figs. 1-7          -> results/ga/figures/
```

The manuscript's own command block lists the first six drivers; `exp_regime.py`
produces `results/ga/regime/regime.json`, from which Table S6 is built, and is included
here for completeness.

| Driver | Paper section | Output |
|---|---|---|
| `exp_correctness.py` | §6.1, Table S1 | `correctness/correctness.{json,csv}` |
| `exp_calibration.py` | §6.2, Fig. 3a, Table S2 | `calibration/calibration*.{json,csv}` |
| `exp_power.py` | §6.2, Fig. 3b, Table S5 | `calibration/calibration_power.{json,csv}` |
| `exp_benchmark.py` | §6.3–6.4, Figs. 4–6, Table 3 | `benchmark/benchmark.{json,csv}` |
| `exp_parallel.py` | §6.5, Fig. 7 | `parallel/parallel.json` |
| `exp_regime.py` | §6.5, Table S6 | `regime/regime.json` |
| `exp_realdata.py` | §6.6, Fig. 1b, Table S3 | `realdata/*` |
| `exp_stability.py` | §6.6, Table S8 | `realdata/*_period_stability.json` |

`make_tables.py` and `make_figures.py` only read the JSON/CSV above; they never rerun a
model, so a figure can be restyled without repeating the computation.

**Runtime.** `exp_correctness`, `exp_calibration`, `exp_realdata` and `exp_stability`
finish in minutes. `exp_benchmark` (up to 10⁶ points, reference *O*(*N*²) baseline) and
`exp_power` are multi-hour runs; both checkpoint after every *N* / grid cell and resume
from what is on disk (`--no-resume` forces a clean pass). The environment the reported
numbers were produced on is frozen in `results/ga/provenance.json`.

**Randomness.** Every driver takes an explicit seed, and the permutation plan — one
seed per permutation drawn from a parent generator — is stored with the results.
A reported *p*-value can therefore be recomputed exactly, and the plan is independent of
the number of worker processes, so serial and parallel runs agree bit for bit.

**Thread oversubscription.** The benchmark and parallel drivers pin `OMP_NUM_THREADS`
and friends to 1 *before* importing NumPy; without this, each forked worker opens its
own OpenBLAS pool and the speed-up curves measure throttling instead of scaling.

## Data

Two preprocessed extracts of the UK Home Office open police data (2023) are shipped:
Avon and Somerset Constabulary (139,001 events, 4 offence categories, 12 monthly
periods) and City of London Police (15,776 events, the same categories and periods).
Both are released under the Open Government Licence v3.0; see `data/README.md` for the
field list, the provenance, and the columns dropped before release.

## Licence

Code: MIT (`LICENSE`). Data: OGL v3.0 (`data/README.md`). The manuscript text is not
covered by either.

## Citation

`CITATION.cff` is provided. Until the article is published, please cite the repository
(<https://github.com/WuxbPy/stclq_gsis>) and the manuscript named at the top of this
file. No DOI has been minted: the repository itself is the single location named in the
`Data availability` and `Code availability` statements of the manuscript, and a Zenodo
snapshot can be added at the revision stage if the editor asks for a persistent
identifier.

## Authors

Xibo Wu (wuxibo@gz.gov.cn, ORCID 0009-0003-7027-6678), Guangzhou Academy of Social
Sciences, Guangzhou, China.
Weiping Hu (huweiping@m.scnu.edu.cn, corresponding author, ORCID 0009-0005-7065-2746),
School of Geography, South China Normal University, Guangzhou, China.

The work received no specific funding.
