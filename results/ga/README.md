# Experiment drivers

Everything in the paper's Results section, namely Figs. 1–7, Tables 1–3 and S1–S8,
is produced from this directory. The experiments are purely spatio-temporal (`w_s · w_t`);
the attribute dimension (`w_a`) is outside the scope of this paper and is not exercised
here.

Run from the repository root so that `ROOT` resolves (`results/ga/<script>.py` relies on
`HERE.parents[1]`):

```bash
PYTHONPATH=src python results/ga/exp_realdata.py
```

## Drivers

| Script | Paper | Output |
|---|---|---|
| `_ref.py` | §6.1 reference | the dense *O*(*N*²) implementation used as the correctness and speed baseline |
| `exp_correctness.py` | §6.1, Table S1 | `correctness/correctness.{json,csv}` |
| `exp_calibration.py` | §6.2, Fig. 3a, Table S2 | `calibration/calibration*.{json,csv}` |
| `exp_power.py` | §6.2, Fig. 3b, Table S5 | `calibration/calibration_power.{json,csv}` |
| `exp_benchmark.py` | §6.3–6.4, Figs. 4–6, Table 3 | `benchmark/benchmark.{json,csv}` |
| `exp_parallel.py` | §6.5, Fig. 7 | `parallel/parallel.json` |
| `exp_regime.py` | §6.5, Table S6 | `regime/regime.json` |
| `exp_realdata.py` | §6.6, Fig. 1b, Table S3 | `realdata/*` |
| `exp_stability.py` | §6.6, Table S8 | `realdata/*_period_stability.json` |
| `calibration_simulation.py` | simulation core imported by the calibration driver | n/a |
| `benchmark_scaling.py` | standalone scaling probe kept for reference | n/a |
| `make_tables.py` | Tables 1–3, S1–S8 | `../manuscript/tables/*` |
| `make_figures.py` | Figs. 1–7 | `figures/*.pdf`, `figures/*.png` |
| `render_manuscript.py` | full text | injects the numbers into `manuscript/GA_manuscript.md` from `GA_manuscript.template.md`; backs up to `_render_backup/` before overwriting |
| `make_refs.py` | reference list | parses `manuscript/references.bib`, emits the GA-style entry list |
| `verify_refs.py` | pre-submission check | Crossref metadata plus a citation-graph audit (dangling keys, zombie entries, author–year collisions) |
| `qa_manuscript.py` | pre-submission check | unfilled tokens, stray numbers, duplicated paragraphs, abstract length, figure-source existence |

`make_tables.py` and `make_figures.py` read only the JSON/CSV below; they never rerun a
model, so a figure can be restyled without repeating the computation.

## Benchmark conventions

The bandwidth is fixed at k̄ ≈ 150 (bisection) so that the scaling curves separate the
algorithmic gain from the growth of the data. Each *N* is measured in its own
subprocess, so the reported peak RSS (`ru_maxrss`) carries no residue from the previous
*N*. The reference *O*(*N*²) implementation runs only up to *N* ≤ 3 × 10⁴; beyond that it
is extrapolated on a log–log fit (Fig. 3), and the fits are stored in
`benchmark/benchmark.json` under `fits`:

| Quantity | Expected slope |
|---|---|
| sparse single permutation, end to end | ≈ 1 (`O(nnz)`) |
| sparse SpMM alone | ≈ 1 |
| reference implementation | ≈ 2 |

Timing is split into `spmm_s`, `permgen_s` and `reduce_s`, so the claim "the cost of a
permutation is the SpMM, the reduction is rounding error" is measurable rather than
asserted. The reduction deliberately avoids BLAS (`bincount` segmented sums): a forked
worker would otherwise open its own OpenBLAS pool.

`exp_benchmark.py` and `exp_parallel.py` pin `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` /
`MKL_NUM_THREADS` / `NUMEXPR_NUM_THREADS` to 1 *before* importing NumPy. Without that
guard, `processes × threads` oversubscription contaminates every parallel measurement.

## Long runs and resumption

`exp_benchmark.py` and `exp_power.py` write incrementally, per *N* and per grid cell,
and resume from whatever is on disk, so an interrupted run loses at most one point;
`exp_benchmark.py --no-resume` forces a clean pass.

## Determinism

All drivers take an explicit seed. The permutation plan (one seed per permutation,
drawn from a parent generator) is stored with the results, so any reported *p*-value
can be recomputed exactly, and the plan does not depend on the number of workers:
serial and parallel runs agree bit for bit.

`provenance.json` freezes the machine the reported numbers were produced on. If you
rerun the experiments on different hardware, update it; otherwise a later re-render
will attribute the results to the wrong machine.
