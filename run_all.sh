#!/usr/bin/env bash
# Reproduce every number, table and figure of the paper from this checkout.
#
#   ./run_all.sh              # full suite (multi-hour: exp_benchmark, exp_power)
#   PYTHON=python3 ./run_all.sh
#
# The drivers checkpoint after every N / grid cell; rerunning resumes.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"
PY="${PYTHON:-python}"

for s in exp_correctness exp_benchmark exp_calibration exp_power \
         exp_parallel exp_regime exp_realdata exp_stability; do
    echo "== $s"
    "$PY" "results/ga/$s.py"
done

echo "== make_tables"
"$PY" results/ga/make_tables.py
echo "== make_figures"
"$PY" results/ga/make_figures.py

echo "done: results/ga/ and manuscript/tables/ are up to date"
