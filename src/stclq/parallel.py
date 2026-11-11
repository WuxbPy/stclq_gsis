"""Parallel permutation engine (contribution 4 of the paper).

Design notes
------------
1. **Parallelism over permutations**: the ``n_perm`` permutations are split into
   blocks, one per child process. A child is created by ``fork`` and **inherits the
   read-only CSR cache already built by the parent** (``indptr`` / ``indices`` /
   ``data``) together with the design-matrix skeleton, so a neighbourhood table that
   easily runs to GBs never has to be serialised. This is exactly where a generic
   parallel framework (which must pickle large objects) cannot simply step in.
2. **Row-private accumulators**: every row of the ``SpMM`` output depends only on
   that row, and rows never write to the same location. The object reduced across
   processes is a **small accumulator** of size ``O(NK)``, not the full ``O(NK²)``
   count tensor.
3. **Reproducible numerics**: the permutation sequence is fixed entirely by the
   per-permutation seed ``plan`` derived from the parent RNG, and children consume no
   global random state. **Serial and parallel runs therefore agree bit for bit at
   any process count** (accumulator entries are integers, so grouped floating-point
   summation is exact).
4. **Zero-copy fallback**: when ``n_jobs <= 1``, ``len(plan) <= 1`` or ``fork`` is
   unavailable, the call falls back to serial with unchanged results.

Platform note: this relies on POSIX ``fork`` (Linux/macOS); on Windows it falls back
to serial.
"""

from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

from .inference import global_sim, local_counts

# Read-only context inherited by forked children (avoids pickling)
_STATE: dict = {}


def resolve_n_jobs(n_jobs) -> int:
    """Normalise ``n_jobs`` to a positive integer (``-1``/``None`` means all cores)."""
    if n_jobs is None or n_jobs == 0:
        return 1
    if n_jobs < 0:
        return os.cpu_count() or 1
    return max(1, int(n_jobs))


def _fork_context():
    """Return an available ``fork`` context, or ``None`` (fall back to serial)."""
    try:
        return mp.get_context("fork")
    except (ValueError, RuntimeError):  # pragma: no cover - e.g. Windows
        return None


def _split(plan: np.ndarray, n_chunks: int):
    """Split the permutation plan into contiguous blocks.

    The sparse work per permutation is **constant** (``nnz`` is fixed and so is the
    design-matrix structure), so oversampling plus dynamic scheduling is not needed
    to balance the load: setting the block count to the number of processes is
    enough, which cuts cross-process round trips and keeps small blocks from being
    swamped by fork/IPC overhead.
    """
    n = len(plan)
    n_chunks = max(1, min(int(n_chunks), n))
    edges = np.linspace(0, n, n_chunks + 1).astype(int)
    return [plan[edges[i]:edges[i + 1]] for i in range(n_chunks) if edges[i + 1] > edges[i]]


#: Total byte budget for child accumulators in the local layer (each block returns
#: two ``(N,K,K)`` arrays, ``ge`` and ``le``)
LOCAL_ACCUMULATOR_BUDGET = 512 << 20


def _local_n_chunks(engine, n_jobs: int) -> int:
    """Block count for the local layer: bounded by the memory budget for the returned
    accumulators rather than by the number of processes alone.
    """
    per_chunk = 2 * int(engine.n) * int(engine.K) ** 2 * 8
    if per_chunk <= 0:
        return max(1, int(n_jobs))
    return max(1, min(int(n_jobs), int(LOCAL_ACCUMULATOR_BUDGET // per_chunk)))


# ---------------------------------------------------------------------------
# Child side: read-only ``_STATE``, no arguments passed in
# ---------------------------------------------------------------------------
def _worker_global(chunk):
    engine, null = _STATE["engine"], _STATE["null"]
    K = engine.K
    out = np.empty((len(chunk), K, K), dtype=np.float64)
    for i, s in enumerate(chunk):
        out[i] = global_sim(engine, s, null)
    return out


def _worker_local(chunk):
    """Child side: block-private accumulator (``(N, K, K)``, integer-valued),
    row-private with no shared writes."""
    engine, null = _STATE["engine"], _STATE["null"]
    observed = _STATE["observed"]
    n, K = engine.n, engine.K
    ge = np.zeros((n, K, K), dtype=np.float64)
    le = np.zeros((n, K, K), dtype=np.float64)
    local_counts(engine, chunk, null, observed, ge, le)
    return ge, le


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------
def parallel_global_sims(engine, plan, null, n_jobs) -> np.ndarray:
    """Compute the global permutation simulation matrix ``(n_perm, K, K)`` in parallel."""
    n_jobs = resolve_n_jobs(n_jobs)
    ctx = _fork_context()
    if n_jobs <= 1 or ctx is None or len(plan) <= 1:      # pragma: no cover
        K = engine.K
        out = np.empty((len(plan), K, K), dtype=np.float64)
        for i, s in enumerate(plan):
            out[i] = global_sim(engine, s, null)
        return out

    chunks = _split(plan, n_jobs)
    _STATE.clear()
    _STATE.update(engine=engine, null=null)
    try:
        with ctx.Pool(processes=min(n_jobs, len(chunks))) as pool:
            parts = pool.map(_worker_global, chunks)
    finally:
        _STATE.clear()
    return np.concatenate(parts, axis=0)


def parallel_local_counts(engine, plan, null, observed, ge, le, n_jobs):
    """Accumulate local permutation counts in parallel: children return block-private
    accumulators and the parent sums them exactly."""
    n_jobs = resolve_n_jobs(n_jobs)
    ctx = _fork_context()
    if n_jobs <= 1 or ctx is None or len(plan) <= 1:      # pragma: no cover
        from .inference import local_counts
        return local_counts(engine, plan, null, observed, ge, le)

    # each local block returns an O(NK²) accumulator, so the block count is budget-bound
    chunks = _split(plan, _local_n_chunks(engine, n_jobs))
    _STATE.clear()
    _STATE.update(engine=engine, null=null, observed=observed)
    try:
        with ctx.Pool(processes=min(n_jobs, len(chunks))) as pool:
            parts = pool.map(_worker_local, chunks)
    finally:
        _STATE.clear()

    np.copyto(ge, 0.0)
    np.copyto(le, 0.0)
    for part_ge, part_le in parts:                        # integer sums: exact
        ge += part_ge
        le += part_le
    return ge, le
