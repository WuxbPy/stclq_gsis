"""Statistical inference layer: permutation tests, multiple-testing correction, classification.

The four null hypotheses (paper §6.3):

=========================  ==============================  ==========================================
``null``                   quantity permuted               proposition tested
=========================  ==============================  ==========================================
``type``                   type labels                     type ⟂ location (global label permutation)
``type_period``            type labels, blocked by period  type ⟂ location | time structure
``attribute``              attributes, pooled over types   attribute ⟂ location
``attribute_within_type``  attributes, within each type    attribute ⟂ location | type
=========================  ==============================  ==========================================

A p-value uses the bias correction ``(count + 1) / (n_perm + 1)`` (Phipson and Smyth 2010).

**Algebra of a permutation** (paper §3–§4): permuting the type labels changes neither the
sparse neighbourhood structure nor ``w_s``/``w_t``, so each permutation only reorders the
rows of the design matrix ``Y`` (an equivalent row permutation), and the statistic follows
from one ``SpMM`` at cost ``O(nnz)``.

**Reproducibility**: the permutation sequence is fixed by a per-permutation seed vector
``plan`` drawn from a parent RNG (``default_rng(seed).integers(...)``). Serial and
many-process runs are therefore bit-for-bit identical, and ``plan`` travels with the
results so that any reported p-value can be audited and recomputed.
"""

from __future__ import annotations

import numpy as np

from .engine import design_matrix

EPS = 1e-12
NULL_CHOICES = ("type", "type_period", "attribute", "attribute_within_type")


# ---------------------------------------------------------------------------
# Multiple-testing correction and classification
# ---------------------------------------------------------------------------
def benjamini_hochberg_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR correction; returns q-values shaped like the input."""
    p = np.asarray(p_values, dtype=np.float64)
    shape = p.shape
    flat = p.ravel()
    m = flat.size
    order = np.argsort(flat, kind="mergesort")
    ranked = flat[order]
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(flat)
    out[order] = q
    return out.reshape(shape)


def bonferroni_correction(p_values: np.ndarray, n_tests: int = None) -> np.ndarray:
    """Bonferroni correction, clipped at 1."""
    p = np.asarray(p_values, dtype=np.float64)
    if n_tests is None:
        n_tests = p.size
    return np.clip(p * n_tests, 0.0, 1.0)


def classify(clq_value, p_value, alpha=0.05):
    """ArcGIS Pro style classification: attraction, repulsion, not significant."""
    sig = p_value is not None and np.isfinite(p_value) and p_value < alpha
    if not sig:
        return {"code": 0, "category": "Not significant", "type_name": "Not Significant",
                "significance": False}
    if clq_value > 1:
        return {"code": 1, "category": "Attraction", "type_name": "Attraction",
                "significance": True}
    if clq_value < 1:
        return {"code": -1, "category": "Repulsion", "type_name": "Repulsion",
                "significance": True}
    return {"code": 0, "category": "Neutral", "type_name": "Neutral",
            "significance": False}


# ---------------------------------------------------------------------------
# Permutation plan and a single permutation (shared by the serial and parallel paths)
# ---------------------------------------------------------------------------
def make_plan(seed, n_perm: int) -> np.ndarray:
    """Derive ``n_perm`` permutation seeds from the parent RNG (recordable, recomputable)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2 ** 63 - 1, size=int(n_perm), dtype=np.int64)


def permuted_state(engine, null: str, rng):
    """Draw one permutation under the chosen null and return ``(kind, array)``.

    * ``kind == 'types'``: ``array`` holds the permuted type labels (length N);
    * ``kind == 'attrs'``: ``array`` holds the permuted attribute matrix (N × n_attr).
    """
    n = engine.n
    if null == "type":
        return "types", rng.permutation(engine.types)
    if null == "type_period":
        t = engine.types.copy()
        for per in np.unique(engine.periods):
            idx = np.where(engine.periods == per)[0]
            t[idx] = t[idx][rng.permutation(idx.size)]
        return "types", t
    if null == "attribute":
        return "attrs", engine.attributes[rng.permutation(n)]
    if null == "attribute_within_type":
        a = engine.attributes.copy()
        for ty in np.unique(engine.types):
            idx = np.where(engine.types == ty)[0]
            if idx.size > 1:
                a[idx] = a[idx][rng.permutation(idx.size)]
        return "attrs", a
    raise ValueError(f"unknown null: {null!r}")


def global_sim(engine, plan_seed, null: str) -> np.ndarray:
    """**Global** statistic of one permutation, ``(K, K)`` (one SpMM plus a small reduction)."""
    rng = np.random.default_rng(int(plan_seed))
    kind, arr = permuted_state(engine, null, rng)
    if kind == "attrs":
        engine.W.data = engine._data_from_attributes(arr)
        return engine.global_clq()
    engine.reset_weights()      # keep the real attribute weights, permute types only
    return engine.global_clq(arr)


def local_counts(engine, plan, null, observed, ge, le):
    """**Local-level** permutation counts (raw semantics, element-wise comparison).

    Each permutation builds an ``(N, K, K)`` array and compares it with the observed
    values. Memory is ``O(NK²)``, which is the size of the **local output itself** and
    has nothing to do with the engine's ``O(nnz)``; paper §5 keeps the two apart.

    The accumulators are **integers**, so chunk sums across processes are exact and the
    serial and parallel paths agree bit for bit.
    """
    for s in plan:
        rng = np.random.default_rng(int(s))
        kind, arr = permuted_state(engine, null, rng)
        if kind == "attrs":
            engine.W.data = engine._data_from_attributes(arr)
            sim = engine.local_clq()
        else:
            engine.reset_weights()
            sim = engine.local_clq(arr)
        ge += (sim >= observed)
        le += (sim <= observed)
    return ge, le


def _pvalues_from_counts(ge, le, n_perm):
    denom = n_perm + 1
    p_attract = (ge + 1.0) / denom
    p_repel = (le + 1.0) / denom
    return p_attract, p_repel


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def permutation_test(engine, observed, level="global", null="type",
                     n_perm=999, seed=42, alpha=0.05, correction=True,
                     n_jobs=1, plan=None, return_sims=False):
    """Monte Carlo permutation test, with optional multiprocessing.

    Parameters
    ----------
    engine : STCLQEngine
    observed : np.ndarray
        (K, K) when ``level='global'``; (n, K, K) when ``level='local'``.
    level : 'global' | 'local'
    null : str
        ``type`` / ``type_period`` / ``attribute`` / ``attribute_within_type``.
    n_perm, seed, alpha : int, int, float
    correction : bool
        Also report the Bonferroni and BH-FDR corrected results.
    n_jobs : int
        Number of processes at the permutation level; ``1`` is serial, ``-1`` uses every
        core. **The result does not depend on this number.**
    plan : np.ndarray or None
        For recomputation: supply the per-permutation seeds directly (length ``n_perm``).
    return_sims : bool
        Also return the simulated global matrix ``(n_perm, K, K)``, for analytic
        approximations and diagnostics; only meaningful when ``level='global'``.

    Returns
    -------
    dict
    """
    if level not in ("global", "local"):
        raise ValueError("level must be 'global' or 'local'")
    if null in ("attribute", "attribute_within_type") and not engine.use_attribute_weights:
        raise ValueError("attribute permutation requires attribute weights")
    if null == "type_period" and engine.periods is None:
        raise ValueError("null='type_period' requires time_periods")
    if null not in NULL_CHOICES:
        raise ValueError(f"null must be one of {NULL_CHOICES}")

    K, n = engine.K, engine.n
    n_perm = int(n_perm)
    if plan is None:
        plan = make_plan(engine_seed(seed), n_perm)
    else:
        plan = np.asarray(plan, dtype=np.int64)
        n_perm = plan.size
    engine.reset_weights()

    if level == "global":
        sims = _global_sims(engine, plan, null, n_jobs)
        engine.reset_weights()
        obs = np.asarray(observed, dtype=np.float64)
        ge = (sims >= obs).sum(axis=0).astype(np.float64)
        le = (sims <= obs).sum(axis=0).astype(np.float64)
        p_attract, p_repel = _pvalues_from_counts(ge, le, n_perm)
        p_two_sided = (np.abs(sims - 1.0) >= np.abs(obs - 1.0)).sum(axis=0)
        p_two_sided = (p_two_sided + 1.0) / (n_perm + 1)
        pseudo_ci = (np.percentile(sims, 2.5, axis=0),
                     np.percentile(sims, 97.5, axis=0))
    else:
        obs = np.asarray(observed, dtype=np.float64)
        ge = np.zeros((n, K, K), dtype=np.float64)
        le = np.zeros((n, K, K), dtype=np.float64)
        ge, le = _local_counts(engine, plan, null, obs, ge, le, n_jobs)
        engine.reset_weights()
        p_attract, p_repel = _pvalues_from_counts(ge, le, n_perm)
        p_two_sided = None
        pseudo_ci = None

    out = {
        "observed": obs,
        "p_value": p_attract,          # compatibility field: one-sided (clustering) test
        "p_attract": p_attract,
        "p_repel": p_repel,
        "p_two_sided": p_two_sided,
        "significant": p_attract < alpha,
        "alpha": alpha,
        "n_permutations": n_perm,
        "level": level,
        "null": null,
        "n_jobs": n_jobs,
        "plan_seeds": plan,            # reproducible: per-permutation seeds
    }
    if return_sims and level == "global":
        out["sims"] = sims
    if correction:
        out["p_attract_fdr"] = benjamini_hochberg_fdr(p_attract)
        out["p_attract_bonferroni"] = bonferroni_correction(p_attract, p_attract.size)
        out["significant_fdr"] = out["p_attract_fdr"] < alpha
        out["significant_bonferroni"] = out["p_attract_bonferroni"] < alpha
    if pseudo_ci is not None:
        out["pseudo_ci_lower"], out["pseudo_ci_upper"] = pseudo_ci
    return out


def engine_seed(seed):
    """Keep the historical default seed when ``seed=None``."""
    return 42 if seed is None else seed


def _global_sims(engine, plan, null, n_jobs) -> np.ndarray:
    """Simulated global matrix ``(n_perm, K, K)``; identical bit for bit when parallel."""
    if not _worth_parallel(engine, len(plan), n_jobs):
        K = engine.K
        out = np.empty((len(plan), K, K), dtype=np.float64)
        for i, s in enumerate(plan):
            out[i] = global_sim(engine, s, null)
        return out
    from .parallel import parallel_global_sims
    return parallel_global_sims(engine, plan, null, n_jobs)


def _local_counts(engine, plan, null, observed, ge, le, n_jobs):
    if not _worth_parallel(engine, len(plan), n_jobs):
        return local_counts(engine, plan, null, observed, ge, le)
    from .parallel import parallel_local_counts
    return parallel_local_counts(engine, plan, null, observed, ge, le, n_jobs)


#: Below this amount of work (permutations x nonzero entries) multiprocessing is not
#: started, so that fork overhead does not dominate the gain.
PARALLEL_WORK_THRESHOLD = 5e6


def _worth_parallel(engine, n_perm, n_jobs) -> bool:
    """Whether multiprocessing is worth starting: enough work and parallelism requested."""
    if n_jobs is None or n_jobs == 1 or n_jobs == 0 or n_perm <= 1:
        return False
    nnz = getattr(getattr(engine, "W", None), "nnz", 0)
    return float(n_perm) * float(nnz) >= PARALLEL_WORK_THRESHOLD
