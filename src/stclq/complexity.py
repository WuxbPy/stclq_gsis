"""Complexity, memory bounds and the **feasible-scale criterion** (paper §5).

Notation
--------
``N``   number of points; ``K`` number of types; ``nnz`` number of CSR nonzero
entries (directed edges);
``ρ``   neighbourhood capture fraction, ``ρ = nnz / N² = k̄ / N`` (``k̄`` is the
mean neighbour count);
``n_perm`` number of permutations; ``c`` cutoff factor (3.0 by default).

Key results
-----------
==================  ==========================================
Stage               Complexity
==================  ==========================================
Build (one-off)     ``O(N log N + nnz)``
One permutation     ``O(nnz)`` (reference: ``O(N²)``)
Reduce per perm     ``O(NK + K²)`` (reference: ``O(N²)``)
Peak memory         ``O(nnz)`` (CSR); reference ``O(N²)``
==================  ==========================================

Amortised cost ``≈ build + n_perm · nnz``; from ``n_perm ≳ 10`` on this is well
below the ``n_perm · N²`` of recomputing from scratch each time.

Feasible-scale criterion
------------------------
The method requires ``ρ ≲ 0.01``; otherwise the neighbourhoods cover almost the whole
domain and the CLQ is mechanically compressed towards 1.
For a region of uniform density and area ``A``, ``ρ ≈ π(3b)² / A``, hence

    ``b_max = sqrt(ρ_max · A) / (c · sqrt(π))``

So the **bandwidth must shrink relative to the study-domain scale**; and since
``nnz = ρN²``, ``nnz`` still grows as ``N²`` even when ``ρ`` is fixed: **the feasible
scale is ultimately set by the criterion, not by the efficiency of the implementation**.
"""

from __future__ import annotations

import math

import numpy as np

#: Per-edge storage in the CSR (bytes): indices (int64) + data (float64)
BYTES_PER_EDGE_CSR = 16
#: Transient build cost per edge (bytes): src/dst (int64) + dist (float64)
BYTES_PER_EDGE_BUILD = 40


# ---------------------------------------------------------------------------
# Complexity bounds
# ---------------------------------------------------------------------------
def theoretical_cost(n: int, nnz: int, n_perm: int, k: int = 2) -> dict:
    """Theoretical bound for each stage (paper Table 2 and budget estimates)."""
    n = int(n)
    nnz = int(nnz)
    n_perm = int(n_perm)
    build_ops = n * math.log2(max(n, 2)) + nnz
    ratio = nnz / (n * n) if n else float("inf")
    return {
        "n": n, "nnz": nnz, "K": int(k), "n_perm": n_perm,
        "rho": nnz / (n * n) if n else float("nan"),
        "build_ops": build_ops,
        "perm_ops_each": nnz,
        "perm_ops_total": nnz * n_perm,
        "reduce_ops_each": n * k + k * k,
        "reference_perm_ops_each": n * n,
        "reference_perm_ops_total": n * n * n_perm,
        "speedup_ops": (n * n * n_perm) / (nnz * n_perm) if nnz else float("inf"),
        "amortized_ops": build_ops + n_perm * nnz,
        "reference_total_ops": n * n * n_perm,
        # asymptotic speed-up per permutation = N²/nnz = 1/ρ
        "per_perm_speedup": (1.0 / ratio) if ratio else float("inf"),
        # how many reference permutations the one-off build is worth: build / N²
        "build_in_reference_perm_equivalents": build_ops / (n * n) if n else float("inf"),
        # the sparse path wins once amortised: n_perm ≥ build/(N² − nnz)
        "amortization_threshold_n_perm":
            build_ops / max(n * n - nnz, 1e-30) if n else float("inf"),
    }


def memory_bytes(nnz: int, n: int, k: int, csr_share: float = 1.0) -> dict:
    """Memory of the CSR path against the ``O(N²)`` dense path (bytes)."""
    csr = BYTES_PER_EDGE_CSR * nnz
    build_peak = BYTES_PER_EDGE_BUILD * nnz
    # dense path: N×N float64 distance matrix + weight matrix
    dense = 8 * n * n
    design = 8 * n * (k + 1)
    return {"csr_bytes": csr, "csr_build_peak_bytes": max(csr, build_peak),
            "dense_bytes": dense, "design_bytes": design,
            "saving_ratio": dense / csr if csr else float("inf")}


# ---------------------------------------------------------------------------
# Feasible-scale criterion
# ---------------------------------------------------------------------------
def analytic_rho(bandwidth: float, domain_area: float,
                 cutoff_factor: float = 3.0) -> float:
    """Analytic ``ρ`` under a uniform density: ``ρ = π(c·b)² / A``."""
    r = float(cutoff_factor) * float(bandwidth)
    return math.pi * r * r / float(domain_area)


def bandwidth_for_rho(rho_max: float, domain_area: float,
                      cutoff_factor: float = 3.0) -> float:
    """Bandwidth upper bound ``b_max`` that satisfies ``ρ ≤ ρ_max``."""
    c = float(cutoff_factor)
    return math.sqrt(float(rho_max) * float(domain_area)) / (c * math.sqrt(math.pi))


def bandwidth_for_knn(kbar: float, n: int, domain_area: float,
                      cutoff_factor: float = 3.0) -> float:
    """Bandwidth at which the mean neighbour count ``k̄`` equals the given value
    (uniform density)."""
    c = float(cutoff_factor)
    return math.sqrt(float(kbar) * float(domain_area) / (math.pi * float(n))) / c


def feasible_scale_table(ns, side: float = 20_000.0, rho_max: float = 0.01,
                         target_knn: float | None = 150.0) -> list:
    """Feasible scale for different values of ``N``.

    When ``target_knn`` is not None, each row also gives the bandwidth needed to hold
    ``k̄`` constant, which shows that *the study radius must shrink with the sample
    size*; otherwise the row gives the bandwidth upper bound imposed by ``ρ_max``.

    Each row returns ``N / b_max at ρ_max / the matching k̄ / nnz / CSR memory /
    dense memory``.
    """
    area = float(side) ** 2
    rows = []
    for n in ns:
        n = int(n)
        b_max = bandwidth_for_rho(rho_max, area)
        kbar_max = rho_max * n
        nnz_max = rho_max * n * n
        row = {
            "n": n,
            "side_m": float(side),
            "rho_max": float(rho_max),
            "b_max_rho": b_max,
            "kbar_at_b_max": kbar_max,
            "nnz_at_b_max": int(nnz_max),
            "csr_mb_at_b_max": memory_bytes(int(nnz_max), n, 2)["csr_bytes"] / 2 ** 20,
            "dense_mb": 8.0 * n * n / 2 ** 20,
        }
        if target_knn:
            b_knn = bandwidth_for_knn(target_knn, n, area)
            row.update({
                "b_for_target_knn": b_knn,
                "rho_at_target_knn": target_knn / n,
                "nnz_at_target_knn": int(target_knn * n),
                "csr_mb_at_target_knn":
                    memory_bytes(int(target_knn * n), n, 2)["csr_bytes"] / 2 ** 20,
            })
        rows.append(row)
    return rows


def rho_curve(bandwidths, side: float = 20_000.0, n: int = 100_000,
              cutoff_factor: float = 3.0) -> np.ndarray:
    """Analytic ``ρ(b)`` curve (plotted against measured points, see Fig.4)."""
    area = float(side) ** 2
    return np.array([analytic_rho(b, area, cutoff_factor) for b in bandwidths])
