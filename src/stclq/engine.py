"""Single computation engine for the STCLQ family.

The statistic (the only place where it is implemented):

    w(i, j) = w_s(i, j) · w_t(i, j) · w_a(i, j)

    global  CLQ_g[a, b] = ( Σ_{i∈a} Σ_{j∈b, j≠i} w ) / ( Σ_{i∈a} Σ_{j≠i} w ) / p_b
    local   CLQ_l[i, b] = ( Σ_{j∈b, j≠i} w ) / ( Σ_{j≠i} w ) / p_b

Setting either component to 1 reduces the statistic to the corresponding member of the
family. Computation runs on the CSR sparse path: a permutation only recomputes
``types``/``attributes``, and the neighbourhood structure is reused (see ``neighbors``).
"""

from __future__ import annotations

import numpy as np

from . import kernels, neighbors
from ._types import build_attribute_arrays


def design_matrix(types: np.ndarray, K: int) -> np.ndarray:
    """**Augmented-column** design matrix ``Y ∈ R^{N×(K+1)}``.

    The first ``K`` columns hold the type one-hot indicators; the **last column is
    always 1**. Writing ``Z = W @ Y`` gives

    * ``Z[:, :K]``: weighted count of each neighbour type (numerator term);
    * ``Z[:, K]``: sum of neighbourhood weights (row sums for the denominator).

    That is, **one SpMM yields the numerator and the row sums at once**, with no
    second traversal. This is the key trick that reduces permutation inference to
    sparse linear algebra (see §3/§4 of the paper).
    """
    n = types.size
    Y = np.zeros((n, K + 1), dtype=np.float64)
    Y[np.arange(n), types - 1] = 1.0
    Y[:, K] = 1.0
    return Y


def _encode_labels(labels: np.ndarray):
    """Map arbitrary comparable labels to consecutive 1..K; return (codes, label table)."""
    uniq = np.unique(labels)
    if uniq.size < 2:
        raise ValueError("types must contain at least two distinct labels")
    lut = {v: i + 1 for i, v in enumerate(uniq)}
    encoded = np.fromiter((lut[v] for v in labels), dtype=np.int64, count=labels.size)
    return encoded, uniq.tolist()


class STCLQEngine:
    """Sparse engine built once and reused across calls."""

    def __init__(
        self,
        coords,
        types,
        bandwidth,
        attributes=None,
        time_periods=None,
        spatial_kernel="gaussian",
        temporal_weight="none",
        temporal_bandwidth=1.0,
        kernel_convention="sigma",
        cutoff_factor=3.0,
        distance_metric="auto",
        projection_constants=None,
        attribute_specs=None,
        use_attribute_weights=False,
        p_denominator="n_minus_1",
    ):
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords must have shape (n, 2)")

        self.n = coords.shape[0]
        self.coords = coords
        self.bandwidth = float(bandwidth)
        self.spatial_kernel = spatial_kernel
        self.kernel_convention = kernel_convention
        self.cutoff_factor = float(cutoff_factor)
        self.distance_metric = distance_metric
        self.projection_constants = projection_constants
        self.temporal_weight = temporal_weight
        self.temporal_bandwidth = float(temporal_bandwidth)
        self.p_denominator = p_denominator

        # --- Types: recode automatically to 1..K ---
        types = np.asarray(types)
        self.types, self.type_labels = _encode_labels(types)
        self.K = len(self.type_labels)

        # --- Time: encode to 0..P-1 ---
        if time_periods is not None:
            tp = np.asarray(time_periods)
            self.period_labels = np.unique(tp).tolist()
            lut = {v: i for i, v in enumerate(self.period_labels)}
            self.periods = np.fromiter((lut[v] for v in tp), dtype=np.int64, count=tp.size)
            self.n_periods = len(self.period_labels)
        else:
            self.periods = None
            self.period_labels = None
            self.n_periods = 1

        # --- Attributes ---
        if attributes is not None:
            attributes = np.asarray(attributes, dtype=np.float64)
            if attributes.ndim == 1:
                attributes = attributes[:, None]
        self.attributes = attributes
        n_attr = 0 if attributes is None else attributes.shape[1]
        self.n_attributes = n_attr
        self.attribute_specs = attribute_specs or []
        (self.attr_types, self.attr_bandwidths,
         self.ordinal_ranges, self.attr_weights) = build_attribute_arrays(
            self.attribute_specs, n_attr)
        self.use_attribute_weights = bool(use_attribute_weights and n_attr > 0)

        # --- Neighbourhood and permutation-invariant weights ---
        src, dst, dist, meta = neighbors.build_edges(
            coords, self.bandwidth, self.cutoff_factor,
            self.distance_metric, self.projection_constants)
        self.src, self.dst, self.dist = src, dst, dist
        self.neighbour_meta = meta

        w_s = kernels.spatial_weights(
            dist, self.bandwidth, self.spatial_kernel, self.kernel_convention)

        if (self.periods is not None and temporal_weight not in (None, "none")):
            dt = np.abs(self.periods[src] - self.periods[dst])
            w_t = kernels.temporal_weights(dt, temporal_weight, self.temporal_bandwidth)
        else:
            w_t = np.ones_like(w_s)
        self.w_s, self.w_t = w_s, w_t

        # Permutation invariants (unchanged under label or time permutation)
        self.base_data = w_s * w_t

        data = self._data_from_attributes(self.attributes) \
            if self.use_attribute_weights else self.base_data
        self.W = neighbors.build_csr(src, dst, data, self.n)

        # Whether an attribute permutation altered the weights, so they can be reset safely
        self._attr_dirty = False

    # ------------------------------------------------------------------
    def _data_from_attributes(self, attributes) -> np.ndarray:
        """Recompute the edge weights ``w_s · w_t · w_a`` from the attribute matrix."""
        if not self.use_attribute_weights:
            return self.base_data
        za = np.asarray(attributes, dtype=np.float64)
        w_a = kernels.attribute_edge_weights(
            za[self.src], za[self.dst],
            self.attr_types, self.attr_bandwidths,
            self.ordinal_ranges, self.attr_weights)
        return self.base_data * w_a

    def reset_weights(self):
        """Restore the CSR ``data`` to the true-attribute / original-label values."""
        if self.use_attribute_weights:
            self.W.data = self._data_from_attributes(self.attributes)
        else:
            self.W.data = self.base_data
        self._attr_dirty = False

    # ------------------------------------------------------------------
    # Invariance self-checks (used for the empirical validation / regression
    # tests in the paper)
    # ------------------------------------------------------------------
    def structure_fingerprint(self) -> tuple:
        """Fingerprint of the neighbourhood structure and the invariant weights
        (it must stay the same under type or time label permutations).

        Returns ``(indptr checksum, indices checksum, data bytes, src/dst bytes)``.
        """
        W = self.W.tocsr()
        return (
            int(W.indptr.sum()),
            int(W.indices.astype(np.int64).sum()),
            W.data.tobytes(),
            self.src.tobytes() + self.dst.tobytes(),
        )

    def invariance_check(self, n_draws=8, seed=0) -> dict:
        """Check that a type label permutation leaves the CSR structure, ``w_s`` and
        ``w_t`` unchanged.

        For several random label permutations, compare :meth:`structure_fingerprint`
        and return ``{'unchanged': bool, 'max_data_delta': float}``.
        """
        base = self.structure_fingerprint()
        rng = np.random.default_rng(seed)
        worst = 0.0
        ok = True
        for _ in range(n_draws):
            t = rng.permutation(self.types)
            _ = self.global_clq(t)              # run a reduction (no cache writes)
            cur = self.structure_fingerprint()
            if cur[0] != base[0] or cur[1] != base[1]:
                ok = False
            if cur[2] != base[2]:
                b = np.frombuffer(base[2], dtype=np.float64)
                c = np.frombuffer(cur[2], dtype=np.float64)
                worst = max(worst, float(np.max(np.abs(b - c))) if b.size else 0.0)
        return {"unchanged": ok, "max_data_delta": worst}

    # ------------------------------------------------------------------
    def _p_b(self, K: int) -> np.ndarray:
        counts = np.bincount(self.types, minlength=K + 1)[1:].astype(np.float64)
        if self.p_denominator == "n_minus_1" and self.n > 1:
            base = self.n - 1
        else:
            base = self.n
        return counts / base

    def spmm(self, Y: np.ndarray) -> np.ndarray:
        """Sparse matrix product ``Z = W @ Y`` (CSR rows in parallel, no shared writes
        between rows).

        This is the **only** `O(nnz)` dense-side cost once a permutation is adopted;
        the ``indptr``/``indices``/``data`` of ``W`` stay read-only throughout, so
        several processes can share them safely (see :mod:`stclq.parallel`).
        """
        return np.asarray(self.W @ Y)

    def reduce_clq(self, Z: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Reduce the global ``CLQ_g`` (K, K) from ``Z = W @ Y`` and 0-based ``labels``.

        **BLAS-free**: ``bincount`` segment sums by source type over ``Z[:, :K]``
        (numerator) and ``Z[:, K]`` (row sums), costing ``O(NK)``. See the note under
        :meth:`global_clq`.
        """
        K = self.K
        Zk = Z[:, :K]
        pnw = np.empty((K, K), dtype=np.float64)
        for b in range(K):
            pnw[:, b] = np.bincount(labels, weights=Zk[:, b], minlength=K)
        denom = np.bincount(labels, weights=Z[:, K], minlength=K)
        p_b = self._p_b(K)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where((denom[:, None] > 0) & (p_b[None, :] > 0),
                            pnw / denom[:, None] / p_b[None, :], 0.0)

    def global_clq(self, types=None) -> np.ndarray:
        """Global STCLQ matrix (K, K).

        The reduction is deliberately written as BLAS-free ``bincount`` segment sums
        (:meth:`reduce_clq`) rather than a dense GEMM of the form ``Yᵀ Z``:

        1. the complexity stays the clean ``O(nnz + NK + K²)`` and no third-party
           BLAS behaviour enters the picture;
        2. under process-level parallelism (:mod:`stclq.parallel`), a BLAS thread
           pool initialised inside each child would oversubscribe the host by a
           factor of ``p × nproc``; that is precisely why the reduction stays at
           the level of basic NumPy operations.
        """
        t = self.types if types is None else np.asarray(types, dtype=np.int64)
        Z = self.spmm(design_matrix(t, self.K))   # (n, K+1): one SpMM
        return self.reduce_clq(Z, t - 1)          # 0..K-1 is valid bincount input

    def local_clq(self, types=None) -> np.ndarray:
        """Local STCLQ, shape (n, K, K); only [i, type_i, :] is nonzero."""
        t = self.types if types is None else np.asarray(types, dtype=np.int64)
        K = self.K
        Y = design_matrix(t, K)
        Z = self.spmm(Y)
        Zk = Z[:, :K]
        rowsum = Z[:, K]
        p_b = self._p_b(K)
        out = np.zeros((self.n, K, K), dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            P = np.divide(Zk, rowsum[:, None],
                          out=np.zeros_like(Zk), where=rowsum[:, None] > 1e-12)
            vals = np.divide(P, p_b[None, :],
                             out=np.zeros_like(P), where=p_b[None, :] > 0)
        out[np.arange(self.n), t - 1, :] = vals
        return out

    # ------------------------------------------------------------------
    def neighbourhood_fraction(self, sample=200000, seed=42) -> float:
        """Neighbourhood capture fraction ρ = mean neighbours per point / N, a
        bandwidth diagnostic.
        """
        from scipy.spatial import cKDTree

        xy, mode, _ = neighbors.project_coords(
            self.coords, self.distance_metric, self.projection_constants)
        cutoff = self.bandwidth * self.cutoff_factor
        tree = cKDTree(xy)
        n = self.n
        if n > sample:
            idx = np.random.default_rng(seed).choice(n, sample, replace=False)
            cnt = tree.query_ball_point(xy[idx], cutoff, return_length=True)
        else:
            cnt = tree.query_ball_point(xy, cutoff, return_length=True)
        return float(np.mean(cnt)) / n
