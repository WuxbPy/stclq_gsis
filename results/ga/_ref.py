"""Reference implementation (the naive ``O(N²)`` path): the single reference for the
correctness check and for the speed baseline.

It mirrors the conventions of the engine exactly (see ``src/stclq``):

* cutoff radius ``cutoff = c · b`` (``c = 3`` by default); weights beyond the radius
  are set to 0;
* Gaussian kernel ``exp(-0.5 (d/b)²)``; box kernel ``1[d ≤ b]``; Epanechnikov
  ``1-z²``; inverse distance ``1/(d+ε)``;
* the temporal weight multiplies ``w_s``: ``1/(1+|Δt|)`` or ``exp(-|Δt|/τ)``;
* normaliser ``p_b = n_b / (n − 1)``; statistic
  ``CLQ_g[a,b] = Σ_{i∈a} Σ_{j∈b,j≠i} w / Σ_{i∈a} Σ_{j≠i} w / p_b``.

Deliberately written the "dumb" way: it expands the dense distance matrix block by
block and reuses no cache at all, so that it forms an independent check on the sparse
engine (§6.2 of the paper, "Correctness").
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def _kernel(d, b, kernel):
    if kernel == "gaussian":
        return np.exp(-0.5 * (d / b) ** 2)
    if kernel == "box":
        return (d <= b).astype(np.float64)
    if kernel == "epanechnikov":
        z = d / b
        return np.where(np.abs(z) <= 1.0, 1.0 - z * z, 0.0)
    if kernel == "inverse_distance":
        return 1.0 / (d + EPS)
    raise ValueError(f"unknown kernel: {kernel!r}")


def _weights(d, b, cutoff, kernel, dt=None, temporal="none", tau=1.0):
    w = _kernel(d, b, kernel)
    w[d > cutoff] = 0.0
    if temporal not in ("none", None) and dt is not None:
        w = w * (1.0 / (1.0 + dt) if temporal == "inverse_dt"
                 else np.exp(-dt / float(tau)))
    return w


def reference_global_clq(coords, types, bandwidth, K,
                         kernel="gaussian", cutoff_factor=3.0,
                         periods=None, temporal="none", tau=1.0, chunk=256):
    """Naive global CLQ ``(K, K)``."""
    coords = np.asarray(coords, dtype=np.float64)
    types = np.asarray(types, dtype=np.int64)
    n = coords.shape[0]
    cutoff = float(bandwidth) * float(cutoff_factor)
    counts = np.bincount(types, minlength=K + 1)[1:].astype(np.float64)
    p_b = counts / (n - 1.0)
    pnw = np.zeros((K, K), dtype=np.float64)
    den = np.zeros(K, dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = np.sqrt(((coords[s:e, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
        dt = None
        if temporal not in ("none", None) and periods is not None:
            per = np.asarray(periods, dtype=np.float64)
            dt = np.abs(per[s:e, None] - per[None, :])
        w = _weights(d, bandwidth, cutoff, kernel, dt, temporal, tau)
        w[np.arange(e - s), s + np.arange(e - s)] = 0.0
        src = types[s:e] - 1
        for b in range(K):
            pnw[:, b] += np.bincount(src, weights=w[:, types == b + 1].sum(axis=1),
                                     minlength=K)
        den += np.bincount(src, weights=w.sum(axis=1), minlength=K)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((den[:, None] > 0) & (p_b[None, :] > 0),
                        pnw / den[:, None] / p_b[None, :], 0.0)


def reference_local_clq(coords, types, bandwidth, K,
                        kernel="gaussian", cutoff_factor=3.0,
                        periods=None, temporal="none", tau=1.0):
    """Naive local CLQ ``(n, K, K)`` (only ``[i, t_i, :]`` is nonzero)."""
    coords = np.asarray(coords, dtype=np.float64)
    types = np.asarray(types, dtype=np.int64)
    n = coords.shape[0]
    cutoff = float(bandwidth) * float(cutoff_factor)
    counts = np.bincount(types, minlength=K + 1)[1:].astype(np.float64)
    p_b = counts / (n - 1.0)
    out = np.zeros((n, K, K), dtype=np.float64)
    for i in range(n):
        d = np.sqrt(((coords - coords[i]) ** 2).sum(axis=1))
        dt = None
        if temporal not in ("none", None) and periods is not None:
            per = np.asarray(periods, dtype=np.float64)
            dt = np.abs(per - per[i])
        w = _weights(d, bandwidth, cutoff, kernel, dt, temporal, tau)
        w[i] = 0.0
        row = np.bincount(types - 1, weights=w, minlength=K)
        tot = row.sum()
        if tot <= 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            vals = np.where(p_b > 0, row / tot / p_b, 0.0)
        out[i, types[i] - 1, :] = vals
    return out
