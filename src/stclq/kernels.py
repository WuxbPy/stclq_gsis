"""STCLQ kernels: the **only place** where the spatial, temporal and attribute
weights are implemented.

Shared conventions (section 4 of the design document):
* Gaussian kernel ``exp(-0.5 * (d/b)**2)`` (``kernel_convention='sigma'``); with
  ``kernel_convention='legacy_pystclq'`` it becomes ``exp(-(d/b)**2)``, which
  reproduces the old values.
* Temporal kernels: ``none`` / ``inverse_dt = 1/(1+|dt|)`` /
  ``exponential = exp(-|dt|/tau)``.
* Attribute kernels are dispatched by type; several attributes are aggregated by a
  weighted mean.

All functions are **vectorised over the edge arrays** and do not depend on Numba
(they still work when Numba is absent).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def spatial_weights(dist, bandwidth, kernel="gaussian", convention="sigma"):
    """Vectorised spatial weights.

    Parameters
    ----------
    dist : np.ndarray
        Pairwise distances (metres).
    bandwidth : float
        Spatial bandwidth (kernel scale, metres).
    kernel : str
        gaussian / box / inverse_distance / epanechnikov.
    convention : str
        sigma (default) or legacy_pystclq.
    """
    d = np.asarray(dist, dtype=np.float64)
    b = float(bandwidth)
    if b <= 0:
        return np.ones_like(d)

    if kernel == "gaussian":
        z = d / b
        if convention == "legacy_pystclq":
            return np.exp(-(z * z))
        return np.exp(-0.5 * (z * z))
    if kernel == "box":
        return (d <= b).astype(np.float64)
    if kernel == "inverse_distance":
        return 1.0 / (d + EPS)
    if kernel == "epanechnikov":
        z = d / b
        w = 1.0 - z * z
        return np.where(np.abs(z) <= 1.0, w, 0.0)
    raise ValueError(f"unknown spatial kernel: {kernel!r}")


def temporal_weights(time_diff, tw_type="none", tau=1.0):
    """Vectorised temporal weights (``time_diff`` is |Δt|)."""
    dt = np.abs(np.asarray(time_diff, dtype=np.float64))
    if tw_type in ("none", None):
        return np.ones_like(dt)
    if tw_type == "inverse_dt":
        return 1.0 / (1.0 + dt)
    if tw_type == "exponential":
        t = float(tau) if tau and tau > 0 else 1.0
        return np.exp(-dt / t)
    raise ValueError(f"unknown temporal weight type: {tw_type!r}")


def attribute_edge_weights(z_src, z_dst, attr_types, bandwidths,
                           ordinal_ranges, attr_weights):
    """Edge-wise attribute similarity weights ``w_a`` (weighted mean over attributes).

    Parameters
    ----------
    z_src, z_dst : np.ndarray (m, n_attrs)
        Attribute vectors at the two endpoints of each edge.
    attr_types : np.ndarray (n_attrs,)
        0=continuous, 1=categorical, 2=ordinal, 3=binary.
    bandwidths, ordinal_ranges, attr_weights : np.ndarray (n_attrs,)
        Per-attribute parameters, one entry per attribute.
    """
    zs = np.asarray(z_src, dtype=np.float64)
    zd = np.asarray(z_dst, dtype=np.float64)
    n_attr = zs.shape[1] if zs.ndim > 1 else 0
    if n_attr == 0:
        return np.ones(zs.shape[0], dtype=np.float64)

    total = np.zeros(zs.shape[0], dtype=np.float64)
    wsum = 0.0
    for k in range(n_attr):
        t = int(attr_types[k])
        omega = float(attr_weights[k])
        if omega == 0.0:
            continue
        if t == 0:  # continuous: Gaussian on |Δz|
            b = float(bandwidths[k])
            if b > 0:
                w = np.exp(-0.5 * ((zs[:, k] - zd[:, k]) / b) ** 2)
            else:
                w = (np.abs(zs[:, k] - zd[:, k]) < 1e-10).astype(np.float64)
        elif t in (1, 3):  # categorical / binary: indicator
            w = (zs[:, k] == zd[:, k]).astype(np.float64)
        elif t == 2:  # ordinal: linear decay
            r = float(ordinal_ranges[k])
            if r > 0:
                w = np.maximum(0.0, 1.0 - np.abs(zs[:, k] - zd[:, k]) / r)
            else:
                w = (zs[:, k] == zd[:, k]).astype(np.float64)
        else:
            w = np.ones(zs.shape[0], dtype=np.float64)
        total += omega * w
        wsum += omega

    if wsum <= 0:
        return np.ones(zs.shape[0], dtype=np.float64)
    return total / wsum
