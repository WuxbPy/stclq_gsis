"""Bandwidth selection and scale diagnostics.

Spatial bandwidth: variogram, Moran and k-nearest-neighbour rules of thumb.
Attribute bandwidth: Silverman, Scott and cross-validation.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from . import neighbors


# ---------------------------------------------------------------------------
# Scale diagnostics
# ---------------------------------------------------------------------------
def neighbourhood_fraction(coords, bandwidth, sample=200000, seed=42,
                           distance_metric="auto", projection_constants=None,
                           cutoff_factor=3.0):
    """Neighbourhood capture fraction ρ = mean neighbour count / N; empirically it
    should be ≲ 0.01.
    """
    xy, _, _ = neighbors.project_coords(coords, distance_metric, projection_constants)
    cutoff = float(bandwidth) * float(cutoff_factor)
    tree = cKDTree(xy)
    n = len(xy)
    if n > sample:
        idx = np.random.default_rng(seed).choice(n, sample, replace=False)
        cnt = tree.query_ball_point(xy[idx], cutoff, return_length=True)
    else:
        cnt = tree.query_ball_point(xy, cutoff, return_length=True)
    return float(np.mean(cnt)) / n


# ---------------------------------------------------------------------------
# Spatial bandwidth
# ---------------------------------------------------------------------------
def _knn_bandwidth(xy, k):
    tree = cKDTree(xy)
    dist, _ = tree.query(xy, k=min(k + 1, len(xy)))
    return float(np.median(dist[:, -1]))


def _rule_of_thumb_bandwidth(xy):
    extent = min(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
    return float(extent * 0.1)


def _indicator_variogram_bandwidth(coords, types, min_bw, max_bw,
                                   distance_metric, projection_constants,
                                   n_sample=3000, n_bins=24):
    """Type indicator variogram: the range is the distance at which the
    semivariogram reaches 95% of the sill.
    """
    xy, mode, _ = neighbors.project_coords(coords, distance_metric, projection_constants)
    n = len(xy)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, min(n_sample, n), replace=False) if n > n_sample else np.arange(n)
    tree = cKDTree(xy)
    edges = np.linspace(0, max_bw, n_bins + 1)
    num = np.zeros(n_bins)
    den = np.zeros(n_bins)
    t = np.asarray(types)
    for i in idx:
        nb = tree.query_ball_point(xy[i], max_bw)
        if not nb:
            continue
        nb = np.asarray(nb)
        nb = nb[nb != i]
        if nb.size == 0:
            continue
        d = np.sqrt(((xy[nb] - xy[i]) ** 2).sum(axis=1))
        dissim = (t[nb] != t[i]).astype(np.float64)
        b = np.clip(np.digitize(d, edges) - 1, 0, n_bins - 1)
        np.add.at(num, b, dissim)
        np.add.at(den, b, 1.0)
    gamma = np.where(den > 0, 0.5 * num / np.maximum(den, 1), np.nan)
    valid = np.isfinite(gamma)
    if not valid.any():
        return float((min_bw + max_bw) / 2)
    centers = (edges[:-1] + edges[1:]) / 2
    sill = np.nanmax(gamma)
    if sill <= 0:
        return float((min_bw + max_bw) / 2)
    reach = np.where(valid & (gamma >= 0.95 * sill))[0]
    rng_bw = centers[reach[0]] if reach.size else centers[valid][-1]
    return float(np.clip(rng_bw, min_bw, max_bw))


def _moran_bandwidth(coords, types, min_bw, max_bw,
                     distance_metric, projection_constants, n_candidates=15):
    """Take the candidate bandwidth that maximises |Moran's I|."""
    xy, _, _ = neighbors.project_coords(coords, distance_metric, projection_constants)
    n = len(xy)
    rng = np.random.default_rng(0)
    if n > 5000:
        idx = rng.choice(n, 5000, replace=False)
        xy, tv = xy[idx], np.asarray(types, dtype=np.float64)[idx]
    else:
        tv = np.asarray(types, dtype=np.float64)
    tree = cKDTree(xy)
    tv = tv - tv.mean()
    den = float((tv ** 2).sum())
    if den <= 0:
        return float((min_bw + max_bw) / 2)
    best_bw, best_abs = (min_bw + max_bw) / 2, -1.0
    m = len(xy)
    for bw in np.logspace(np.log10(min_bw), np.log10(max_bw), n_candidates):
        r = bw * 3.0
        num = 0.0
        for i in range(min(m, 1000)):
            nb = tree.query_ball_point(xy[i], r)
            for j in nb:
                if j != i:
                    num += tv[i] * tv[j]
        moran = num / den
        if abs(moran) > best_abs:
            best_abs, best_bw = abs(moran), float(bw)
    return float(np.clip(best_bw, min_bw, max_bw))


def fit_spatial_bandwidth(coords, types=None, method="auto",
                          min_bw=200.0, max_bw=5000.0, n_candidates=15,
                          distance_metric="auto", projection_constants=None):
    """Automatic spatial bandwidth selection.

    method : auto / variogram / moran / knn / rule_of_thumb
    """
    coords = np.asarray(coords, dtype=np.float64)
    xy, _, _ = neighbors.project_coords(coords, distance_metric, projection_constants)
    n = len(xy)

    if method == "auto":
        if types is not None and len(np.unique(types)) > 3:
            method = "moran"
        elif n > 50000:
            method = "variogram"
        else:
            method = "rule_of_thumb"

    if method == "variogram" and types is not None:
        bw = _indicator_variogram_bandwidth(coords, types, min_bw, max_bw,
                                            distance_metric, projection_constants)
    elif method == "moran" and types is not None:
        bw = _moran_bandwidth(coords, types, min_bw, max_bw,
                              distance_metric, projection_constants, n_candidates)
    elif method == "knn":
        k = max(10, min(int(math.sqrt(n)), 100))
        bw = _knn_bandwidth(xy, k)
    else:
        bw = _rule_of_thumb_bandwidth(xy)

    bw = float(np.clip(bw, min_bw, max_bw))
    return {"optimal_bandwidth": bw, "method": method,
            "n_points": n, "n_types": None if types is None else int(len(np.unique(types)))}


# ---------------------------------------------------------------------------
# Attribute bandwidth
# ---------------------------------------------------------------------------
def silverman_bandwidth(data) -> float:
    x = np.asarray(data, dtype=np.float64)
    n = x.size
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    q1, q3 = np.percentile(x, [25, 75])
    iqr = (q3 - q1) / 1.34
    a = min(sd, iqr) if iqr > 0 else sd
    if a <= 0:
        return 0.0
    return float(0.9 * a * n ** (-1.0 / 5.0))


def scott_bandwidth(data) -> float:
    x = np.asarray(data, dtype=np.float64)
    n = x.size
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    if sd <= 0:
        return 0.0
    return float(1.06 * sd * n ** (-1.0 / 5.0))


def cv_bandwidth(data, candidates=None) -> float:
    """Least-squares cross-validation bandwidth for a one-dimensional Gaussian KDE."""
    x = np.asarray(data, dtype=np.float64)
    n = x.size
    if n < 2:
        return 0.0
    sd = float(np.std(x, ddof=1))
    if sd <= 0:
        return 0.0
    if candidates is None:
        candidates = sd * np.logspace(-1.5, 0.5, 30)
    X = x[:, None] - x[None, :]
    np.fill_diagonal(X, np.nan)
    best_h, best_score = float(candidates[0]), np.inf
    for h in candidates:
        K = np.exp(-0.5 * (X / h) ** 2) / (h * math.sqrt(2 * math.pi))
        term1 = np.nansum(K) * 2.0 / (n * (n - 1))
        K2 = np.exp(-0.25 * (X / h) ** 2) / (2 * h * math.sqrt(math.pi))
        term2 = np.nansum(K2) / (n * (n - 1))
        score = term2 - term1
        if score < best_score:
            best_score, best_h = score, float(h)
    return best_h


def fit_attribute_bandwidth(data, method="silverman") -> float:
    """Attribute bandwidth selection: silverman / scott / cv / auto."""
    if method == "auto":
        method = "cv" if np.asarray(data).size <= 20000 else "silverman"
    if method == "silverman":
        return silverman_bandwidth(data)
    if method == "scott":
        return scott_bandwidth(data)
    if method == "cv":
        return cv_bandwidth(data)
    raise ValueError(f"unknown attribute bandwidth method: {method!r}")
