"""Neighbourhood construction and the CSR cache.

Design notes:
* A KD-tree collects the point pairs within the cutoff radius and compresses them
  into CSR; ``data`` holds the permutation-invariant weights (``w_s`` or
  ``w_s·w_t``).
* A type or time label permutation leaves both the neighbourhood structure and
  those weights **unchanged**, so the cache is built once and each permutation
  reduces to a single sparse matrix product.

The module only maps coordinates to the sparse structure; it does not compute the
statistic.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

# Shared convention: equirectangular approximation constants
METERS_PER_DEG_LON = 111320.0
METERS_PER_DEG_LAT = 110540.0


def _looks_like_lonlat(coords: np.ndarray) -> bool:
    lon = np.abs(coords[:, 0]).max()
    lat = np.abs(coords[:, 1]).max()
    return lon <= 180.0 and lat <= 90.0


def project_coords(coords, distance_metric="auto", projection_constants=None):
    """Project coordinates onto a metric plane.

    distance_metric : auto / equirectangular / projected / haversine
      * auto: lon/lat ranges select equirectangular, otherwise the input is taken
        as already projected metric coordinates;
      * haversine: neighbourhoods and distances stay on the sphere (the lon/lat
        values are returned as they are, see build_edges);
      * projected: return unchanged.

    Returns
    -------
    xy : np.ndarray (n, 2)
    mode : str  the metric actually used
    scale : tuple(lon_scale, lat_scale) or None
    """
    c = np.asarray(coords, dtype=np.float64)
    if distance_metric == "auto":
        distance_metric = "equirectangular" if _looks_like_lonlat(c) else "projected"

    if distance_metric == "projected":
        return c, "projected", None
    if distance_metric == "haversine":
        return c, "haversine", None

    lon_m, lat_m = projection_constants if projection_constants else (
        METERS_PER_DEG_LON, METERS_PER_DEG_LAT)
    lat_mean = float(c[:, 1].mean())
    scale_lon = float(lon_m) * math.cos(math.radians(lat_mean))
    scale_lat = float(lat_m)
    xy = np.column_stack([c[:, 0] * scale_lon, c[:, 1] * scale_lat])
    return xy, "equirectangular", (scale_lon, scale_lat)


def _haversine(lon1, lat1, lon2, lat2):
    r = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def build_edges(coords, bandwidth, cutoff_factor=3.0,
                distance_metric="auto", projection_constants=None):
    """Build the **directed edge arrays** within the cutoff radius (expanded both
    ways, self-loops removed).

    Returns
    -------
    src, dst : np.ndarray (m,) int64
    dist : np.ndarray (m,) float64
    meta : dict  {mode, scale, cutoff}
    """
    c = np.asarray(coords, dtype=np.float64)
    xy, mode, scale = project_coords(c, distance_metric, projection_constants)
    cutoff = float(bandwidth) * float(cutoff_factor)

    # Candidate search: in haversine mode, prefilter on an equirectangular plane at
    # the smaller scale so that the candidate set stays conservative
    if mode == "haversine":
        lat_mean = float(c[:, 1].mean())
        cand_scale = min(METERS_PER_DEG_LON * math.cos(math.radians(lat_mean)),
                         METERS_PER_DEG_LAT)
        xy_c = np.column_stack([c[:, 0] * cand_scale, c[:, 1] * cand_scale])
        r_query = cutoff
    else:
        xy_c = xy
        r_query = cutoff

    tree = cKDTree(xy_c)
    pairs = tree.query_pairs(r_query, output_type="ndarray")
    if pairs.size == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i.copy(), np.empty(0, dtype=np.float64), \
            {"mode": mode, "scale": scale, "cutoff": cutoff}

    src = np.concatenate([pairs[:, 0], pairs[:, 1]]).astype(np.int64)
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]]).astype(np.int64)

    if mode == "haversine":
        dist = _haversine(c[src, 0], c[src, 1], c[dst, 0], c[dst, 1])
    else:
        d = xy[src] - xy[dst]
        dist = np.sqrt((d * d).sum(axis=1))

    keep = (dist <= cutoff) & (src != dst)
    src, dst, dist = src[keep], dst[keep], dist[keep]

    # Sort by (src, dst) lexicographic order so that the data order of csr_matrix
    # matches the edge array order exactly; later code can then replace W.data in
    # place, which the permutation test depends on.
    order = np.lexsort((dst, src))
    src, dst, dist = src[order], dst[order], dist[order]
    return src, dst, dist, {"mode": mode, "scale": scale, "cutoff": cutoff}


def build_csr(src, dst, data, n):
    """Build the CSR sparse matrix ``W`` (n×n) from the edge arrays.

    Requires ``src``/``dst`` to be sorted in (src, dst) lexicographic order (see
    :func:`build_edges`); ``csr_matrix`` then leaves ``data`` in place, so
    ``W.data`` maps one-to-one onto the edge arrays.
    """
    return csr_matrix((np.asarray(data, dtype=np.float64), (src, dst)), shape=(n, n))
