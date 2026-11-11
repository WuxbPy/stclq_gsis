"""Data loading and result export (pandas imported lazily)."""

from __future__ import annotations

import numpy as np


def load_npz(path, region_id=None, period=None):
    """Load from ``.npz``; return (coords, types, time_periods, region_id).

    Expected keys: ``coords`` (n,2), ``types`` (n,), and optionally ``time_periods``
    and ``region_id``.
    """
    d = np.load(path, allow_pickle=True)
    coords = np.asarray(d["coords"], dtype=np.float64)
    types = np.asarray(d["types"])
    tp = d["time_periods"] if "time_periods" in d.files else None
    region = d["region_id"] if "region_id" in d.files else None

    mask = np.ones(len(coords), dtype=bool)
    if region_id is not None and region is not None:
        mask &= np.isin(region, np.atleast_1d(region_id))
    if period is not None and tp is not None:
        mask &= np.isin(tp, np.atleast_1d(period))

    return (coords[mask], types[mask],
            (tp[mask] if tp is not None else None),
            (region[mask] if region is not None else None))


def load_csv(path, lon_col="longitude", lat_col="latitude", type_col="type_id",
             time_col="period_id", region_col="region_id",
             region_id=None, period=None):
    """Load from a CSV; return (coords, types, time_periods, region_id)."""
    import pandas as pd

    df = pd.read_csv(path, low_memory=False)
    if region_id is not None and region_col in df.columns:
        df = df[df[region_col].isin(np.atleast_1d(region_id))]
    if period is not None and time_col in df.columns:
        df = df[df[time_col].isin(np.atleast_1d(period))]

    coords = df[[lon_col, lat_col]].to_numpy(dtype=np.float64)
    types = df[type_col].to_numpy()
    tp = df[time_col].to_numpy() if time_col in df.columns else None
    region = df[region_col].to_numpy() if region_col in df.columns else None
    return coords, types, tp, region


def global_to_dataframe(gclq, p=None, type_names=None, alpha=0.05):
    """Global matrix as a table of type pairs."""
    import pandas as pd
    from .inference import classify

    n = gclq.shape[0]
    if type_names is None:
        type_names = [f"Type_{i + 1}" for i in range(n)]
    rows = []
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            pv = None if p is None else float(p[a, b])
            cls = classify(float(gclq[a, b]), pv, alpha)
            rows.append({
                "type_a_id": a + 1, "type_b_id": b + 1,
                "type_a": type_names[a], "type_b": type_names[b],
                "stclq": round(float(gclq[a, b]), 4),
                "p_value": None if pv is None else round(pv, 4),
                "significant": cls["significance"],
                "category": cls["category"],
            })
    return pd.DataFrame(rows)


def local_to_dataframe(coords, types, lclq, p=None, type_names=None, alpha=0.05):
    """Local results as a per-point table."""
    import pandas as pd
    from .inference import classify

    n = len(coords)
    K = lclq.shape[1]
    if type_names is None:
        type_names = [f"Type_{i + 1}" for i in range(K)]
    # type labels need not be 1..K, so the source type follows the codes passed in
    rows = []
    for i in range(n):
        a = int(types[i]) - 1
        rec = {"point_id": i + 1, "longitude": coords[i, 0], "latitude": coords[i, 1],
               "type": int(types[i]), "type_name": type_names[a]}
        for b in range(K):
            pv = None if p is None else float(p[i, a, b])
            cls = classify(float(lclq[i, a, b]), pv, alpha)
            rec[f"stclq_to_{b + 1}"] = round(float(lclq[i, a, b]), 4)
            rec[f"p_to_{b + 1}"] = None if pv is None else round(pv, 4)
            rec[f"sig_to_{b + 1}"] = int(cls["significance"])
        rows.append(rec)
    return pd.DataFrame(rows)
