"""STCLQ type definitions: enums, attribute specifications and code tables.

This module holds data shapes only; it contains no computation logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class AttributeType(Enum):
    """Attribute variable type."""

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    BINARY = "binary"


class KernelType(Enum):
    """Spatial kernel type."""

    GAUSSIAN = "gaussian"
    BOX = "box"
    INVERSE_DISTANCE = "inverse_distance"
    EPANECHNIKOV = "epanechnikov"


class TemporalWeightType(Enum):
    """Temporal weight type.

    Note: ``INVERSE_N`` / ``INVERSE_N_SQUARED`` are period-level constant factors and
    cancel out of a ratio-type statistic, so STCLQ does not provide them (see section
    4 of the design document on unified conventions).
    """

    NONE = "none"
    INVERSE_DT = "inverse_dt"
    EXPONENTIAL = "exponential"


@dataclass
class AttributeSpec:
    """Attribute variable specification.

    Parameters
    ----------
    name : str
        Column name.
    type : AttributeType
        Variable type; this decides which kernel is used.
    bandwidth : float, optional
        Gaussian kernel bandwidth for a continuous attribute.
    ordinal_range : float, optional
        Decay range for an ordinal attribute; falls back to ``bandwidth``.
    weight : float
        Weight in the multi-attribute aggregate (1.0 by default, i.e. equal weights).
    categories : list, optional
        Category values of a categorical or binary attribute (optional, used for
        validation).
    """

    name: str
    type: AttributeType = AttributeType.CONTINUOUS
    bandwidth: Optional[float] = None
    ordinal_range: Optional[float] = None
    weight: float = 1.0
    categories: Optional[List] = None


# --- Integer codes (for the vectorised and optional Numba kernels) -----------

ATTR_TYPE_CODE = {
    "continuous": 0,
    "categorical": 1,
    "ordinal": 2,
    "binary": 3,
}

KERNEL_CODE = {
    "gaussian": 0,
    "box": 1,
    "inverse_distance": 2,
    "epanechnikov": 3,
}

TW_CODE = {
    "none": 0,
    "inverse_dt": 1,
    "exponential": 2,
}


def build_attribute_arrays(specs: List[AttributeSpec], n_attrs: int):
    """Turn a list of ``AttributeSpec`` into equal-length numeric arrays.

    Returns
    -------
    attr_types : np.ndarray (n_attrs,)  int64
    bandwidths : np.ndarray (n_attrs,)  float64
    ordinal_ranges : np.ndarray (n_attrs,)  float64
    attr_weights : np.ndarray (n_attrs,)  float64
    """
    import numpy as np

    attr_types = np.zeros(n_attrs, dtype=np.int64)
    bandwidths = np.ones(n_attrs, dtype=np.float64)
    ordinal_ranges = np.ones(n_attrs, dtype=np.float64)
    attr_weights = np.ones(n_attrs, dtype=np.float64)

    for k in range(min(len(specs), n_attrs)):
        spec = specs[k]
        attr_types[k] = ATTR_TYPE_CODE[spec.type.value]
        if spec.bandwidth is not None and spec.bandwidth > 0:
            bandwidths[k] = float(spec.bandwidth)
        else:
            bandwidths[k] = 0.0
        r = spec.ordinal_range if spec.ordinal_range is not None else spec.bandwidth
        ordinal_ranges[k] = float(r) if r is not None else 0.0
        attr_weights[k] = float(spec.weight)

    return attr_types, bandwidths, ordinal_ranges, attr_weights
