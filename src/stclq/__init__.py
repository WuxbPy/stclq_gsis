"""STCLQ: a toolkit for the spatio-temporal co-location quotient.

One weight engine, ``w = w_s · w_t · w_a``, covers the co-location quotient family
(GCLQ / LCLQ / GTWCLQ / attribute-conditional members), with permutation inference,
multiple-testing correction and scale diagnostics built in.

Quick start
-----------
    from stclq import STCLQ

    model = STCLQ(spatial_kernel='gaussian', temporal_weight='none', n_jobs=-1)
    res = model.run(coords, types, bandwidth=500.0, level='both', n_perm=999)
    print(res['gclq'])

Backward compatibility
----------------------
    from stclq.compat import pySTCLQ, MCCLQ, load_npz, load_csv
"""

from __future__ import annotations

__version__ = "0.1.0"

from ._types import (
    AttributeSpec,
    AttributeType,
    KernelType,
    TemporalWeightType,
)
from .engine import STCLQEngine, design_matrix
from .api import STCLQ
from .inference import (
    benjamini_hochberg_fdr,
    bonferroni_correction,
    classify,
    make_plan,
    permutation_test,
)
from .parallel import resolve_n_jobs
from .complexity import (
    analytic_rho,
    bandwidth_for_knn,
    bandwidth_for_rho,
    feasible_scale_table,
    memory_bytes,
    theoretical_cost,
)
from .bandwidth import (
    fit_attribute_bandwidth,
    fit_spatial_bandwidth,
    neighbourhood_fraction,
)
from .diagnostics import (
    DEFAULT_RHO_MAX,
    DiagnosticResult,
    format_report,
    run_diagnostics,
)
from .io import (
    global_to_dataframe,
    load_csv,
    load_npz,
    local_to_dataframe,
)
from .compat import (
    MCCLQ,
    create_attribute_specs_from_df,
    prepare_asw_stclq_data,
    prepare_mcclq_data,
    pySTCLQ,
)

# Optional acceleration backend: its absence changes nothing (the SciPy sparse path
# is what the package uses by default)
try:  # pragma: no cover
    import numba  # noqa: F401
    HAS_NUMBA = True
except Exception:  # pragma: no cover
    HAS_NUMBA = False

__all__ = [
    "STCLQ",
    "STCLQEngine",
    "design_matrix",
    "make_plan",
    "resolve_n_jobs",
    "theoretical_cost",
    "memory_bytes",
    "analytic_rho",
    "bandwidth_for_rho",
    "bandwidth_for_knn",
    "feasible_scale_table",
    "AttributeSpec",
    "AttributeType",
    "KernelType",
    "TemporalWeightType",
    "classify",
    "permutation_test",
    "benjamini_hochberg_fdr",
    "bonferroni_correction",
    "fit_spatial_bandwidth",
    "fit_attribute_bandwidth",
    "neighbourhood_fraction",
    "DiagnosticResult",
    "run_diagnostics",
    "format_report",
    "DEFAULT_RHO_MAX",
    "load_npz",
    "load_csv",
    "global_to_dataframe",
    "local_to_dataframe",
    "pySTCLQ",
    "MCCLQ",
    "create_attribute_specs_from_df",
    "prepare_mcclq_data",
    "prepare_asw_stclq_data",
    "HAS_NUMBA",
    "__version__",
]
