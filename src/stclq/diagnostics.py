"""Active diagnostics: warn when a result would not be interpretable instead of
returning it silently.

Severity (section 2 of the design document):
* ``warning`` / ``error``: affects interpretability, triggers ``warnings.warn``;
* ``info``: advisory only (e.g. a parameter that cancels out of a ratio-type
  statistic), no warning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from . import bandwidth as _bw

DEFAULT_RHO_MAX = 0.01
DEFAULT_MIN_GROUP = 20


@dataclass
class DiagnosticResult:
    name: str
    severity: str  # info / warning / error
    message: str

    @property
    def is_problem(self) -> bool:
        return self.severity in ("warning", "error")

    def __str__(self) -> str:  # pragma: no cover - for printing
        return f"[{self.severity.upper()}] {self.name}: {self.message}"


def check_bandwidth_scale(rho: float, rho_max: float = DEFAULT_RHO_MAX) -> DiagnosticResult:
    if rho > rho_max:
        return DiagnosticResult(
            "bandwidth_scale", "warning",
            f"Neighbourhood capture fraction ρ={rho:.4f} exceeds the limit {rho_max:.4f}; "
            f"the CLQ is then mechanically compressed towards 1, so read it descriptively.")
    return DiagnosticResult(
        "bandwidth_scale", "info", f"ρ={rho:.4f} is within the recommended range.")


def check_group_sizes(types, min_group: int = DEFAULT_MIN_GROUP) -> List[DiagnosticResult]:
    out = []
    vals, cnts = np.unique(np.asarray(types), return_counts=True)
    small = [(v, int(c)) for v, c in zip(vals, cnts) if c < min_group]
    if small:
        detail = ", ".join(f"{v}:{c}" for v, c in small)
        out.append(DiagnosticResult(
            "group_sizes", "warning",
            f"These types have fewer than {min_group} points, so the permutation "
            f"test is unreliable: {detail}"))
    else:
        out.append(DiagnosticResult("group_sizes", "info", "Every type has enough points."))
    return out


def check_inert_parameters(temporal_weight, n_periods, has_time) -> DiagnosticResult:
    if (temporal_weight not in (None, "none")) and (not has_time or n_periods <= 1):
        return DiagnosticResult(
            "inert_parameters", "info",
            "A temporal weight is enabled but the data hold a single period; the "
            "weight is then a constant factor and does not affect the result.")
    return DiagnosticResult("inert_parameters", "info", "No parameter cancels out.")


def run_diagnostics(coords, types, bandwidth, attributes=None,
                    temporal_weight="none", n_periods=1,
                    distance_metric="auto", projection_constants=None,
                    cutoff_factor=3.0, rho_max=DEFAULT_RHO_MAX,
                    min_group=DEFAULT_MIN_GROUP, rho: float = None) -> List[DiagnosticResult]:
    """Run every diagnostic and return the list of results (warnings come first)."""
    if rho is None:
        rho = _bw.neighbourhood_fraction(
            coords, bandwidth, distance_metric=distance_metric,
            projection_constants=projection_constants, cutoff_factor=cutoff_factor)
    results = [check_bandwidth_scale(rho, rho_max)]
    results.extend(check_group_sizes(types, min_group))
    results.append(check_inert_parameters(temporal_weight, n_periods,
                                          n_periods > 1))
    return results


def format_report(results: List[DiagnosticResult]) -> str:
    lines = ["STCLQ diagnostic report:"]
    icon = {"warning": "!", "error": "X", "info": "i"}
    for r in results:
        lines.append(f"  [{icon.get(r.severity, '?')}] {r.name} - {r.message}")
    return "\n".join(lines)
