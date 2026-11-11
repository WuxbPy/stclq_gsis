"""Backward-compatibility shell: existing ``pySTCLQ`` / ``MCCLQ`` calls migrate to
STCLQ unchanged.

The shell **always calls the new engine** and keeps no second kernel (principle 5 of
the design document).

Numeric conventions:
* the legacy layer uses ``kernel_convention='legacy_pystclq'`` and
  ``p_denominator='n_minus_1'`` so that it reproduces the old values;
* the ``MCCLQ`` layer uses ``kernel_convention='sigma'`` and ``p_denominator='n'``.
"""

from __future__ import annotations

import numpy as np

from ._types import AttributeSpec, AttributeType
from .api import STCLQ


# ===========================================================================
# Legacy compatibility layer
# ===========================================================================
class pySTCLQ:
    """Compatibility wrapper that preserves the legacy call signature."""

    def __init__(self, n_jobs: int = -1):
        self.n_jobs = n_jobs

    def _model(self, kernel, diagnostics=False):
        return STCLQ(spatial_kernel=kernel, use_attribute_weights=False,
                     temporal_weight="none", kernel_convention="legacy_pystclq",
                     p_denominator="n_minus_1", n_jobs=self.n_jobs,
                     diagnostics=diagnostics)

    def global_clq(self, coords, types, bandwidth, kernel="gaussian",
                   chunk_size=20000, n_jobs=None):
        return self._model(kernel).global_clq(coords, types, bandwidth)["gclq"]

    def local_clq(self, coords, types, bandwidth, kernel="gaussian",
                  chunk_size=20000, n_jobs=None):
        return self._model(kernel).local_clq(coords, types, bandwidth)["lclq"]

    def permutation_test(self, coords, types, observed, bandwidth,
                         kernel="gaussian", level="global", n_perm=999,
                         seed=42, chunk_size=20000, n_jobs=None):
        res = self._model(kernel).permutation_test(
            coords, types, observed, bandwidth, level=level, null="type",
            n_perm=n_perm, seed=seed)
        return res["p_value"]

    def fit_bandwidth(self, coords, types=None, method="auto",
                      min_bw=200.0, max_bw=5000.0, n_candidates=20):
        return self._model("gaussian").fit_bandwidth(
            coords, types, method=method, min_bw=min_bw, max_bw=max_bw,
            n_candidates=n_candidates)

    def rho(self, coords, bandwidth):
        return self._model("gaussian").rho(coords, bandwidth)

    @staticmethod
    def classify(clq_value, p_value, alpha=0.05):
        return STCLQ.classify(clq_value, p_value, alpha)

    def run(self, coords, types, bandwidth, kernel="gaussian",
            analysis="both", n_perm=999, seed=42, chunk_size=20000, n_jobs=None):
        res = self._model(kernel).run(
            coords, types, bandwidth, level=analysis, null="type",
            n_perm=n_perm, seed=seed)
        out = {"bandwidth": bandwidth, "n_types": res["n_types"],
               "n_points": res["n_points"], "rho": res["rho"]}
        if analysis in ("global", "both"):
            out["gclq"] = res["gclq"]
            out["p_global"] = res["global_test"]["p_value"]
        if analysis in ("local", "both"):
            out["lclq"] = res["lclq"]
            out["p_local"] = res["local_test"]["p_value"]
        return out


# ===========================================================================
# MCCLQ compatibility layer
# ===========================================================================
class MCCLQ:
    """Compatibility class with the same signature as ``MCCLQ/MCCLQ.py``
    (subset: analysis, tests, export)."""

    def __init__(self, attribute_specs, spatial_kernel="gaussian",
                 temporal_weight_type="inverse_dt", temporal_bandwidth=1.0,
                 spatial_cutoff_factor=3.0, n_jobs=-1, use_attribute_weights=True,
                 chunk_size=10000, max_memory_gb=2.0, verbose=True):
        self.attribute_specs = attribute_specs
        self.spatial_kernel = spatial_kernel
        self.temporal_weight_type = temporal_weight_type
        self.temporal_bandwidth = temporal_bandwidth
        self.spatial_cutoff_factor = spatial_cutoff_factor
        self.n_jobs = n_jobs
        self.use_attribute_weights = use_attribute_weights
        self.chunk_size = chunk_size
        self.max_memory_gb = max_memory_gb
        self.verbose = verbose
        self.results_ = {}

    def _model(self):
        return STCLQ(
            attribute_specs=self.attribute_specs,
            spatial_kernel=self.spatial_kernel,
            temporal_weight=self.temporal_weight_type,
            temporal_bandwidth=self.temporal_bandwidth,
            cutoff_factor=self.spatial_cutoff_factor,
            kernel_convention="sigma", p_denominator="n",
            use_attribute_weights=self.use_attribute_weights,
            n_jobs=self.n_jobs, diagnostics=False, verbose=self.verbose)

    # --- Analysis modes ---
    def single_period_global_mcclq(self, coords, types, attributes, bandwidth):
        r = self._model().global_clq(coords, types, bandwidth, attributes=attributes)
        return {"mc_gclq": r["gclq"], "type_names": r["type_names"],
                "n_points": r["n_points"], "n_types": r["n_types"],
                "bandwidth": bandwidth}

    def single_period_local_mcclq(self, coords, types, attributes, bandwidth):
        r = self._model().local_clq(coords, types, bandwidth, attributes=attributes)
        return {"mc_lclq": r["lclq"], "coords": r["coords"], "types": r["types"],
                "attributes": np.asarray(attributes), "bandwidth": bandwidth}

    def multi_period_global_mcclq(self, coords, types, attributes, time_periods, bandwidth):
        r = self._model().global_clq(coords, types, bandwidth,
                                     attributes=attributes, time_periods=time_periods)
        return {"mc_gclq": r["gclq"], "type_names": r["type_names"],
                "n_points": r["n_points"], "n_types": r["n_types"],
                "bandwidth": bandwidth}

    def multi_period_local_mcclq(self, coords, types, attributes, time_periods, bandwidth):
        r = self._model().local_clq(coords, types, bandwidth,
                                    attributes=attributes, time_periods=time_periods)
        return {"mc_lclq": r["lclq"], "coords": r["coords"], "types": r["types"],
                "attributes": np.asarray(attributes), "time_periods": time_periods,
                "bandwidth": bandwidth}

    # --- Significance testing ---
    def monte_carlo_test_global(self, coords, types, attributes, bandwidth,
                                n_permutations=999, time_periods=None, alpha=0.05):
        model = self._model()
        obs = model.global_clq(coords, types, bandwidth,
                               attributes=attributes, time_periods=time_periods)["gclq"]
        res = model.permutation_test(coords, types, obs, bandwidth,
                                     attributes=attributes, time_periods=time_periods,
                                     level="global", null="type",
                                     n_perm=n_permutations, alpha=alpha)
        out = {"observed_gclq": obs, "p_values": res["p_attract"],
               "significant": res["significant"], "p_attract": res["p_attract"],
               "p_repel": res["p_repel"], "p_two_sided": res["p_two_sided"],
               "p_attract_bonferroni": res.get("p_attract_bonferroni"),
               "p_attract_fdr": res.get("p_attract_fdr"),
               "alpha": alpha, "n_permutations": n_permutations}
        if "pseudo_ci_lower" in res:
            out["pseudo_ci_lower"] = res["pseudo_ci_lower"]
            out["pseudo_ci_upper"] = res["pseudo_ci_upper"]
        return out

    def monte_carlo_test_local(self, coords, types, attributes, bandwidth,
                               n_permutations=99, time_periods=None, alpha=0.05):
        model = self._model()
        obs = model.local_clq(coords, types, bandwidth,
                              attributes=attributes, time_periods=time_periods)["lclq"]
        res = model.permutation_test(coords, types, obs, bandwidth,
                                     attributes=attributes, time_periods=time_periods,
                                     level="local", null="type",
                                     n_perm=n_permutations, alpha=alpha)
        return {"observed_lclq": obs, "p_values": res["p_attract"],
                "significant": res["significant"], "p_attract": res["p_attract"],
                "p_repel": res["p_repel"], "alpha": alpha,
                "n_permutations": n_permutations}

    # --- Export ---
    def export_results_to_csv(self, results, output_path, analysis_type="global"):
        import pandas as pd
        if analysis_type == "global":
            g = results["mc_gclq"]
            rows = [{"Source_Type": a + 1, "Target_Type": b + 1, "MC_CLQ": float(g[a, b])}
                    for a in range(g.shape[0]) for b in range(g.shape[1])]
            if "p_values" in results:
                for r in rows:
                    r["p_value"] = float(results["p_values"][r["Source_Type"] - 1,
                                                             r["Target_Type"] - 1])
            pd.DataFrame(rows).to_csv(output_path, index=False)
        else:
            l = results["mc_lclq"]
            coords = results["coords"]
            types = results["types"]
            rows = []
            for i in range(l.shape[0]):
                a = int(types[i]) - 1
                for b in range(l.shape[2]):
                    rows.append({"Point_ID": i + 1, "Lon": coords[i, 0], "Lat": coords[i, 1],
                                 "Source_Type": a + 1, "Target_Type": b + 1,
                                 "MC_CLQ": float(l[i, a, b])})
            pd.DataFrame(rows).to_csv(output_path, index=False)
        return output_path

    def summary_report(self, results, analysis_type="global") -> str:
        lines = ["=" * 60, "STCLQ (MCCLQ-compatible) Summary", "=" * 60]
        if analysis_type == "global":
            g = results["mc_gclq"]
            lines.append(f"Global matrix {g.shape}; mean={np.nanmean(g):.3f}")
        else:
            l = results["mc_lclq"]
            lines.append(f"Local tensor {l.shape}")
        return "\n".join(lines)


# ===========================================================================
# Module-level helpers
# ===========================================================================
def create_attribute_specs_from_df(df, attr_columns, attr_types):
    """Build a list of ``AttributeSpec`` from a DataFrame (MCCLQ-compatible)."""
    specs = []
    for col, atype in zip(attr_columns, attr_types):
        specs.append(AttributeSpec(
            name=col, type=AttributeType(atype),
            categories=df[col].unique().tolist() if atype in ("categorical", "binary") else None))
    return specs


def prepare_mcclq_data(df, lon_col, lat_col, type_col, attr_cols, time_col=None):
    """Reshape a DataFrame into (coords, types, attributes[, time_periods]), coding
    types as 1..K.
    """
    coords = df[[lon_col, lat_col]].to_numpy(dtype=np.float64)
    raw = df[type_col].to_numpy()
    if not np.issubdtype(np.asarray(raw).dtype, np.integer):
        uniq = sorted(set(raw.tolist()))
        lut = {v: i + 1 for i, v in enumerate(uniq)}
        types = np.array([lut[v] for v in raw], dtype=np.int64)
    else:
        types = np.asarray(raw, dtype=np.int64)
    attributes = df[attr_cols].to_numpy(dtype=np.float64)
    if time_col is not None:
        tp = df[time_col].to_numpy()
        if not np.issubdtype(tp.dtype, np.integer):
            uniq = sorted(set(tp.tolist()))
            lut = {v: i + 1 for i, v in enumerate(uniq)}
            tp = np.array([lut[v] for v in tp], dtype=np.int64)
        return coords, types, attributes, tp
    return coords, types, attributes


# legacy alias for asw_stclq
prepare_asw_stclq_data = prepare_mcclq_data
