"""The STCLQ main class: the only public entry point (recommended for new code).

Design principle (section 5 of the design document): **the statistical logic may live
in exactly one place**. This class only arranges parameters and wraps results; every
computation is delegated to ``engine`` / ``inference`` / ``bandwidth`` / ``diagnostics``.
"""

from __future__ import annotations

import warnings

import numpy as np

from . import bandwidth as _bw
from . import diagnostics as _dg
from . import inference as _inf
from .engine import STCLQEngine


class STCLQ:
    """Spatio-Temporal Co-location Quotient.

    ``w = w_s · w_t · w_a``; setting either component to 1 reduces the statistic to
    the corresponding member of the family.
    """

    def __init__(
        self,
        attribute_specs=None,
        spatial_kernel="gaussian",
        temporal_weight="none",
        temporal_bandwidth=1.0,
        kernel_convention="sigma",
        cutoff_factor=3.0,
        distance_metric="auto",
        projection_constants=None,
        p_denominator="n_minus_1",
        use_attribute_weights=True,
        n_jobs=-1,
        chunk_size=20000,
        random_state=42,
        diagnostics=True,
        rho_max=_dg.DEFAULT_RHO_MAX,
        verbose=False,
    ):
        self.attribute_specs = attribute_specs or []
        self.spatial_kernel = spatial_kernel
        self.temporal_weight = temporal_weight
        self.temporal_bandwidth = temporal_bandwidth
        self.kernel_convention = kernel_convention
        self.cutoff_factor = cutoff_factor
        self.distance_metric = distance_metric
        self.projection_constants = projection_constants
        self.p_denominator = p_denominator
        self.use_attribute_weights = use_attribute_weights
        self.n_jobs = n_jobs
        self.chunk_size = chunk_size
        self.random_state = random_state
        self.diagnostics = diagnostics
        self.rho_max = rho_max
        self.verbose = verbose
        self.last_diagnostics_ = None

    # ------------------------------------------------------------------
    def _make_engine(self, coords, types, bandwidth, attributes, time_periods):
        return STCLQEngine(
            coords, types, bandwidth,
            attributes=attributes, time_periods=time_periods,
            spatial_kernel=self.spatial_kernel,
            temporal_weight=self.temporal_weight,
            temporal_bandwidth=self.temporal_bandwidth,
            kernel_convention=self.kernel_convention,
            cutoff_factor=self.cutoff_factor,
            distance_metric=self.distance_metric,
            projection_constants=self.projection_constants,
            attribute_specs=self.attribute_specs,
            use_attribute_weights=self.use_attribute_weights,
            p_denominator=self.p_denominator,
        )

    def _diagnose(self, coords, types, bandwidth, attributes, engine, rho=None):
        """Run diagnostics once, on the observed values only, never in the permutation
        loop."""
        if not self.diagnostics:
            self.last_diagnostics_ = []
            return []
        res = _dg.run_diagnostics(
            coords, types, bandwidth, attributes=attributes,
            temporal_weight=self.temporal_weight,
            n_periods=getattr(engine, "n_periods", 1),
            distance_metric=self.distance_metric,
            projection_constants=self.projection_constants,
            cutoff_factor=self.cutoff_factor, rho_max=self.rho_max, rho=rho)
        self.last_diagnostics_ = res
        if any(r.is_problem for r in res):
            warnings.warn(_dg.format_report(res), UserWarning, stacklevel=3)
        return res

    # ------------------------------------------------------------------
    # Analysis modes
    # ------------------------------------------------------------------
    def global_clq(self, coords, types, bandwidth, attributes=None, time_periods=None):
        """Global STCLQ matrix, returned as a dict."""
        engine = self._make_engine(coords, types, bandwidth, attributes, time_periods)
        rho = engine.neighbourhood_fraction()
        self._diagnose(coords, types, bandwidth, attributes, engine, rho=rho)
        g = engine.global_clq()
        return {"gclq": g, "type_names": [f"Type_{v}" for v in engine.type_labels],
                "n_points": engine.n, "n_types": engine.K,
                "bandwidth": bandwidth, "rho": rho, "engine": engine}

    def local_clq(self, coords, types, bandwidth, attributes=None, time_periods=None):
        """Local STCLQ of shape (n, K, K), returned as a dict."""
        engine = self._make_engine(coords, types, bandwidth, attributes, time_periods)
        rho = engine.neighbourhood_fraction()
        self._diagnose(coords, types, bandwidth, attributes, engine, rho=rho)
        l = engine.local_clq()
        return {"lclq": l, "coords": np.asarray(coords), "types": engine.types,
                "type_names": [f"Type_{v}" for v in engine.type_labels],
                "bandwidth": bandwidth, "rho": rho, "engine": engine}

    def multi_period_global_clq(self, coords, types, attributes, time_periods, bandwidth):
        return self.global_clq(coords, types, bandwidth, attributes, time_periods)

    def multi_period_local_clq(self, coords, types, attributes, time_periods, bandwidth):
        return self.local_clq(coords, types, bandwidth, attributes, time_periods)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def permutation_test(self, coords, types, observed, bandwidth,
                         attributes=None, time_periods=None,
                         level="global", null="type", n_perm=999, seed=None,
                         alpha=0.05, correction=True, n_jobs=None, plan=None):
        """Monte Carlo permutation test (four nulls: label, period-block and
        attribute-conditional; runs in parallel when requested).
        """
        engine = self._make_engine(coords, types, bandwidth, attributes, time_periods)
        return _inf.permutation_test(
            engine, observed, level=level, null=null, n_perm=n_perm,
            seed=self.random_state if seed is None else seed,
            alpha=alpha, correction=correction,
            n_jobs=self.n_jobs if n_jobs is None else n_jobs, plan=plan)

    def run(self, coords, types, bandwidth, attributes=None, time_periods=None,
            level="both", null="type", n_perm=999, seed=None, alpha=0.05,
            n_jobs=None):
        """One-call analysis: observed values plus significance tests."""
        engine = self._make_engine(coords, types, bandwidth, attributes, time_periods)
        rho = engine.neighbourhood_fraction()
        self._diagnose(coords, types, bandwidth, attributes, engine, rho=rho)

        seed = self.random_state if seed is None else seed
        jobs = self.n_jobs if n_jobs is None else n_jobs
        out = {"bandwidth": bandwidth, "rho": rho, "n_points": engine.n,
               "n_types": engine.K, "type_names": [f"Type_{v}" for v in engine.type_labels],
               "n_periods": engine.n_periods}
        if level in ("global", "both"):
            g = engine.global_clq()
            out["gclq"] = g
            out["global_test"] = _inf.permutation_test(
                engine, g, level="global", null=null, n_perm=n_perm,
                seed=seed, alpha=alpha, n_jobs=jobs)
        if level in ("local", "both"):
            l = engine.local_clq()
            out["lclq"] = l
            out["local_test"] = _inf.permutation_test(
                engine, l, level="local", null=null, n_perm=n_perm,
                seed=seed, alpha=alpha, n_jobs=jobs)
        return out

    # ------------------------------------------------------------------
    # Bandwidth and diagnostics
    # ------------------------------------------------------------------
    def fit_bandwidth(self, coords, types=None, method="auto",
                      min_bw=200.0, max_bw=5000.0, n_candidates=15):
        return _bw.fit_spatial_bandwidth(
            coords, types, method=method, min_bw=min_bw, max_bw=max_bw,
            n_candidates=n_candidates, distance_metric=self.distance_metric,
            projection_constants=self.projection_constants)

    def fit_attribute_bandwidths(self, attributes, specs=None, method="silverman"):
        """Estimate a bandwidth for each continuous attribute from its spec; returns a
        dict."""
        specs = specs or self.attribute_specs
        attrs = np.asarray(attributes, dtype=np.float64)
        if attrs.ndim == 1:
            attrs = attrs[:, None]
        out = {}
        for k, spec in enumerate(specs):
            if k >= attrs.shape[1]:
                break
            if getattr(spec.type, "value", None) == "continuous":
                out[spec.name] = _bw.fit_attribute_bandwidth(attrs[:, k], method=method)
        return out

    def rho(self, coords, bandwidth):
        return _bw.neighbourhood_fraction(
            coords, bandwidth, distance_metric=self.distance_metric,
            projection_constants=self.projection_constants,
            cutoff_factor=self.cutoff_factor)

    @staticmethod
    def classify(clq_value, p_value, alpha=0.05):
        return _inf.classify(clq_value, p_value, alpha)
