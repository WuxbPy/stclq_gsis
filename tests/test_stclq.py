"""Unit tests for STCLQ (runnable directly, also works under pytest)."""

from __future__ import annotations

import numpy as np

from stclq import STCLQ, STCLQEngine, AttributeSpec, AttributeType
from stclq.compat import pySTCLQ, MCCLQ, prepare_mcclq_data


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    c1 = rng.normal(0.0, 0.01, (n, 2))
    c2 = rng.normal(0.01, 0.01, (n, 2))
    coords = np.vstack([c1, c2]) + np.array([-0.1, 51.5])
    types = np.concatenate([np.ones(n, dtype=np.int64), 2 * np.ones(n, dtype=np.int64)])
    return coords, types


def test_global_shape_and_identity():
    coords, types = _data()
    m = STCLQ(diagnostics=False)
    g = m.global_clq(coords, types, 1500.0)["gclq"]
    assert g.shape == (2, 2)
    # Identity: Σ_b CLQ[a,b]·p_b = 1 (approximate; denominator large, no truncation)
    engine = m._make_engine(coords, types, 1500.0, None, None)
    p_b = engine._p_b(2)
    resid = np.abs((g * p_b[None, :]).sum(axis=1) - 1.0)
    assert float(np.max(resid)) < 1e-6


def test_local_only_source_row_nonzero():
    coords, types = _data(n=150)
    m = STCLQ(diagnostics=False)
    l = m.local_clq(coords, types, 1500.0)["lclq"]
    n = coords.shape[0]
    assert l.shape == (n, 2, 2)
    # A point is nonzero only in the row of its own type
    for i in range(0, n, 37):
        a = int(types[i]) - 1
        other = 1 - a
        assert np.allclose(l[i, other, :], 0.0)


def test_permutation_pvalues_in_range():
    coords, types = _data(n=200)
    m = STCLQ(diagnostics=False)
    out = m.run(coords, types, 1500.0, level="global", n_perm=99, seed=3)
    p = out["global_test"]["p_attract"]
    assert p.shape == (2, 2)
    assert np.all((p > 0) & (p <= 1))


def test_type_auto_reencode():
    coords, types = _data(n=120)
    relabeled = np.where(types == 1, 10, 42)  # non-contiguous labels
    m = STCLQ(diagnostics=False)
    g = m.global_clq(coords, relabeled, 1500.0)["gclq"]
    assert g.shape == (2, 2)


def test_attribute_weights_run():
    coords, types = _data(n=200)
    attrs = np.random.default_rng(1).normal(0, 1, (coords.shape[0], 2))
    specs = [AttributeSpec("a", AttributeType.CONTINUOUS, bandwidth=1.0),
             AttributeSpec("b", AttributeType.CATEGORICAL)]
    m = STCLQ(attribute_specs=specs, use_attribute_weights=True, diagnostics=False)
    g = m.global_clq(coords, types, 1500.0, attributes=attrs)["gclq"]
    assert g.shape == (2, 2)
    # the attribute-conditional permutation path runs
    res = m.permutation_test(coords, types, g, 1500.0, attributes=attrs,
                             level="global", null="attribute", n_perm=19, seed=5)
    assert res["p_attract"].shape == (2, 2)


def test_csr_data_order_matches_edges():
    """The CSR data order must match the edge array, or an in-place W.data swap misaligns."""
    coords, types = _data(n=200)
    attrs = np.random.default_rng(7).normal(0, 1, (coords.shape[0], 2))
    specs = [AttributeSpec("a", AttributeType.CONTINUOUS, bandwidth=1.0),
             AttributeSpec("b", AttributeType.CATEGORICAL)]
    m = STCLQ(attribute_specs=specs, use_attribute_weights=True, diagnostics=False)
    eng = m._make_engine(coords, types, 1500.0, attrs, None)
    assert np.array_equal(eng._data_from_attributes(eng.attributes), eng.W.data)
    # still consistent after reset
    eng.reset_weights()
    assert np.array_equal(eng._data_from_attributes(eng.attributes), eng.W.data)


def test_compat_pystclq_returns_array():
    coords, types = _data(n=150)
    old = pySTCLQ()
    g = old.global_clq(coords, types, 1500.0)
    assert isinstance(g, np.ndarray) and g.shape == (2, 2)
    p = old.permutation_test(coords, types, g, 1500.0, n_perm=19)
    assert p.shape == (2, 2)


def test_compat_mcclq_keys():
    coords, types = _data(n=150)
    attrs = np.random.default_rng(2).normal(0, 1, (coords.shape[0], 1))
    specs = [AttributeSpec("x", AttributeType.CONTINUOUS, bandwidth=1.0)]
    model = MCCLQ(attribute_specs=specs, verbose=False)
    res = model.single_period_global_mcclq(coords, types, attrs, 1500.0)
    assert "mc_gclq" in res and res["mc_gclq"].shape == (2, 2)


# ---------------------------------------------------------------------------
# Regression tests for contributions 1–4 of the paper
# ---------------------------------------------------------------------------
def test_augmented_design_matrix_one_spmm():
    """Augmented column: one SpMM gives the numerator and the row sums, matching two passes."""
    from stclq import design_matrix
    coords, types = _data(n=200)
    m = STCLQ(diagnostics=False)
    eng = m._make_engine(coords, types, 1500.0, None, None)
    K = eng.K
    Y = design_matrix(eng.types, K)
    assert Y.shape == (eng.n, K + 1)
    assert np.allclose(Y[:, K], 1.0)
    Z = eng.spmm(Y)
    # the row sum from the augmented column == summing per type
    assert np.allclose(Z[:, K], Z[:, :K].sum(axis=1))
    # the global matrix from one SpMM matches the separately computed one exactly
    g_direct = eng.global_clq()
    Yk = Y[:, :K]
    pnw = Yk.T @ Z[:, :K]
    den = Yk.T @ Z[:, K]
    p_b = eng._p_b(K)
    g_manual = pnw / den[:, None] / p_b[None, :]
    assert np.allclose(g_direct, g_manual)


def test_invariance_of_csr_under_label_permutation():
    """Invariance theorem: a label permutation leaves the CSR structure and w_s·w_t unchanged."""
    coords, types = _data(n=300)
    m = STCLQ(diagnostics=False)
    eng = m._make_engine(coords, types, 1500.0, None, None)
    res = eng.invariance_check(n_draws=5, seed=1)
    assert res["unchanged"] and res["max_data_delta"] == 0.0


def test_local_counts_match_full_kernel():
    """Local counts: calling the counting function directly matches the permutation path."""
    from stclq.inference import permutation_test, local_counts
    coords, types = _data(n=220)
    m = STCLQ(diagnostics=False)
    eng = m._make_engine(coords, types, 1500.0, None, None)
    obs = eng.local_clq()
    plan = np.arange(11, dtype=np.int64)
    ge = np.zeros_like(obs)
    le = np.zeros_like(obs)
    local_counts(eng, plan, "type", obs, ge, le)
    r = permutation_test(eng, obs, level="local", null="type",
                         n_perm=len(plan), plan=plan, correction=False)
    assert np.array_equal(r["p_attract"], (ge + 1.0) / (len(plan) + 1.0))
    assert np.array_equal(r["p_repel"], (le + 1.0) / (len(plan) + 1.0))


def test_parallel_global_bitwise_identical():
    """Parallel permutation: results are **bit-for-bit identical** to serial, whatever the process count."""
    from stclq.inference import permutation_test
    coords, types = _data(n=400)
    m = STCLQ(diagnostics=False)
    eng = m._make_engine(coords, types, 1500.0, None, None)
    obs = eng.global_clq()
    plan = np.arange(5, 5 + 41, dtype=np.int64)
    serial = permutation_test(eng, obs, level="global", null="type",
                              n_perm=len(plan), plan=plan, correction=False,
                              n_jobs=1)
    par = permutation_test(eng, obs, level="global", null="type",
                           n_perm=len(plan), plan=plan, correction=False,
                           n_jobs=4)
    assert np.array_equal(serial["p_attract"], par["p_attract"])
    assert np.array_equal(serial["p_repel"], par["p_repel"])
    # do not depend on the _worth_parallel threshold: test the parallel entry point directly
    from stclq.parallel import parallel_global_sims
    from stclq.inference import global_sim
    ref = np.stack([global_sim(eng, s, "type") for s in plan])
    got = parallel_global_sims(eng, plan, "type", 4)
    assert np.array_equal(ref, got)


def test_parallel_local_bitwise_identical():
    from stclq.inference import permutation_test
    from stclq.parallel import parallel_local_counts
    coords, types = _data(n=200)
    m = STCLQ(diagnostics=False)
    eng = m._make_engine(coords, types, 1500.0, None, None)
    obs = eng.local_clq()
    plan = np.arange(100, 100 + 13, dtype=np.int64)
    a = permutation_test(eng, obs, level="local", null="type",
                         n_perm=len(plan), plan=plan, correction=False,
                         n_jobs=1)
    b = permutation_test(eng, obs, level="local", null="type",
                         n_perm=len(plan), plan=plan, correction=False,
                         n_jobs=3)
    assert np.array_equal(a["p_attract"], b["p_attract"])
    ge1 = np.zeros_like(obs); le1 = np.zeros_like(obs)
    ge2 = np.zeros_like(obs); le2 = np.zeros_like(obs)
    from stclq.inference import local_counts
    local_counts(eng, plan, "type", obs, ge1, le1)
    parallel_local_counts(eng, plan, "type", obs, ge2, le2, 3)
    assert np.array_equal(ge1, ge2) and np.array_equal(le1, le2)


def test_plan_seeds_reproducible():
    """The same plan recomputes to the same result; the output carries plan_seeds for audit."""
    from stclq.inference import permutation_test, make_plan
    coords, types = _data(n=250)
    m = STCLQ(diagnostics=False)
    eng = m._make_engine(coords, types, 1500.0, None, None)
    obs = eng.global_clq()
    plan = make_plan(7, 21)
    r1 = permutation_test(eng, obs, level="global", n_perm=21, plan=plan,
                          correction=False)
    r2 = permutation_test(eng, obs, level="global", n_perm=21, plan=plan,
                          correction=False)
    assert np.array_equal(r1["p_attract"], r2["p_attract"])
    assert np.array_equal(r1["plan_seeds"], plan)


def test_theoretical_bounds_and_rho_criterion():
    from stclq import theoretical_cost, feasible_scale_table, analytic_rho
    cost = theoretical_cost(n=10_000, nnz=1_500_000, n_perm=999, k=8)
    assert cost["speedup_ops"] > 50
    rows = feasible_scale_table([1_000, 100_000], side=20_000.0, rho_max=0.01)
    assert len(rows) == 2 and rows[0]["b_max_rho"] > 0
    # the analytic ρ is self-consistent with nnz = ρN²
    rho = analytic_rho(rows[0]["b_max_rho"], 20_000.0 ** 2)
    assert abs(rho - 0.01) < 1e-12


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"all {len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
