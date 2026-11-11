"""Example 03: multi-period spatio-temporal CLQ (GTWCLQ), temporal weight in the statistic.

The same points carry a time label. With ``temporal_weight='inverse_dt'`` the global STCLQ
is computed under joint spatio-temporal weights, then compared with ignoring time.

    python examples/03_multi_period_gtwclq.py
"""

import numpy as np

from stclq import STCLQ


def make_data(n=300, n_periods=4, seed=2):
    rng = np.random.default_rng(seed)
    coords = rng.normal(0.0, 0.02, (n, 2)) + np.array([-0.10, 51.50])
    types = rng.integers(1, 3, size=n)              # two types
    periods = rng.integers(1, n_periods + 1, size=n)
    return coords, types, periods


def main():
    coords, types, periods = make_data()

    # (a) Ignore time: reduces to the classical spatial CLQ
    m_plain = STCLQ(temporal_weight="none", diagnostics=False)
    g_plain = m_plain.global_clq(coords, types, bandwidth=1500.0)["gclq"]

    # (b) Joint: pairs close in space and in period get a higher weight
    m_gt = STCLQ(temporal_weight="inverse_dt", diagnostics=False)
    g_gt = m_gt.global_clq(coords, types, bandwidth=1500.0,
                           time_periods=periods)["gclq"]

    print("Classical spatial CLQ:\n", np.round(g_plain, 3))
    print("GTWCLQ (inverse_dt):\n", np.round(g_gt, 3))

    # significance (label permutation; the temporal structure is held fixed)
    out = m_gt.run(coords, types, bandwidth=1500.0, time_periods=periods,
                   level="global", n_perm=99, seed=7)
    print("p_attract:\n", np.round(out["global_test"]["p_attract"], 4))
    print("n_periods =", out["n_periods"])


if __name__ == "__main__":
    main()
