"""Example 01: the shortest path, build data and compute global and local STCLQ.

    python examples/01_quickstart.py
"""

import numpy as np

from stclq import STCLQ


def make_data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    c1 = rng.normal(0.0, 0.01, (n, 2))
    c2 = rng.normal(0.01, 0.01, (n, 2))
    coords = np.vstack([c1, c2]) + np.array([-0.10, 51.50])
    types = np.concatenate([np.ones(n, dtype=np.int64), 2 * np.ones(n, dtype=np.int64)])
    return coords, types


def main():
    coords, types = make_data()
    model = STCLQ(diagnostics=False)

    res = model.global_clq(coords, types, bandwidth=1000.0)
    print("Global STCLQ matrix (K, K):")
    print(np.round(res["gclq"], 3))
    print("type_names:", res["type_names"])
    print("rho =", round(res["rho"], 4))

    loc = model.local_clq(coords, types, bandwidth=1000.0)
    print("Local STCLQ shape:", loc["lclq"].shape)


if __name__ == "__main__":
    main()
