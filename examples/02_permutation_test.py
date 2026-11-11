"""Example 02: permutation significance test, multiple-testing correction, classification.

    python examples/02_permutation_test.py
"""

import numpy as np

from stclq import STCLQ


def make_data(n=400, seed=1):
    rng = np.random.default_rng(seed)
    c1 = rng.normal(0.0, 0.01, (n, 2))
    c2 = rng.normal(0.02, 0.01, (n, 2))
    coords = np.vstack([c1, c2]) + np.array([-0.10, 51.50])
    types = np.concatenate([np.ones(n, dtype=np.int64), 2 * np.ones(n, dtype=np.int64)])
    return coords, types


def main():
    coords, types = make_data()
    model = STCLQ(diagnostics=False)

    out = model.run(coords, types, bandwidth=1000.0,
                    level="global", null="type", n_perm=199, seed=42)
    test = out["global_test"]
    g = out["gclq"]

    print("Observed matrix:\n", np.round(g, 3))
    print("p_attract:\n", np.round(test["p_attract"], 4))
    print("p_attract_fdr:\n", np.round(test["p_attract_fdr"], 4))

    for a in range(g.shape[0]):
        for b in range(g.shape[1]):
            if a == b:
                continue
            cls = model.classify(g[a, b], test["p_attract_fdr"][a, b])
            print(f"Type {a+1} -> {b+1}: STCLQ={g[a,b]:.3f}  {cls['category']}")


if __name__ == "__main__":
    main()
