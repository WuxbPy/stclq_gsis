"""Example 04: two null hypotheses under attribute weights, label vs attribute permutation.

Shows how `null='type'` and `null='attribute'` differ: once attribute weights enter the
kernel, permuting the group labels alone is miscalibrated, so the attribute-conditional
permutation is required to keep the test nominal.

    python examples/04_calibrated_attribute_test.py
"""

import numpy as np

from stclq import STCLQ, AttributeSpec, AttributeType


def make_data(n=400, seed=3):
    rng = np.random.default_rng(seed)
    coords = rng.normal(0.0, 0.02, (n, 2)) + np.array([-0.10, 51.50])
    types = rng.integers(1, 3, size=n)
    severity = rng.normal(0.0, 1.0, n)              # continuous attribute
    area = rng.integers(0, 3, n)                    # categorical attribute
    attrs = np.column_stack([severity, area]).astype(np.float64)
    return coords, types, attrs


def main():
    coords, types, attrs = make_data()
    specs = [AttributeSpec("severity", AttributeType.CONTINUOUS, bandwidth=1.0),
             AttributeSpec("area", AttributeType.CATEGORICAL)]
    model = STCLQ(attribute_specs=specs, use_attribute_weights=True, diagnostics=False)

    g = model.global_clq(coords, types, bandwidth=1500.0, attributes=attrs)["gclq"]
    print("Attribute-weighted STCLQ:\n", np.round(g, 3))

    naive = model.permutation_test(coords, types, g, 1500.0, attributes=attrs,
                                   level="global", null="type", n_perm=199, seed=11)
    calib = model.permutation_test(coords, types, g, 1500.0, attributes=attrs,
                                   level="global", null="attribute", n_perm=199, seed=11)

    print("Label permutation p_attract:\n", np.round(naive["p_attract"], 4))
    print("Attribute-conditional permutation p_attract:\n", np.round(calib["p_attract"], 4))


if __name__ == "__main__":
    main()
