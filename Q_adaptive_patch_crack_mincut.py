"""Run method 2 with the quadratic coefficient corrected to ``-2 * Wij``.

All preprocessing, reconstruction, label-orientation, metrics, and command-line
arguments come from :mod:`Q_adaptive_patch_crack`.  The only intentional change
is the pairwise coefficient convention used to construct the dimod BQM.

For each edge, this module constructs

    Wij * (xi + xj - 2 * xi * xj) = Wij * |xi - xj|,

which is the signed minimum-cut QUBO when each dimod interaction ``(i, j)`` is
stored once.
"""

from __future__ import annotations

import numpy as np

import Q_adaptive_patch_crack as method2


def get_linear_quadratic_dict_mincut(W):
    """Build the single-counted dimod coefficients for signed min-cut."""
    n = W.shape[0]
    linear = {}
    quadratic = {}
    for i in range(n):
        linear[i] = float(np.sum(W[i]))
        for j in range(i + 1, n):
            if W[i, j] != 0:
                quadratic[(i, j)] = float(-2.0 * W[i, j])
    return linear, quadratic


def main() -> None:
    # The original solver resolves this function through its module globals,
    # so replacing it here leaves every other method-2 operation unchanged.
    method2.get_linear_quadratic_dict = get_linear_quadratic_dict_mincut
    method2.main()


if __name__ == "__main__":
    main()
