"""Legacy pairwise formulations, preserving their exact coefficient order."""
from __future__ import annotations

from typing import Iterable, Mapping, Tuple

import numpy as np

from .base import Coefficients, Edge, Formulation, add_quadratic


class StandardPairwise(Formulation):
    def __init__(self, name: str, pairwise_factor: float):
        self.name = name
        self.pairwise_factor = float(pairwise_factor)

    def coefficients(
        self,
        shape: Tuple[int, int],
        edges: Iterable[Edge],
        features: Mapping[str, np.ndarray],
    ) -> Coefficients:
        linear = np.zeros(shape[0] * shape[1], dtype=np.float64)
        quadratic = {}
        for i, j, weight in edges:
            linear[i] += weight
            linear[j] += weight
            add_quadratic(quadratic, i, j, self.pairwise_factor * weight)
        return Coefficients(
            linear, quadratic, {"base": (linear.copy(), dict(quadratic))}
        )

    def config(self):
        return {
            **super().config(),
            "base_pairwise_formulation": self.name,
            "quadratic_factor": self.pairwise_factor,
        }
