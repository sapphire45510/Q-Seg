"""Shared coefficient-only interface for QUBO formulations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np

Edge = Tuple[int, int, float]
Quadratic = Dict[Tuple[int, int], float]


@dataclass
class Coefficients:
    """Coefficients owned by a formulation (the runner adds balance terms)."""

    linear: np.ndarray
    quadratic: Quadratic
    components: Dict[str, Tuple[np.ndarray, Quadratic]]
    diagnostics: Dict[str, float] = field(default_factory=dict)


class Formulation:
    name = "base"
    pairwise_factor = None
    formal_result = "raw"

    def prepare_features(self, gray: np.ndarray) -> Mapping[str, np.ndarray]:
        return {}

    def coefficients(
        self,
        shape: Tuple[int, int],
        edges: Iterable[Edge],
        features: Mapping[str, np.ndarray],
    ) -> Coefficients:
        raise NotImplementedError

    def config(self) -> Dict[str, object]:
        return {"formulation_module": f"{type(self).__module__}.{type(self).__name__}"}


def add_quadratic(target: Quadratic, i: int, j: int, value: float) -> None:
    key = (min(i, j), max(i, j))
    target[key] = target.get(key, 0.0) + float(value)


def evaluate(linear: np.ndarray, quadratic: Quadratic, sample: np.ndarray) -> float:
    result = float(np.dot(linear, sample))
    return result + sum(value * sample[i] * sample[j] for (i, j), value in quadratic.items())
