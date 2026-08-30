"""Full-image Frangi/Hessian features and directional crack coefficients."""
from __future__ import annotations

import warnings
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import frangi

from .base import Coefficients, Edge, Formulation, add_quadratic

DIAGONAL_FACTOR = 1.0 / np.sqrt(2.0)


class FrangiDirectional(Formulation):
    name = "frangi_directional"
    pairwise_factor = -2.0

    def __init__(self, lambda_line: float, lambda_parallel: float,
                 lambda_perpendicular: float, frangi_sigmas: Sequence[float],
                 orientation_sigma: float, directional_base: str = "mincut"):
        values = (lambda_line, lambda_parallel, lambda_perpendicular)
        if any(value < 0 for value in values):
            raise ValueError("Frangi lambda values must be non-negative")
        if not frangi_sigmas or any(value <= 0 for value in frangi_sigmas):
            raise ValueError("frangi_sigmas must be a non-empty positive sequence")
        if orientation_sigma <= 0:
            raise ValueError("orientation_sigma must be positive")
        if directional_base not in ("method2", "mincut"):
            raise ValueError("directional_base must be 'method2' or 'mincut'")
        self.lambda_line, self.lambda_parallel, self.lambda_perpendicular = map(float, values)
        self.frangi_sigmas = tuple(float(value) for value in frangi_sigmas)
        self.orientation_sigma = float(orientation_sigma)
        self.directional_base = directional_base
        self.pairwise_factor = -1.0 if directional_base == "method2" else -2.0

    def prepare_features(self, gray: np.ndarray) -> Mapping[str, np.ndarray]:
        gray = np.asarray(gray, dtype=np.float64)
        if gray.ndim != 2 or not np.all(np.isfinite(gray)):
            raise ValueError("gray must be a finite 2-D image")
        if gray.min() < 0 or gray.max() > 1:
            raise ValueError("gray must be normalized to [0, 1]")
        # skimage 0.25: black_ridges=True detects dark ridges on a bright field.
        score = frangi(gray, sigmas=self.frangi_sigmas, black_ridges=True)
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        low, high = float(score.min()), float(score.max())
        score = (score - low) / (high - low) if high - low > 1e-12 else np.zeros_like(score)
        score = np.clip(score, 0.0, 1.0)
        if float(score.max()) < 1e-6 or float(np.mean(score)) < 1e-8:
            warnings.warn("Frangi line score is almost entirely zero", RuntimeWarning)

        sigma = self.orientation_sigma
        ixx = gaussian_filter(gray, sigma=sigma, order=(0, 2), mode="reflect")
        ixy = gaussian_filter(gray, sigma=sigma, order=(1, 1), mode="reflect")
        iyy = gaussian_filter(gray, sigma=sigma, order=(2, 0), mode="reflect")
        hessian = np.empty(gray.shape + (2, 2), dtype=np.float64)
        hessian[..., 0, 0], hessian[..., 0, 1] = ixx, ixy
        hessian[..., 1, 0], hessian[..., 1, 1] = ixy, iyy
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        index = np.argmin(np.abs(eigenvalues), axis=-1)
        tangent = np.take_along_axis(eigenvectors, index[..., None, None], axis=-1)[..., 0]
        # Components are (x, y), because the Hessian matrix above is ordered x,y.
        orientation = np.mod(np.arctan2(tangent[..., 1], tangent[..., 0]), np.pi)
        return {"line_score": score, "orientation": orientation}

    def coefficients(self, shape: Tuple[int, int], edges: Iterable[Edge],
                     features: Mapping[str, np.ndarray]) -> Coefficients:
        height, width = shape
        score = np.asarray(features["line_score"], dtype=np.float64)
        theta = np.asarray(features["orientation"], dtype=np.float64)
        if score.shape != shape or theta.shape != shape:
            raise ValueError("feature patch shape does not match image patch")

        base_linear = np.zeros(height * width, dtype=np.float64)
        base_quadratic = {}
        for i, j, weight in edges:
            base_linear[i] += weight
            base_linear[j] += weight
            add_quadratic(base_quadratic, i, j, self.pairwise_factor * weight)
        line_linear = (-self.lambda_line * score).reshape(-1)
        shape_quadratic = {}
        for y in range(height):
            for x in range(width):
                i = y * width + x
                for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    yy, xx = y + dy, x + dx
                    if not (0 <= yy < height and 0 <= xx < width):
                        continue
                    j = yy * width + xx
                    phi = np.arctan2(dy, dx)
                    alignment = 0.5 * (
                        np.cos(theta[y, x] - phi) ** 2
                        + np.cos(theta[yy, xx] - phi) ** 2
                    )
                    confidence = score[y, x] * score[yy, xx]
                    value = confidence * (
                        self.lambda_perpendicular * (1.0 - alignment)
                        - self.lambda_parallel * alignment
                    )
                    if dy and dx:
                        value *= DIAGONAL_FACTOR
                    # Do not materialize zero interactions: all-zero directional
                    # weights must hash exactly like the legacy mincut BQM.
                    if value != 0.0:
                        add_quadratic(shape_quadratic, i, j, value)
        linear = base_linear + line_linear
        quadratic = dict(base_quadratic)
        for (i, j), value in shape_quadratic.items():
            add_quadratic(quadratic, i, j, value)
        shape_values = np.asarray(list(shape_quadratic.values()) or [0.0])
        return Coefficients(linear, quadratic, {
            "base": (base_linear, base_quadratic),
            "line": (line_linear, {}),
            "shape": (np.zeros_like(linear), shape_quadratic),
        }, {
            "line_score_mean": float(score.mean()), "line_score_max": float(score.max()),
            "line_linear_min": float(line_linear.min()), "line_linear_max": float(line_linear.max()),
            "shape_quadratic_min": float(shape_values.min()),
            "shape_quadratic_max": float(shape_values.max()),
            "shape_quadratic_mean": float(shape_values.mean()),
        })

    def config(self):
        return {**super().config(), "lambda_line": self.lambda_line,
                "lambda_parallel": self.lambda_parallel,
                "lambda_perpendicular": self.lambda_perpendicular,
                "frangi_sigmas": list(self.frangi_sigmas),
                "orientation_sigma": self.orientation_sigma,
                "diagonal_factor": float(DIAGONAL_FACTOR),
                "feature_calculation_scope": "full image",
                "formal_evaluation_output": "raw",
                "overlap_dark_outputs": "diagnostic only; may flip class semantics",
                "base_pairwise_formulation": self.directional_base,
                "directional_base": self.directional_base,
                "quadratic_factor": self.pairwise_factor}
