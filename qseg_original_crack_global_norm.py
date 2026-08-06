"""
qseg_original_crack.py

CrackForest adaptation of the ORIGINAL Q-Seg formulation.

Source-faithful core
--------------------
This script follows the original Q-Seg paper and public repository:

1. Each pixel is one node in a four-neighbor grid graph.
2. Raw edge dissimilarity:
       w'(i,j) = 1 - exp(-(I_i-I_j)^2 / (2*sigma^2))
3. Edge weights are min-max normalized to [-1, 1] and multiplied by -1:
       similar pixels    -> positive weights
       dissimilar pixels -> negative weights
4. Signed minimum-cut objective:
       E(x) = sum_(i,j) w_ij |x_i-x_j|
            = sum_(i,j) w_ij (x_i + x_j - 2 x_i x_j)
5. The QUBO is solved with neal.SimulatedAnnealingSampler as a local
   substitute for the D-Wave quantum annealer.
6. No target ratio, balance penalty, Otsu prior, unary prior, or morphology.

Crack-specific adaptation
-------------------------
- Reads CFD .seg ground truth.
- Supports a full image split into independent patches because a 480x320
  image is too large for a direct annealing experiment.
- Saves both:
    raw             : the solver's label orientation
    dark_oriented   : the darker partition is called crack
  The orientation rule is necessary because binary cuts are label-symmetric;
  it is not an original Q-Seg term.
- Reconstructs a complete image for non-overlapping or overlapping patches.

Important distinction
---------------------
The original Q-Seg paper DOES contain sigma in its Gaussian edge-weight
formula. What is not part of original Q-Seg is the later target-ratio /
balance-penalty machinery used in Q_adaptive_patch_crack.py.

Example
-------
python qseg_original_crack.py ^
  --image_path "CrackForest-dataset\\image\\001.jpg" ^
  --gt_path "CrackForest-dataset\\seg\\001.seg" ^
  --patch_size 32 ^
  --stride 32 ^
  --sigma 0.5 ^
  --n_samples 200 ^
  --output_dir "qseg_original_001"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import dimod
import matplotlib.pyplot as plt
import neal
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------

def load_gray_image(
    path: str | Path,
    resize_width: int = 0,
    resize_height: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
        gray_u8 : uint8 image
        gray    : float32 image in [0,1]
    """
    image = Image.open(path).convert("L")

    if (resize_width > 0) != (resize_height > 0):
        raise ValueError(
            "resize_width and resize_height must both be zero or both be positive."
        )

    if resize_width > 0 and resize_height > 0:
        image = image.resize(
            (resize_width, resize_height),
            resample=Image.Resampling.BOX,
        )

    gray_u8 = np.asarray(image, dtype=np.uint8)
    gray = gray_u8.astype(np.float32) / 255.0
    return gray_u8, gray


def load_cfd_seg_mask(path: str | Path) -> np.ndarray:
    """
    CFD .seg format after 'data':
        label y x_start x_end

    Label 1 is treated as crack, matching the existing project convention.
    """
    path = Path(path)

    width: Optional[int] = None
    height: Optional[int] = None
    in_data = False
    runs: List[Tuple[int, int, int, int]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("width "):
                width = int(line.split()[1])
                continue

            if line.startswith("height "):
                height = int(line.split()[1])
                continue

            if line == "data":
                in_data = True
                continue

            if not in_data:
                continue

            parts = line.split()
            if len(parts) == 4:
                runs.append(tuple(map(int, parts)))

    if width is None or height is None:
        raise ValueError(f"Cannot read width/height from {path}")

    mask = np.zeros((height, width), dtype=np.uint8)

    for label, y, x_start, x_end in runs:
        if label != 1:
            continue

        if not 0 <= y < height:
            raise ValueError(f"Invalid y={y} in {path}")

        x_start = max(0, x_start)
        x_end = min(width - 1, x_end)

        if x_start <= x_end:
            mask[y, x_start:x_end + 1] = 1

    return mask


def load_ground_truth(
    path: str | Path,
    target_shape: Tuple[int, int],
) -> np.ndarray:
    path = Path(path)

    if path.suffix.lower() == ".seg":
        mask = load_cfd_seg_mask(path)
    else:
        image = Image.open(path).convert("L")
        mask = (np.asarray(image, dtype=np.uint8) > 0).astype(np.uint8)

    if mask.shape != target_shape:
        target_height, target_width = target_shape
        image = Image.fromarray(mask * 255)
        image = image.resize(
            (target_width, target_height),
            resample=Image.Resampling.NEAREST,
        )
        mask = (np.asarray(image, dtype=np.uint8) > 0).astype(np.uint8)

    return mask


# ---------------------------------------------------------------------
# Original Q-Seg graph construction
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class WeightedEdge:
    node_i: int
    node_j: int
    weight: float


def gaussian_dissimilarity(
    value_a: float,
    value_b: float,
    sigma: float,
) -> float:
    """
    Original Q-Seg Eq. (2):
        1 - exp(-(a-b)^2 / (2 sigma^2))
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    difference = float(value_a) - float(value_b)
    return float(
        1.0
        - np.exp(
            -(difference * difference) / (2.0 * sigma * sigma)
        )
    )



def raw_gaussian_edge_range(
    gray_image: np.ndarray,
    sigma: float,
) -> Tuple[float, float]:
    """Compute one raw Gaussian-dissimilarity min/max over the full image."""
    height, width = gray_image.shape
    values: List[float] = []

    for y in range(height):
        for x in range(width):
            if y > 0:
                values.append(
                    gaussian_dissimilarity(
                        gray_image[y, x],
                        gray_image[y - 1, x],
                        sigma,
                    )
                )
            if x > 0:
                values.append(
                    gaussian_dissimilarity(
                        gray_image[y, x],
                        gray_image[y, x - 1],
                        sigma,
                    )
                )

    if not values:
        return 0.0, 0.0

    values_array = np.asarray(values, dtype=np.float64)
    return float(values_array.min()), float(values_array.max())


def original_qseg_edges(
    gray_patch: np.ndarray,
    sigma: float,
    normalization_min: Optional[float] = None,
    normalization_max: Optional[float] = None,
) -> List[WeightedEdge]:
    """
    Reproduce the public graph_utils.py convention:
    - up and left edges only, yielding one copy of each undirected edge
    - raw Gaussian dissimilarity
    - global min-max normalization per input patch
    - map to [-1,1], then multiply by -1

    Therefore:
      small dissimilarity -> approximately +1
      large dissimilarity -> approximately -1
    """
    height, width = gray_patch.shape
    raw_edges: List[Tuple[int, int, float]] = []

    for y in range(height):
        for x in range(width):
            node_i = y * width + x

            if y > 0:
                node_j = (y - 1) * width + x
                raw_edges.append(
                    (
                        node_i,
                        node_j,
                        gaussian_dissimilarity(
                            gray_patch[y, x],
                            gray_patch[y - 1, x],
                            sigma,
                        ),
                    )
                )

            if x > 0:
                node_j = y * width + (x - 1)
                raw_edges.append(
                    (
                        node_i,
                        node_j,
                        gaussian_dissimilarity(
                            gray_patch[y, x],
                            gray_patch[y, x - 1],
                            sigma,
                        ),
                    )
                )

    if not raw_edges:
        return []

    raw_values = np.asarray(
        [edge[2] for edge in raw_edges],
        dtype=np.float64,
    )

    if (normalization_min is None) != (normalization_max is None):
        raise ValueError(
            "normalization_min and normalization_max must be supplied together."
        )

    if normalization_min is None:
        minimum = float(raw_values.min())
        maximum = float(raw_values.max())
    else:
        minimum = float(normalization_min)
        maximum = float(normalization_max)

    normalized_edges: List[WeightedEdge] = []

    if maximum > minimum:
        for node_i, node_j, raw_weight in raw_edges:
            mapped = (
                2.0 * (raw_weight - minimum) / (maximum - minimum)
                - 1.0
            )
            signed_weight = -mapped
            normalized_edges.append(
                WeightedEdge(
                    node_i=node_i,
                    node_j=node_j,
                    weight=float(np.round(signed_weight, 4)),
                )
            )
    elif maximum == 0.0 and minimum == 0.0:
        # Exact behavior in the public repository.
        normalized_edges = [
            WeightedEdge(node_i, node_j, 1.0)
            for node_i, node_j, _ in raw_edges
        ]
    else:
        normalized_edges = [
            WeightedEdge(
                node_i,
                node_j,
                float(-np.round(raw_weight, 4)),
            )
            for node_i, node_j, raw_weight in raw_edges
        ]

    return normalized_edges


def original_mincut_bqm(
    number_of_nodes: int,
    edges: Sequence[WeightedEdge],
) -> dimod.BinaryQuadraticModel:
    """
    Signed minimum-cut QUBO:

        E = sum_(i,j) w_ij |x_i-x_j|
          = sum_(i,j) w_ij (x_i + x_j - 2 x_i x_j)

    Thus, for each edge:
        linear[i] += w
        linear[j] += w
        quadratic[i,j] += -2w

    This is the symmetric cut indicator. It matches the Qiskit Maxcut
    construction on the negated adjacency matrix used by the official
    tutorial, after converting its maximization objective to minimization.
    """
    linear = {node: 0.0 for node in range(number_of_nodes)}
    quadratic: Dict[Tuple[int, int], float] = {}

    for edge in edges:
        i, j, weight = edge.node_i, edge.node_j, edge.weight

        linear[i] += weight
        linear[j] += weight

        key = (min(i, j), max(i, j))
        quadratic[key] = quadratic.get(key, 0.0) - 2.0 * weight

    return dimod.BinaryQuadraticModel(
        linear,
        quadratic,
        0.0,
        dimod.BINARY,
    )


# ---------------------------------------------------------------------
# Solve and label orientation
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class PatchSolution:
    raw_mask: np.ndarray
    dark_oriented_mask: np.ndarray
    flipped: bool
    energy: float
    solve_seconds: float
    positive_ratio_raw: float
    positive_ratio_dark: float
    edge_min: float
    edge_max: float
    edge_mean: float


def orient_darker_partition_as_crack(
    raw_mask: np.ndarray,
    gray_patch: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """
    Cut labels are symmetric: x and 1-x describe the same partition.

    Crack-specific reporting convention:
      the partition with lower average grayscale intensity is crack=1.
    """
    zero_region = raw_mask == 0
    one_region = raw_mask == 1

    if not np.any(zero_region) or not np.any(one_region):
        return raw_mask.astype(np.uint8), False

    zero_mean = float(gray_patch[zero_region].mean())
    one_mean = float(gray_patch[one_region].mean())

    if one_mean <= zero_mean:
        return raw_mask.astype(np.uint8), False

    return (1 - raw_mask).astype(np.uint8), True


def solve_patch(
    gray_patch: np.ndarray,
    sigma: float,
    n_samples: int,
    seed: Optional[int],
    normalization_min: Optional[float] = None,
    normalization_max: Optional[float] = None,
) -> PatchSolution:
    height, width = gray_patch.shape
    edges = original_qseg_edges(
        gray_patch,
        sigma=sigma,
        normalization_min=normalization_min,
        normalization_max=normalization_max,
    )
    bqm = original_mincut_bqm(
        number_of_nodes=height * width,
        edges=edges,
    )

    sampler = neal.SimulatedAnnealingSampler()
    sampler_arguments: Dict[str, object] = {
        "num_reads": n_samples,
    }
    if seed is not None:
        sampler_arguments["seed"] = seed

    start = time.perf_counter()
    sample_set = sampler.sample(bqm, **sampler_arguments)
    solve_seconds = time.perf_counter() - start

    best = sample_set.first
    raw_mask = np.asarray(
        [int(best.sample[node]) for node in range(height * width)],
        dtype=np.uint8,
    ).reshape(height, width)

    dark_mask, flipped = orient_darker_partition_as_crack(
        raw_mask,
        gray_patch,
    )

    edge_values = np.asarray(
        [edge.weight for edge in edges],
        dtype=np.float64,
    )

    if edge_values.size:
        edge_min = float(edge_values.min())
        edge_max = float(edge_values.max())
        edge_mean = float(edge_values.mean())
    else:
        edge_min = edge_max = edge_mean = 0.0

    return PatchSolution(
        raw_mask=raw_mask,
        dark_oriented_mask=dark_mask,
        flipped=flipped,
        energy=float(best.energy),
        solve_seconds=solve_seconds,
        positive_ratio_raw=float(raw_mask.mean()),
        positive_ratio_dark=float(dark_mask.mean()),
        edge_min=edge_min,
        edge_max=edge_max,
        edge_mean=edge_mean,
    )


# ---------------------------------------------------------------------
# Patch extraction and reconstruction
# ---------------------------------------------------------------------

def patch_starts(
    length: int,
    patch_size: int,
    stride: int,
) -> List[int]:
    if length <= patch_size:
        return [0]

    starts = list(range(0, length - patch_size + 1, stride))
    final_start = length - patch_size

    if starts[-1] != final_start:
        starts.append(final_start)

    return starts


def center_weight_map(size: int) -> np.ndarray:
    """
    Smooth weight used only when patches overlap.
    It does not alter the QUBO; it only combines repeated predictions.
    """
    if size <= 1:
        return np.ones((size, size), dtype=np.float64)

    axis = np.hanning(size)
    weights = np.outer(axis, axis)
    weights = np.maximum(weights, 1e-3)
    return weights / weights.max()


@dataclass(frozen=True)
class ReconstructionResult:
    raw_mask: np.ndarray
    raw_score: np.ndarray
    dark_mask: np.ndarray
    dark_score: np.ndarray
    patch_rows: List[Dict[str, object]]
    total_solve_seconds: float


def reconstruct_image(
    gray_image: np.ndarray,
    patch_size: int,
    stride: int,
    sigma: float,
    n_samples: int,
    seed: int,
    normalization_min: Optional[float] = None,
    normalization_max: Optional[float] = None,
) -> ReconstructionResult:
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive.")
    if stride > patch_size:
        raise ValueError("stride must not exceed patch_size.")

    height, width = gray_image.shape

    if patch_size > height or patch_size > width:
        raise ValueError(
            f"patch_size={patch_size} exceeds image shape={gray_image.shape}"
        )

    y_starts = patch_starts(height, patch_size, stride)
    x_starts = patch_starts(width, patch_size, stride)

    raw_vote_sum = np.zeros((height, width), dtype=np.float64)
    dark_vote_sum = np.zeros((height, width), dtype=np.float64)
    weight_sum = np.zeros((height, width), dtype=np.float64)

    weights = (
        np.ones((patch_size, patch_size), dtype=np.float64)
        if stride == patch_size
        else center_weight_map(patch_size)
    )

    patch_rows: List[Dict[str, object]] = []
    total_solve_seconds = 0.0
    patch_index = 0

    for y0 in y_starts:
        for x0 in x_starts:
            patch = gray_image[
                y0:y0 + patch_size,
                x0:x0 + patch_size,
            ]

            patch_index += 1
            solution = solve_patch(
                gray_patch=patch,
                sigma=sigma,
                n_samples=n_samples,
                seed=seed + patch_index,
                normalization_min=normalization_min,
                normalization_max=normalization_max,
            )

            y1 = y0 + patch_size
            x1 = x0 + patch_size

            raw_vote_sum[y0:y1, x0:x1] += (
                solution.raw_mask * weights
            )
            dark_vote_sum[y0:y1, x0:x1] += (
                solution.dark_oriented_mask * weights
            )
            weight_sum[y0:y1, x0:x1] += weights

            total_solve_seconds += solution.solve_seconds

            patch_rows.append(
                {
                    "patch_index": patch_index,
                    "y0": y0,
                    "x0": x0,
                    "energy": solution.energy,
                    "solve_seconds": solution.solve_seconds,
                    "label_flipped_by_darkness": solution.flipped,
                    "raw_positive_ratio": solution.positive_ratio_raw,
                    "dark_positive_ratio": solution.positive_ratio_dark,
                    "edge_min": solution.edge_min,
                    "edge_max": solution.edge_max,
                    "edge_mean": solution.edge_mean,
                }
            )

    safe_weight_sum = np.maximum(weight_sum, 1e-12)
    raw_score = raw_vote_sum / safe_weight_sum
    dark_score = dark_vote_sum / safe_weight_sum

    raw_mask = (raw_score >= 0.5).astype(np.uint8)
    dark_mask = (dark_score >= 0.5).astype(np.uint8)

    return ReconstructionResult(
        raw_mask=raw_mask,
        raw_score=raw_score,
        dark_mask=dark_mask,
        dark_score=dark_score,
        patch_rows=patch_rows,
        total_solve_seconds=total_solve_seconds,
    )


# ---------------------------------------------------------------------
# Metrics and output
# ---------------------------------------------------------------------

def calculate_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> Dict[str, float]:
    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)

    tp = int(np.sum(pred & gt))
    tn = int(np.sum(~pred & ~gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))

    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "IoU": tp / (tp + fp + fn + eps),
        "Dice": 2.0 * tp / (2.0 * tp + fp + fn + eps),
        "F1": 2.0 * precision * recall / (precision + recall + eps),
        "Precision": precision,
        "Recall": recall,
        "Specificity": tn / (tn + fp + eps),
        "Pixel Accuracy": (
            (tp + tn) / (tp + tn + fp + fn + eps)
        ),
        "Predicted positive ratio": float(pred.mean()),
        "Ground-truth positive ratio": float(gt.mean()),
    }


def save_binary_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def save_score(path: Path, score: np.ndarray) -> None:
    image = np.clip(score * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def save_comparison(
    path: Path,
    gray_u8: np.ndarray,
    raw_mask: np.ndarray,
    dark_mask: np.ndarray,
    ground_truth: Optional[np.ndarray],
) -> None:
    panels: List[Tuple[str, np.ndarray]] = [
        ("Input", gray_u8),
        ("Q-Seg raw", raw_mask),
        ("Q-Seg dark-oriented", dark_mask),
    ]

    if ground_truth is not None:
        panels.append(("Ground truth", ground_truth))

    columns = 2
    rows = math.ceil(len(panels) / columns)

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, 5 * rows),
    )
    axes_array = np.asarray(axes).reshape(-1)

    for axis, (title, image) in zip(axes_array, panels):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255 if title == "Input" else 1)
        axis.set_title(title)
        axis.axis("off")

    for axis in axes_array[len(panels):]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_csv(
    path: Path,
    rows: Sequence[Dict[str, object]],
) -> None:
    if not rows:
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Original Q-Seg formulation adapted to CFD crack segmentation."
        )
    )

    parser.add_argument("--image_path", required=True)
    parser.add_argument("--gt_path", default="")

    parser.add_argument(
        "--resize_width",
        type=int,
        default=0,
        help="0 keeps original width.",
    )
    parser.add_argument(
        "--resize_height",
        type=int,
        default=0,
        help="0 keeps original height.",
    )

    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=32)

    parser.add_argument(
        "--sigma",
        type=float,
        default=0.5,
        help=(
            "Original public repository default is 0.5 "
            "for grayscale values normalized to [0,1]."
        ),
    )
    parser.add_argument(
        "--normalization_scope",
        choices=["patch", "global"],
        default="global",
        help=(
            "patch: each patch uses its own min/max; "
            "global: all patches share the full-image min/max."
        ),
    )
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output_dir",
        default="qseg_original_crack_results",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_total = time.perf_counter()

    gray_u8, gray = load_gray_image(
        args.image_path,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
    )

    ground_truth = (
        load_ground_truth(args.gt_path, gray.shape)
        if args.gt_path
        else None
    )

    if args.normalization_scope == "global":
        normalization_min, normalization_max = raw_gaussian_edge_range(
            gray,
            sigma=args.sigma,
        )
    else:
        normalization_min = None
        normalization_max = None

    print("Image shape:", gray.shape)
    print("Patch size:", args.patch_size)
    print("Stride:", args.stride)
    print("Sigma:", args.sigma)
    print("Normalization scope:", args.normalization_scope)
    if normalization_min is not None:
        print(
            "Global raw edge range:",
            f"min={normalization_min:.8f}, max={normalization_max:.8f}",
        )
    print("QUBO: signed minimum cut; no balance penalty or target ratio")

    reconstruction = reconstruct_image(
        gray_image=gray,
        patch_size=args.patch_size,
        stride=args.stride,
        sigma=args.sigma,
        n_samples=args.n_samples,
        seed=args.seed,
        normalization_min=normalization_min,
        normalization_max=normalization_max,
    )

    elapsed_total = time.perf_counter() - start_total

    Image.fromarray(gray_u8).save(output_dir / "input_gray.png")
    save_binary_mask(
        output_dir / "qseg_original_raw_mask.png",
        reconstruction.raw_mask,
    )
    save_binary_mask(
        output_dir / "qseg_original_dark_oriented_mask.png",
        reconstruction.dark_mask,
    )
    save_score(
        output_dir / "qseg_original_raw_score.png",
        reconstruction.raw_score,
    )
    save_score(
        output_dir / "qseg_original_dark_score.png",
        reconstruction.dark_score,
    )

    if ground_truth is not None:
        save_binary_mask(
            output_dir / "ground_truth.png",
            ground_truth,
        )

    save_comparison(
        output_dir / "comparison.png",
        gray_u8=gray_u8,
        raw_mask=reconstruction.raw_mask,
        dark_mask=reconstruction.dark_mask,
        ground_truth=ground_truth,
    )

    write_csv(
        output_dir / "patch_log.csv",
        reconstruction.patch_rows,
    )

    metric_rows: List[Dict[str, object]] = []

    if ground_truth is not None:
        for method_name, mask in (
            ("Q-Seg original raw", reconstruction.raw_mask),
            (
                "Q-Seg original dark-oriented",
                reconstruction.dark_mask,
            ),
        ):
            row: Dict[str, object] = {
                "Method": method_name,
                **calculate_metrics(mask, ground_truth),
            }
            metric_rows.append(row)

            print(
                f"{method_name:<34} "
                f"IoU={row['IoU']:.4f} "
                f"Dice={row['Dice']:.4f} "
                f"Precision={row['Precision']:.4f} "
                f"Recall={row['Recall']:.4f} "
                f"PredRatio={row['Predicted positive ratio']:.4f}"
            )

        write_csv(
            output_dir / "metrics.csv",
            metric_rows,
        )

    configuration = {
        **vars(args),
        "image_shape": list(gray.shape),
        "patch_count": len(reconstruction.patch_rows),
        "normalization_scope": args.normalization_scope,
        "normalization_min": normalization_min,
        "normalization_max": normalization_max,
        "total_solve_seconds": reconstruction.total_solve_seconds,
        "total_elapsed_seconds": elapsed_total,
        "formulation": (
            "Original Q-Seg Gaussian dissimilarity + signed [-1,1] "
            "normalization + signed minimum-cut QUBO"
        ),
        "added_crack_orientation_rule": (
            "Darker partition is reported as crack=1"
        ),
    }

    with (output_dir / "config.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("Patch count:", len(reconstruction.patch_rows))
    print("Total solver time:", reconstruction.total_solve_seconds)
    print("Total elapsed time:", elapsed_total)
    print("Results:", output_dir.resolve())


if __name__ == "__main__":
    main()
