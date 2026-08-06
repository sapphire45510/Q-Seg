"""
paper_style_qseg_baseline.py

A paper-style Q-Seg baseline inspired by:
"Benefiting from Quantum? A Comparative Study of Q-Seg,
Quantum-Inspired Techniques, and U-Net for Crack Segmentation"

What this script intentionally follows
--------------------------------------
1. Grayscale 32x32 inputs.
2. One independent lattice graph per patch.
3. Four-neighbor spatial connectivity.
4. Edge weights based directly on squared grayscale differences:
       w_ij = (I_i - I_j)^2
5. Canonical weighted Max-Cut QUBO.
6. No target_ratio, no balance penalty, no Otsu prior.
7. No overlap voting, cosine weighting, or full-image patch alignment.
8. Simulated annealing (neal) as a classical substitute for the paper's
   D-Wave quantum annealer.
9. Standard IoU/F1 and an optional Boundary Proximity Metric (BPM).

Important limitations
---------------------
- The comparison paper does not provide its complete implementation or every
  QUBO coefficient convention. This script uses the canonical weighted
  Max-Cut minimization QUBO.
- Max-Cut labels are symmetric. For crack evaluation, the darker of the two
  output groups is assigned to label 1 (crack). This orientation rule is a
  practical convention and is not explicitly specified in the paper.
- The paper defines BPM using skeletonization and dilation radius r, but the
  exact r used in its table is not stated in the supplied text. This script
  therefore exposes --bpm_radius.
- CFD is not the paper's concrete-patch dataset. Use this script to compare
  pipelines, not to claim exact reproduction of the paper's reported numbers.

Dependencies
------------
pip install numpy pillow matplotlib pandas networkx dimod dwave-neal scipy scikit-image

Example: one CFD image, evaluate crack-containing 32x32 patches
----------------------------------------------------------------
python paper_style_qseg_baseline.py ^
  --image_path "CrackForest-dataset\\image\\001.jpg" ^
  --gt_path "CrackForest-dataset\\seg\\001.seg" ^
  --patch_size 32 ^
  --patch_selection positive ^
  --n_samples 200 ^
  --bpm_radius 2 ^
  --output_dir "paper_qseg_001"

Example: use exactly the U-Net test-image split
-----------------------------------------------
python paper_style_qseg_baseline.py ^
  --dataset_dir "CrackForest-dataset" ^
  --split_json "unet_results\\dataset_split.json" ^
  --split_name test ^
  --patch_size 32 ^
  --patch_selection positive ^
  --n_samples 200 ^
  --bpm_radius 2 ^
  --output_dir "paper_qseg_test"
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import dimod
import matplotlib.pyplot as plt
import neal
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.morphology import disk, skeletonize


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ImageMaskPair:
    image_path: Path
    gt_path: Path
    sample_id: str


def load_cfd_seg_mask(seg_path: str | Path) -> np.ndarray:
    """Read CFD .seg; label 1 is treated as crack."""
    seg_path = Path(seg_path)
    width: Optional[int] = None
    height: Optional[int] = None
    in_data = False
    runs: List[Tuple[int, int, int, int]] = []

    with seg_path.open("r", encoding="utf-8", errors="ignore") as file:
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
        raise ValueError(f"Cannot read width/height from {seg_path}")

    mask = np.zeros((height, width), dtype=np.uint8)

    for label, y, x_start, x_end in runs:
        if label != 1:
            continue
        if not 0 <= y < height:
            raise ValueError(f"Invalid y={y} in {seg_path}")
        x_start = max(0, x_start)
        x_end = min(width - 1, x_end)
        if x_start <= x_end:
            mask[y, x_start:x_end + 1] = 1

    return mask


def load_binary_mask(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".seg":
        return load_cfd_seg_mask(path)

    image = Image.open(path).convert("L")
    return (np.asarray(image, dtype=np.uint8) > 0).astype(np.uint8)


def load_gray_image(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0

def resize_image_and_mask(
    gray_image: np.ndarray,
    ground_truth: np.ndarray,
    target_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    將完整灰階影像與 Ground Truth 縮放成 target_size × target_size。

    灰階影像使用 BOX：
        適合縮小影像，會平均局部紋理。

    Ground Truth 使用 NEAREST：
        避免產生介於 0 和 1 之間的新標籤。
    """
    gray_u8 = np.clip(gray_image * 255.0, 0, 255).astype(np.uint8)
    gray_pil = Image.fromarray(gray_u8)

    resized_gray = gray_pil.resize(
        (target_size, target_size),
        resample=Image.Resampling.BOX,
    )

    gt_u8 = (ground_truth > 0).astype(np.uint8) * 255
    gt_pil = Image.fromarray(gt_u8)

    resized_gt = gt_pil.resize(
        (target_size, target_size),
        resample=Image.Resampling.NEAREST,
    )

    gray_array = (
        np.asarray(resized_gray, dtype=np.float32) / 255.0
    )

    gt_array = (
        np.asarray(resized_gt, dtype=np.uint8) > 0
    ).astype(np.uint8)

    return gray_array, gt_array

def discover_pairs(dataset_dir: str | Path) -> List[ImageMaskPair]:
    dataset_dir = Path(dataset_dir)
    image_dir = dataset_dir / "image"
    seg_dir = dataset_dir / "seg"

    pairs: List[ImageMaskPair] = []
    for extension in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        for image_path in sorted(image_dir.glob(extension)):
            gt_path = seg_dir / f"{image_path.stem}.seg"
            if gt_path.exists():
                pairs.append(
                    ImageMaskPair(
                        image_path=image_path,
                        gt_path=gt_path,
                        sample_id=image_path.stem,
                    )
                )

    if not pairs:
        raise RuntimeError(f"No CFD image/.seg pairs found in {dataset_dir}")

    return pairs


def pairs_from_split_json(
    split_json: str | Path,
    split_name: str,
) -> List[ImageMaskPair]:
    with Path(split_json).open("r", encoding="utf-8") as file:
        data = json.load(file)

    aliases = {
        "val": "validation",
        "validation": "validation",
        "train": "train",
        "test": "test",
    }
    key = aliases[split_name]

    if key not in data:
        raise KeyError(f"Split '{key}' not found in {split_json}")

    pairs: List[ImageMaskPair] = []
    for item in data[key]:
        pairs.append(
            ImageMaskPair(
                image_path=Path(item["image_path"]),
                gt_path=Path(item["mask_path"]),
                sample_id=str(item["sample_id"]),
            )
        )
    return pairs


# ---------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class PatchRecord:
    sample_id: str
    patch_id: str
    y0: int
    x0: int
    image: np.ndarray
    ground_truth: np.ndarray


def patch_starts(length: int, patch_size: int) -> List[int]:
    """
    Non-overlapping patches. Include a final boundary-aligned patch when the
    image dimension is not divisible by patch_size.
    """
    if length <= patch_size:
        return [0]

    starts = list(range(0, length - patch_size + 1, patch_size))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def extract_patches(
    image: np.ndarray,
    ground_truth: np.ndarray,
    sample_id: str,
    patch_size: int,
    selection: str,
    minimum_crack_pixels: int,
) -> List[PatchRecord]:
    if image.shape != ground_truth.shape:
        raise ValueError(
            f"Image shape {image.shape} != GT shape {ground_truth.shape}"
        )

    patches: List[PatchRecord] = []
    for y0 in patch_starts(image.shape[0], patch_size):
        for x0 in patch_starts(image.shape[1], patch_size):
            image_patch = image[y0:y0 + patch_size, x0:x0 + patch_size]
            gt_patch = ground_truth[y0:y0 + patch_size, x0:x0 + patch_size]

            if image_patch.shape != (patch_size, patch_size):
                continue

            crack_pixels = int(gt_patch.sum())
            if selection == "positive" and crack_pixels < minimum_crack_pixels:
                continue
            if selection == "negative" and crack_pixels >= minimum_crack_pixels:
                continue

            patch_id = f"{sample_id}_y{y0:03d}_x{x0:03d}"
            patches.append(
                PatchRecord(
                    sample_id=sample_id,
                    patch_id=patch_id,
                    y0=y0,
                    x0=x0,
                    image=image_patch,
                    ground_truth=gt_patch,
                )
            )

    return patches


# ---------------------------------------------------------------------
# Paper-style graph and Max-Cut QUBO
# ---------------------------------------------------------------------

def squared_difference_edges(gray_patch: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Four-neighbor lattice. Direct squared grayscale difference, no Gaussian,
    no sigma, and no per-patch min-max normalization.
    """
    height, width = gray_patch.shape
    edges: List[Tuple[int, int, float]] = []

    for y in range(height):
        for x in range(width):
            i = y * width + x

            if x + 1 < width:
                j = y * width + (x + 1)
                difference = float(gray_patch[y, x] - gray_patch[y, x + 1])
                edges.append((i, j, difference * difference))

            if y + 1 < height:
                j = (y + 1) * width + x
                difference = float(gray_patch[y, x] - gray_patch[y + 1, x])
                edges.append((i, j, difference * difference))

    return edges


def maxcut_bqm(
    number_of_nodes: int,
    edges: Sequence[Tuple[int, int, float]],
    edge_scale: float,
) -> dimod.BinaryQuadraticModel:
    """
    Weighted Max-Cut:
        maximize sum_(i,j) w_ij [x_i + x_j - 2 x_i x_j]

    Equivalent minimization QUBO:
        minimize sum_(i,j) [-w_ij x_i - w_ij x_j + 2w_ij x_i x_j]
    """
    linear = {i: 0.0 for i in range(number_of_nodes)}
    quadratic: Dict[Tuple[int, int], float] = {}

    for i, j, raw_weight in edges:
        weight = edge_scale * float(raw_weight)
        linear[i] -= weight
        linear[j] -= weight
        quadratic[(i, j)] = quadratic.get((i, j), 0.0) + 2.0 * weight

    return dimod.BinaryQuadraticModel(
        linear,
        quadratic,
        0.0,
        vartype=dimod.BINARY,
    )


def orient_darker_class_as_crack(
    raw_mask: np.ndarray,
    gray_patch: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """
    Max-Cut is label-symmetric. Assign the darker partition to crack=1.
    """
    one_pixels = raw_mask == 1
    zero_pixels = raw_mask == 0

    if not np.any(one_pixels) or not np.any(zero_pixels):
        return raw_mask.astype(np.uint8), False

    mean_one = float(gray_patch[one_pixels].mean())
    mean_zero = float(gray_patch[zero_pixels].mean())

    if mean_one > mean_zero:
        return (1 - raw_mask).astype(np.uint8), True

    return raw_mask.astype(np.uint8), False


@dataclass(frozen=True)
class SolveResult:
    raw_mask: np.ndarray
    oriented_mask: np.ndarray
    flipped: bool
    energy: float
    solve_seconds: float


def solve_patch(
    gray_patch: np.ndarray,
    n_samples: int,
    edge_scale: float,
    seed: Optional[int],
) -> SolveResult:
    height, width = gray_patch.shape
    edges = squared_difference_edges(gray_patch)
    bqm = maxcut_bqm(height * width, edges, edge_scale=edge_scale)

    sampler = neal.SimulatedAnnealingSampler()
    kwargs = {"num_reads": n_samples}
    if seed is not None:
        kwargs["seed"] = seed

    start = time.perf_counter()
    sampleset = sampler.sample(bqm, **kwargs)
    solve_seconds = time.perf_counter() - start

    best = sampleset.first
    raw = np.asarray(
        [int(best.sample[i]) for i in range(height * width)],
        dtype=np.uint8,
    ).reshape(height, width)

    oriented, flipped = orient_darker_class_as_crack(raw, gray_patch)

    return SolveResult(
        raw_mask=raw,
        oriented_mask=oriented,
        flipped=flipped,
        energy=float(best.energy),
        solve_seconds=solve_seconds,
    )


# ---------------------------------------------------------------------
# Standard and BPM metrics
# ---------------------------------------------------------------------

def confusion_counts(prediction: np.ndarray, ground_truth: np.ndarray) -> Dict[str, int]:
    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)

    return {
        "TP": int(np.sum(pred & gt)),
        "TN": int(np.sum(~pred & ~gt)),
        "FP": int(np.sum(pred & ~gt)),
        "FN": int(np.sum(~pred & gt)),
    }


def metrics_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    tp, tn, fp, fn = (
        counts["TP"],
        counts["TN"],
        counts["FP"],
        counts["FN"],
    )
    eps = 1e-12

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    return {
        **counts,
        "IoU": tp / (tp + fp + fn + eps),
        "F1": 2.0 * precision * recall / (precision + recall + eps),
        "Dice": 2.0 * tp / (2.0 * tp + fp + fn + eps),
        "Precision": precision,
        "Recall": recall,
        "Specificity": tn / (tn + fp + eps),
        "Pixel Accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
        "Predicted positive ratio": (tp + fp) / (tp + tn + fp + fn + eps),
        "Ground-truth positive ratio": (tp + fn) / (tp + tn + fp + fn + eps),
    }


def bpm_confusion_counts(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    radius: int,
) -> Dict[str, int]:
    """
    Symmetric tolerance evaluation inspired by the paper's BPM:
    - skeletonize prediction and GT
    - a predicted skeleton pixel is TP when within radius of GT skeleton
    - an unmatched predicted skeleton pixel is FP
    - an unmatched GT skeleton pixel is FN
    - TN fills the remaining patch pixels

    This is an operational interpretation of the paper's BPM description.
    """
    pred_skeleton = skeletonize(prediction.astype(bool))
    gt_skeleton = skeletonize(ground_truth.astype(bool))

    if radius < 0:
        raise ValueError("BPM radius must be non-negative.")

    footprint = disk(radius)
    dilated_gt = binary_dilation(gt_skeleton, structure=footprint)
    dilated_pred = binary_dilation(pred_skeleton, structure=footprint)

    tp = int(np.sum(pred_skeleton & dilated_gt))
    fp = int(np.sum(pred_skeleton & ~dilated_gt))
    fn = int(np.sum(gt_skeleton & ~dilated_pred))

    total_pixels = prediction.size
    tn = max(0, total_pixels - tp - fp - fn)

    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def save_binary_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def save_patch_figure(
    path: Path,
    gray_patch: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    for axis, title, array in zip(
        axes,
        ("Input", "Ground truth", "Paper-style Q-Seg"),
        (gray_patch, ground_truth, prediction),
    ):
        axis.imshow(array, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def save_full_image_figure(
    path: Path,
    gray_image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> None:
    """
    儲存完整影像、完整 GT 與重建後 Q-Seg mask。
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    panels = [
        ("Input image", gray_image),
        ("Ground truth", ground_truth),
        ("Paper-style Q-Seg", prediction),
    ]

    for axis, (title, array) in zip(axes, panels):
        axis.imshow(array, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return

    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    return float(array.mean()), float(array.std(ddof=0))


def summarize_patch_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    summary: Dict[str, object] = {"patch_id": "MEAN_STD", "patch_count": len(rows)}

    for metric in (
        "IoU",
        "F1",
        "Dice",
        "Precision",
        "Recall",
        "BPM IoU",
        "BPM F1",
        "solve_seconds",
    ):
        values = [float(row[metric]) for row in rows]
        mean, std = mean_std(values)
        summary[f"{metric} mean"] = mean
        summary[f"{metric} std"] = std

    return summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-style 32x32 Q-Seg baseline with squared-difference weights."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image_path", type=str)
    input_group.add_argument("--dataset_dir", type=str)

    parser.add_argument(
        "--gt_path",
        type=str,
        default="",
        help="Required with --image_path.",
    )
    parser.add_argument(
        "--split_json",
        type=str,
        default="",
        help="Optional U-Net dataset_split.json for identical image split.",
    )
    parser.add_argument(
        "--split_name",
        choices=["train", "val", "validation", "test"],
        default="test",
    )

    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument(
        "--patch_selection",
        choices=["all", "positive", "negative"],
        default="positive",
        help=(
            "positive approximates a curated crack-patch dataset. "
            "It uses GT only to select evaluation patches, not in Q-Seg."
        ),
    )
    parser.add_argument(
        "--minimum_crack_pixels",
        type=int,
        default=1,
    )

    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument(
        "--edge_scale",
        type=float,
        default=1.0,
        help="Global multiplier for squared-difference edge weights.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bpm_radius", type=int, default=2)
    parser.add_argument(
        "--max_patches",
        type=int,
        default=0,
        help="0 means all selected patches; useful for quick tests.",
    )
    parser.add_argument(
        "--save_all_figures",
        action="store_true",
        help="Save a three-panel figure for every evaluated patch.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="paper_style_qseg_results",
    )

    return parser.parse_args()


def resolve_pairs(args: argparse.Namespace) -> List[ImageMaskPair]:
    if args.image_path:
        if not args.gt_path:
            raise ValueError("--gt_path is required with --image_path.")
        return [
            ImageMaskPair(
                image_path=Path(args.image_path),
                gt_path=Path(args.gt_path),
                sample_id=Path(args.image_path).stem,
            )
        ]

    if args.split_json:
        return pairs_from_split_json(args.split_json, args.split_name)

    return discover_pairs(args.dataset_dir)


def main() -> None:
    args = parse_args()

    if args.patch_size != 32:
        print(
            "Warning: the comparison paper uses 32x32 patches; "
            f"you selected {args.patch_size}x{args.patch_size}."
        )

    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "masks"
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    pairs = resolve_pairs(args)
    all_patches: List[PatchRecord] = []

    # 儲存每張原始完整影像、GT 與之後的重建結果
    full_images: Dict[str, np.ndarray] = {}
    full_ground_truths: Dict[str, np.ndarray] = {}
    full_predictions: Dict[str, np.ndarray] = {}
    full_coverage: Dict[str, np.ndarray] = {}

    for pair in pairs:
        image = load_gray_image(pair.image_path)
        gt = load_binary_mask(pair.gt_path)

        # 將完整的 480×320 影像與 GT 縮成 32×32
        image, gt = resize_image_and_mask(
            gray_image=image,
            ground_truth=gt,
            target_size=32,
        )

        print(
            f"{pair.sample_id}: resized image shape={image.shape}, "
            f"GT shape={gt.shape}, GT crack pixels={int(gt.sum())}"
        )
        
        full_images[pair.sample_id] = image
        full_ground_truths[pair.sample_id] = gt

        full_predictions[pair.sample_id] = np.zeros(
            image.shape,
            dtype=np.uint8,
        )

        full_coverage[pair.sample_id] = np.zeros(
            image.shape,
            dtype=np.uint8,
        )

        patches = extract_patches(
            image=image,
            ground_truth=gt,
            sample_id=pair.sample_id,
            patch_size=args.patch_size,
            selection=args.patch_selection,
            minimum_crack_pixels=args.minimum_crack_pixels,
        )
        all_patches.extend(patches)

    if args.max_patches > 0:
        all_patches = all_patches[: args.max_patches]

    if not all_patches:
        raise RuntimeError("No patches matched the requested selection.")

    print(
        f"Images: {len(pairs)} | selected patches: {len(all_patches)} | "
        f"selection={args.patch_selection}"
    )
    print(
        "Method: squared differences + canonical Max-Cut QUBO + neal SA; "
        "no balance penalty and no target ratio."
    )

    patch_rows: List[Dict[str, object]] = []

    for index, patch in enumerate(all_patches, start=1):
        result = solve_patch(
            gray_patch=patch.image,
            n_samples=args.n_samples,
            edge_scale=args.edge_scale,
            seed=args.seed + index,
        )

        patch_height, patch_width = result.oriented_mask.shape

        y1 = patch.y0 + patch_height
        x1 = patch.x0 + patch_width

        full_predictions[patch.sample_id][
            patch.y0:y1,
            patch.x0:x1,
        ] = result.oriented_mask

        full_coverage[patch.sample_id][
            patch.y0:y1,
            patch.x0:x1,
        ] = 1

        standard = metrics_from_counts(
            confusion_counts(result.oriented_mask, patch.ground_truth)
        )
        bpm = metrics_from_counts(
            bpm_confusion_counts(
                result.oriented_mask,
                patch.ground_truth,
                radius=args.bpm_radius,
            )
        )

        row: Dict[str, object] = {
            "sample_id": patch.sample_id,
            "patch_id": patch.patch_id,
            "y0": patch.y0,
            "x0": patch.x0,
            "energy": result.energy,
            "solve_seconds": result.solve_seconds,
            "label_flipped_by_darkness": result.flipped,
            **standard,
            "BPM radius": args.bpm_radius,
            "BPM TP": bpm["TP"],
            "BPM TN": bpm["TN"],
            "BPM FP": bpm["FP"],
            "BPM FN": bpm["FN"],
            "BPM IoU": bpm["IoU"],
            "BPM F1": bpm["F1"],
            "BPM Precision": bpm["Precision"],
            "BPM Recall": bpm["Recall"],
        }
        patch_rows.append(row)

        save_binary_mask(
            mask_dir / f"{patch.patch_id}_prediction.png",
            result.oriented_mask,
        )
        save_binary_mask(
            mask_dir / f"{patch.patch_id}_ground_truth.png",
            patch.ground_truth,
        )

        if args.save_all_figures or index <= 20:
            save_patch_figure(
                figure_dir / f"{patch.patch_id}_comparison.png",
                patch.image,
                patch.ground_truth,
                result.oriented_mask,
            )

        print(
            f"[{index:4d}/{len(all_patches)}] {patch.patch_id} | "
            f"IoU={standard['IoU']:.4f}, F1={standard['F1']:.4f}, "
            f"BPM IoU={bpm['IoU']:.4f}, BPM F1={bpm['F1']:.4f}"
        )

    full_image_rows: List[Dict[str, object]] = []

    for sample_id in full_predictions:
        full_prediction = full_predictions[sample_id]
        full_gt = full_ground_truths[sample_id]
        full_image = full_images[sample_id]
        coverage = full_coverage[sample_id]

        coverage_ratio = float(np.mean(coverage))

        if coverage_ratio < 1.0:
            print(
                f"Warning: {sample_id} only has "
                f"{coverage_ratio:.2%} pixel coverage. "
                "Use --patch_selection all and do not use --max_patches "
                "for full-image evaluation."
            )

        standard_full = metrics_from_counts(
            confusion_counts(
                full_prediction,
                full_gt,
            )
        )

        bpm_full = metrics_from_counts(
            bpm_confusion_counts(
                full_prediction,
                full_gt,
                radius=args.bpm_radius,
            )
        )

        full_row: Dict[str, object] = {
            "sample_id": sample_id,
            "coverage_ratio": coverage_ratio,
            **standard_full,
            "BPM radius": args.bpm_radius,
            "BPM TP": bpm_full["TP"],
            "BPM TN": bpm_full["TN"],
            "BPM FP": bpm_full["FP"],
            "BPM FN": bpm_full["FN"],
            "BPM IoU": bpm_full["IoU"],
            "BPM F1": bpm_full["F1"],
            "BPM Precision": bpm_full["Precision"],
            "BPM Recall": bpm_full["Recall"],
        }

        full_image_rows.append(full_row)

        save_binary_mask(
            output_dir / f"{sample_id}_full_prediction.png",
            full_prediction,
        )

        save_binary_mask(
            output_dir / f"{sample_id}_full_ground_truth.png",
            full_gt,
        )

        save_full_image_figure(
            output_dir / f"{sample_id}_full_comparison.png",
            gray_image=full_image,
            ground_truth=full_gt,
            prediction=full_prediction,
        )

        print(
            f"\nFull image {sample_id} | "
            f"IoU={standard_full['IoU']:.4f}, "
            f"F1={standard_full['F1']:.4f}, "
            f"Precision={standard_full['Precision']:.4f}, "
            f"Recall={standard_full['Recall']:.4f}, "
            f"BPM IoU={bpm_full['IoU']:.4f}, "
            f"BPM F1={bpm_full['F1']:.4f}"
        )

    write_csv(
        output_dir / "full_image_metrics.csv",
        full_image_rows,
    )

    summary = summarize_patch_rows(patch_rows)
    write_csv(output_dir / "patch_metrics.csv", patch_rows)
    write_csv(output_dir / "summary.csv", [summary])

    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, ensure_ascii=False)

    print("\nSummary (mean ± population std across patches)")
    print(
        f"IoU: {summary['IoU mean']:.4f} ± {summary['IoU std']:.4f}\n"
        f"F1 : {summary['F1 mean']:.4f} ± {summary['F1 std']:.4f}\n"
        f"BPM IoU: {summary['BPM IoU mean']:.4f} ± "
        f"{summary['BPM IoU std']:.4f}\n"
        f"BPM F1 : {summary['BPM F1 mean']:.4f} ± "
        f"{summary['BPM F1 std']:.4f}"
    )
    print("Saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
