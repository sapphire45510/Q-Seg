"""Evaluate completed segmentation masks with the paper's BPM procedure.

BPM skeletonizes the prediction and ground truth, then tolerates skeleton
pixels that lie within a disk of radius r around the other skeleton.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.morphology import disk, skeletonize


def load_cfd_seg(path: Path, crack_label: int = 1) -> np.ndarray:
    width = height = None
    in_data = False
    runs: list[tuple[int, int, int, int]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("width "):
                width = int(line.split()[1])
            elif line.startswith("height "):
                height = int(line.split()[1])
            elif line == "data":
                in_data = True
            elif in_data:
                parts = line.split()
                if len(parts) == 4:
                    runs.append(tuple(map(int, parts)))

    if width is None or height is None:
        raise ValueError(f"Cannot read width/height from {path}")

    mask = np.zeros((height, width), dtype=bool)
    for label, y, x_start, x_end in runs:
        if label == crack_label and 0 <= y < height:
            start, end = max(0, x_start), min(width - 1, x_end)
            if start <= end:
                mask[y, start : end + 1] = True
    return mask


def load_mask(
    path: Path,
    threshold: float,
    crack_label: int,
    target_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Mask not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".seg":
        mask = load_cfd_seg(path, crack_label)
    elif suffix == ".npy":
        array = np.load(path)
        array = np.squeeze(array)
        if array.ndim != 2:
            raise ValueError(f"Mask must be 2D after squeeze, got {array.shape}: {path}")
        mask = array > threshold
    else:
        array = np.asarray(Image.open(path).convert("L"))
        mask = array > threshold

    if target_shape is not None and mask.shape != target_shape:
        resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (target_shape[1], target_shape[0]), Image.Resampling.NEAREST
        )
        mask = np.asarray(resized) > 0
    return mask.astype(bool)


def standard_counts(prediction: np.ndarray, ground_truth: np.ndarray) -> Dict[str, int]:
    return {
        "TP": int(np.sum(prediction & ground_truth)),
        "TN": int(np.sum(~prediction & ~ground_truth)),
        "FP": int(np.sum(prediction & ~ground_truth)),
        "FN": int(np.sum(~prediction & ground_truth)),
    }


def bpm_counts(
    prediction: np.ndarray, ground_truth: np.ndarray, radius: int
) -> Dict[str, int]:
    """Recalculate confusion counts using Eq. (3)-(4) of the paper."""
    if radius < 0:
        raise ValueError("BPM radius must be non-negative")

    pred_skeleton = skeletonize(prediction)
    gt_skeleton = skeletonize(ground_truth)
    footprint = disk(radius)
    gt_tolerance = binary_dilation(gt_skeleton, structure=footprint)
    pred_tolerance = binary_dilation(pred_skeleton, structure=footprint)

    # A predicted skeleton point is correct if it is close to the GT skeleton.
    tp = int(np.sum(pred_skeleton & gt_tolerance))
    fp = int(np.sum(pred_skeleton & ~gt_tolerance))
    # GT skeleton points with no nearby prediction remain false negatives.
    fn = int(np.sum(gt_skeleton & ~pred_tolerance))
    tn = max(0, prediction.size - tp - fp - fn)
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def metrics(counts: Dict[str, int]) -> Dict[str, float | int]:
    tp, tn, fp, fn = (counts[key] for key in ("TP", "TN", "FP", "FN"))

    def divide(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    return {
        **counts,
        "IoU": divide(tp, tp + fp + fn),
        "Dice": divide(2 * tp, 2 * tp + fp + fn),
        "Precision": divide(tp, tp + fp),
        "Recall": divide(tp, tp + fn),
        "Specificity": divide(tn, tn + fp),
        "Pixel Accuracy": divide(tp + tn, tp + tn + fp + fn),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate standard and Boundary Proximity Metric results."
    )
    parser.add_argument("--gt", required=True, help="Ground truth (.seg, .npy, or image)")
    parser.add_argument(
        "--prediction",
        required=True,
        action="append",
        help="Completed Q-Seg mask; repeat this option to evaluate multiple masks",
    )
    parser.add_argument(
        "--radius", type=int, default=2, help="BPM disk radius r in pixels (default: 2)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0,
        help="Values greater than this are crack pixels (default: 0)",
    )
    parser.add_argument("--crack_label", type=int, default=1)
    parser.add_argument("--invert_gt", action="store_true")
    parser.add_argument("--invert_prediction", action="store_true")
    parser.add_argument(
        "--resize_prediction",
        action="store_true",
        help="Nearest-neighbor resize predictions to the GT shape",
    )
    parser.add_argument("--output_csv", default="bpm_metrics.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.radius < 0:
        raise ValueError("BPM radius must be non-negative")

    gt_path = Path(args.gt)
    ground_truth = load_mask(gt_path, args.threshold, args.crack_label)
    if args.invert_gt:
        ground_truth = ~ground_truth

    rows = []
    for prediction_name in args.prediction:
        prediction_path = Path(prediction_name)
        target_shape = ground_truth.shape if args.resize_prediction else None
        prediction = load_mask(
            prediction_path, args.threshold, args.crack_label, target_shape
        )
        if args.invert_prediction:
            prediction = ~prediction
        if prediction.shape != ground_truth.shape:
            raise ValueError(
                f"Shape mismatch for {prediction_path}: prediction={prediction.shape}, "
                f"ground_truth={ground_truth.shape}. Use --resize_prediction only if "
                "nearest-neighbor resizing is appropriate."
            )

        standard = metrics(standard_counts(prediction, ground_truth))
        bpm = metrics(bpm_counts(prediction, ground_truth, args.radius))
        row: Dict[str, object] = {
            "Prediction": str(prediction_path),
            "Ground truth": str(gt_path),
            "BPM radius": args.radius,
        }
        row.update({f"Standard {key}": value for key, value in standard.items()})
        row.update({f"BPM {key}": value for key, value in bpm.items()})
        rows.append(row)

        print(f"\nPrediction: {prediction_path}")
        print(f"Ground truth: {gt_path}")
        print(f"BPM radius: {args.radius} pixels")
        print(
            f"Standard  IoU={standard['IoU']:.6f}  Dice={standard['Dice']:.6f}  "
            f"Precision={standard['Precision']:.6f}  Recall={standard['Recall']:.6f}"
        )
        print(
            f"With BPM  IoU={bpm['IoU']:.6f}  Dice={bpm['Dice']:.6f}  "
            f"Precision={bpm['Precision']:.6f}  Recall={bpm['Recall']:.6f}"
        )
        print(
            f"BPM counts: TP={bpm['TP']} TN={bpm['TN']} "
            f"FP={bpm['FP']} FN={bpm['FN']}"
        )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
