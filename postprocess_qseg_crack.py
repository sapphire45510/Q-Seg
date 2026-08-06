from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_binary_mask(path: str) -> np.ndarray:
    """讀取二值 mask，回傳 0/1 uint8。"""
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.uint8)
    return (array > 127).astype(np.uint8)


def save_binary_mask(path: Path, mask: np.ndarray) -> None:
    """將 0/1 mask 儲存成黑白 PNG。"""
    output = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(output).save(path)


def remove_small_components(
    mask: np.ndarray,
    min_area: int = 10,
    connectivity: int = 8,
) -> np.ndarray:
    """
    刪除面積小於 min_area 的白色連通區域。

    mask:
        0 = background
        1 = crack / positive
    """
    binary = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=connectivity,
    )

    cleaned = np.zeros_like(binary)

    # label 0 是背景，因此從 1 開始
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area >= min_area:
            cleaned[labels == label_id] = 1

    return cleaned


def apply_closing(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """以 closing 連接小斷點並填補細小孔洞。"""
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size 必須是正奇數，例如 3 或 5。")

    binary = (mask > 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=iterations,
    )

    return (closed > 0).astype(np.uint8)


def load_ground_truth(path: str, target_shape: tuple[int, int]) -> np.ndarray:
    """
    讀取 PNG ground truth。
    若尺寸不同，以 nearest-neighbor 調整。
    """
    image = Image.open(path).convert("L")

    height, width = target_shape
    if image.size != (width, height):
        image = image.resize(
            (width, height),
            resample=Image.Resampling.NEAREST,
        )

    return (np.asarray(image, dtype=np.uint8) > 127).astype(np.uint8)


def calculate_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> Dict[str, float]:
    pred = (prediction > 0).astype(np.uint8)
    gt = (ground_truth > 0).astype(np.uint8)

    if pred.shape != gt.shape:
        raise ValueError(
            f"Prediction shape {pred.shape} "
            f"與 ground truth shape {gt.shape} 不一致。"
        )

    tp = int(np.sum((pred == 1) & (gt == 1)))
    tn = int(np.sum((pred == 0) & (gt == 0)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))

    eps = 1e-12

    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    pixel_accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "IoU": iou,
        "Dice": dice,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "Pixel Accuracy": pixel_accuracy,
        "Predicted positive ratio": float(np.mean(pred)),
        "Ground-truth positive ratio": float(np.mean(gt)),
    }


def save_comparison(
    output_path: Path,
    raw: np.ndarray,
    component_filtered: np.ndarray,
    final_mask: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    panels = [
        ("Q-Seg raw", raw),
        ("Remove small components", component_filtered),
        ("Components + closing", final_mask),
    ]

    for axis, (title, image) in zip(axes, panels):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_metrics_csv(
    output_path: Path,
    rows: list[Dict[str, object]],
) -> None:
    if not rows:
        return

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process an existing Q-Seg crack mask."
    )

    parser.add_argument(
        "--raw_mask",
        required=True,
        help="先前輸出的 qseg_raw_mask.png",
    )

    parser.add_argument(
        "--gt_path",
        default="",
        help="可選：ground-truth PNG，用於重新計算指標",
    )

    parser.add_argument(
        "--min_area",
        type=int,
        default=10,
        help="刪除面積小於此值的連通區域",
    )

    parser.add_argument(
        "--closing_kernel",
        type=int,
        default=3,
        help="Closing kernel 大小，必須為奇數",
    )

    parser.add_argument(
        "--closing_iterations",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output_dir",
        default="qseg_postprocess_results",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_binary_mask(args.raw_mask)

    component_filtered = remove_small_components(
        raw,
        min_area=args.min_area,
    )

    final_mask = apply_closing(
        component_filtered,
        kernel_size=args.closing_kernel,
        iterations=args.closing_iterations,
    )

    save_binary_mask(
        output_dir / "qseg_raw_copy.png",
        raw,
    )

    save_binary_mask(
        output_dir / "qseg_remove_small_components.png",
        component_filtered,
    )

    save_binary_mask(
        output_dir / "qseg_components_closing.png",
        final_mask,
    )

    save_comparison(
        output_dir / "postprocess_comparison.png",
        raw,
        component_filtered,
        final_mask,
    )

    if args.gt_path:
        gt = load_ground_truth(args.gt_path, raw.shape)

        rows = []

        for method_name, mask in [
            ("Q-Seg raw", raw),
            ("Q-Seg + component filtering", component_filtered),
            ("Q-Seg + component filtering + closing", final_mask),
        ]:
            row: Dict[str, object] = {
                "Method": method_name,
                **calculate_metrics(mask, gt),
            }
            rows.append(row)

            print(
                f"{method_name:<45} "
                f"IoU={row['IoU']:.4f} "
                f"Dice={row['Dice']:.4f} "
                f"Precision={row['Precision']:.4f} "
                f"Recall={row['Recall']:.4f}"
            )

        write_metrics_csv(
            output_dir / "postprocess_metrics.csv",
            rows,
        )

    print("Finished.")
    print("Results:", output_dir.resolve())


if __name__ == "__main__":
    main()