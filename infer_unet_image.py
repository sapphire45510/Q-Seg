"""Run a trained CFD UNet checkpoint on one image."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train_unet_cfd import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    width = int(checkpoint.get("resize_width", 480))
    height = int(checkpoint.get("resize_height", 320))

    model = UNet(int(checkpoint.get("base_channels", 32))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    original = Image.open(args.input).convert("RGB")
    resized = original.resize((width, height), Image.Resampling.BILINEAR)
    gray = resized.convert("L")
    array = np.asarray(gray, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        probability_tensor = torch.sigmoid(model(tensor))
        probability = probability_tensor[0, 0].cpu().numpy()
    mask = probability >= args.threshold

    # Evaluate at the input image's original resolution. Resize probabilities
    # first and threshold afterward, avoiding interpolation of a binary mask.
    original_width, original_height = original.size
    probability_original = torch.nn.functional.interpolate(
        probability_tensor,
        size=(original_height, original_width),
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()
    mask_original = probability_original >= args.threshold

    args.output_dir.mkdir(parents=True, exist_ok=True)
    resized.save(args.output_dir / "input_resized_480x320.png")
    Image.fromarray(np.uint8(probability * 255), mode="L").save(
        args.output_dir / "unet_probability.png"
    )
    Image.fromarray(np.uint8(mask) * 255, mode="L").save(
        args.output_dir / "unet_mask.png"
    )

    rgb = np.asarray(resized, dtype=np.float32)
    red = np.zeros_like(rgb)
    red[..., 0] = 255
    overlay = rgb.copy()
    overlay[mask] = 0.55 * rgb[mask] + 0.45 * red[mask]
    Image.fromarray(np.uint8(np.clip(overlay, 0, 255))).save(
        args.output_dir / "unet_overlay.png"
    )
    np.save(args.output_dir / "unet_probability.npy", probability)
    Image.fromarray(np.uint8(probability_original * 255), mode="L").save(
        args.output_dir / "unet_probability_original_size.png"
    )
    Image.fromarray(np.uint8(mask_original) * 255, mode="L").save(
        args.output_dir / "unet_mask_original_size.png"
    )
    np.save(args.output_dir / "unet_probability_original_size.npy", probability_original)

    metrics = None
    if args.ground_truth:
        gt_image = Image.open(args.ground_truth).convert("L")
        if gt_image.size != original.size:
            gt_image = gt_image.resize(original.size, Image.Resampling.NEAREST)
        ground_truth = np.asarray(gt_image, dtype=np.uint8) > 0

        tp = int(np.sum(mask_original & ground_truth))
        tn = int(np.sum(~mask_original & ~ground_truth))
        fp = int(np.sum(mask_original & ~ground_truth))
        fn = int(np.sum(~mask_original & ground_truth))

        def divide(numerator: float, denominator: float) -> float:
            return numerator / denominator if denominator else float("nan")

        metrics = {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "IoU": divide(tp, tp + fp + fn),
            "Dice": divide(2 * tp, 2 * tp + fp + fn),
            "Precision": divide(tp, tp + fp),
            "Recall": divide(tp, tp + fn),
            "Specificity": divide(tn, tn + fp),
            "Accuracy": divide(tp + tn, tp + tn + fp + fn),
        }
        with (args.output_dir / "metrics_original_size.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=metrics.keys())
            writer.writeheader()
            writer.writerow(metrics)

    print(f"device={device}")
    print(f"checkpoint_epoch={checkpoint.get('epoch', 'unknown')}")
    print(f"checkpoint_best_val_dice={checkpoint.get('best_val_dice', 'unknown')}")
    print(f"size={width}x{height}")
    print(f"threshold={args.threshold}")
    print(f"foreground_pixels={int(mask.sum())}/{mask.size} ({mask.mean():.4%})")
    print(
        "original_size_foreground_pixels="
        f"{int(mask_original.sum())}/{mask_original.size} ({mask_original.mean():.4%})"
    )
    if metrics:
        print("confusion_matrix=" + ", ".join(f"{k}={metrics[k]}" for k in ("TP", "TN", "FP", "FN")))
        print("metrics=" + ", ".join(f"{k}={metrics[k]:.6f}" for k in ("IoU", "Dice", "Precision", "Recall", "Specificity", "Accuracy")))
    print(f"output_dir={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
