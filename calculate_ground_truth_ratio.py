"""Calculate the crack-pixel ratio of a CFD ground-truth mask.

The reported global ratio can be passed to Q_adaptive_patch_crack.py as
``--target_ratio`` when ``--target_ratio_mode fixed`` is used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_cfd_seg_mask(path: Path, crack_label: int = 1) -> np.ndarray:
    width = height = None
    data_started = False
    runs: list[tuple[int, int, int, int]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for raw_line in stream:
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
                data_started = True
                continue
            if data_started:
                parts = line.split()
                if len(parts) == 4:
                    runs.append(tuple(map(int, parts)))

    if width is None or height is None:
        raise ValueError(f"Cannot read width/height from {path}")

    mask = np.zeros((height, width), dtype=np.uint8)
    for label, y, x_start, x_end in runs:
        if label != crack_label or not 0 <= y < height:
            continue
        start = max(0, x_start)
        end = min(width - 1, x_end)
        if start <= end:
            mask[y, start : end + 1] = 1
    return mask


def load_raster_mask(path: Path, threshold: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    return (np.asarray(image, dtype=np.uint8) > threshold).astype(np.uint8)


def patch_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate crack ratio from a CFD .seg or raster GT mask."
    )
    parser.add_argument("gt_path", help="Path to a .seg file or binary mask image")
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert crack/background after loading the mask",
    )
    parser.add_argument(
        "--crack_label",
        type=int,
        default=1,
        help="Crack label in a CFD .seg file (default: 1)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="Raster values greater than this are crack pixels (default: 0)",
    )
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--min_ratio", type=float, default=0.005)
    parser.add_argument("--max_ratio", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.gt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Ground truth not found: {path}")
    if args.patch_size <= 0 or args.stride <= 0:
        raise ValueError("patch_size and stride must be positive")
    if args.stride > args.patch_size:
        raise ValueError("stride must not exceed patch_size")
    if not 0.0 <= args.min_ratio <= args.max_ratio <= 1.0:
        raise ValueError("Require 0 <= min_ratio <= max_ratio <= 1")

    if path.suffix.lower() == ".seg":
        mask = load_cfd_seg_mask(path, crack_label=args.crack_label)
    else:
        mask = load_raster_mask(path, threshold=args.threshold)
    if args.invert:
        mask = 1 - mask

    height, width = mask.shape
    crack_pixels = int(mask.sum())
    total_pixels = int(mask.size)
    global_ratio = crack_pixels / total_pixels
    effective_ratio = float(np.clip(global_ratio, args.min_ratio, args.max_ratio))

    ratios = []
    for y0 in patch_starts(height, args.patch_size, args.stride):
        for x0 in patch_starts(width, args.patch_size, args.stride):
            patch = mask[y0 : y0 + args.patch_size, x0 : x0 + args.patch_size]
            ratios.append(float(patch.mean()))
    patch_ratios = np.asarray(ratios, dtype=float)

    print(f"Ground truth : {path.resolve()}")
    print(f"Mask shape   : {height} x {width}")
    print(f"Crack pixels : {crack_pixels}")
    print(f"Total pixels : {total_pixels}")
    print(f"Global ratio : {global_ratio:.10f} ({global_ratio * 100:.6f}%)")
    print(f"Patch count  : {patch_ratios.size}")
    print(
        "Patch ratios : "
        f"min={patch_ratios.min():.10f}, "
        f"mean={patch_ratios.mean():.10f}, "
        f"median={np.median(patch_ratios):.10f}, "
        f"max={patch_ratios.max():.10f}"
    )
    print(f"Clipped ratio: {effective_ratio:.10f}")
    print()
    print("Use in Q_adaptive_patch_crack.py:")
    print(
        "  --target_ratio_mode fixed "
        f"--target_ratio {global_ratio:.10f} "
        f"--min_ratio {args.min_ratio:g} --max_ratio {args.max_ratio:g}"
    )
    if effective_ratio != global_ratio:
        print(
            "Note: the current min/max bounds change this to "
            f"{effective_ratio:.10f} during reconstruction."
        )


if __name__ == "__main__":
    main()
