"""Patch-based grayscale Q-Seg for CrackForest Dataset (CFD).

Outputs:
- Otsu baseline
- Q-Seg raw
- Q-Seg with overlap label alignment
- Q-Seg with dark-pixel label alignment
- metrics.csv when a ground-truth mask is supplied

Example (Windows cmd):
python run_qseg_patch_crack.py ^
  --image_path CrackForest-dataset\image\001.jpg ^
  --gt_path CrackForest-dataset\groundTruthPng\001.png ^
  --patch_size 32 --stride 16 --n_samples 200 ^
  --output_dir crack_results\001
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import dimod
import matplotlib.pyplot as plt
import neal
import networkx as nx
import numpy as np
from PIL import Image, ImageFilter


def load_gray(path: str, median_size: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    image = Image.open(path).convert("L")
    if median_size > 1:
        if median_size % 2 == 0:
            raise ValueError("median_size must be odd")
        image = image.filter(ImageFilter.MedianFilter(size=median_size))
    u8 = np.asarray(image, dtype=np.uint8)
    return u8, u8.astype(np.float32) / 255.0


def load_mask(path: str, shape: Tuple[int, int], invert: bool = False) -> np.ndarray:

    suffix = Path(path).suffix.lower()

    # ---------- CFD SEG ----------
    if suffix == ".seg":

        width = None
        height = None
        data_start = False

        with open(path, "r") as f:
            lines = f.readlines()

        for line in lines:

            line = line.strip()

            if line.startswith("width"):
                width = int(line.split()[1])
                continue

            if line.startswith("height"):
                height = int(line.split()[1])
                continue

            if line == "data":
                data_start = True
                break

        mask = np.zeros((height, width), dtype=np.uint8)

        start = lines.index("data\n") + 1

        for line in lines[start:]:

            line = line.strip()

            if line == "":
                continue

            label, y, x1, x2 = map(int, line.split())

            if label == 1:
                mask[y, x1:x2 + 1] = 1

    # ---------- PNG / JPG ----------
    else:

        image = Image.open(path).convert("L")

        if image.size != (shape[1], shape[0]):
            image = image.resize(
                (shape[1], shape[0]),
                Image.Resampling.NEAREST,
            )

        mask = (np.asarray(image) > 0).astype(np.uint8)

    if invert:
        mask = 1 - mask

    return mask


def otsu_crack_mask(gray_u8: np.ndarray) -> Tuple[np.ndarray, float]:
    threshold, mask = cv2.threshold(
        gray_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return (mask > 0).astype(np.uint8), float(threshold)


def gray_dissimilarity(a: float, b: float, sigma: float) -> float:
    d = float(a - b)
    return float(1.0 - np.exp(-(d * d) / (2.0 * sigma * sigma)))


def build_graph(patch: np.ndarray, sigma: float) -> nx.Graph:
    h, w = patch.shape
    graph = nx.Graph()
    graph.add_nodes_from(range(h * w))
    for y in range(h):
        for x in range(w):
            i = y * w + x
            if x + 1 < w:
                j = i + 1
                graph.add_edge(i, j, weight=gray_dissimilarity(patch[y, x], patch[y, x + 1], sigma))
            if y + 1 < h:
                j = i + w
                graph.add_edge(i, j, weight=gray_dissimilarity(patch[y, x], patch[y + 1, x], sigma))
    return graph


def maxcut_qubo(graph: nx.Graph) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float]]:
    """Minimize the negative weighted Max-Cut objective."""
    linear = {int(i): 0.0 for i in graph.nodes}
    quadratic: Dict[Tuple[int, int], float] = {}
    for i, j, data in graph.edges(data=True):
        weight = float(data["weight"])
        linear[int(i)] -= weight
        linear[int(j)] -= weight
        key = (int(min(i, j)), int(max(i, j)))
        quadratic[key] = quadratic.get(key, 0.0) + 2.0 * weight
    return linear, quadratic


def add_balance_penalty(
    linear: Dict[int, float],
    quadratic: Dict[Tuple[int, int], float],
    n: int,
    ratio: float,
    penalty: float,
) -> None:
    """Add penalty * (sum(x)-ratio*n)^2."""
    if penalty <= 0:
        return
    k = float(np.clip(ratio, 0.0, 1.0)) * n
    for i in range(n):
        linear[i] = linear.get(i, 0.0) + penalty * (1.0 - 2.0 * k)
    for i in range(n):
        for j in range(i + 1, n):
            quadratic[(i, j)] = quadratic.get((i, j), 0.0) + 2.0 * penalty


def solve_patch(
    patch: np.ndarray,
    sigma: float,
    n_samples: int,
    target_ratio: float,
    balance_penalty: float,
    seed: Optional[int],
) -> Tuple[np.ndarray, float, float]:
    h, w = patch.shape
    graph = build_graph(patch, sigma)
    linear, quadratic = maxcut_qubo(graph)
    add_balance_penalty(linear, quadratic, h * w, target_ratio, balance_penalty)
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)
    sampler = neal.SimulatedAnnealingSampler()
    kwargs = {"num_reads": n_samples}
    if seed is not None:
        kwargs["seed"] = seed
    start = time.time()
    result = sampler.sample(bqm, **kwargs)
    elapsed = time.time() - start
    best = result.first
    values = np.asarray([best.sample[i] for i in range(h * w)], dtype=np.uint8)
    return values.reshape(h, w), float(best.energy), elapsed


def patch_starts(length: int, patch_size: int, stride: int) -> List[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def center_weight(h: int, w: int) -> np.ndarray:
    wy = np.hanning(h) if h > 1 else np.ones(1)
    wx = np.hanning(w) if w > 1 else np.ones(1)
    return np.maximum(np.outer(wy, wx), 1e-3).astype(np.float32)


def align_overlap(mask, vote_sum, weight_sum, y0, y1, x0, x1):
    existing_weight = weight_sum[y0:y1, x0:x1]
    overlap = existing_weight > 1e-12
    if not np.any(overlap):
        return mask, False
    existing = vote_sum[y0:y1, x0:x1] / np.maximum(existing_weight, 1e-12)
    existing = (existing >= 0.5).astype(np.uint8)
    d0 = np.mean(mask[overlap] != existing[overlap])
    d1 = np.mean((1 - mask[overlap]) != existing[overlap])
    return (1 - mask, True) if d1 < d0 else (mask, False)


def align_dark(mask: np.ndarray, gray_patch: np.ndarray):
    ones, zeros = mask == 1, mask == 0
    if not np.any(ones) or not np.any(zeros):
        return mask, False
    mean_one = float(np.mean(gray_patch[ones]))
    mean_zero = float(np.mean(gray_patch[zeros]))
    return (1 - mask, True) if mean_one > mean_zero else (mask, False)


def reconstruct(
    gray: np.ndarray,
    otsu: np.ndarray,
    patch_size: int,
    stride: int,
    sigma: float,
    n_samples: int,
    target_ratio_mode: str,
    fixed_ratio: float,
    min_ratio: float,
    max_ratio: float,
    balance_penalty: float,
    seed: Optional[int],
):
    h, w = gray.shape
    methods = ("raw", "overlap", "dark")
    votes = {m: np.zeros((h, w), np.float32) for m in methods}
    weights = {m: np.zeros((h, w), np.float32) for m in methods}
    logs = []
    ys, xs = patch_starts(h, patch_size, stride), patch_starts(w, patch_size, stride)
    total = len(ys) * len(xs)
    pid = 0

    for y0 in ys:
        for x0 in xs:
            pid += 1
            y1, x1 = min(y0 + patch_size, h), min(x0 + patch_size, w)
            patch = gray[y0:y1, x0:x1]
            otsu_patch = otsu[y0:y1, x0:x1]
            ratio = fixed_ratio if target_ratio_mode == "fixed" else float(np.mean(otsu_patch))
            ratio = float(np.clip(ratio, min_ratio, max_ratio))
            print(f"[{pid}/{total}] y={y0}:{y1}, x={x0}:{x1}, ratio={ratio:.4f}")
            raw, energy, seconds = solve_patch(
                patch, sigma, n_samples, ratio, balance_penalty,
                None if seed is None else seed + pid,
            )
            overlap, overlap_flipped = align_overlap(
                raw, votes["overlap"], weights["overlap"], y0, y1, x0, x1
            )
            dark, dark_flipped = align_dark(raw, patch)
            masks = {"raw": raw, "overlap": overlap, "dark": dark}
            weight = center_weight(y1 - y0, x1 - x0)
            for method, mask in masks.items():
                votes[method][y0:y1, x0:x1] += mask * weight
                weights[method][y0:y1, x0:x1] += weight
            logs.append({
                "patch_id": pid, "y0": y0, "y1": y1, "x0": x0, "x1": x1,
                "target_ratio": ratio, "otsu_ratio": float(np.mean(otsu_patch)),
                "raw_ratio": float(np.mean(raw)), "energy": energy,
                "sampling_time": seconds, "overlap_flipped": overlap_flipped,
                "dark_flipped": dark_flipped,
            })

    outputs = {}
    for method in methods:
        score = votes[method] / np.maximum(weights[method], 1e-12)
        outputs[method] = {"score": score, "mask": (score >= 0.5).astype(np.uint8)}
    return outputs, logs


def metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    p, g = pred.astype(bool), gt.astype(bool)
    tp, tn = int(np.sum(p & g)), int(np.sum(~p & ~g))
    fp, fn = int(np.sum(p & ~g)), int(np.sum(~p & g))
    eps = 1e-12
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "IoU": tp / (tp + fp + fn + eps),
        "Dice": 2 * tp / (2 * tp + fp + fn + eps),
        "Precision": tp / (tp + fp + eps),
        "Recall": tp / (tp + fn + eps),
        "Specificity": tn / (tn + fp + eps),
        "Pixel Accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
        "Predicted positive ratio": float(np.mean(p)),
        "Ground-truth positive ratio": float(np.mean(g)),
    }


def write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_binary(path: Path, mask: np.ndarray):
    Image.fromarray((mask * 255).astype(np.uint8)).save(path)


def save_outputs(outdir: Path, gray_u8, otsu, outputs, gt=None):
    Image.fromarray(gray_u8).save(outdir / "input_gray.png")
    save_binary(outdir / "otsu_baseline.png", otsu)
    for name, data in outputs.items():
        save_binary(outdir / f"qseg_{name}_mask.png", data["mask"])
        plt.imsave(outdir / f"qseg_{name}_score.png", data["score"], cmap="gray", vmin=0, vmax=1)
    if gt is not None:
        save_binary(outdir / "ground_truth.png", gt)

    panels = [("Input", gray_u8), ("Otsu", otsu)]
    panels += [(f"Q-Seg {name}", data["mask"]) for name, data in outputs.items()]
    if gt is not None:
        panels.append(("Ground truth", gt))
    cols, rows = 3, int(np.ceil(len(panels) / 3))
    plt.figure(figsize=(12, 4 * rows))
    for i, (title, image) in enumerate(panels, 1):
        ax = plt.subplot(rows, cols, i)
        ax.imshow(image, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(outdir / "comparison.png", dpi=200, bbox_inches="tight")
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description="Patch-based grayscale Q-Seg for CFD")
    p.add_argument("--image_path", required=True)
    p.add_argument("--gt_path", default="")
    p.add_argument("--invert_gt", action="store_true")
    p.add_argument("--patch_size", type=int, default=32)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--median_size", type=int, default=0)
    p.add_argument("--sigma", type=float, default=0.10)
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--target_ratio_mode", choices=["fixed", "otsu"], default="otsu")
    p.add_argument("--target_ratio", type=float, default=0.05)
    p.add_argument("--min_ratio", type=float, default=0.01)
    p.add_argument("--max_ratio", type=float, default=0.30)
    p.add_argument("--balance_penalty", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output_dir", default="patch_qseg_crack_results")
    return p.parse_args()


def main():
    args = parse_args()
    if args.stride > args.patch_size:
        raise ValueError("stride must not exceed patch_size")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    gray_u8, gray = load_gray(args.image_path, args.median_size)
    otsu, threshold = otsu_crack_mask(gray_u8)
    gt = load_mask(args.gt_path, gray.shape, args.invert_gt) if args.gt_path else None
    print("Image shape:", gray.shape)
    print("Otsu threshold:", threshold, "Otsu ratio:", float(np.mean(otsu)))
    outputs, logs = reconstruct(
        gray, otsu, args.patch_size, args.stride, args.sigma, args.n_samples,
        args.target_ratio_mode, args.target_ratio, args.min_ratio, args.max_ratio,
        args.balance_penalty, args.seed,
    )
    save_outputs(outdir, gray_u8, otsu, outputs, gt)
    write_csv(outdir / "patch_log.csv", logs)
    if gt is not None:
        rows = [{"Method": "Otsu baseline", **metrics(otsu, gt)}]
        rows += [{"Method": f"Q-Seg {name}", **metrics(data["mask"], gt)}
                 for name, data in outputs.items()]
        write_csv(outdir / "metrics.csv", rows)
        for row in rows:
            print(f"{row['Method']:<20} IoU={row['IoU']:.4f} Dice={row['Dice']:.4f} "
                  f"Precision={row['Precision']:.4f} Recall={row['Recall']:.4f}")
    print(f"Finished in {time.time() - start:.2f} seconds")
    print("Results:", outdir.resolve())


if __name__ == "__main__":
    main()
