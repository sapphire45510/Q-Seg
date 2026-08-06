"""Patch-based grayscale Q-Seg for CrackForest Dataset (CFD).

Derived from Q_adaptive_patch.py. The original QUBO formulation, balance
penalty, simulated annealing solver, patch extraction, cosine weighting, and
overlap voting are preserved. Crack-specific changes are grayscale input,
CFD .seg ground truth, optional Otsu ratio estimation, and crack metrics.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import dimod
import matplotlib.pyplot as plt
import neal
import networkx as nx
import numpy as np
from PIL import Image, ImageFilter

try:
    from qseg.utils import decode_binary_string
except Exception:
    def decode_binary_string(binary_string, height, width):
        return np.asarray(binary_string, dtype=int).reshape((height, width))


# ===== Crack modification: grayscale feature =====
def gaussian_dissimilarity_gray(a: float, b: float, sigma: float = 0.10) -> float:
    d = float(a) - float(b)
    return float(1.0 - np.exp(-(d * d) / (2.0 * sigma * sigma)))

# 改點1
def gaussian_similarity_gray(a, b, sigma=0.10):

    d = a - b

    return np.exp(-(d*d)/(2*sigma*sigma))

def load_full_image_gray(image_path: str, target_size=None, median_size: int = 0):
    img = Image.open(image_path).convert("L")
    if median_size and median_size > 1:
        if median_size % 2 == 0:
            raise ValueError("median_size must be odd")
        img = img.filter(ImageFilter.MedianFilter(size=median_size))
    if target_size is not None:
        img = img.resize((target_size, target_size), resample=Image.Resampling.BOX)
    gray_u8 = np.asarray(img, dtype=np.uint8)
    gray = gray_u8.astype(np.float32) / 255.0
    return gray_u8, gray


def image_to_grid_graph_gray(gray_img: np.ndarray, sigma: float = 0.10):
    """Same edge normalization/sign convention as Q_adaptive_patch.py."""
    h, w = gray_img.shape
    raw_edges = []
    min_weight = float("inf")
    max_weight = float("-inf")

    for y in range(h):
        for x in range(w):
            i = y * w + x
            if y > 0:
                j = (y - 1) * w + x
                weight = gaussian_dissimilarity_gray(gray_img[y, x], gray_img[y - 1, x], sigma)
                raw_edges.append((i, j, weight))
                min_weight = min(min_weight, weight)
                max_weight = max(max_weight, weight)
            if x > 0:
                j = y * w + (x - 1)
                weight = gaussian_dissimilarity_gray(gray_img[y, x], gray_img[y, x - 1], sigma)
                raw_edges.append((i, j, weight))
                min_weight = min(min_weight, weight)
                max_weight = max(max_weight, weight)

    normalized_edges = []
    a, b = -1, 1
    for i, j, weight in raw_edges:
        if max_weight - min_weight > 1e-12:
            normalized = ((b - a) * ((weight - min_weight) / (max_weight - min_weight))) + a
            normalized = -1.0 * np.round(normalized, 4)
        elif max_weight == 0 and min_weight == 0:
            normalized = 1.0
        else:
            normalized = -1.0 * np.round(weight, 4)
        normalized_edges.append((i, j, float(normalized)))
    return normalized_edges


def build_otsu_crack_prior(gray_u8: np.ndarray):
    threshold, mask = cv2.threshold(
        gray_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return (mask > 0).astype(np.float32), float(threshold)


# ===== Preserved QUBO formulation =====
def get_linear_quadratic_dict(W):
    n = W.shape[0]
    linear = {}
    quadratic = {}
    for i in range(n):
        linear[i] = float(np.sum(W[i]))
        for j in range(i + 1, n):
            if W[i, j] != 0:
                quadratic[(i, j)] = float(-W[i, j])
    return linear, quadratic


def add_balance_penalty(linear, quadratic, n, target_ratio=0.5, penalty=0.01):
    k = target_ratio * n
    for i in range(n):
        linear[i] = linear.get(i, 0.0) + penalty * (1 - 2 * k)
    for i in range(n):
        for j in range(i + 1, n):
            quadratic[(i, j)] = quadratic.get((i, j), 0.0) + 2 * penalty
    return linear, quadratic


def simulated_annealer_solver(
    G, n_samples=2000, target_ratio=0.05, balance_penalty=0.01,
    seed: Optional[int] = None,
):
    start_time = time.time()
    W = nx.adjacency_matrix(G).todense()
    linear, quadratic = get_linear_quadratic_dict(W)
    n = W.shape[0]
    linear, quadratic = add_balance_penalty(
        linear, quadratic, n=n, target_ratio=target_ratio, penalty=balance_penalty
    )
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)
    problem_formulation_time = time.time() - start_time

    sampler = neal.SimulatedAnnealingSampler()
    kwargs = {"num_reads": n_samples}
    if seed is not None:
        kwargs["seed"] = seed
    start_time = time.time()
    sample_set = sampler.sample(bqm, **kwargs)
    response_time = time.time() - start_time

    info = {
        "problem_formulation_time": problem_formulation_time,
        "response_time": response_time,
        "num_variables": n,
        "num_quadratic_terms": len(quadratic),
        "target_ratio": target_ratio,
        "balance_penalty": balance_penalty,
    }
    return sample_set.to_pandas_dataframe(), info


def solve_one_patch_gray(
    gray_patch, n_samples=2000, target_ratio=0.05,
    balance_penalty=0.01, sigma=0.10, seed: Optional[int] = None,
):
    patch_h, patch_w = gray_patch.shape
    edge_list = image_to_grid_graph_gray(gray_patch, sigma=sigma)
    G = nx.Graph()
    G.add_nodes_from(range(patch_h * patch_w))
    G.add_weighted_edges_from(edge_list)
    samples_df, info = simulated_annealer_solver(
        G, n_samples=n_samples, target_ratio=target_ratio,
        balance_penalty=balance_penalty, seed=seed,
    )
    best_sample = samples_df.sort_values("energy").iloc[0]
    solution = best_sample.drop(
        labels=["energy", "num_occurrences", "chain_break_fraction"], errors="ignore"
    ).astype(int).to_numpy()
    mask = decode_binary_string(solution, patch_h, patch_w)
    return mask.astype(np.float32), info, float(best_sample["energy"])


def get_patch_starts(length, patch_size, stride):
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def cosine_weight_2d(h, w, eps=1e-3):
    wy = np.hanning(h) if h > 1 else np.ones(1)
    wx = np.hanning(w) if w > 1 else np.ones(1)
    return np.maximum(np.outer(wy, wx), eps).astype(np.float32)


def align_patch_label_with_overlap(mask, vote_sum, weight_sum, y0, y1, x0, x1):
    existing_weight = weight_sum[y0:y1, x0:x1]
    overlap = existing_weight > 1e-12
    if not np.any(overlap):
        return mask, False
    existing = vote_sum[y0:y1, x0:x1] / np.maximum(existing_weight, 1e-12)
    existing = (existing >= 0.5).astype(np.float32)
    d0 = np.mean(np.abs(mask[overlap] - existing[overlap]))
    d1 = np.mean(np.abs((1.0 - mask[overlap]) - existing[overlap]))
    return (1.0 - mask, True) if d1 < d0 else (mask, False)


def align_patch_label_with_darkness(mask, gray_patch):
    ones, zeros = mask == 1, mask == 0
    if not np.any(ones) or not np.any(zeros):
        return mask, False
    mean_one = float(np.mean(gray_patch[ones]))
    mean_zero = float(np.mean(gray_patch[zeros]))
    return (1.0 - mask, True) if mean_one > mean_zero else (mask, False)


def choose_target_ratio(mode, fixed_ratio, otsu_patch, min_ratio, max_ratio):
    ratio = fixed_ratio if mode == "fixed" else float(np.mean(otsu_patch))
    return float(np.clip(ratio, min_ratio, max_ratio))


def patch_qseg_reconstruct_gray(
    gray, otsu_prior, patch_size=32, stride=16, n_samples=2000,
    target_ratio_mode="fixed", target_ratio=0.05,
    min_ratio=0.005, max_ratio=0.30, balance_penalty=0.01,
    sigma=0.10, use_center_weight=True, seed: Optional[int] = 1234,
):
    H, W = gray.shape
    methods = ("raw", "overlap", "dark")
    votes = {m: np.zeros((H, W), np.float32) for m in methods}
    weights = {m: np.zeros((H, W), np.float32) for m in methods}
    ys = get_patch_starts(H, patch_size, stride)
    xs = get_patch_starts(W, patch_size, stride)
    total = len(ys) * len(xs)
    patch_infos = []
    pid = 0

    for y0 in ys:
        for x0 in xs:
            pid += 1
            y1, x1 = min(y0 + patch_size, H), min(x0 + patch_size, W)
            patch = gray[y0:y1, x0:x1]
            otsu_patch = otsu_prior[y0:y1, x0:x1]
            ratio = choose_target_ratio(
                target_ratio_mode, target_ratio, otsu_patch, min_ratio, max_ratio
            )
            print(f"Solving patch {pid}/{total}: y={y0}:{y1}, x={x0}:{x1}, ratio={ratio:.4f}")
            raw, info, energy = solve_one_patch_gray(
                patch, n_samples=n_samples, target_ratio=ratio,
                balance_penalty=balance_penalty, sigma=sigma,
                seed=None if seed is None else seed + pid,
            )
            overlap, overlap_flipped = align_patch_label_with_overlap(
                raw, votes["overlap"], weights["overlap"], y0, y1, x0, x1
            )
            dark, dark_flipped = align_patch_label_with_darkness(raw, patch)
            masks = {"raw": raw, "overlap": overlap, "dark": dark}
            weight = cosine_weight_2d(*raw.shape) if use_center_weight else np.ones_like(raw)
            for method, mask in masks.items():
                votes[method][y0:y1, x0:x1] += mask * weight
                weights[method][y0:y1, x0:x1] += weight
            patch_infos.append({
                "patch_id": pid, "y0": y0, "y1": y1, "x0": x0, "x1": x1,
                "energy": energy, "adaptive_target_ratio": ratio,
                "otsu_ratio_in_patch": float(np.mean(otsu_patch)),
                "raw_positive_ratio": float(np.mean(raw)),
                "overlap_flipped": overlap_flipped, "dark_flipped": dark_flipped,
                **info,
            })

    outputs = {}
    for method in methods:
        score = votes[method] / np.maximum(weights[method], 1e-12)
        outputs[method] = {"score_map": score, "mask": (score >= 0.5).astype(np.uint8)}
    return outputs, patch_infos


# ===== Crack modification: CFD .seg ground truth =====
def load_cfd_seg_mask(seg_path, expected_shape=None, crack_label=1, invert=False):
    width = height = None
    data_started = False
    runs = []
    with open(seg_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("width "):
                width = int(line.split()[1]); continue
            if line.startswith("height "):
                height = int(line.split()[1]); continue
            if line == "data":
                data_started = True; continue
            if not data_started:
                continue
            parts = line.split()
            if len(parts) == 4:
                runs.append(tuple(map(int, parts)))
    if width is None or height is None:
        raise ValueError(f"Cannot read width/height from {seg_path}")
    mask = np.zeros((height, width), dtype=np.uint8)
    for label, y, x_start, x_end in runs:
        if label == crack_label:
            mask[y, max(0, x_start):min(width - 1, x_end) + 1] = 1
    if invert:
        mask = 1 - mask
    if expected_shape is not None and mask.shape != expected_shape:
        raise ValueError(
            f"GT shape {mask.shape} != image shape {expected_shape}. "
            "Use --target_size 0 with CFD .seg files."
        )
    return mask


def load_ground_truth(path, shape, invert=False):
    if Path(path).suffix.lower() == ".seg":
        return load_cfd_seg_mask(path, shape, invert=invert)
    img = Image.open(path).convert("L")
    if img.size != (shape[1], shape[0]):
        img = img.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    mask = (np.asarray(img, dtype=np.uint8) > 0).astype(np.uint8)
    return 1 - mask if invert else mask


def calculate_metrics(prediction, ground_truth):
    p, g = prediction.astype(bool), ground_truth.astype(bool)
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


def save_binary(path, mask):
    Image.fromarray((mask * 255).astype(np.uint8)).save(path)


def make_boundary_overlay(gray_u8, mask):
    rgb = np.repeat(gray_u8[:, :, None], 3, axis=2)
    boundary = np.zeros_like(mask, dtype=bool)
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[1:, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    boundary[:, 1:] |= mask[:, :-1] != mask[:, 1:]
    rgb[boundary] = [255, 0, 0]
    return rgb


def write_csv(path, rows: List[Dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def save_outputs(outdir, gray_u8, otsu, outputs, gt=None):
    os.makedirs(outdir, exist_ok=True)
    Image.fromarray(gray_u8).save(os.path.join(outdir, "input_gray.png"))
    save_binary(os.path.join(outdir, "otsu_baseline_mask.png"), otsu.astype(np.uint8))
    for name, result in outputs.items():
        save_binary(os.path.join(outdir, f"qseg_{name}_mask.png"), result["mask"])
        plt.imsave(os.path.join(outdir, f"qseg_{name}_score.png"), result["score_map"], cmap="gray", vmin=0, vmax=1)
        Image.fromarray(make_boundary_overlay(gray_u8, result["mask"])).save(
            os.path.join(outdir, f"qseg_{name}_boundary_overlay.png")
        )
    if gt is not None:
        save_binary(os.path.join(outdir, "ground_truth.png"), gt)
    panels = [("Input", gray_u8), ("Otsu", otsu)] + [
        (f"Q-Seg {name}", result["mask"]) for name, result in outputs.items()
    ]
    if gt is not None:
        panels.append(("Ground truth", gt))
    cols, rows = 3, int(np.ceil(len(panels) / 3))
    plt.figure(figsize=(12, 4 * rows))
    for i, (title, image) in enumerate(panels, 1):
        ax = plt.subplot(rows, cols, i); ax.imshow(image, cmap="gray"); ax.set_title(title); ax.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "comparison.png"), dpi=200, bbox_inches="tight"); plt.close()


def save_config(args, image_shape, total_patches, otsu_threshold, elapsed):
    with open(os.path.join(args.output_dir, "config.txt"), "w", encoding="utf-8") as f:
        f.write("Patch-based Crack Q-Seg Configuration\n" + "=" * 50 + "\n")
        f.write(f"Time                : {datetime.now()}\n")
        f.write(f"Image               : {args.image_path}\n")
        f.write(f"Ground truth        : {args.gt_path}\n")
        f.write(f"Image shape         : {image_shape}\n")
        f.write(f"Patch size          : {args.patch_size}\n")
        f.write(f"Stride              : {args.stride}\n")
        f.write(f"Total patches       : {total_patches}\n")
        f.write(f"Sigma               : {args.sigma}\n")
        f.write(f"n_samples           : {args.n_samples}\n")
        f.write(f"Target ratio mode   : {args.target_ratio_mode}\n")
        f.write(f"Target ratio        : {args.target_ratio}\n")
        f.write(f"Balance penalty     : {args.balance_penalty}\n")
        f.write(f"Otsu threshold      : {otsu_threshold}\n")
        f.write(f"Execution time (s)  : {elapsed:.2f}\n")
        f.write("Label convention    : 1=crack, 0=background\n")


def parse_args():
    p = argparse.ArgumentParser(description="Patch-based grayscale Q-Seg for CFD")
    p.add_argument("--image_path", required=True)
    p.add_argument("--gt_path", default="")
    p.add_argument("--invert_gt", action="store_true")
    p.add_argument("--target_size", type=int, default=0, help="0 keeps original size; required for .seg GT")
    p.add_argument("--patch_size", type=int, default=32)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--median_size", type=int, default=0)
    p.add_argument("--sigma", type=float, default=0.10)
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--target_ratio_mode", choices=["fixed", "otsu"], default="fixed")
    p.add_argument("--target_ratio", type=float, default=0.05)
    p.add_argument("--min_ratio", type=float, default=0.005)
    p.add_argument("--max_ratio", type=float, default=0.30)
    p.add_argument("--balance_penalty", type=float, default=0.01)
    p.add_argument("--no_center_weight", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output_dir", default="patch_qseg_crack_results")
    return p.parse_args()


def main():
    args = parse_args()
    if args.stride > args.patch_size:
        raise ValueError("stride must not exceed patch_size")
    target_size = None if args.target_size == 0 else args.target_size
    start = time.time()
    gray_u8, gray = load_full_image_gray(args.image_path, target_size, args.median_size)
    otsu, otsu_threshold = build_otsu_crack_prior(gray_u8)
    gt = load_ground_truth(args.gt_path, gray.shape, args.invert_gt) if args.gt_path else None
    print("Image shape:", gray.shape)
    print("Otsu threshold:", otsu_threshold, "Otsu ratio:", float(np.mean(otsu)))

    outputs, logs = patch_qseg_reconstruct_gray(
        gray, otsu, patch_size=args.patch_size, stride=args.stride,
        n_samples=args.n_samples, target_ratio_mode=args.target_ratio_mode,
        target_ratio=args.target_ratio, min_ratio=args.min_ratio, max_ratio=args.max_ratio,
        balance_penalty=args.balance_penalty, sigma=args.sigma,
        use_center_weight=not args.no_center_weight, seed=args.seed,
    )
    elapsed = time.time() - start
    os.makedirs(args.output_dir, exist_ok=True)
    save_outputs(args.output_dir, gray_u8, otsu, outputs, gt)
    write_csv(os.path.join(args.output_dir, "patch_infos.csv"), logs)
    for name, result in outputs.items():
        np.save(os.path.join(args.output_dir, f"qseg_{name}_mask.npy"), result["mask"])
        np.save(os.path.join(args.output_dir, f"qseg_{name}_score.npy"), result["score_map"])

    if gt is not None:
        rows = [{"Method": "Otsu baseline", **calculate_metrics(otsu, gt)}]
        rows += [{"Method": f"Q-Seg {name}", **calculate_metrics(result["mask"], gt)} for name, result in outputs.items()]
        write_csv(os.path.join(args.output_dir, "metrics.csv"), rows)
        for row in rows:
            print(f"{row['Method']:<22} IoU={row['IoU']:.4f} Dice={row['Dice']:.4f} Precision={row['Precision']:.4f} Recall={row['Recall']:.4f}")

    save_config(args, gray.shape, len(logs), otsu_threshold, elapsed)
    print(f"Finished in {elapsed:.2f} seconds")
    print("Results:", Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
