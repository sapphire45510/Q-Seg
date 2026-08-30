"""Controlled Q-Seg experiments with interchangeable QUBO formulations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import dimod
import matplotlib.pyplot as plt
import neal
import numpy as np
from PIL import Image

import Q_adaptive_patch_crack as common
from calculate_bpm_metrics import bpm_counts, metrics as counts_to_metrics
from qubo_formulations import NAMES, create
from qubo_formulations.base import Coefficients, evaluate

EXPERIMENT_SEEDS: Tuple[int, ...] = tuple(range(1234, 1244))
REPRESENTATIVE_SEED = 1235
FORMULATIONS = {"method2": -1.0, "mincut": -2.0, "frangi_directional": None}


def save_qseg_comparison(path: Path, gt: np.ndarray, outputs: Mapping[str, Mapping[str, np.ndarray]]):
    """Save the ground truth and three Q-Seg masks in a single row."""
    panels = [
        ("GT", gt),
        ("Q-Seg raw", outputs["raw"]["mask"]),
        ("Q-Seg overlap", outputs["overlap"]["mask"]),
        ("Q-Seg dark", outputs["dark"]["mask"]),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_qseg_only_comparison(path: Path, outputs: Mapping[str, Mapping[str, np.ndarray]]):
    """Save the three Q-Seg masks in a single row."""
    panels = [
        ("Q-Seg raw", outputs["raw"]["mask"]),
        ("Q-Seg overlap", outputs["overlap"]["mask"]),
        ("Q-Seg dark", outputs["dark"]["mask"]),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def canonical_edges(gray_patch: np.ndarray, sigma: float):
    edges = common.image_to_grid_graph_gray(gray_patch, sigma=sigma)
    return sorted((min(i, j), max(i, j), float(w)) for i, j, w in edges)


def add_balance(number_of_nodes: int, target_ratio: float, penalty: float):
    linear = np.full(number_of_nodes, penalty * (1.0 - 2.0 * target_ratio * number_of_nodes))
    quadratic = {}
    if penalty:
        for i in range(number_of_nodes):
            for j in range(i + 1, number_of_nodes):
                quadratic[(i, j)] = 2.0 * penalty
    return linear, quadratic


def build_canonical_bqm(number_of_nodes: int, coefficients: Coefficients,
                        target_ratio: float, balance_penalty: float):
    """Add the shared balance term, then insert all coefficients canonically."""
    if not 0.0 <= target_ratio <= 1.0:
        raise ValueError("target_ratio must be in [0, 1]")
    if balance_penalty < 0.0:
        raise ValueError("balance_penalty must be non-negative")
    balance_linear, balance_quadratic = add_balance(
        number_of_nodes, target_ratio, balance_penalty
    )
    linear = coefficients.linear.copy() + balance_linear
    quadratic = dict(coefficients.quadratic)
    for key, value in balance_quadratic.items():
        quadratic[key] = quadratic.get(key, 0.0) + value
    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
    for node in range(number_of_nodes):
        bqm.add_variable(node, float(linear[node]))
    for i, j in sorted(quadratic):
        bqm.add_interaction(i, j, float(quadratic[(i, j)]))
    return bqm, (balance_linear, balance_quadratic)


def bqm_hash(bqm):
    digest = hashlib.sha256()
    for node in sorted(bqm.variables):
        digest.update(f"L,{node},{bqm.linear[node]:.17g}\n".encode("ascii"))
    interactions = sorted((min(i, j), max(i, j), float(value))
                          for (i, j), value in bqm.quadratic.items())
    for i, j, value in interactions:
        digest.update(f"Q,{i},{j},{value:.17g}\n".encode("ascii"))
    return digest.hexdigest()


def solve_patch(gray_patch, formulation, feature_patch, target_ratio,
                balance_penalty, sigma, num_reads, solver_seed):
    shape = gray_patch.shape
    coefficients = formulation.coefficients(
        shape, canonical_edges(gray_patch, sigma), feature_patch
    )
    bqm, balance = build_canonical_bqm(
        shape[0] * shape[1], coefficients, target_ratio, balance_penalty
    )
    started = time.perf_counter()
    samples = neal.SimulatedAnnealingSampler().sample(
        bqm, num_reads=num_reads, seed=solver_seed
    )
    elapsed = time.perf_counter() - started
    best = samples.first
    vector_size = shape[0] * shape[1]
    vector = np.asarray([int(best.sample[node]) for node in range(vector_size)])
    raw = vector.astype(np.uint8).reshape(shape)
    energies = {}
    for name in ("base", "line", "shape"):
        component = coefficients.components.get(name, (np.zeros(vector_size), {}))
        energies[f"{name}_energy"] = evaluate(*component, vector)
    energies["balance_energy"] = evaluate(*balance, vector)
    energies["total_energy"] = float(best.energy)
    component_sum = sum(energies[f"{name}_energy"] for name in ("base", "balance", "line", "shape"))
    energies["energy_component_error"] = component_sum - energies["total_energy"]
    tolerance = 1e-8 * max(1.0, abs(energies["total_energy"]))
    if abs(energies["energy_component_error"]) > tolerance:
        raise RuntimeError("Energy components do not sum to the BQM energy")
    diagnostics = {"line_score_mean": 0.0, "line_score_max": 0.0,
                   "line_linear_min": 0.0, "line_linear_max": 0.0,
                   "shape_quadratic_min": 0.0, "shape_quadratic_max": 0.0,
                   "shape_quadratic_mean": 0.0, **coefficients.diagnostics}
    return raw, elapsed, bqm_hash(bqm), energies, diagnostics


def reconstruct(gray, otsu, args, experiment_seed, formulation, features):
    height, width = gray.shape
    y_starts = common.get_patch_starts(height, args.patch_size, args.stride)
    x_starts = common.get_patch_starts(width, args.patch_size, args.stride)
    methods = ("raw", "overlap", "dark")
    votes = {name: np.zeros((height, width), np.float32) for name in methods}
    weights = {name: np.zeros((height, width), np.float32) for name in methods}
    logs, patch_id = [], 0
    for y0 in y_starts:
        for x0 in x_starts:
            patch_id += 1
            y1, x1 = min(y0 + args.patch_size, height), min(x0 + args.patch_size, width)
            patch, otsu_patch = gray[y0:y1, x0:x1], otsu[y0:y1, x0:x1]
            feature_patch = {key: value[y0:y1, x0:x1] for key, value in features.items()}
            ratio = common.choose_target_ratio(args.target_ratio_mode, args.target_ratio,
                                               otsu_patch, args.min_ratio, args.max_ratio)
            solver_seed = experiment_seed + patch_id - 1
            raw, seconds, coefficient_hash, energies, diagnostics = solve_patch(
                patch, formulation, feature_patch, ratio, args.balance_penalty,
                args.sigma, args.n_samples, solver_seed
            )
            overlap, overlap_flipped = common.align_patch_label_with_overlap(
                raw, votes["overlap"], weights["overlap"], y0, y1, x0, x1)
            dark, dark_flipped = common.align_patch_label_with_darkness(raw, patch)
            patch_weight = (common.cosine_weight_2d(*raw.shape) if not args.no_center_weight
                            else np.ones_like(raw, dtype=np.float32))
            for name, mask in {"raw": raw, "overlap": overlap, "dark": dark}.items():
                votes[name][y0:y1, x0:x1] += mask * patch_weight
                weights[name][y0:y1, x0:x1] += patch_weight
            logs.append({"experiment_seed": experiment_seed, "patch_id": patch_id,
                         "solver_seed": solver_seed, "y0": y0, "y1": y1,
                         "x0": x0, "x1": x1, "target_ratio": ratio,
                         "raw_positive_ratio": float(raw.mean()),
                         "energy": energies["total_energy"], "solve_seconds": seconds,
                         "overlap_flipped": overlap_flipped, "dark_flipped": dark_flipped,
                         "bqm_sha256": coefficient_hash, **diagnostics, **energies})
    outputs = {}
    for name in methods:
        score = votes[name] / np.maximum(weights[name], 1e-12)
        outputs[name] = {"score_map": score, "mask": (score >= 0.5).astype(np.uint8)}
    return outputs, logs


def save_features(outdir: Path, gray_u8: np.ndarray, features: Mapping[str, np.ndarray]):
    if not features:
        return
    score, orientation = features["line_score"], features["orientation"]
    np.save(outdir / "frangi_line_score.npy", score)
    np.save(outdir / "hessian_orientation.npy", orientation)
    Image.fromarray(np.round(score * 255).astype(np.uint8)).save(outdir / "frangi_line_score.png")
    hue = orientation / np.pi
    hsv = np.stack((hue, score, score), axis=-1)
    import matplotlib.colors as colors
    rgb = np.round(colors.hsv_to_rgb(hsv) * 255).astype(np.uint8)
    Image.fromarray(rgb).save(outdir / "hessian_orientation_hsv.png")
    overlay = np.repeat(gray_u8[..., None], 3, axis=-1).astype(np.float64)
    overlay[..., 0] = np.maximum(overlay[..., 0], score * 255)
    overlay[..., 1:] *= (1.0 - 0.6 * score[..., None])
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(outdir / "frangi_overlay.png")


def write_csv(path, rows):
    if not rows: return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def summarize(rows):
    summary = []
    metrics = ["IoU", "Dice", "Precision", "Recall", "Specificity",
               "Pixel Accuracy", "Predicted positive ratio"]
    for method in ("raw", "overlap", "dark"):
        selected = [row for row in rows if row["Method"] == method]
        result = {"Method": method, "Result role": "formal" if method == "raw" else "diagnostic only",
                  "Number of seeds": len(selected)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected])
            result[f"{metric} mean"], result[f"{metric} std"] = float(values.mean()), float(values.std(ddof=1))
        summary.append(result)
    return summary


def summarize_bpm(rows):
    """Summarize the standard-to-BPM metric changes over all seeds."""
    summary = []
    metric_names = ("IoU", "Dice", "Precision", "Recall", "Specificity", "Pixel Accuracy")
    for method in ("raw", "overlap", "dark"):
        selected = [row for row in rows if row["Method"] == method]
        result = {"Method": method,
                  "Result role": "formal" if method == "raw" else "diagnostic only",
                  "Number of seeds": len(selected), "BPM radius": selected[0]["BPM radius"]}
        for metric in metric_names:
            standard = np.asarray([float(row[f"Standard {metric}"]) for row in selected])
            bpm = np.asarray([float(row[f"BPM {metric}"]) for row in selected])
            delta = bpm - standard
            result[f"Standard {metric} mean"] = float(standard.mean())
            result[f"BPM {metric} mean"] = float(bpm.mean())
            result[f"Delta {metric} mean"] = float(delta.mean())
        summary.append(result)
    return summary


def positive_floats(values):
    parsed = tuple(map(float, values))
    if not parsed or any(value <= 0 for value in parsed):
        raise argparse.ArgumentTypeError("values must be positive")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubo", choices=NAMES, required=True)
    parser.add_argument("--image_path", required=True); parser.add_argument("--gt_path", required=True)
    parser.add_argument("--target_size", type=int, default=0); parser.add_argument("--invert_gt", action="store_true")
    parser.add_argument("--patch_size", type=int, default=32); parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--median_size", type=int, default=0); parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--bpm_radius", type=int, default=3,
                        help="BPM tolerance radius in pixels (default: 3)")
    parser.add_argument("--target_ratio_mode", choices=("fixed", "otsu"), default="fixed")
    parser.add_argument("--target_ratio", type=float, default=0.3); parser.add_argument("--min_ratio", type=float, default=0.005)
    parser.add_argument("--max_ratio", type=float, default=0.3); parser.add_argument("--balance_penalty", type=float, default=0.01)
    parser.add_argument("--no_center_weight", action="store_true"); parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--lambda_line", type=float, default=0.0)
    parser.add_argument("--lambda_parallel", type=float, default=0.0)
    parser.add_argument("--lambda_perpendicular", type=float, default=0.0)
    parser.add_argument("--frangi_sigmas", nargs="+", type=float, default=(1.0, 2.0, 3.0))
    parser.add_argument("--orientation_sigma", type=float, default=1.5)
    parser.add_argument("--directional_base", choices=("method2", "mincut"),
                        default="mincut",
                        help="Pairwise base used only by frangi_directional")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stride > args.patch_size: raise ValueError("stride must not exceed patch_size")
    if args.bpm_radius < 0: raise ValueError("bpm_radius must be non-negative")
    formulation = create(args.qubo, args)
    target_size = None if args.target_size == 0 else args.target_size
    gray_u8, gray = common.load_full_image_gray(args.image_path, target_size, args.median_size)
    otsu, otsu_threshold = common.build_otsu_crack_prior(gray_u8)
    gt = common.load_ground_truth(args.gt_path, gray.shape, args.invert_gt)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = formulation.prepare_features(gray)  # exactly once, before the seed loop
    save_features(args.output_dir, gray_u8, features)
    metric_rows, bpm_rows, patch_rows, representative_rows = [], [], [], []
    started = time.perf_counter()
    for run_number, experiment_seed in enumerate(EXPERIMENT_SEEDS, 1):
        print(f"[{run_number}/10] experiment_seed={experiment_seed}, qubo={args.qubo}", flush=True)
        outputs, logs = reconstruct(gray, otsu, args, experiment_seed, formulation, features)
        patch_rows.extend(logs)
        for method, output in outputs.items():
            row = {"Experiment seed": experiment_seed, "Method": method,
                   "Result role": "formal" if method == "raw" else "diagnostic only",
                   **common.calculate_metrics(output["mask"], gt)}
            metric_rows.append(row)
            if experiment_seed == REPRESENTATIVE_SEED: representative_rows.append(row)
            bpm = counts_to_metrics(bpm_counts(output["mask"].astype(bool), gt.astype(bool),
                                               args.bpm_radius))
            bpm_row = {"Experiment seed": experiment_seed, "Method": method,
                       "Result role": row["Result role"], "BPM radius": args.bpm_radius}
            for metric in ("IoU", "Dice", "Precision", "Recall", "Specificity", "Pixel Accuracy"):
                bpm_row[f"Standard {metric}"] = row[metric]
                bpm_row[f"BPM {metric}"] = bpm[metric]
                bpm_row[f"Delta {metric}"] = bpm[metric] - row[metric]
            bpm_row.update({f"BPM {key}": bpm[key] for key in ("TP", "TN", "FP", "FN")})
            bpm_rows.append(bpm_row)
        seed_bpm_rows = [row for row in bpm_rows if row["Experiment seed"] == experiment_seed]
        seed_dir = args.output_dir / f"seed_{experiment_seed}"
        seed_dir.mkdir(exist_ok=True)
        write_csv(seed_dir / "bpm_metrics.csv", seed_bpm_rows)
        if experiment_seed == REPRESENTATIVE_SEED:
            common.save_outputs(str(seed_dir), gray_u8, otsu, outputs, gt)
            save_qseg_comparison(seed_dir / "comparison_qseg_methods.png", gt, outputs)
            save_qseg_only_comparison(seed_dir / "comparison_qseg_only.png", outputs)
            for method, output in outputs.items():
                np.save(seed_dir / f"qseg_{method}_mask.npy", output["mask"])
                np.save(seed_dir / f"qseg_{method}_score.npy", output["score_map"])
    summary_rows = summarize(metric_rows)
    bpm_summary_rows = summarize_bpm(bpm_rows)
    write_csv(args.output_dir / "metrics_all_seeds.csv", metric_rows)
    write_csv(args.output_dir / f"metrics_seed_{REPRESENTATIVE_SEED}.csv", representative_rows)
    write_csv(args.output_dir / "metrics_summary.csv", summary_rows)
    write_csv(args.output_dir / "bpm_metrics_all_seeds.csv", bpm_rows)
    write_csv(args.output_dir / "bpm_metrics_summary.csv", bpm_summary_rows)
    write_csv(args.output_dir / "patch_logs_all_seeds.csv", patch_rows)
    feature_ranges = ({key: [float(value.min()), float(value.max())] for key, value in features.items()} if features else {})
    config = {"qubo": args.qubo, "lambda_line": None, "lambda_parallel": None,
              "lambda_perpendicular": None, "frangi_sigmas": None,
              "orientation_sigma": None, "diagonal_factor": None,
              "feature_calculation_scope": None, "formal_evaluation_output": "raw",
              "overlap_dark_outputs": "diagnostic only",
              "bpm_radius": args.bpm_radius,
              **formulation.config(), "fixed_experiment_seeds": list(EXPERIMENT_SEEDS),
              "representative_seed": REPRESENTATIVE_SEED,
              "patch_seed_rule": "solver_seed = experiment_seed + one_based_patch_id - 1",
              "image_path": args.image_path, "gt_path": args.gt_path, "image_shape": list(gray.shape),
              "otsu_threshold": otsu_threshold, "patch_size": args.patch_size, "stride": args.stride,
              "sigma": args.sigma, "n_samples": args.n_samples, "target_ratio_mode": args.target_ratio_mode,
              "target_ratio": args.target_ratio, "min_ratio": args.min_ratio, "max_ratio": args.max_ratio,
              "balance_penalty": args.balance_penalty, "feature_coefficient_ranges": feature_ranges,
              "label_semantics": "x=1 crack, x=0 background", "elapsed_seconds": time.perf_counter() - started}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nseed={REPRESENTATIVE_SEED} metrics")
    for row in representative_rows: print(f"  {row['Method']:<7} [{row['Result role']}] IoU={row['IoU']:.6f} Dice={row['Dice']:.6f}")
    print("\n10-seed mean +/- sample std")
    for row in summary_rows: print(f"  {row['Method']:<7} [{row['Result role']}] IoU={row['IoU mean']:.6f}+/-{row['IoU std']:.6f}")
    print(f"\n10-seed standard -> BPM mean changes (radius={args.bpm_radius})")
    for row in bpm_summary_rows:
        print(f"  {row['Method']:<7} [{row['Result role']}] "
              f"IoU {row['Standard IoU mean']:.6f} -> {row['BPM IoU mean']:.6f} "
              f"({row['Delta IoU mean']:+.6f}); "
              f"Dice {row['Standard Dice mean']:.6f} -> {row['BPM Dice mean']:.6f} "
              f"({row['Delta Dice mean']:+.6f})")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__": main()
