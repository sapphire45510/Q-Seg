import os
import time
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import dimod
import neal
from PIL import Image, ImageFilter
from matplotlib.colors import rgb_to_hsv, ListedColormap
from datetime import datetime
try:
    from qseg.utils import decode_binary_string
except Exception:
    def decode_binary_string(binary_string, height, width):
        return np.asarray(binary_string, dtype=int).reshape((height, width))


def hsv_distance2(a, b, h_weight=2.0, s_weight=1.0, v_weight=0.5):
    dh = abs(a[0] - b[0])
    dh = min(dh, 1.0 - dh)
    ds = a[1] - b[1]
    dv = a[2] - b[2]
    return h_weight * dh * dh + s_weight * ds * ds + v_weight * dv * dv


def gaussian_dissimilarity_hsv(a, b, sigma=0.25):
    d2 = hsv_distance2(a, b)
    return 1.0 - np.exp(-d2 / (2.0 * sigma * sigma))


def load_full_image_hsv(image_path, target_size=None, median_size=5):
    """
    Load original image, optionally resize to target_size, then convert RGB -> HSV.
    target_size=None means keep original resolution.
    """
    img = Image.open(image_path).convert("RGB")

    if median_size and median_size > 1:
        img = img.filter(ImageFilter.MedianFilter(size=median_size))

    if target_size is not None:
        img = img.resize((target_size, target_size), resample=Image.Resampling.BOX)

    rgb = np.array(img).astype(float) / 255.0
    hsv = rgb_to_hsv(rgb)
    return rgb, hsv


def image_to_grid_graph_hsv(hsv_img, sigma=0.25):
    h, w, _ = hsv_img.shape
    raw_edges = []
    min_weight = float("inf")
    max_weight = float("-inf")

    for y in range(h):
        for x in range(w):
            i = y * w + x

            if y > 0:
                j = (y - 1) * w + x
                weight = gaussian_dissimilarity_hsv(hsv_img[y, x], hsv_img[y - 1, x], sigma=sigma)
                raw_edges.append((i, j, weight))
                min_weight = min(min_weight, weight)
                max_weight = max(max_weight, weight)

            if x > 0:
                j = y * w + (x - 1)
                weight = gaussian_dissimilarity_hsv(hsv_img[y, x], hsv_img[y, x - 1], sigma=sigma)
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

def circular_hue_distance(H, center):
    d = np.abs(H - center)
    return np.minimum(d, 1.0 - d)


def build_global_vegetation_prior(
    hsv,
    target_ratio=0.3,
    green_center=1.0 / 3.0,
    green_sigma=0.10,
):
    """
    Build a vegetation score and a binary non-vegetation prior.

    vegetation_score:
        Higher value means the pixel looks more like vegetation.

    nonveg_prior:
        0 = vegetation
        1 = non-vegetation

    target_ratio still means the expected full-image non-vegetation ratio.
    """
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]

    green_dist = circular_hue_distance(H, green_center)
    green_similarity = np.exp(
        -(green_dist ** 2) / (2.0 * green_sigma ** 2)
    )

    # Green hue plus sufficient saturation gives a high vegetation score.
    vegetation_score = S * green_similarity

    # The lowest-scoring target_ratio pixels are marked as non-vegetation.
    threshold = np.quantile(vegetation_score, target_ratio)
    nonveg_prior = (vegetation_score < threshold).astype(np.float32)

    return nonveg_prior, vegetation_score

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
    """
    Add penalty * (sum_i x_i - k)^2 to avoid all-0 or all-1 trivial segmentation.
    """
    k = target_ratio * n

    for i in range(n):
        linear[i] = linear.get(i, 0.0) + penalty * (1 - 2 * k)

    for i in range(n):
        for j in range(i + 1, n):
            quadratic[(i, j)] = quadratic.get((i, j), 0.0) + 2 * penalty

    return linear, quadratic


def simulated_annealer_solver(G, n_samples=2000, target_ratio=0.5, balance_penalty=0.01):
    start_time = time.time()

    W = nx.adjacency_matrix(G).todense()
    linear, quadratic = get_linear_quadratic_dict(W)
    n = W.shape[0]

    linear, quadratic = add_balance_penalty(
        linear,
        quadratic,
        n=n,
        target_ratio=target_ratio,
        penalty=balance_penalty,
    )

    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)
    problem_formulation_time = time.time() - start_time

    start_time = time.time()
    sampler = neal.SimulatedAnnealingSampler()
    sample_set = sampler.sample(bqm, num_reads=n_samples)
    response_time = time.time() - start_time

    samples_df = sample_set.to_pandas_dataframe()
    info_dict = {
        "problem_formulation_time": problem_formulation_time,
        "response_time": response_time,
        "num_variables": n,
        "num_quadratic_terms": len(quadratic),
        "target_ratio": target_ratio,
        "balance_penalty": balance_penalty,
    }
    return samples_df, info_dict


def solve_one_patch_hsv(hsv_patch, n_samples=2000, target_ratio=0.3, balance_penalty=0.01, sigma=0.25):
    patch_h, patch_w = hsv_patch.shape[:2]
    edge_list = image_to_grid_graph_hsv(hsv_patch, sigma=sigma)

    G = nx.Graph()
    G.add_nodes_from(range(patch_h * patch_w))
    G.add_weighted_edges_from(edge_list)

    samples_df, info = simulated_annealer_solver(
        G,
        n_samples=n_samples,
        target_ratio=target_ratio,
        balance_penalty=balance_penalty,
    )

    best_sample = samples_df.sort_values("energy").iloc[0]
    solution_binary_string = best_sample.drop(
        labels=["energy", "num_occurrences", "chain_break_fraction"],
        errors="ignore",
    ).astype(int).to_numpy()

    mask = decode_binary_string(solution_binary_string, patch_h, patch_w)
    return mask.astype(np.float32), info, float(best_sample["energy"])


def get_patch_starts(length, patch_size, stride):
    """
    Ensure the last patch covers the image boundary even when length is not divisible by stride.
    """
    if length <= patch_size:
        return [0]

    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def cosine_weight_2d(h, w, eps=1e-3):
    """
    Give center pixels larger weights and patch-border pixels smaller weights.
    This reduces block artifacts when overlapping patches are merged.
    """
    wy = np.hanning(h) if h > 1 else np.ones(1)
    wx = np.hanning(w) if w > 1 else np.ones(1)
    weight = np.outer(wy, wx)
    weight = np.maximum(weight, eps)
    return weight.astype(np.float32)


def align_patch_label_with_overlap(
    patch_mask,
    vote_sum,
    weight_sum,
    y0,
    y1,
    x0,
    x1,
):
    """
    Decide whether to flip a patch by comparing it with the already reconstructed
    mask in the overlap region.

    This method does not use soil/vegetation semantics. It only attempts to keep
    adjacent patches' binary labels consistent.
    """
    existing_weight = weight_sum[y0:y1, x0:x1]
    overlap = existing_weight > 1e-12

    # First patch, or no overlap with a reconstructed patch.
    if not np.any(overlap):
        return patch_mask.copy(), False, np.nan, np.nan

    existing_score = (
        vote_sum[y0:y1, x0:x1]
        / np.maximum(existing_weight, 1e-12)
    )
    existing_mask = (existing_score >= 0.5).astype(np.float32)

    diff_original = float(
        np.mean(np.abs(patch_mask[overlap] - existing_mask[overlap]))
    )
    diff_flipped = float(
        np.mean(np.abs((1.0 - patch_mask[overlap]) - existing_mask[overlap]))
    )

    if diff_flipped < diff_original:
        return 1.0 - patch_mask, True, diff_original, diff_flipped

    return patch_mask.copy(), False, diff_original, diff_flipped


def align_patch_label_with_class_prior(patch_mask, prior_patch):
    """
    Decide whether to flip a patch by comparing its binary labels with the
    vegetation-score non-vegetation prior.

    Convention after alignment:
        1 = non-vegetation
        0 = vegetation

    The prior is used only to choose the global 0/1 orientation of the patch.
    It does not replace individual Q-Seg pixel labels.
    """
    prior_binary = (prior_patch >= 0.5).astype(np.float32)

    agreement_original = float(np.mean(patch_mask == prior_binary))
    agreement_flipped = float(np.mean((1.0 - patch_mask) == prior_binary))

    if agreement_flipped > agreement_original:
        return 1.0 - patch_mask, True, agreement_original, agreement_flipped

    return patch_mask.copy(), False, agreement_original, agreement_flipped


def add_patch_vote(vote_sum, weight_sum, patch_mask, weight, y0, y1, x0, x1):
    """Accumulate one patch into a weighted voting canvas."""
    vote_sum[y0:y1, x0:x1] += patch_mask * weight
    weight_sum[y0:y1, x0:x1] += weight


def finalize_vote(vote_sum, weight_sum):
    """Convert accumulated weighted votes into score map and binary mask."""
    score_map = vote_sum / np.maximum(weight_sum, 1e-12)
    final_mask = (score_map >= 0.5).astype(np.uint8)
    return final_mask, score_map


def patch_qseg_reconstruct_three_methods(
    hsv,
    global_prior=None,
    patch_size=32,
    stride=16,
    n_samples=2000,
    target_ratio=0.3,
    balance_penalty=0.01,
    sigma=0.25,
    use_center_weight=True,
):
    """
    Solve every patch only once, then reconstruct the same patch solutions with
    three label-alignment strategies:

    1. raw:
       No patch-level label flipping.

    2. overlap:
       Flip a patch only when doing so improves consistency with already merged
       neighboring patches in the overlap region.

    3. soil:
       Flip a patch only when doing so improves agreement with the binary
       vegetation-score prior.

    All three results use the same simulated-annealing patch solutions, so the
    comparison is not affected by different random SA runs.
    """
    H, W = hsv.shape[:2]

    vote_sums = {
        "raw": np.zeros((H, W), dtype=np.float32),
        "overlap": np.zeros((H, W), dtype=np.float32),
        "vegetation": np.zeros((H, W), dtype=np.float32),
    }
    weight_sums = {
        "raw": np.zeros((H, W), dtype=np.float32),
        "overlap": np.zeros((H, W), dtype=np.float32),
        "vegetation": np.zeros((H, W), dtype=np.float32),
    }

    y_starts = get_patch_starts(H, patch_size, stride)
    x_starts = get_patch_starts(W, patch_size, stride)

    patch_infos = []
    total_patches = len(y_starts) * len(x_starts)
    patch_id = 0

    for y0 in y_starts:
        for x0 in x_starts:
            patch_id += 1
            y1 = min(y0 + patch_size, H)
            x1 = min(x0 + patch_size, W)

            hsv_patch = hsv[y0:y1, x0:x1]
            print(
                f"Solving patch {patch_id}/{total_patches}: "
                f"y={y0}:{y1}, x={x0}:{x1}"
            )

            if global_prior is not None:
                prior_patch = global_prior[y0:y1, x0:x1]
                adaptive_target_ratio = float(np.mean(prior_patch))
            else:
                prior_patch = None
                adaptive_target_ratio = target_ratio

            raw_patch_mask, info, energy = solve_one_patch_hsv(
                hsv_patch,
                n_samples=n_samples,
                target_ratio=adaptive_target_ratio,
                balance_penalty=balance_penalty,
                sigma=sigma,
            )

            # Method 1: no flipping.
            raw_aligned = raw_patch_mask.copy()

            # Method 2: overlap-based flipping.
            overlap_aligned, overlap_flipped, overlap_diff_original, overlap_diff_flipped = (
                align_patch_label_with_overlap(
                    raw_patch_mask,
                    vote_sums["overlap"],
                    weight_sums["overlap"],
                    y0,
                    y1,
                    x0,
                    x1,
                )
            )

            # Method 3: vegetation-prior-based flipping.
            if prior_patch is not None:
                vegetation_aligned, vegetation_flipped, vegetation_agree_original, vegetation_agree_flipped = (
                    align_patch_label_with_class_prior(
                        raw_patch_mask,
                        prior_patch,
                    )
                )
            else:
                vegetation_aligned = raw_patch_mask.copy()
                vegetation_flipped = False
                vegetation_agree_original = np.nan
                vegetation_agree_flipped = np.nan

            ph, pw = raw_patch_mask.shape
            if use_center_weight:
                weight = cosine_weight_2d(ph, pw)
            else:
                weight = np.ones((ph, pw), dtype=np.float32)

            add_patch_vote(
                vote_sums["raw"],
                weight_sums["raw"],
                raw_aligned,
                weight,
                y0,
                y1,
                x0,
                x1,
            )
            add_patch_vote(
                vote_sums["overlap"],
                weight_sums["overlap"],
                overlap_aligned,
                weight,
                y0,
                y1,
                x0,
                x1,
            )
            add_patch_vote(
                vote_sums["vegetation"],
                weight_sums["vegetation"],
                vegetation_aligned,
                weight,
                y0,
                y1,
                x0,
                x1,
            )

            patch_infos.append({
                "patch_id": patch_id,
                "y0": y0,
                "y1": y1,
                "x0": x0,
                "x1": x1,
                "energy": energy,
                "adaptive_target_ratio": adaptive_target_ratio,
                "raw_ratio": float(np.mean(raw_patch_mask)),
                "overlap_flipped": overlap_flipped,
                "overlap_diff_original": overlap_diff_original,
                "overlap_diff_flipped": overlap_diff_flipped,
                "vegetation_flipped": vegetation_flipped,
                "vegetation_agreement_original": vegetation_agree_original,
                "vegetation_agreement_flipped": vegetation_agree_flipped,
                **info,
            })

    results = {}
    for method in ("raw", "overlap", "vegetation"):
        final_mask, score_map = finalize_vote(
            vote_sums[method],
            weight_sums[method],
        )
        results[method] = {
            "mask": final_mask,
            "score_map": score_map,
        }

    return results, patch_infos

def make_boundary_overlay(rgb, mask):
    h, w = mask.shape
    overlay = rgb.copy()
    boundary = np.zeros_like(mask, dtype=bool)

    for y in range(h):
        for x in range(w):
            if y + 1 < h and mask[y, x] != mask[y + 1, x]:
                boundary[y, x] = True
                boundary[y + 1, x] = True
            if x + 1 < w and mask[y, x] != mask[y, x + 1]:
                boundary[y, x] = True
                boundary[y, x + 1] = True

    overlay[boundary] = [1.0, 0.0, 0.0]
    return overlay


def save_single_method_images(rgb, mask, score_map, method_name, output_dir):
    """Save score map, colored mask, and boundary overlay for one method."""
    seg_cmap = ListedColormap(["#2E8B57", "#C2B280"])
    # 0 = vegetation (green), 1 = non-vegetation (tan)

    pretty_names = {
        "raw": "No label alignment",
        "overlap": "Overlap label alignment",
        "vegetation": "Vegetation-prior label alignment",
    }
    title_name = pretty_names[method_name]

    plt.figure()
    plt.imshow(score_map, cmap="gray", vmin=0, vmax=1)
    plt.title(f"Patch voting score: {title_name}")
    plt.axis("off")
    plt.savefig(
        os.path.join(output_dir, f"{method_name}_voting_score_map.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure()
    plt.imshow(mask, cmap=seg_cmap, vmin=0, vmax=1)
    plt.title(f"Patch Q-Seg: {title_name}")
    plt.axis("off")
    plt.savefig(
        os.path.join(output_dir, f"{method_name}_qseg_mask.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    overlay = make_boundary_overlay(rgb, mask)
    plt.figure()
    plt.imshow(overlay)
    plt.title(f"Boundary overlay: {title_name}")
    plt.axis("off")
    plt.savefig(
        os.path.join(output_dir, f"{method_name}_boundary_overlay.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_comparison_figure(rgb, results, nonveg_prior, output_dir):
    """Save a single figure for side-by-side qualitative comparison."""
    seg_cmap = ListedColormap(["#2E8B57", "#C2B280"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Input image")

    axes[0, 1].imshow(nonveg_prior, cmap=seg_cmap, vmin=0, vmax=1)
    axes[0, 1].set_title("HSV vegetation-score baseline")

    axes[0, 2].axis("off")

    axes[1, 0].imshow(results["raw"]["mask"], cmap=seg_cmap, vmin=0, vmax=1)
    axes[1, 0].set_title("Q-Seg: no alignment")

    axes[1, 1].imshow(results["overlap"]["mask"], cmap=seg_cmap, vmin=0, vmax=1)
    axes[1, 1].set_title("Q-Seg: overlap alignment")

    axes[1, 2].imshow(results["vegetation"]["mask"], cmap=seg_cmap, vmin=0, vmax=1)
    axes[1, 2].set_title("Q-Seg: vegetation-prior alignment")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "comparison_all_methods.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_result_images(rgb, results, nonveg_prior, vegetation_score, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    seg_cmap = ListedColormap(["#2E8B57", "#C2B280"])

    plt.figure()
    plt.imshow(rgb)
    plt.title("Input image: median blur + HSV")
    plt.axis("off")
    plt.savefig(
        os.path.join(output_dir, "input_full_resolution.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Pure classical baseline.
    plt.figure()
    plt.imshow(vegetation_score, cmap="viridis")
    plt.title("HSV vegetation score")
    plt.axis("off")
    plt.colorbar()
    plt.savefig(
        os.path.join(output_dir, "vegetation_score_map.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure()
    plt.imshow(nonveg_prior, cmap=seg_cmap, vmin=0, vmax=1)
    plt.title("HSV vegetation-score baseline mask")
    plt.axis("off")
    plt.savefig(
        os.path.join(output_dir, "vegetation_score_baseline_mask.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    for method_name, method_result in results.items():
        save_single_method_images(
            rgb,
            method_result["mask"],
            method_result["score_map"],
            method_name,
            output_dir,
        )

    save_comparison_figure(rgb, results, nonveg_prior, output_dir)

def save_config(args, image_path, output_dir, elapsed_time):
    config_path = os.path.join(output_dir, "config.txt")

    with open(config_path, "w", encoding="utf-8") as f:
        f.write("Patch-based Q-Seg Configuration\n")
        f.write("=" * 50 + "\n")
        f.write(f"Time                : {datetime.now()}\n")
        f.write(f"Image               : {image_path}\n")
        f.write(f"Target size         : {args.target_size}\n")
        f.write(f"Patch size          : {args.patch_size}\n")
        f.write(f"Stride              : {args.stride}\n")
        f.write(f"Target ratio        : {args.target_ratio}\n")
        f.write(f"Green center        : {args.green_center}\n")
        f.write(f"Green sigma         : {args.green_sigma}\n")
        f.write(f"Balance penalty     : {args.balance_penalty}\n")
        f.write(f"n_samples           : {args.n_samples}\n")
        f.write("Sampler             : SimulatedAnnealingSampler\n")
        f.write(f"Overlap             : {args.patch_size - args.stride}\n")

        if args.target_size > 0:
            num_patch_x = len(
                get_patch_starts(args.target_size, args.patch_size, args.stride)
            )
            num_patch_y = num_patch_x
            f.write(f"Patch grid          : {num_patch_x} x {num_patch_y}\n")
            f.write(f"Total patches       : {num_patch_x * num_patch_y}\n")

        f.write(f"Execution time (s)  : {elapsed_time:.2f}\n")
        f.write(
            "Compared methods    : raw / overlap alignment / "
            "vegetation-prior alignment\n"
        )
        f.write("Label convention    : 0=vegetation, 1=non-vegetation\n")

def process_one_image(image_path, args):
    if not os.path.exists(image_path):
        print(f"\n[SKIP] Image not found: {image_path}")
        return None

    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    image_output_dir = os.path.join(args.output_dir, image_stem)
    os.makedirs(image_output_dir, exist_ok=True)

    target_size = None if args.target_size == 0 else args.target_size
    start_total = time.time()

    rgb, hsv = load_full_image_hsv(
        image_path,
        target_size=target_size,
        median_size=args.median_size,
    )

    print("\n" + "=" * 70)
    print("Image:", image_path)
    print("Image shape:", rgb.shape)
    print("Patch size:", args.patch_size)
    print("Stride:", args.stride)
    print("n_samples:", args.n_samples)

    global_prior, vegetation_score = build_global_vegetation_prior(
        hsv,
        target_ratio=args.target_ratio,
        green_center=args.green_center,
        green_sigma=args.green_sigma,
    )

    print(
        "Global non-vegetation prior ratio:",
        float(np.mean(global_prior)),
    )

    results, patch_infos = patch_qseg_reconstruct_three_methods(
        hsv,
        global_prior=global_prior,
        patch_size=args.patch_size,
        stride=args.stride,
        n_samples=args.n_samples,
        target_ratio=args.target_ratio,
        balance_penalty=args.balance_penalty,
        sigma=args.sigma,
        use_center_weight=not args.no_center_weight,
    )

    total_time = time.time() - start_total

    for method_name, method_result in results.items():
        print(f"\n{method_name} mask values:")
        print(np.unique(method_result["mask"], return_counts=True))
        print(
            f"{method_name} final non-vegetation ratio:",
            float(np.mean(method_result["mask"])),
        )

    overlap_flip_count = sum(
        bool(row["overlap_flipped"]) for row in patch_infos
    )
    vegetation_flip_count = sum(
        bool(row["vegetation_flipped"]) for row in patch_infos
    )

    print("\nTotal patches:", len(patch_infos))
    print("Overlap-aligned flipped patches:", overlap_flip_count)
    print(
        "Vegetation-prior-aligned flipped patches:",
        vegetation_flip_count,
    )
    print("Total time:", total_time)

    for method_name, method_result in results.items():
        np.save(
            os.path.join(
                image_output_dir,
                f"{method_name}_qseg_mask.npy",
            ),
            method_result["mask"],
        )
        np.save(
            os.path.join(
                image_output_dir,
                f"{method_name}_voting_score_map.npy",
            ),
            method_result["score_map"],
        )

    np.save(
        os.path.join(image_output_dir, "global_nonveg_prior.npy"),
        global_prior,
    )
    np.save(
        os.path.join(image_output_dir, "global_vegetation_score.npy"),
        vegetation_score,
    )

    info_path = os.path.join(image_output_dir, "patch_infos.csv")
    if patch_infos:
        keys = list(patch_infos[0].keys())
        with open(info_path, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for row in patch_infos:
                f.write(
                    ",".join(str(row.get(k, "")) for k in keys) + "\n"
                )

    save_result_images(
        rgb,
        results,
        global_prior,
        vegetation_score,
        image_output_dir,
    )
    save_config(
        args,
        image_path,
        image_output_dir,
        total_time,
    )

    print("\nSaved results to:", image_output_dir)
    print("- comparison_all_methods.png")
    print("- vegetation_score_map.png")
    print("- vegetation_score_baseline_mask.png")
    print("- raw_qseg_mask.png / .npy")
    print("- overlap_qseg_mask.png / .npy")
    print("- vegetation_qseg_mask.png / .npy")
    print("- patch_infos.csv")

    return {
        "image": image_path,
        "output_dir": image_output_dir,
        "elapsed_time": total_time,
        "num_patches": len(patch_infos),
        "overlap_flips": overlap_flip_count,
        "vegetation_flips": vegetation_flip_count,
    }


def save_batch_summary(rows, output_dir):
    if not rows:
        return

    summary_path = os.path.join(output_dir, "batch_summary.csv")
    keys = list(rows[0].keys())

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")

    print("\nBatch summary saved to:", summary_path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch patch-based HSV Q-Seg using vegetation-score prior "
            "and three label-alignment strategies."
        )
    )
    parser.add_argument(
        "--image_paths",
        nargs="+",
        default=[
            "71619_sat_26.jpg",
            "71619_sat_30.jpg",
            "71619_sat_44.jpg",
            "72807_sat_88.jpg",
        ],
        help="One or more input image paths.",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=256,
        help=(
            "Resize each image to target_size x target_size before patching. "
            "Use 0 to keep original size."
        ),
    )
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--median_size", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument(
        "--target_ratio",
        type=float,
        default=0.3,
        help="Expected full-image non-vegetation ratio.",
    )
    parser.add_argument(
        "--green_center",
        type=float,
        default=1.0 / 3.0,
        help="HSV hue center used as representative green.",
    )
    parser.add_argument(
        "--green_sigma",
        type=float,
        default=0.10,
        help="Width of the green-hue Gaussian similarity.",
    )
    parser.add_argument("--balance_penalty", type=float, default=0.01)
    parser.add_argument("--no_center_weight", action="store_true")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vegetation_qseg_batch_results",
        help="Root output directory; each image gets its own subdirectory.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    batch_rows = []
    batch_start = time.time()

    for image_path in args.image_paths:
        row = process_one_image(image_path, args)
        if row is not None:
            batch_rows.append(row)

    total_batch_time = time.time() - batch_start
    save_batch_summary(batch_rows, args.output_dir)

    print("\n" + "=" * 70)
    print("Finished images:", len(batch_rows))
    print("Total batch time (s):", total_batch_time)


if __name__ == "__main__":
    main()
