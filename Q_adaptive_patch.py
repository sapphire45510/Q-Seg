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


def build_global_soil_prior(hsv, target_ratio=0.3):
    """
    target_ratio 在這裡是全圖 soil 比例。
    prior = 1 表示比較像 soil。
    """
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    soil_dist = circular_hue_distance(H, 0.10)
    green_dist = circular_hue_distance(H, 1.0 / 3.0)

    soil_score = -soil_dist + 0.35 * S + 0.20 * V + 0.35 * green_dist

    threshold = np.quantile(soil_score, 1.0 - target_ratio)
    soil_prior = (soil_score >= threshold).astype(np.float32)

    return soil_prior, soil_score

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

'''#方法3新增用來反轉的程式
def align_patch_label_with_overlap(patch_mask, vote_sum, weight_sum, y0, y1, x0, x1):
    """
    Align patch 0/1 labels using overlap with already reconstructed area.

    If flipping the patch labels makes it more consistent with the existing
    voting result in the overlap region, flip the patch mask.
    """
    existing_weight = weight_sum[y0:y1, x0:x1]

    overlap = existing_weight > 1e-12

    # 第一個 patch 或沒有重疊區域時，沒有參考對象，不做反轉
    if not np.any(overlap):
        return patch_mask, False

    existing_score = vote_sum[y0:y1, x0:x1] / np.maximum(existing_weight, 1e-12)
    existing_mask = (existing_score >= 0.5).astype(np.float32)

    diff_original = np.mean(np.abs(patch_mask[overlap] - existing_mask[overlap]))
    diff_flipped = np.mean(np.abs((1.0 - patch_mask[overlap]) - existing_mask[overlap]))

    if diff_flipped < diff_original:
        return 1.0 - patch_mask, True

    return patch_mask, False

#方法3用的
def patch_qseg_reconstruct_hsv(
    hsv,
    patch_size=32,
    stride=16,
    n_samples=2000,
    target_ratio=0.3,
    balance_penalty=0.01,
    sigma=0.25,
    use_center_weight=True,
):
    """
    Split full image into overlapping 32x32 patches, solve each patch independently,
    and reconstruct full-resolution mask by weighted voting.
    """
    H, W = hsv.shape[:2]
    vote_sum = np.zeros((H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

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
            print(f"Solving patch {patch_id}/{total_patches}: y={y0}:{y1}, x={x0}:{x1}")

            patch_mask, info, energy = solve_one_patch_hsv(
                hsv_patch,
                n_samples=n_samples,
                target_ratio=target_ratio,
                balance_penalty=balance_penalty,
                sigma=sigma,
            )

            patch_mask, flipped = align_patch_label_with_overlap(
    patch_mask,
    vote_sum,
    weight_sum,
    y0,
    y1,
    x0,
    x1,
)

            ph, pw = patch_mask.shape
            if use_center_weight:
                weight = cosine_weight_2d(ph, pw)
            else:
                weight = np.ones((ph, pw), dtype=np.float32)

            vote_sum[y0:y1, x0:x1] += patch_mask * weight
            weight_sum[y0:y1, x0:x1] += weight

            patch_infos.append({
                "patch_id": patch_id,
                "y0": y0,
                "y1": y1,
                "x0": x0,
                "x1": x1,
                "energy": energy,
                "flipped": flipped,
                **info,
            })

    score_map = vote_sum / np.maximum(weight_sum, 1e-12)
    final_mask = (score_map >= 0.5).astype(np.uint8)
    return final_mask, score_map, patch_infos
'''
def patch_qseg_reconstruct_hsv(
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
    Split full image into overlapping 32x32 patches, solve each patch independently,
    and reconstruct full-resolution mask by weighted voting.
    """
    H, W = hsv.shape[:2]
    vote_sum = np.zeros((H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

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
            print(f"Solving patch {patch_id}/{total_patches}: y={y0}:{y1}, x={x0}:{x1}")

            prior_patch = global_prior[y0:y1, x0:x1]

            adaptive_ratio = np.mean(prior_patch)

            if global_prior is not None:
                prior_patch = global_prior[y0:y1, x0:x1]
                adaptive_target_ratio = float(np.mean(prior_patch))
            else:
                adaptive_target_ratio = target_ratio

            patch_mask, info, energy = solve_one_patch_hsv(
                hsv_patch,
                n_samples=n_samples,
                target_ratio=adaptive_target_ratio,
                balance_penalty=balance_penalty,
                sigma=sigma,
            )  

            ph, pw = patch_mask.shape
            if use_center_weight:
                weight = cosine_weight_2d(ph, pw)
            else:
                weight = np.ones((ph, pw), dtype=np.float32)

            vote_sum[y0:y1, x0:x1] += patch_mask * weight
            weight_sum[y0:y1, x0:x1] += weight

            patch_infos.append({
                "patch_id": patch_id,
                "y0": y0,
                "y1": y1,
                "x0": x0,
                "x1": x1,
                "energy": energy,
                "adaptive_target_ratio": adaptive_target_ratio,
                "prior_ratio_in_patch": adaptive_target_ratio,
                **info,
            })

    score_map = vote_sum / np.maximum(weight_sum, 1e-12)
    final_mask = (score_map >= 0.5).astype(np.uint8)
    return final_mask, score_map, patch_infos


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


def save_result_images(rgb, final_mask, score_map, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    seg_cmap = ListedColormap(["#2E8B57", "#C2B280"])  # 綠色、土黃色

    plt.figure()
    plt.imshow(rgb)
    plt.title("Input image: median blur + HSV, no 32x32 global resize")
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, "input_full_resolution.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(score_map, cmap="gray", vmin=0, vmax=1)
    plt.title("Patch voting score map")
    plt.axis("off")
    #plt.colorbar()
    plt.savefig(os.path.join(output_dir, "patch_voting_score_map.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(final_mask, cmap=seg_cmap, vmin=0, vmax=1)
    plt.title("Patch-based Q-Seg mask: green / soil")
    plt.axis("off")
    #plt.colorbar()
    plt.savefig(os.path.join(output_dir, "patch_qseg_mask.png"), dpi=300, bbox_inches="tight")
    plt.close()

    overlay = make_boundary_overlay(rgb, final_mask)
    plt.figure()
    plt.imshow(overlay)
    plt.title("Boundary overlay: patch-based Q-Seg")
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, "patch_qseg_boundary_overlay.png"), dpi=300, bbox_inches="tight")
    plt.close()

def save_config(args, output_dir, elapsed_time):
    config_path = os.path.join(output_dir, "config.txt")

    with open(config_path, "w", encoding="utf-8") as f:
        f.write("Patch-based Q-Seg Configuration\n")
        f.write("=" * 50 + "\n")
        f.write(f"Time                : {datetime.now()}\n")
        f.write(f"Image               : {args.image_path}\n")
        f.write(f"Target size         : {args.target_size}\n")
        f.write(f"Patch size          : {args.patch_size}\n")
        f.write(f"Stride              : {args.stride}\n")
        f.write(f"Target ratio        : {args.target_ratio}\n")
        f.write(f"Balance penalty     : {args.balance_penalty}\n")
        f.write(f"n_samples           : {args.n_samples}\n")
        f.write(f"Sampler             : SimulatedAnnealingSampler\n")
        f.write(f"Overlap             : {args.patch_size - args.stride}\n")

        num_patch_x = (args.target_size - args.patch_size) // args.stride + 1
        num_patch_y = (args.target_size - args.patch_size) // args.stride + 1
        f.write(f"Patch grid          : {num_patch_x} x {num_patch_y}\n")
        f.write(f"Total patches       : {num_patch_x * num_patch_y}\n")
        f.write(f"Execution time (s)  : {elapsed_time:.2f}\n")

def main():
    parser = argparse.ArgumentParser(description="Patch-based HSV Q-Seg with overlap voting.")
    parser.add_argument("--image_path", type=str, default="71619_sat_30.jpg")
    parser.add_argument("--target_size", type=int, default=256, help="Resize full image to target_size x target_size before patching. Use 0 to keep original size.")
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--median_size", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--target_ratio", type=float, default=0.3)
    parser.add_argument("--balance_penalty", type=float, default=0.01)
    parser.add_argument("--no_center_weight", action="store_true")
    parser.add_argument("--output_dir", type=str, default="patch_qseg_results")
    args = parser.parse_args()

    target_size = None if args.target_size == 0 else args.target_size

    start_total = time.time()

    rgb, hsv = load_full_image_hsv(
        args.image_path,
        target_size=target_size,
        median_size=args.median_size,
    )

    print("Image shape:", rgb.shape)
    print("Patch size:", args.patch_size)
    print("Stride:", args.stride)

    global_prior, soil_score = build_global_soil_prior(
        hsv,
        target_ratio=args.target_ratio,
    )

    print("Global soil prior ratio:", float(np.mean(global_prior)))
    final_mask, score_map, patch_infos = patch_qseg_reconstruct_hsv(
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

    print("\nUnique final mask values:")
    print(np.unique(final_mask, return_counts=True))

    print("\nTotal patches:", len(patch_infos))
    print("Total time:", total_time)

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "patch_qseg_mask.npy"), final_mask)
    np.save(os.path.join(args.output_dir, "patch_voting_score_map.npy"), score_map)
    np.save(os.path.join(args.output_dir, "global_soil_prior.npy"), global_prior)
    np.save(os.path.join(args.output_dir, "global_soil_score.npy"), soil_score)

    # Save patch running information as CSV without requiring pandas.
    info_path = os.path.join(args.output_dir, "patch_infos.csv")
    if patch_infos:
        keys = list(patch_infos[0].keys())
        with open(info_path, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for row in patch_infos:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")

    save_result_images(rgb, final_mask, score_map, args.output_dir)
    save_config(args, args.output_dir, total_time)

    print("\nSaved results to:", args.output_dir)
    print("- input_full_resolution.png")
    print("- patch_voting_score_map.png")
    print("- patch_qseg_mask.png")
    print("- patch_qseg_boundary_overlay.png")
    print("- patch_qseg_mask.npy")
    print("- patch_voting_score_map.npy")
    print("- patch_infos.csv")


if __name__ == "__main__":
    main()
