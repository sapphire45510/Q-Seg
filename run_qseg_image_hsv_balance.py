import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import dimod
import neal
from PIL import Image, ImageFilter
from matplotlib.colors import rgb_to_hsv

from qseg.graph_utils import draw_graph_cut_edges
from qseg.utils import decode_binary_string


def hsv_distance2(a, b, h_weight=2.0, s_weight=1.0, v_weight=0.5):
    # Hue 是環狀，0.99 和 0.01 應該很接近
    dh = abs(a[0] - b[0])
    dh = min(dh, 1.0 - dh)

    ds = a[1] - b[1]
    dv = a[2] - b[2]

    return h_weight * dh * dh + s_weight * ds * ds + v_weight * dv * dv


def gaussian_dissimilarity_hsv(a, b, sigma=0.25):
    d2 = hsv_distance2(a, b)
    return 1.0 - np.exp(-d2 / (2.0 * sigma * sigma))


def load_and_preprocess_image_hsv(image_path, size=32, median_size=5):
    img = Image.open(image_path).convert("RGB")

    # 論文 Forest Cover preprocessing 有提到 median blurring
    img = img.filter(ImageFilter.MedianFilter(size=median_size))

    # Downsample to 32x32
    img = img.resize(
        (size, size),
        resample=Image.Resampling.BOX
    )

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
                weight = gaussian_dissimilarity_hsv(
                    hsv_img[y, x],
                    hsv_img[y - 1, x],
                    sigma=sigma
                )
                raw_edges.append((i, j, weight))
                min_weight = min(min_weight, weight)
                max_weight = max(max_weight, weight)

            if x > 0:
                j = y * w + (x - 1)
                weight = gaussian_dissimilarity_hsv(
                    hsv_img[y, x],
                    hsv_img[y, x - 1],
                    sigma=sigma
                )
                raw_edges.append((i, j, weight))
                min_weight = min(min_weight, weight)
                max_weight = max(max_weight, weight)

    normalized_edges = []

    # 模仿 Q-Seg graph_utils.py 的 [-1, 1] normalization + 負號
    a = -1
    b = 1

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


def get_linear_quadratic_dict(W):
    n = W.shape[0]
    linear = {}
    quadratic = {}

    for i in range(n):
        linear[i] = float(np.sum(W[i]))
        for j in range(n):
            if i < j and W[i, j] != 0:
                quadratic[(i, j)] = float(-W[i, j])

    return linear, quadratic


def add_balance_penalty(linear, quadratic, n, target_ratio=0.5, penalty=0.01):
    """
    Add penalty * (sum_i x_i - k)^2
    to avoid all-0 or all-1 trivial segmentation.
    """
    k = target_ratio * n

    for i in range(n):
        linear[i] = linear.get(i, 0.0) + penalty * (1 - 2 * k)

    for i in range(n):
        for j in range(i + 1, n):
            quadratic[(i, j)] = quadratic.get((i, j), 0.0) + 2 * penalty

    return linear, quadratic


def simulated_annealer_solver(
    G,
    n_samples=2000,
    target_ratio=0.5,
    balance_penalty=0.01
):
    start_time = time.time()

    W = nx.adjacency_matrix(G).todense()
    linear, quadratic = get_linear_quadratic_dict(W)

    n = W.shape[0]

    linear, quadratic = add_balance_penalty(
        linear,
        quadratic,
        n=n,
        target_ratio=target_ratio,
        penalty=balance_penalty
    )

    bqm = dimod.BinaryQuadraticModel(
        linear,
        quadratic,
        0.0,
        dimod.BINARY
    )

    problem_formulation_time = time.time() - start_time

    start_time = time.time()
    sampler = neal.SimulatedAnnealingSampler()
    sample_set = sampler.sample(
        bqm,
        num_reads=n_samples
    )
    response_time = time.time() - start_time

    samples_df = sample_set.to_pandas_dataframe()

    info_dict = {
        "problem_formulation_time": problem_formulation_time,
        "response_time": response_time,
        "num_variables": n,
        "num_quadratic_terms": len(quadratic),
        "target_ratio": target_ratio,
        "balance_penalty": balance_penalty
    }

    return samples_df, info_dict


def index_mask_to_xy_mask(mask):
    """
    目前 graph node 是 0,1,...,1023。
    draw_graph_cut_edges 需要 node 是 (y,x)，所以這裡只用在畫 mask，不改 graph。
    """
    return mask


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


def main():
    image_path = "71619_sat_26.jpg"
    resize_size = 32

    rgb, hsv = load_and_preprocess_image_hsv(
        image_path,
        size=resize_size,
        median_size=5
    )

    height, width = hsv.shape[:2]

    plt.figure()
    plt.imshow(rgb)
    plt.title("Input image: median blur + HSV + resize 32x32")
    plt.axis("off")
    plt.savefig("input_hsv_resized_32x32_2.png", dpi=300, bbox_inches="tight")
    plt.show()

    normalized_nx_elist = image_to_grid_graph_hsv(
        hsv,
        sigma=0.25
    )

    G = nx.Graph()
    G.add_nodes_from(range(height * width))
    G.add_weighted_edges_from(normalized_nx_elist)

    start_time = time.time()

    samples_dataframe, info_dict = simulated_annealer_solver(
        G,
        n_samples=2000,
        target_ratio=0.3,
        balance_penalty=0.01
    )

    info_dict["total_time"] = time.time() - start_time

    print("Execution info:")
    print(info_dict)

    print("\nSamples:")
    print(samples_dataframe.head())

    best_sample = samples_dataframe.sort_values("energy").iloc[0]

    solution_binary_string = best_sample.drop(
        labels=["energy", "num_occurrences", "chain_break_fraction"],
        errors="ignore"
    ).astype(int).to_numpy()

    print("\nBest binary solution:")
    print(solution_binary_string)

    segmentation_mask = decode_binary_string(
        solution_binary_string,
        height,
        width
    )

    print("\nSegmentation mask:")
    print(segmentation_mask)

    print("\nUnique mask values:")
    print(np.unique(segmentation_mask, return_counts=True))

    plt.figure()
    plt.imshow(segmentation_mask, cmap="gray", vmin=0, vmax=1)
    plt.title("Segmentation mask: HSV + SA + balance penalty")
    plt.axis("off")
    plt.colorbar()
    plt.savefig("segmentation_mask_hsv_balance_2.png", dpi=300, bbox_inches="tight")
    plt.show()

    overlay = make_boundary_overlay(rgb, segmentation_mask)

    plt.figure()
    plt.imshow(overlay)
    plt.title("Boundary overlay: HSV + SA + balance penalty")
    plt.axis("off")
    plt.savefig("boundary_overlay_hsv_balance_2.png", dpi=300, bbox_inches="tight")
    plt.show()

    cut_edges = [
        (u, v)
        for (u, v, d) in G.edges(data=True)
        if segmentation_mask[u // width, u % width] != segmentation_mask[v // width, v % width]
    ]

    print("\nNumber of cut edges:")
    print(len(cut_edges))

    print("\nFirst 50 cut edges:")
    print(cut_edges[:50])


if __name__ == "__main__":
    main()