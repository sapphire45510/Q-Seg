import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import dimod
import neal
from PIL import Image

from qseg.graph_utils import image_to_grid_graph, draw_graph_cut_edges
from qseg.utils import decode_binary_string


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


def load_and_preprocess_image(image_path, size=32):
    img = Image.open(image_path).convert("RGB")

    # RGB -> grayscale
    img_gray = img.convert("L")

    # Downsample to 32x32
    img_gray = img_gray.resize(
        (size, size),
        resample=Image.Resampling.BOX
    )

    image = np.array(img_gray).astype(float) / 255.0

    return image

def add_seed_constraints(linear, image, penalty=10.0):
    h, w = image.shape

    flat = image.flatten()

    dark_idx = int(np.argmin(flat))
    bright_idx = int(np.argmax(flat))

    # dark pixel forced to 1
    linear[dark_idx] = linear.get(dark_idx, 0.0) - penalty

    # bright pixel forced to 0
    linear[bright_idx] = linear.get(bright_idx, 0.0) + penalty

    return linear, dark_idx, bright_idx

def main():
    image_path = "72807_sat_88.jpg"
    resize_size = 32

    image = load_and_preprocess_image(
        image_path,
        size=resize_size
    )

    height, width = image.shape

    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.title("Input image resized to 32x32")
    plt.savefig("input_resized_32x32.png", dpi=300, bbox_inches="tight")
    plt.show()

    normalized_nx_elist = image_to_grid_graph(image)

    G = nx.grid_2d_graph(height, width)
    G.add_weighted_edges_from(normalized_nx_elist)

    start_time = time.time()

    samples_dataframe, info_dict = simulated_annealer_solver(
        G,
        n_samples=2000,
        target_ratio=0.5,
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
    plt.title("Segmentation mask by simulated annealing + balance penalty")
    plt.colorbar()
    plt.savefig("segmentation_mask_balance.png", dpi=300, bbox_inches="tight")
    plt.show()

    cut_edges = [
        (u, v)
        for (u, v, d) in G.edges(data=True)
        if segmentation_mask[u] != segmentation_mask[v]
    ]

    print("\nNumber of cut edges:")
    print(len(cut_edges))

    print("\nCut edges:")
    print(cut_edges)

    if len(cut_edges) > 0:
        draw_graph_cut_edges(G, image, cut_edges)
        plt.title("Cut edges")
        plt.savefig("cut_edges_balance.png", dpi=300, bbox_inches="tight")
        plt.show()
    else:
        print("\nNo cut edges found.")


if __name__ == "__main__":
    main()