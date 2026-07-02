import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import dimod
import neal
from PIL import Image, ImageFilter
from matplotlib.colors import rgb_to_hsv


def load_forest_image_hsv(image_path, size=32, median_size=5):
    img = Image.open(image_path).convert("RGB")

    # median blurring
    img = img.filter(ImageFilter.MedianFilter(size=median_size))

    # downscale to 32x32
    img = img.resize((size, size), resample=Image.Resampling.BOX)

    rgb = np.asarray(img).astype(np.float32) / 255.0
    hsv = rgb_to_hsv(rgb)

    return rgb, hsv


def hsv_distance2(a, b, h_weight=2.0, s_weight=1.0, v_weight=0.5):
    # Hue is circular: distance between 0.99 and 0.01 should be small
    dh = abs(a[0] - b[0])
    dh = min(dh, 1.0 - dh)

    ds = a[1] - b[1]
    dv = a[2] - b[2]

    return (
        h_weight * dh * dh +
        s_weight * ds * ds +
        v_weight * dv * dv
    )


def gaussian_dissimilarity_from_hsv(a, b, sigma=0.25):
    d2 = hsv_distance2(a, b)
    return 1.0 - np.exp(-d2 / (2.0 * sigma * sigma))


def build_grid_graph_from_hsv(hsv, sigma=0.25):
    h, w, _ = hsv.shape
    raw_edges = []

    for y in range(h):
        for x in range(w):
            i = y * w + x

            if y > 0:
                j = (y - 1) * w + x
                weight = gaussian_dissimilarity_from_hsv(
                    hsv[y, x],
                    hsv[y - 1, x],
                    sigma=sigma
                )
                raw_edges.append((i, j, weight))

            if x > 0:
                j = y * w + (x - 1)
                weight = gaussian_dissimilarity_from_hsv(
                    hsv[y, x],
                    hsv[y, x - 1],
                    sigma=sigma
                )
                raw_edges.append((i, j, weight))

    weights = np.array([e[2] for e in raw_edges])
    min_w = weights.min()
    max_w = weights.max()

    edges = []

    for i, j, w0 in raw_edges:
        if max_w > min_w:
            scaled = 2.0 * (w0 - min_w) / (max_w - min_w) - 1.0
            # Same convention as Q-Seg code:
            # similar edges become positive, dissimilar edges become negative
            wij = -scaled
        else:
            wij = 1.0

        edges.append((i, j, float(wij)))

    G = nx.Graph()
    G.add_nodes_from(range(h * w))
    G.add_weighted_edges_from(edges)

    return G


def graph_to_cut_qubo(G):
    """
    Minimize sum_{(i,j)} w_ij * [x_i != x_j]

    For binary variables:
    [x_i != x_j] = x_i + x_j - 2 x_i x_j

    Therefore:
    linear[i] += w_ij
    linear[j] += w_ij
    quadratic[(i,j)] += -2*w_ij
    """
    linear = {i: 0.0 for i in G.nodes()}
    quadratic = {}

    for i, j, data in G.edges(data=True):
        w = float(data["weight"])

        linear[i] += w
        linear[j] += w

        key = (min(i, j), max(i, j))
        quadratic[key] = quadratic.get(key, 0.0) - 2.0 * w

    return linear, quadratic


def solve_with_simulated_annealing(G, n_reads=3000):
    start = time.time()

    linear, quadratic = graph_to_cut_qubo(G)

    bqm = dimod.BinaryQuadraticModel(
        linear,
        quadratic,
        0.0,
        dimod.BINARY
    )

    formulation_time = time.time() - start

    start = time.time()
    sampler = neal.SimulatedAnnealingSampler()
    sample_set = sampler.sample(bqm, num_reads=n_reads)
    solve_time = time.time() - start

    samples_df = sample_set.to_pandas_dataframe()

    info = {
        "num_variables": len(linear),
        "num_quadratic_terms": len(quadratic),
        "formulation_time": formulation_time,
        "solve_time": solve_time,
        "n_reads": n_reads
    }

    return samples_df, info


def decode_solution(sample_row, height, width):
    x = sample_row.drop(
        labels=["energy", "num_occurrences", "chain_break_fraction"],
        errors="ignore"
    ).astype(int).to_numpy()

    mask = x.reshape(height, width)
    return mask


def make_boundary(mask):
    h, w = mask.shape
    boundary = np.zeros_like(mask, dtype=bool)

    for y in range(h):
        for x in range(w):
            if y + 1 < h and mask[y, x] != mask[y + 1, x]:
                boundary[y, x] = True
                boundary[y + 1, x] = True

            if x + 1 < w and mask[y, x] != mask[y, x + 1]:
                boundary[y, x] = True
                boundary[y, x + 1] = True

    return boundary


def main():
    image_path = "72807_sat_88.jpg"
    size = 32

    rgb, hsv = load_forest_image_hsv(
        image_path,
        size=size,
        median_size=5
    )

    height, width = hsv.shape[:2]

    plt.figure()
    plt.imshow(rgb)
    plt.title("Input image: median blur + resize 32x32")
    plt.axis("off")
    plt.savefig("forest_input_32x32.png", dpi=300, bbox_inches="tight")
    plt.show()

    G = build_grid_graph_from_hsv(
        hsv,
        sigma=0.25
    )

    samples_df, info = solve_with_simulated_annealing(
        G,
        n_reads=3000
    )

    print("Execution info:")
    print(info)

    print("\nSamples:")
    print(samples_df.head())

    best_sample = samples_df.sort_values("energy").iloc[0]
    mask = decode_solution(best_sample, height, width)

    print("\nBest energy:")
    print(best_sample["energy"])

    print("\nUnique mask values:")
    print(np.unique(mask, return_counts=True))

    plt.figure()
    plt.imshow(mask, cmap="gray", vmin=0, vmax=1)
    plt.title("Q-Seg-like mask using HSV + simulated annealing")
    plt.axis("off")
    plt.savefig("forest_mask_hsv_sa.png", dpi=300, bbox_inches="tight")
    plt.show()

    boundary = make_boundary(mask)

    overlay = rgb.copy()
    overlay[boundary] = [1.0, 0.0, 0.0]

    plt.figure()
    plt.imshow(overlay)
    plt.title("Boundary overlay")
    plt.axis("off")
    plt.savefig("forest_boundary_overlay_hsv_sa.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()