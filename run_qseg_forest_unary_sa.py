import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import dimod
import neal
from PIL import Image, ImageFilter
from matplotlib.colors import rgb_to_hsv


def normalize01(x):
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def otsu_threshold(x):
    vals = (normalize01(x).flatten() * 255).astype(np.uint8)
    hist = np.bincount(vals, minlength=256).astype(np.float64)
    prob = hist / hist.sum()

    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]

    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    t = np.argmax(sigma_b) / 255.0
    return t


def load_image(image_path, size=32, median_size=5):
    img = Image.open(image_path).convert("RGB")
    img = img.filter(ImageFilter.MedianFilter(size=median_size))
    img = img.resize((size, size), resample=Image.Resampling.BOX)

    rgb = np.asarray(img).astype(np.float32) / 255.0
    hsv = rgb_to_hsv(rgb)
    return rgb, hsv


def vegetation_score(rgb, hsv):
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    # Excess Green: forest / vegetation cue
    exg = 2 * g - r - b
    exg = normalize01(exg)

    # Saturation helps separate vegetation from pale road/cloud/soil
    sat = hsv[:, :, 1]
    sat = normalize01(sat)

    score = 0.75 * exg + 0.25 * sat
    return normalize01(score)


def build_graph(rgb, feature, sigma=0.20):
    h, w = feature.shape
    G = nx.Graph()
    G.add_nodes_from(range(h * w))

    for y in range(h):
        for x in range(w):
            i = y * w + x

            for dy, dx in [(-1, 0), (0, -1)]:
                yy, xx = y + dy, x + dx
                if yy < 0 or xx < 0:
                    continue

                j = yy * w + xx

                # feature similarity
                diff = feature[y, x] - feature[yy, xx]
                sim = np.exp(-(diff * diff) / (2 * sigma * sigma))

                G.add_edge(i, j, weight=float(sim))

    return G


def build_qubo_with_unary(G, feature, unary_strength=2.0, smooth_strength=1.0):
    """
    Energy:
        unary: encourage vegetation pixels to x=1, non-vegetation to x=0
        smoothness: penalize cutting similar neighboring pixels
    """
    h, w = feature.shape
    n = h * w

    threshold = otsu_threshold(feature)

    linear = {i: 0.0 for i in range(n)}
    quadratic = {}

    flat = feature.flatten()

    for i in range(n):
        p = flat[i]

        # D0 = cost if x=0, D1 = cost if x=1
        # high vegetation score -> prefer x=1
        D1 = (1.0 - p) * unary_strength
        D0 = p * unary_strength

        # QUBO linear coefficient for D0*(1-x)+D1*x
        linear[i] += D1 - D0

    for i, j, data in G.edges(data=True):
        w_ij = smooth_strength * data["weight"]

        # w * [x_i != x_j] = w*(x_i + x_j - 2 x_i x_j)
        linear[i] += w_ij
        linear[j] += w_ij

        key = (min(i, j), max(i, j))
        quadratic[key] = quadratic.get(key, 0.0) - 2.0 * w_ij

    return linear, quadratic, threshold


def solve_qubo(linear, quadratic, n_reads=1000):
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)
    sampler = neal.SimulatedAnnealingSampler()
    sample_set = sampler.sample(bqm, num_reads=n_reads)
    return sample_set.to_pandas_dataframe()


def boundary_overlay(rgb, mask):
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
    size = 32

    start = time.time()

    rgb, hsv = load_image(image_path, size=size, median_size=5)
    score = vegetation_score(rgb, hsv)

    G = build_graph(rgb, score, sigma=0.20)

    linear, quadratic, threshold = build_qubo_with_unary(
        G,
        score,
        unary_strength=2.0,
        smooth_strength=1.0
    )

    samples_df = solve_qubo(linear, quadratic, n_reads=1000)

    best = samples_df.sort_values("energy").iloc[0]

    x = best.drop(
        labels=["energy", "num_occurrences", "chain_break_fraction"],
        errors="ignore"
    ).astype(int).to_numpy()

    mask = x.reshape(size, size)

    print("Runtime:", time.time() - start)
    print("Best energy:", best["energy"])
    print("Unique mask values:", np.unique(mask, return_counts=True))
    print("Otsu threshold:", threshold)
    print("num_variables:", len(linear))
    print("num_quadratic_terms:", len(quadratic))

    plt.figure()
    plt.imshow(rgb)
    plt.title("Input: median blur + resize 32x32")
    plt.axis("off")
    plt.savefig("forest_input_32x32.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure()
    plt.imshow(score, cmap="gray")
    plt.title("Vegetation score")
    plt.axis("off")
    plt.colorbar()
    plt.savefig("forest_vegetation_score.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure()
    plt.imshow(mask, cmap="gray", vmin=0, vmax=1)
    plt.title("Forest mask: QUBO + simulated annealing")
    plt.axis("off")
    plt.savefig("forest_mask_unary_sa.png", dpi=300, bbox_inches="tight")
    plt.show()

    overlay = boundary_overlay(rgb, mask)

    plt.figure()
    plt.imshow(overlay)
    plt.title("Boundary overlay")
    plt.axis("off")
    plt.savefig("forest_boundary_overlay_unary_sa.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()