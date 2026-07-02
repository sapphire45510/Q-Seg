import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import dimod
import neal

from qseg.graph_utils import image_to_grid_graph, draw, draw_graph_cut_edges
from qseg.utils import decode_binary_string


def get_linear_quadratic_dict(W):
    n = W.shape[0]
    linear = {}
    quadratic = {}

    for i in range(n):
        linear[i] = np.sum(W[i])
        for j in range(n):
            if i < j and W[i, j] != 0:
                quadratic[(i, j)] = -W[i, j]

    return linear, quadratic


def simulated_annealer_solver(G, n_samples=2000):
    start_time = time.time()

    W = nx.adjacency_matrix(G).todense()
    linear, quadratic = get_linear_quadratic_dict(W)

    bqm = dimod.BinaryQuadraticModel(
        linear,
        quadratic,
        0.0,
        dimod.BINARY
    )

    problem_formulation_time = time.time() - start_time

    start_time = time.time()
    sampler = neal.SimulatedAnnealingSampler()
    sample_set = sampler.sample(bqm, num_reads=n_samples)
    response_time = time.time() - start_time

    samples_df = sample_set.to_pandas_dataframe()

    info_dict = {
        "problem_formulation_time": problem_formulation_time,
        "response_time": response_time
    }

    return samples_df, info_dict


def main():
    height, width = 3, 3

    image = np.array([
        [0.82, 0.10, 0.99],
        [0.83, 0.20, 0.95],
        [0.10, 0.05, 0.98]
    ])

    plt.figure()
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title("Input image")
    plt.show()

    normalized_nx_elist = image_to_grid_graph(image)

    G = nx.grid_2d_graph(height, width)
    G.add_weighted_edges_from(normalized_nx_elist)

    draw(G, image)
    plt.title("Grid graph")
    plt.show()

    start_time = time.time()
    samples_dataframe, info_dict = simulated_annealer_solver(
        G,
        n_samples=2000
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

    plt.figure()
    plt.imshow(segmentation_mask, cmap=plt.cm.gray)
    plt.title("Segmentation mask by simulated annealing")
    plt.show()

    cut_edges = [
        (u, v)
        for (u, v, d) in G.edges(data=True)
        if segmentation_mask[u] != segmentation_mask[v]
    ]

    print("\nCut edges:")
    print(cut_edges)

    draw_graph_cut_edges(G, image, cut_edges)
    plt.title("Cut edges")
    plt.show()


if __name__ == "__main__":
    main()