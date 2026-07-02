import time
import dimod
import networkx as nx
import numpy as np

try:
    import neal
except ImportError:
    neal = None


def get_linear_quadratic_dict(W):
    """Computes the QUBO matrix for the Minimum Cut problem given a weight matrix W."""
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
    if neal is None:
        raise ImportError(
            "Missing package: dwave-neal. Install it with: pip install dwave-neal"
        )

    start_time = time.time()

    W = nx.adjacency_matrix(G).todense()
    linear, quadratic = get_linear_quadratic_dict(W)

    problem_formulation_time = time.time() - start_time

    bqm = dimod.BinaryQuadraticModel(
        linear,
        quadratic,
        0.0,
        dimod.BINARY
    )

    start_time = time.time()

    sampler = neal.SimulatedAnnealingSampler()
    sample_set = sampler.sample(
        bqm,
        num_reads=n_samples
    )

    response_time = time.time() - start_time

    start_time = time.time()
    samples_df = sample_set.to_pandas_dataframe()
    sample_fetch_time = time.time() - start_time

    info_dict = {
        "problem_formulation_time": problem_formulation_time,
        "response_time": response_time,
        "sample_fetch_time": sample_fetch_time
    }

    return samples_df, info_dict