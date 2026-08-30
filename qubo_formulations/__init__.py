"""Factory for interchangeable Q-Seg coefficient formulations."""
from .frangi_directional import FrangiDirectional
from .standard_pairwise import StandardPairwise

NAMES = ("method2", "mincut", "frangi_directional")


def create(name, args):
    if name == "method2":
        return StandardPairwise(name, -1.0)
    if name == "mincut":
        return StandardPairwise(name, -2.0)
    if name == "frangi_directional":
        return FrangiDirectional(args.lambda_line, args.lambda_parallel,
                                 args.lambda_perpendicular, args.frangi_sigmas,
                                 args.orientation_sigma, args.directional_base)
    raise ValueError(f"Unknown QUBO formulation: {name}")
