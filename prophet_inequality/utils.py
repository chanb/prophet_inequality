from typing import Tuple

import numpy as np


def generate_truncated_gaussians(
    num_dists: int,
    bounds_lim: np.ndarray,
    locs_lim: np.ndarray,
    scales_lim: float,
    seed: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)

    locs = rng.rand(num_dists) * (locs_lim[1] - locs_lim[0]) - 0.5 * (
        locs_lim[0] + locs_lim[1]
    )

    scales = (1 - rng.rand(num_dists)) * scales_lim

    lbounds = rng.rand(num_dists) * (bounds_lim[1] - bounds_lim[0]) - 0.5 * (
        bounds_lim[0] + bounds_lim[1]
    )
    ubounds = lbounds + rng.rand(num_dists) * (bounds_lim[1] - bounds_lim[0])
    bounds = list(zip(lbounds, ubounds))

    return locs, scales, bounds
