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


def generate_truncated_gaussians_bounded_difference(
    num_dists: int,
    bounds_lim: np.ndarray,
    loc_init: float,
    scales_lim: float,
    bounded_diff: float,
    seed: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)

    locs = np.zeros(num_dists)
    locs[0] = loc_init

    for i in range(1, len(locs)):
        locs[i] = locs[i - 1] + np.random.rand() * 2 * bounded_diff - bounded_diff

    scales = (1 - rng.rand(num_dists)) * scales_lim

    lbounds = rng.rand(num_dists) * (bounds_lim[1] - bounds_lim[0]) - 0.5 * (
        bounds_lim[0] + bounds_lim[1]
    )
    ubounds = lbounds + rng.rand(num_dists) * (bounds_lim[1] - bounds_lim[0])
    bounds = list(zip(lbounds, ubounds))

    return locs, scales, bounds
