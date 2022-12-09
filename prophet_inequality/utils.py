from typing import Tuple

import matplotlib.pyplot as plt
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


def plot_average_returns(
    oracle_rewards, agent_rewards, best_locs_rewards, average_window=500
):
    smoothed_oracle = []
    smoothed_agent = []
    smoothed_best_locs = []

    for idx in range(len(oracle_rewards) - average_window + 1):
        smoothed_oracle.append(np.mean(oracle_rewards[idx : idx + average_window]))
        smoothed_agent.append(np.mean(agent_rewards[idx : idx + average_window]))
        smoothed_best_locs.append(
            np.mean(best_locs_rewards[idx : idx + average_window])
        )

    plt.title("Return over Updates")
    plt.plot(np.arange(len(smoothed_oracle)), smoothed_oracle, label="Offline Best")
    plt.plot(np.arange(len(smoothed_agent)), smoothed_agent, label="Learner")
    plt.plot(
        np.arange(len(smoothed_best_locs)),
        smoothed_best_locs,
        label="Best Distribution",
    )
    plt.xlabel("Timesteps")
    plt.ylabel("Average Return")
    plt.legend()
    plt.show()
