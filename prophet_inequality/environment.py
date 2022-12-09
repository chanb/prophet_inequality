from abc import ABC, abstractclassmethod
from typing import Iterable, Tuple, Any, Dict

import numpy as np
from scipy.stats import truncnorm

from prophet_inequality.constants import ACCEPT


class Environment(ABC):
    def __init__(self, num_dists: int, num_accepts: int):
        assert num_dists > 0, "There must be at least one distribution"
        assert num_accepts > 0, "There must be at least one accept"
        self._num_dists = num_dists
        self._num_accepts = num_accepts

        self._num_accepts_remaining = num_accepts
        self._curr_step = 0

    def reset(self, *args, **kwargs):
        self._num_accepts_remaining = self._num_accepts
        self._curr_step = 0

    def step(self, action: str) -> Tuple[Any, float, bool, bool, Dict]:
        truncation = False
        termination = False
        reward = 0.0
        if action == ACCEPT:
            self._num_accepts_remaining -= 1
            reward = self._sampled_reward[self._curr_step]

        self._curr_step += 1
        if self._num_accepts_remaining <= 0 or self._curr_step >= self._num_dists:
            termination = True

        state = (
            self._sampled_reward[self._curr_step]
            if self._curr_step < self._num_dists
            else None
        )
        return state, reward, truncation, termination, {}

    @abstractclassmethod
    def get_max_rewards(self) -> np.ndarray:
        raise NotImplementedError


class TruncatedGaussianReward(Environment):
    def __init__(
        self,
        locs: Iterable[float],
        scales: Iterable[float],
        bounds: Iterable[Tuple[float, float]],
        num_accepts: int,
    ):
        assert (
            len(locs) == len(scales) == len(bounds)
        ), "The number of parameters must match"
        for scale in scales:
            assert scale > 0, "Variance must be positive"
        for (lo, hi) in bounds:
            assert lo <= hi, "Specified bounds [lo, hi] must be proper (i.e. lo <= hi)"

        super().__init__(len(bounds), num_accepts)
        self._sampled_reward = None
        self._bounds = np.asarray(bounds)
        self._locs = np.asarray(locs)
        self._scales = np.asarray(scales)

    def reset(self, seed: int = None):
        super().reset()
        self._sampled_reward = truncnorm.rvs(
            (self._bounds[:, 0] - self._locs) / self._scales,
            (self._bounds[:, 1] - self._locs) / self._scales,
            loc=self._locs,
            scale=self._scales,
            random_state=seed,
        )
        return self._sampled_reward[self._curr_step]

    def get_max_rewards(self) -> np.ndarray:
        return np.sort(self._sampled_reward)[-self._num_accepts :]

    def get_top_medians(self) -> np.ndarray:
        medians = truncnorm.median(
            (self._bounds[:, 0] - self._locs) / self._scales,
            (self._bounds[:, 1] - self._locs) / self._scales,
            loc=self._locs,
            scale=self._scales,
        )
        return np.sort(medians)[-self._num_accepts :]
