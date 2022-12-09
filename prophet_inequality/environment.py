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

    def get_state(self):
        state = (
            self._sampled_reward[self._curr_step]
            if self._curr_step < self._num_dists
            else None
        )
        return state

    def step(self, action: str) -> Tuple[Any, float, bool, bool, Dict]:
        truncation = False
        termination = False
        reward = 0.0
        if action == ACCEPT:
            self._num_accepts_remaining -= 1
            reward = self._sampled_reward[self._curr_step]

        self._curr_step += 1
        if self._num_accepts_remaining <= 0 or self._curr_step + 1 >= self._num_dists:
            termination = True

        state = self.get_state()
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
        state = self.get_state()
        return state

    def get_sampled_reward(self):
        return self._sampled_reward

    def get_max_dists_indices(self) -> np.ndarray:
        return np.argsort(self._sampled_reward)[-self._num_accepts :]

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


class RLTruncatedGaussianReward(TruncatedGaussianReward):
    def __init__(
        self,
        locs: Iterable[float],
        scales: Iterable[float],
        bounds: Iterable[Tuple[float, float]],
        num_accepts: int,
        history: int = None,
        subtract_max: bool = False,
        normalize: bool = False,
    ):
        super().__init__(locs, scales, bounds, num_accepts)
        self._obs_dim = 3  # curr_reward, time_limit, remaining_accepts
        self._baseline = 0

        self._state_mean = 0.0
        self._state_std = 1.0
        if normalize:
            self._state_mean = np.mean(locs)
            self._state_std = np.std(locs)

        if subtract_max:
            self._baseline = np.max(locs)

        # include a window of past rewards
        self._history = None
        if history and history > 0:
            self._obs_dim += history
            self._history = [0] * history

    def reset(self, seed: int = None):
        if self._history is not None:
            self._history = [0] * len(self._history)
        state = super().reset()
        return state

    def get_state(self):
        curr_reward = (
            self._sampled_reward[self._curr_step]
            if self._curr_step < self._num_dists
            else None
        )
        curr_reward = (curr_reward - self._state_mean) / self._state_std
        time_limit = (self._num_dists - self._curr_step) / self._num_dists
        num_accepts = self._num_accepts_remaining / self._num_accepts
        if self._history is not None:
            state = self._history + [curr_reward, time_limit, num_accepts]
            self._history = self._history[1:] + [curr_reward]
        else:
            state = [curr_reward, time_limit, num_accepts]
        return state

    def step(self, action: str) -> Tuple[Any, float, bool, bool, Dict]:
        state, reward, truncation, termination, info = super().step(action)
        reward -= self._baseline
        return state, reward, truncation, termination, info
