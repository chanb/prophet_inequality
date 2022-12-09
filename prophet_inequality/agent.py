from abc import ABC, abstractclassmethod
from typing import Any

from prophet_inequality.constants import ACCEPT, REJECT

import numpy as np


class Agent(ABC):
    @abstractclassmethod
    def act(self, state: Any) -> Any:
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(
        self, accept_probability: float, rng: np.random.RandomState
    ):
        self._accept_probability = accept_probability
        self._rng = rng

    def act(self, state: float) -> str:
        if self._rng.rand() < self._accept_probability:
            return ACCEPT
        return REJECT


class ThresholdAgent(Agent):
    def __init__(self, threshold: float):
        self._threshold = threshold

    def act(self, state: float) -> str:
        if state >= self._threshold:
            return ACCEPT
        return REJECT


class ProbabilisticThresholdAgent(Agent):
    def __init__(
        self, threshold: float, accept_probability: float, rng: np.random.RandomState
    ):
        self._threshold = threshold
        self._accept_probability = accept_probability
        self._rng = rng

    def act(self, state: float) -> str:
        if state >= self._threshold or self._rng.rand() < self._accept_probability:
            return ACCEPT
        return REJECT
