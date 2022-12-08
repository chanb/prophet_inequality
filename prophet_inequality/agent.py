from abc import ABC, abstractclassmethod
from typing import Any

from prophet_inequality.constants import ACCEPT, REJECT


class Agent(ABC):
    @abstractclassmethod
    def act(self, state: Any) -> Any:
        raise NotImplementedError


class ThresholdAgent(Agent):
    def __init__(self, threshold: float):
        self._threshold = threshold

    def act(self, state: float) -> str:
        if state >= self._threshold:
            return ACCEPT
        return REJECT
