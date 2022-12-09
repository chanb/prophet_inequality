from abc import ABC, abstractclassmethod
from typing import Any, Iterable

from prophet_inequality.constants import ACCEPT, REJECT

import numpy as np
import torch


class Agent(ABC):
    @abstractclassmethod
    def act(self, state: Any) -> Any:
        raise NotImplementedError

    def update(self, pbar: Iterable) -> bool:
        return False

    def store(
        self,
        state: Any,
        action: Any,
        reward: float,
        truncated: bool,
        terminated: bool,
        next_state: Any,
        *args,
        **kwargs
    ):
        pass


class RandomAgent(Agent):
    def __init__(self, accept_probability: float, rng: np.random.RandomState):
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


class REINFORCEAgent(Agent):
    def __init__(
        self,
        model: torch.nn.Module,
        opt: torch.optim.Optimizer,
        batch_size: int,
        ent_coef: float = 0.0,
        decay_factor: float = 0.0,
    ):
        self._model = model
        self._opt = opt
        self._batch_size = batch_size
        self._buffer = []
        self._curr_traj = []
        self._curr_count = 0
        self._ent_coef = ent_coef
        self._decay_factor = decay_factor

    def act(self, state: Any) -> str:
        with torch.no_grad():
            p = self._model(torch.tensor(state).float())
            act = torch.distributions.Bernoulli(logits=p).sample()
        if act:
            return ACCEPT
        return REJECT

    def _compute_return(
        self, rewards: Iterable[float], dones: Iterable[bool]
    ) -> np.ndarray:
        returns = [0] * (len(rewards) + 1)
        for step in reversed(range(len(rewards))):
            returns[step] = returns[step + 1] * (1 - dones[step]) + rewards[step]
        return returns[:-1]

    def update(self, pbar: Iterable) -> bool:
        if self._curr_count == self._batch_size:
            states = []
            actions = []
            returns = []

            self._opt.zero_grad()
            for traj in self._buffer:
                (traj_states, traj_actions, traj_rewards, traj_dones) = list(zip(*traj))
                traj_returns = self._compute_return(traj_rewards, traj_dones)
                states.extend(traj_states)
                actions.extend(traj_actions)
                returns.extend(traj_returns)

            states = torch.tensor(states).float()
            actions = torch.tensor(actions).float()
            returns = torch.tensor(returns).float()
            returns = (returns - torch.mean(returns)) / (torch.std(returns) + 1e-5)

            logits = self._model(states)
            dists = torch.distributions.Bernoulli(logits=logits)
            log_probs = dists.log_prob(actions)

            reinforce_loss = -torch.mean(log_probs * returns)
            entropy_loss = -torch.mean(dists.entropy())  # Maximize entropy
            pbar.set_postfix(
                {
                    "reinforce_loss": reinforce_loss.detach().numpy().item(),
                    "entropy_loss": entropy_loss.detach().numpy().item(),
                }
            )
            total_loss = reinforce_loss + self._ent_coef * entropy_loss
            self._ent_coef = self._ent_coef * self._decay_factor
            total_loss.backward()
            self._opt.step()
            self._buffer = []
            self._curr_count = 0
            return True
        return False

    def store(
        self,
        state: Any,
        action: Any,
        reward: float,
        truncated: bool,
        terminated: bool,
        next_state: Any,
        *args,
        **kwargs
    ):
        done = truncated or terminated
        self._curr_traj.append((state, 1 if ACCEPT else 0, reward, done))
        self._curr_count += 1
        if done:
            self._buffer.append(self._curr_traj)
            self._curr_traj = []
