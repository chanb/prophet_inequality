from prophet_inequality.agent import Agent
from prophet_inequality.constants import MAX_INT
from prophet_inequality.environment import Environment

from tqdm import tqdm
from typing import Tuple

import numpy as np


def evaluate(
    env: Environment,
    agent: Agent,
    num_trials: int,
    seed: int = None,
    skip_agent: bool = False,
    reduction: str = "mean",
) -> Tuple[float, float]:
    rng = np.random.RandomState(seed=seed)

    agent_rewards = []
    oracle_rewards = []
    for _ in tqdm(range(num_trials)):
        state = env.reset(rng.randint(low=0, high=MAX_INT))
        if not skip_agent:
            done = False
            curr_trial_reward = 0
            while not done:
                action = agent.act(state)
                next_state, reward, truncated, terminated, _ = env.step(action)
                done = truncated or terminated
                curr_trial_reward += reward
                agent.store(state, action, reward, truncated, terminated, next_state)
                state = next_state
                agent.update()
            agent_rewards.append(curr_trial_reward)
        oracle_rewards.append(np.sum(env.get_max_rewards()))

    if reduction == "mean":
        return np.mean(oracle_rewards), np.mean(agent_rewards)
    elif reduction is None:
        return np.asarray(oracle_rewards), np.asarray(agent_rewards)
    else:
        raise NotImplementedError
