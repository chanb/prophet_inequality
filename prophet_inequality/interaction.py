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
) -> Tuple[float, float]:
    rng = np.random.RandomState(seed=seed)

    oracle_rewards = 0.0
    for _ in tqdm(range(num_trials)):
        env.reset(rng.randint(low=0, high=MAX_INT))
        oracle_rewards += np.sum(env.get_max_rewards()) / num_trials

    agent_rewards = 0.0
    if not skip_agent:
        for _ in tqdm(range(num_trials)):
            state = env.reset(rng.randint(low=0, high=MAX_INT))
            done = False
            curr_trial_reward = 0
            while not done:
                decision = agent.act(state)
                state, reward, truncated, terminated, _ = env.step(decision)
                done = truncated or terminated
                curr_trial_reward += reward
            agent_rewards += curr_trial_reward / num_trials

    return oracle_rewards, agent_rewards
