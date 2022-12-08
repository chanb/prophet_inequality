from prophet_inequality.agent import ThresholdAgent
from prophet_inequality.environment import TruncatedGaussianReward
from prophet_inequality.interaction import evaluate

import numpy as np


seed = 42
num_trials = 100000

rng = np.random.RandomState(seed)
env = TruncatedGaussianReward(
    locs=(0, 1, 2), scales=(1, 2, 3), bounds=((0, 1), (3, 10), (-10, 10)), num_accepts=1
)
agent = ThresholdAgent(threshold=0.5)

oracle_rewards, agent_rewards = evaluate(env, agent, num_trials, seed)
reward_ratio = np.mean(agent_rewards) / np.mean(oracle_rewards)
print(
    "The empirical expected ratio between ALG and max_i X_i is {}".format(reward_ratio)
)
