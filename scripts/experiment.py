from prophet_inequality.agent import ThresholdAgent
from prophet_inequality.environment import TruncatedGaussianReward
from prophet_inequality.interaction import evaluate
from prophet_inequality.utils import generate_truncated_gaussians

import numpy as np


seed = 42
trials_for_lambda = 500000
num_trials = 100000
num_dists = 100

# Settings for valid truncated normals
bounds_lim = [-100, 100]
locs_lim = [-100, 100]
scales_lim = 100

locs, scales, bounds = generate_truncated_gaussians(
    num_dists, bounds_lim, locs_lim, scales_lim, seed
)

env = TruncatedGaussianReward(locs=locs, scales=scales, bounds=bounds, num_accepts=1)

oracle_rewards, _ = evaluate(env, None, trials_for_lambda, seed - 1, skip_agent=True)
threshold = 0.5 * oracle_rewards
print("Obtained lambda: {}".format(threshold))

agent = ThresholdAgent(threshold=threshold)
oracle_rewards, agent_rewards = evaluate(env, agent, num_trials, seed)
reward_ratio = agent_rewards / oracle_rewards
print(
    "The empirical expected ratio between ALG and max_i X_i is {}".format(reward_ratio)
)
