from prophet_inequality.agent import REINFORCEAgent
from prophet_inequality.constants import MAX_LOC
from prophet_inequality.environment import RLTruncatedGaussianReward
from prophet_inequality.interaction import evaluate
from prophet_inequality.utils import (
    generate_truncated_gaussians_bounded_difference,
    plot_average_returns,
)

import numpy as np
import torch

seed = 42
trials_for_lambda = 500000
num_trials = 1000000
num_dists = 100

# Settings for valid truncated normals
bounds_lim = [-1, 1]
loc_init = 0.0
scales_lim = 1.0
bounded_diff = 10.0

# Set seed
np.random.seed(seed)
torch.manual_seed(seed)
locs, scales, bounds = generate_truncated_gaussians_bounded_difference(
    num_dists, bounds_lim, loc_init, scales_lim, bounded_diff, seed
)

env = RLTruncatedGaussianReward(
    locs=locs,
    scales=scales,
    bounds=bounds,
    num_accepts=1,
    history=3,
    subtract_max=True,
    normalize=True,
)

# Linear model
# model = torch.nn.Linear(6, 1)

# Non-linear model
model = torch.nn.Sequential(
    torch.nn.Linear(6, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)
)

# Basic SGD
opt = torch.optim.SGD(model.parameters(), lr=1e-3)

# Number of episodes per update
batch_size = 8192
ent_coef = np.max(np.abs(locs)) * 10
decay_factor = 0.999

agent = REINFORCEAgent(model, opt, batch_size, ent_coef, decay_factor)
oracle_rewards, agent_rewards, info = evaluate(
    env, agent, num_trials, seed, reduction=None
)
best_locs_rewards = np.asarray(info[MAX_LOC])

plot_average_returns(oracle_rewards, agent_rewards, best_locs_rewards)
