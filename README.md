# CMPUT 676 Assignment 1 Part (b) Winter 2022

## Installation
### Prerequisites:
- Python 3.10

You may install the package using pip (preferably under virtual environment):
```
cd ${PATH_TO_PROJECT_DIR}
pip install -e .
```

### Experiments
The experiments reside in the `scripts` directory. You may simply run the files for corresponding experiments:
```
# For first part
python experiment_single_item.py

# For second/third part
python experiment_multi_item.py

# For RL experiment
python experiment_rl.py
```
Note that for `experiment_multi_item.py`, we will need to change `num_accepts` to modify `k`.

### Code Structure
```
.
├── prophet_inequality/
│   ├── agent.py # policies/learners
│   ├── constants.py # shared constant variables
│   ├── environment.py # various problem formulations of prophet inequalities
│   ├── interaction.py # code to interact with the problem instances
│   └── utils.py # helper code
└── scripts/
    ├── experiment_single_item.py # experiment script for (i)
    └── experiment_multi_item.py # experiment script for (ii) & (iii)
```

#### Agents
We provide multiple agents for prophet inequalities:
- `Threshold`: Accept as long as value is above specified threshold
- `Randomized`: Accept based on a bernoulli distribution
- `RandomizedThreshold`: Accept when value is above specified threshold or when we get a `1` from bernoulli distribution
- `REINFORCE`: A policy is learned based on experiences with different problem instances
