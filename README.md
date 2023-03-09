# RLalgcomparison
Comparison of different reinforcement learning algorithms on different game environments.

## Simulation
To run an algorithm, simply run ```python .\test_algorithm.py --task "Twenty48-version_x"``` e.g. to run the dqn algorithm for the Twenty48-v0 environment run ```python .\test_dqn.py --task "Twenty48-v0"```

## Versions
### Twenty48-v0
The standard implementation for a game of 2048 (stochastic environment)

### Twenty48-v1
The same tile movement mechanics as v0 but all randomness removed (deterministic environment)

## Tensorboard Log
run ```tensorboard --logdir .\log\Twenty48-version_x\algorithm``` to access analytics relating to the training efficiency of an algorithm on a particular version e.g. to access the analysis for the dqn algorithm for the Twenty48-v0 environment run ```tensorboard --logdir .\log\Twenty48-v0\dqn```
