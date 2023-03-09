# RLalgcomparison
Comparison of different reinforcement learning algorithms on different game environments.

## Simulation
To run an algorithm on a particular environment, simply run ```python .\test_algorithm.py --task "Twenty48-version_x"``` e.g. to run the dqn algorithm for the Twenty48-v0 environment run ```python .\test_dqn.py --task "Twenty48-v0"```.

## Versions
### Twenty48-v0
The standard implementation for a game of 2048 (stochastic environment).

### Twenty48-v1
The same tile movement mechanics as v0 but all randomness removed (deterministic environment).

## Tensorboard Log
To access analytics relating to the training efficiency of an algorithm on a particular version run ```tensorboard --logdir .\log\Twenty48-version_x\algorithm``` e.g. to access the analysis for the dqn algorithm for the Twenty48-v0 environment run ```tensorboard --logdir .\log\Twenty48-v0\dqn```.

## Reward Function
The reward is calculated as per the standard implementation of 2048. After a move, the resulting tile values from the newly merged tiles are summed and that becomes the reward for that step. Additionally, any move that does not change the state is rewarded with a negative reward (-0.1).

## Manual Testing
A file called manual_instance.py is provided to easily test the functionality of the game environment.
