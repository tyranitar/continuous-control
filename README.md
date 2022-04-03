# Continuous Control

| A2C | D4PG |
| :-: | :-: |
| ![A2C](images/a2c_perf.gif) | ![D4PG](images/d4pg_perf.gif) |

This project includes implementations of A2C and D4PG for continuous control.

A2C: Advantage Actor-Critic

D4PG: Distributed Distributional Deep Deterministic Policy Gradients (a.k.a. Sally sells seashells by the seashore)

## Overview

In this reinforcement learning task, the agent is a double-jointed arm that is rewarded for keeping its hand inside a moving target location. The environment specifications are:

- `33` continuous-valued states corresponding to the position, rotation, velocity, and angular velocity of the arm.
- `4` continuous-valued actions corresponding to the torques applied to the two joints that fall in the range `[-1, 1]`.
- A reward of `0.1` for each time step that the agent has its hand in the target location, and a reward of `0` otherwise.

The agent’s goal is to achieve an average score of `30` over a window of 100 episodes.

## Setup

Follow the instructions [here](https://github.com/udacity/deep-reinforcement-learning/tree/dc65050c8f47b365560a30a112fb84f762005c6b#dependencies) to:

1. Create a `conda` environment.
2. Clone the Udacity Deep RL repository.
3. Install Python packages into the environment.
4. Create an IPython kernel using the environment.

The OpenAI Gym instructions can be skipped.

In order to watch the agent play the game, you also need to download the environment by following the instructions [here](https://github.com/udacity/deep-reinforcement-learning/tree/dc65050c8f47b365560a30a112fb84f762005c6b/p2_continuous-control#getting-started).

## Training the agent

Once you've completed the setup, you can:

1. Open `Continuous_Control.ipynb`.
2. Select the kernel created during setup.
3. Run all the cells in the notebook to train the agent.

## Watching the agent

Follow the instructions [here](https://github.com/udacity/deep-reinforcement-learning/tree/dc65050c8f47b365560a30a112fb84f762005c6b/p2_continuous-control), load the saved neural network weights (`a2c_policy.pth` for A2C or `d4pg_actor.pth` and `d4pg_critic.pth` for D4PG), and watch the trained agent interact with the environment!
