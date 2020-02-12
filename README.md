
# Adaptive Temporal Difference Learning (AdaTD)

Implemented AdaTD and compared it with other optimization methods in temporal difference learning.
Used in the paper [Adaptive Temporal Difference Learning with Linear Function Approximation](link).

## Getting started:

- Dependencies: Python (3.7.3), OpenAI gym (0.10.5), numpy (1.14.5), matplotlib (3.0.3), tensorflow (2.0.0), multiagent

- To install dpendency 'multiagent': Download 'multiagent-particle-envs-master' provided in folder 'AdaTD-master'. Then cd into its root directory and install it using 'pip install -e multiagent-particle-envs-master'

## Code structure

- `optimizers_linear.py`: Code for running AdaTD, ALRR-TD, and vanilla-TD simultaneously under linear value function approximation.

- `optimzers_nonlinear.py`: Code for running AdaTD and vanilla-TD simultaneously under nonlinear value function approximation.

## Acknowlegement

The dependency 'multiagent (0.0.1)' used in this repo is a modified version of [Multi-Agent Particle Environment](https://github.com/openai/multiagent-particle-envs/blob/master/README.md#multi-agent-particle-environment)
