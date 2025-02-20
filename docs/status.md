---
layout: default
title: Status
---
## Project Summary
Our project, JARVIS, focuses on solving the Obstacle Tower challenge using machine learning. Obstacle Tower is a procedurally generated environment designed to evaluate an AI agent's ability to generalize across multiple tasks such as vision, locomotion, planning, and puzzle-solving. The agent's goal is to navigate through increasingly complex floors, adapting to new obstacles and challenges. Our aim is to develop a reinforcement learning-based solution capable of learning effective strategies to improve performance across different tower configurations.

## Approach
We employ reinforcement learning (RL), leveraging deep learning architectures to train an agent capable of navigating the Obstacle Tower environment. Our methodology consists of the following key components:
*Environment Setup*: We integrate the official Gym interface of Obstacle Tower, ensuring compatibility with reinforcement learning frameworks like Stable Baselines3 and RLlib.
*Model Selection*:
- Deep Q-Networks (DQN): Used for early-stage learning, particularly for discrete action spaces.
- Proximal Policy Optimization (PPO): Our primary algorithm, as PPO is well-suited for environments with continuous and discrete action spaces.
- A3C (Asynchronous Advantage Actor-Critic): Experimented for multi-threaded training.
*Training Process*:
We define states as the RGB image observations and actions as discrete movement choices (e.g., move forward, jump, rotate, etc.).
The reward function incentivizes progression to new floors while penalizing unnecessary movements or collisions.
We train the agent using a curriculum approach, increasing the difficulty gradually.
*Hyperparameter Tuning*:

## Evaluation
Our evaluation consists of quantitative and qualitative metrics:
### Quantitative Metrics

Success Rate: Percentage of times the agent successfully reaches a new floor.

Average Episode Reward: Measures learning improvement over time.

Episode Length: Tracks how efficiently the agent solves a level.

Generalization Performance: Tested by training on specific floors and evaluating on unseen ones.

### Qualitative Analysis

Visualization of Agent Behavior: Screenshots and videos show how the agent learns better strategies over time.

Failure Analysis: Identifies common mistakes, such as getting stuck in loops, misjudging jumps, or inefficient movements.

We plot performance graphs to illustrate the learning curve of different RL models.

## Remaining Goals and Challenges
### End of Quarter Goals
- Fine-tuning the Reward Function: Experiment with reward shaping to guide better decision-making.

- Implementing Curriculum Learning: Gradually increase complexity to ease training.

- Comparing with Human Performance: Use a baseline comparison to evaluate model efficiency.

- Training with Different Seeds: Ensure robustness by evaluating the agent’s adaptability to unseen conditions.

### Anticipated Challenges

- Exploration vs. Exploitation Trade-off: Ensuring the agent efficiently explores new strategies while retaining learned ones.

- Sparse Rewards: Addressing the challenge of long-term dependencies in credit assignment.

- Computational Constraints: Reinforcement learning requires significant training time, so optimizing resources is crucial.

## Video Summary

## Resources Used
- [Obstacle Tower GitHub Repository](https://github.com/Unity-Technologies/obstacle-tower-env)