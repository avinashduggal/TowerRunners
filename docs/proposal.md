---
layout: default
title: Proposal
---

## Summary of the Project
The goal of this project is to develop an AI agent in the Obstacle Tower environment that can go up through the floors using reinforcement learning, with the bonus of added-on noise. The agent's task is to navigate through multiple floors of the tower, solving puzzles and overcoming obstacles along the way. The agent's input will take the current state of each floor, including the layout of obstacles, the position of keys and doors, and its own position and inventory status. The output will be the sequence of actions that the agent performs, such as moving, jumping, picking up items, and unlocking doors. This project has practical applications in developing intelligent agents capable of handling complex, dynamic environments and exploring the effects of noise on agent performance. It also serves as a challenging testbed for multi-step decision-making and task prioritization in reinforcement learning.

## AI/ML Algorithms
An algorithm that the team anticipates on using for our project is 

## Evaluation Plan
We are planning to use reinforcement learning for this project. Some rewards we have thought of are:
- +1 point for getting to the next floor
- +.5 points going through doors
- +.5 points for picking up key
- +.5 points for moving the block to the correct tile to unlock the door
- -.5 points for running into the door (needs to be unlocked)
- -.5 points for moving the block to the wrong tile to unlock the door
- -1 going down a floor

The agent is successful every time they go up to the next floor.

For the quantitative evaluations of our agent, we're manly focusing on the expected rewards that the agent is able to accumulate while training. This may be affected by the amount of computation and time that will be used in order to achieve an adequate average reward. Another metric is the completed rate which is the proportion of floors the agent is able to successfully navigate through. This lets us know if the agent is even capable of achieving the task. A baseline model we'll be using to obtain insights on how our agent is affected by various hyperparameters (e.g. Learning rate, steps, batch size, etc.) is the Proximal Policy Optimization model that uses the Multi-Layer Perception Policy (MLP Policy). More importantly, we can learn how adjustments to these fields will help our agent with task completion. We expect our RL agent to improve by 20% compared to the baseline.

For the qualitative evaluations, we want the agent to be able to distinguish between the different types of doors and understand the requirements (e.g. unlocking door with keys) of traveling through it. This is important because if it's familiar with the doors, then it's able to replicate successful behaviors instead of repeating the same mistakes. We also want to observe whether the agent is spending its time exploring parts of the environment that will get them closer to the next floor versus taking random actions and hoping to be successful. Another detail we want to observe is how consistent our agent is throughout the various floors. As it climbs the different floors, it becomes increasingly difficult so we want to know if its behavior changes (positively or negatively) accordingly.

## Meet the Instructor
The earliest date the team plans to meet with the instructor is Febuary 26th.

## AI Tool Usage
At the moment of this proposal creation, we currently do not know if we will use AI tools, and in the event we do, we do not know which AI Tools we shall utilize. Though one AI tool we might use is ChatGPT to ask it also the feasibility of our current approaches which should help us determine a solution.
