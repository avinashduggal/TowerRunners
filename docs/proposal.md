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

For quantitative evaluations, we'll be looking at our agent's ability to solve puzzles and navigate through the floors. One metric is the completion rate, which is the proportion of floors the agent successfully navigates. If the agent fails to reach the next floor, it indicates the agent did not solve the puzzle correctly or navigate efficiently. Another source of evaluation is the time taken to complete each floor. A potential baseline we'll be using is a PPO model with a Multi-Layer Perceptron Policy (MlpPolicy) to compare the quantitative values between the agents. We expect our RL agent to improve by 20% compared to the baseline.

For qualitative evaluations, we want to check if the agent is able to distinguish between the different types of doors to understand the requirements of going through the door. This could be a basic door with no criteria to unlock it or a door that requires some sort of key. 

## Meet the Instructor
The earliest date the team plans to meet with the instructor is January 28th.

## AI Tool Usage
At the moment of this proposal creation, we currently do not know if we will use AI tools, and in the event we do, we do not know which AI Tools we shall utilize. Though one AI tool we might use is ChatGPT to ask it also the feasibility of our current approaches which should help us determine a solution.
