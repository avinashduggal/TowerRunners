---
layout: default
title: Proposal
---

## Summary of the Project
The goal of this project is to develop an AI agent in Minecraft that can automatically manage a small farm using reinforcement learning. The agent's task is to walk through a predefined farming area, detect when crops (e.g., wheat, carrots, or potatoes) are fully grown, harvest them, and immediately replant seeds in the harvested spots. The agent will take as input the current state of each block in the farming area, such as whether a crop is fully grown, freshly planted, or empty, as well as its own position and inventory status. The output will be the sequence of actions the agent performs, such as harvesting, replanting, or moving to a new block. This project has practical applications in simulating automated farming systems and exploring resource optimization in controlled environments. It also serves as a simple yet effective testbed for multi-step decision-making and task prioritization in RL.

## AI/ML Algorithms
An algorithm that the team anticipates on using for our project is Q-learning that uses a model free approach for traing the agent to maximize crops harvested.

## Evaluation Plan
We are planning to use reinforcement learning for this project. Some rewards we have thought of are:
- +1 points if broke a fully grown crop
- +.5 points if did pick up the crop droppings from the ground
- +.5 points if replanted the crop
- -1 points if broke crop prematurely
- -.5 points if did not pick up the crop droppings
- -.5 if did not replant the crop
- -2 points if non-crop block is broken

The harvest is successful if it has the entire variety of crops in its inventory.

For quantitative evalutions, we'll be looking at our agent's replanting success rate which is the proportion of blocks with crops on them. If there are blocks without crops on them, this indicates the agent harvested a crop and unsuccessfully replanted them. Another source of evaluation is harvesting success rate, which is the proportion of blocks that the agent broke which were crops. A potential baseline that we'll be using is a random action policy to compare the quantitivate values between the agents. We expect the the RL agent to improve by 30% compared to the baseline. 

For qualitative evaluations, we want to check if the agent is able to distinguish between fully grown crops or partially grown crops to ensure that it's not making simple mistakes. One thing we can do is to have a field that has a crops at different harvesting levels. We're also looking into checking whether the agent is adhering to the objectives and targeting only crops (not other blocks) by using a heatmap. I would also set up the field so there are some sparsity between crops to see if the agent is wandering around until it detects crops.

## Meet the Instructor
The earliest date the team plans to meet with the instructor is January 28th.

## AI Tool Usage
At the moment of this proposal creation, we currently do not know if we will use AI tools, and in the event we do, we do not know which AI Tools we shall utilize. Though one AI tool we might use is ChatGPT to ask it also the feasibility of our current approaches which should help us determine a solution.
