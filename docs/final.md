---
layout: default
title: Final Report
---

## Video
## Project Summary
The goal of this project for our team, Tower Runners, is to developing a strategy using machine learning to solve the intricate puzzles introduced by the game Obstacle Towers. The problem we set out to solve was figuring out a method to develop a model that is capable of navigating throughout randomly generated puzzles, and obstacles in order to ascend to higher floors of the game (Insert images of the Obstacle Tower environment, keys, and puzzles). Considering Obstacle Towers was developed with the intention of examining the capabilities of reinforcement learning through its increasingly difficult puzzles, obstacles, and needs for computer vision and generalization skills, the team didn't need to perform additional set up to begin these tasks. 

## Approach
A baseline model that used to benchmark the other methods we planned to use was by using a Random Policy to sample moves to our agent. This model omitted the need for training because it would continuously decide on random actions to take until the end of each episode. 

Mention PPO, PPO with rewards (sprase and dense rewards), and Rainbow models. Include the reasons for why we decided to use these models and implemented changes to the environment. What was the objective and how would using these methods help us in completing our task.

### PPO with Rewards
A method we used to train our reinforcement learning model was using the Proximal Policy Policy algorithm with reward shaping to guide the agent to completing objectives e.g. collecting keys, solving puzzles, pathing through the doors, and ascending to higher floors. In the early stages of reward shaping, we implemented a straight forward way to let the agent know the type of actions we wanted to see more of by rewarding it when it would ascend to higher floors and the count for the keys collected increased. After several sessions of training the agent with this simple process of rewarding the agent, we noticed a major issue where the agent was having a difficult time finding pathing to the doors and exploring the rooms in a manner that would help them collect the keys. Therefore, it meant there were only a few occasions over several episodes of training where the agent would accomplish these tasks.

## Evaluation
## References
## AI Tool Usage
