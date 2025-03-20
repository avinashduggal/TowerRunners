---
layout: default
title: Final Report
---

## Video
## Project Summary
Our team, Tower Runners, worked on developing a reinforcement learning agent to navigate through the Obstacle Towers environment. Obstacle Towers was a game built for an AI and machine learning competition to benchmark the capabilities of the participant's agents. These capabilities include being able to solve intricate puzzles, path finding in an environment with few yet sparse rewards, computer vision, and determining how to balance exploration and exploitation. Within the Obstacle Tower itself, the agent is tasked with the overarching goal of ascending to the highest floor possible within a predetermined amount of time. This task is comprised of navigating through the rooms, collecting keys or solving puzzles to unlock doors (when necessary), and ascending to higher floors, where difficulty increases as the agent reaches higher floors.

The difficulty of the task primarily lies within the nature of sparse rewards and the agent's ability to make generalizations in a randomly generated environment. In reinforcement learning, agents are rewarded for behavior that lead to ideal states and outcomes, and are penalized for the opposite. It's crucial to be able

## Approach
A baseline model that used to benchmark the other methods we planned to use was by using a Random Policy to sample moves to our agent. This model omitted the need for training because it would continuously decide on random actions to take until the end of each episode. 

Mention PPO, PPO with rewards (sprase and dense rewards), and Rainbow models. Include the reasons for why we decided to use these models and implemented changes to the environment. What was the objective and how would using these methods help us in completing our task.

### PPO with Rewards
A method we used to train our reinforcement learning model was using the Proximal Policy Policy algorithm with reward shaping to guide the agent to completing objectives e.g. collecting keys, solving puzzles, pathing through the doors, and ascending to higher floors. In the early stages of reward shaping, we implemented a straight forward way to let the agent know the type of actions we wanted to see more of by rewarding it when it would ascend to higher floors and the count for the keys collected increased. After several sessions of training the agent with this simple process of rewarding the agent, we noticed a major issue where the agent was having a difficult time finding pathing to the doors and exploring the rooms in a manner that would help them collect the keys. Therefore, it meant there were only a few occasions over several episodes of training where the agent would accomplish these tasks.

## Evaluation
## References
## AI Tool Usage
