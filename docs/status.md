---
layout: default
title: Status
---
## Project Summary
Our project, Tower Runners, focuses on solving the Obstacle Tower challenge using machine learning. Obstacle Tower is a procedurally generated environment designed to evaluate an AI agent's ability to generalize across multiple tasks such as vision, locomotion, planning, and puzzle-solving. The agent's goal is to navigate through increasingly complex floors, adapting to new obstacles and challenges. Our aim is to develop a reinforcement learning-based solution capable of learning effective strategies to improve performance across different tower configurations.

## Approach
We employ reinforcement learning (RL), invoking computer vision and leveraging deep learning architectures to train an agent capable of navigating the Obstacle Tower environment. Our baseline methodology consists of integrating the official Gym interface of Obstacle Tower, ensuring compatibility with reinforcement learning frameworks like Stable-Baselines3 and RLlib. As of right now, we are mostly working with Stable-Baselines3. In attempts to train our model, we are running the Multi-Layer Perceptron (Mlp) policy and the Convolutional Neural Network (Cnn) policy. The inputs we are using to help obtain the action the agent should take are the states of the environment and rewards. Once put in a new environment, the agent will analyze the environment and take action. It will analyze and take an action every step of the way. The trained model will need to predict the best action it can take to work towards its goal. In conjunction with the Mlp policy, we have also attempted to run a computer vision preprocessor using models such as ResNet, as it was reported that CNNs help improve decision-making (Trying to navigate...). We are looking towards this approach because the agent needs to visually identify its location within the current state of the environment, and learn where to go to proceed to the next floor.

As of right now, training in the environment takes an immense amount of time. I suspect this is because it is a 3D environment as opposed to 2D from the exercises. Since we are running the environment from our local machines, we are leaving our machines running for hours on end up to 100,000 timesteps to see how the model trains and then testing the model in our evaluation script. The hyperparameters we are using right now are:
        learning_rate = 3e-4,
        n_steps = 2048,
        batch_size = 64,
        n_epochs = 10,
        gamma = 0.99,
        gae_lambda: = 0.95,
        clip_range = 0.2,
        clip_range_vf = None,
        ent_coef = 0.0.

We have tried tweaking hyperparameters such as learning rate, gamma, and clip range +/- 2.

## Evaluation
We are able to run and test different algorithms in our model to see how they perform, so the environment does run correctly, albeit takes an immensely large amount of time to run. Our evaluation consists of quantitative and qualitative metrics:
### Quantitative Metrics
For quantitative metrics, we evaluated our project observing the ep_rew_mean collected in our training, and the average floors climbed and rewards for testing the classifiers. In the training phase, the rewards, length, and time spent per iteration are saved into a CSV file, and you can observe that as time goes on and more iterations are completed, each of the different RL algorithms becomes more successful by displaying higher rewards, obviously at different rates. Another observation we have made is that when the reward increases, so does the length of the episode. This is because going to the next floor gives you additional time, hence increasing the length of the episode.

Success Rate: Percentage of times the agent successfully reaches a new floor.

Average Episode Reward: Measures learning improvement over time.

Episode Length: Tracks how efficiently the agent solves a level.

Generalization Performance: Tested by training on seeds/towers and evaluating on unseen ones.

### Qualitative Analysis
Our agent in all different algorithms often struggles to make it past the first floor. It is jumping in every direction and many times get's stuck in corners or loops. This is something that needs to be directly addressed and has been our biggest issue. In all of our runs, the agent has yet to reach the second floor , 

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
- [Obstacle Tower: A Generalization Challenge in Vision, Control, and Planning](https://arxiv.org/abs/1902.01378)
- [PPO Dash: Improving Generalization in Deep Reinforcement Learning](https://arxiv.org/abs/1907.06704)
- [Trying to navigate in the Obstacle Tower environment with Reinforcement Learning](https://smartcat.io/tech-blog/data-science/trying-to-navigate-in-the-obstacle-tower-environment-with-reinforcement-learning/)
