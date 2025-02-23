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

Average Episode Reward: Measures learning improvement over time.

Episode Length: Tracks how efficiently the agent solves a level.

Generalization Performance: Tested by training on seeds/towers and evaluating on unseen ones.

In order to measure the performance of our RL agent using the Proximal Policy Optimization model, we ran several training sessions and recorded the average reward that it accumulated over time. Below, we've included the plot of one model that used the Multi-layer Perception policy, and the second model that used a residual network (Convolution Neural Network) policy. Before performing analysis of the models that we trained, we hoped the average reward accumulated by the agent using the residual network would've outperformed the agent that didn't. The original model that's barebones using only PPO obtained higher mean rewards and it also increased throughout it's training process. So the next thing we have to explore is how to fine-tune this portion of our model so the agent can fully utilize feature detection of the symbols and colors on the doors of various levels. Overall, we believe there's more to be understood about the affects of the hyperparameters e.g. learning rate, gamma (far-sightedness), batch size, and so on because our model was having trouble making progress by moving through the doors. We also need to increase the number of timesteps to train the models even longer.

![MLP and CNN Policy Models](https://github.com/user-attachments/assets/a51570ce-e213-4a55-89cd-79edc50da0e9)

### Qualitative Analysis

Among the different models that we trained using the algorithms that we've described, it often struggles with making past the first floor. Even compared to the model that use a random policy, our agent is continuously jumping in every direction and walks around in loops within a section of the room. This is a major concern that we have and it will be immediately addressed. In all the runs, the agent hasn't been able to reach the second floor because of the difficulty in navigating in the environment and making actions that prevents it from being stuck.

![example_agent_stuck](https://github.com/user-attachments/assets/b1058ff1-7f42-45dc-a706-5558fa6e8fc3)

There are occasions where the agent paths towards the door but doesn't fully go through and is walking back and forth. While we were experimenting with our methods, we were hoping the agent would consistently make its way to the doors, but it's repeated the same mistakes. After some period of time the agent even walks away from the door and ends up in the opposite side of the room. 

![example_agent_paths_to_door](https://github.com/user-attachments/assets/08b7196a-4a1d-41c0-b2a0-7c530c1a41f1)

Visualization of Agent Behavior: Screenshots and videos show how the agent learns better strategies over time.

Failure Analysis: Identifies common mistakes, such as getting stuck in loops, misjudging jumps, or inefficient movements.

We plot performance graphs to illustrate the learning curve of different RL models.

## Remaining Goals and Challenges
### End of Quarter Goals

Some ways that we believe our prototype is limited is its accuracy and its ability to showcase that its behaving "intelligently" throughout the process of training. I believe some contributing factors to this is the amount of time that the model spends training, the reward function it utilizes to guide decision making, and the lack of time it spends exploring its environment. As a result, some goals that we have in mind for the remainder for the remainder of the quarter is to fine-tune the reward function that our agent uses by increasing the complexity of how it determines the agent should be rewarded for its actions. It would also be more realistic if we randomized the seeds while training to ensure the agent is capable of adapting to conditions that it hasn't encountered before, thus it's not overfitting to the same environment and floors that it trains on. We're also hoping to continue learning about the affects of the hyperparameters of the model we're using from stable-baselines3 and playing around with them to achieve higher average rewards. We're certainly going to explore other algorithms to understand to serve as a means of comparison. We'd hope to analyze the components of the algorithms we're currently using and future ones so see how different implementations alters the way the agent is trying to complete the task.

### Anticipated Challenges

So far, challenges we believe we'll be facing throughout by the time of the final report is ensuring we're allocating the right resources to the methods we're using. We also need to find the right balance of making somewhat quick decisions and making the sacrifice of taking actions that seem unfavorable in the short-run, but actually benefitting it in traversing through the rooms and floors. As mentioned earlier, we want to work in a direction where our data is suggesting our model is performly well, but overfits to the seeds that we train it on. Resultingly, it would affect the way the model can generalize to new scenarios. Since we're also in the process of designing a reward function, we're worried that a poor reward function will create unintended behavior, so this is more so on the top of the list of worries compared to fine-tuning hyperparameters. Another concern we have is the complexity of the methods we're using because it may lead to long periods of training. This depends on factors that we've mentioned already, but it would give us less time to work on other things if we our model can't train in a reasonable amount of time. When it comes to developing our own reward function and changing hyperparameters, we will definitely be utilizing office hours to consult the individuals that are more knowledgeable in these areas. It would save us a lot of time if we have a better idea of the direction we should be working towards.

### Resources Used
In the early stages of development, we read the Unity documentation for Obstacle Towers on the public repository to help us set up the environment. It required some set up in the beginning by downloading the necessary packages and playing around with the examples to ensure everything was working as intended.

Since the code base for Obstacle Towers was deprecated, we struggled alot with version conflicts when installing dependencies. We had to look at the official documentation for various packages like stable-baselines3, numpy, mlagents, etc. to see which versions were compatible with one another. There were instances where we had to downgrade to lower versions because higher versions couldn't support the old packages used by the repository.

To help with the agent's task of differentiating between the colors of the doors to climb the floors, we did research on how to add this level of complexity of incoporating computer vision to train our model. But as a team, we weren't familiar with computer vision or convultional neural networks, so we utilized ChatGPT to guide us through the process. It suggested a method called residual networks (ResNet), so we did some research about the advantages that it provides, and we learned there's an issue with vanishing gradients. During optimization, if it's unable to make proper updates to the parameters of the model, then it'll have a difficult time training. Additionally, we were able to find a ResNet model through PyTorch and we read through the documentation on how to set it up.
  
- [Obstacle Tower GitHub Repository](https://github.com/Unity-Technologies/obstacle-tower-env)
- [Obstacle Tower: A Generalization Challenge in Vision, Control, and Planning](https://arxiv.org/abs/1902.01378)
- [PPO Dash: Improving Generalization in Deep Reinforcement Learning](https://arxiv.org/abs/1907.06704)
- [Trying to navigate in the Obstacle Tower environment with Reinforcement Learning](https://smartcat.io/tech-blog/data-science/trying-to-navigate-in-the-obstacle-tower-environment-with-reinforcement-learning/)
- [ResNet Deep Learning: PyTorch Documentation](https://pytorch.org/vision/main/models/resnet.html)
  
## Video Summary

<iframe width="560" height="315" src="https://www.youtube.com/embed/icrvvXN5Vi0?si=kRrAqUXKMrOgvkbi" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
