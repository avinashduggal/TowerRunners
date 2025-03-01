from datetime import datetime
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gym.spaces import Box
from obstacle_tower_env import ObstacleTowerEnv

# Specify a policy for the model
policy_name = "CnnPolicy"

# Declare an an OT_logs directory to store the logs of the training sessions
experiment_logdir = f"OT_logs/{policy_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# Initialize the Obstacle Tower environment for the RL agent
env = ObstacleTowerEnv('./ObstacleTower/obstacletower', retro=True, realtime_mode=False)

# Wrap the environment with monitor to log the statistics e.g. mean rewards, timesteps, etc.
env = Monitor(env, filename=experiment_logdir)
env = DummyVecEnv([lambda: env])

# Iniitialize a Proximal Policy Optimization (PPO) model
model = PPO(policy_name, env, verbose=1,
            learning_rate=1e-3,             # Lower learning rate for cautious optimization to the model parameters
            n_steps=2048,                   # Taking n_steps before updating its policy
            batch_size=128,                 # Higher batch size to make updates based on generalized results over many batches of actions (samples)
            gamma=0.995)                    # Higher far sightedness. Possibly making poor decisions in the short run to gain big rewards in long-run

# Specify the total timesteps the model should train for
model.learn(total_timesteps=100000)

# Save the trained model so it can be used for evaluation
model.save(f"models/{policy_name}_obstacle_tower_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
