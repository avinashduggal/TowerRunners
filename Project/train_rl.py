from datetime import datetime
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gym.spaces import Box
from obstacle_tower_env import ObstacleTowerEnv

policy_name = "CnnPolicy"
experiment_logdir = f"OT_logs/{policy_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
env = ObstacleTowerEnv('./ObstacleTower/obstacletower', retro=True, realtime_mode=False)
env = Monitor(env, filename=experiment_logdir)

# Wrap the environment in a DummyVecEnv for stable-baselines
env = DummyVecEnv([lambda: env])

# Now, the PPO model should work with the wrapped environment
model = PPO(policy_name, env, verbose=1)
model.learn(total_timesteps=100000)

# Save the model
model.save(f"models/{policy_name}_obstacle_tower_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
