import os

from datetime import datetime
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gym.spaces import Box
from obstacle_tower_env import ObstacleTowerEnv

from config import config_data

# ./models/MlpPolicy_SparseRewards_obstacle_tower_2025-03-09_21-11-33
# ./models/MlpPolicy_Dense+CustomSparseRewards_obstacle_tower_2025-03-16_13-26-21.zip
# ./models/MlpPolicy_Dense+CustomSparseRewards_obstacle_tower_2025-03-16_08-49-49.zips
model_path = "./models/MlpPolicy_Dense+CustomSparseRewards_obstacle_tower_2025-03-16_08-49-49.zip"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model directory '{model_path}' not found!")

def run_episode(env, model, render=True):
    """
    run_episode will have the model predict an action based on the agent's latest observation of the environment. It passes the action
    to the agent and the environment returns 4 things:

    - obs: new observation of the environment
    - reward: reward for action taken
    - done: if the episode is over, the agent will exit the environment
    - info: additional information that we won't be using here

    The funtion will output the total_rewards accumulated over one episode using the model.
    """

    obs = env.reset()[0]
    total_reward = 0
    done = False

    while not done:
        # Retrieve action from the model
        action, states = model.predict(obs)
        action = int(action[0]) if isinstance(action, np.ndarray) else action

        # Perform the selected action
        obs, reward, done, info = env.step([action])

        done = done[0]

        # Add to the cumulative reward
        total_reward += reward[0]

        # Optionally render the environment to see the agent in real time
        if render:
            env.render()

    print(f"Episode finished with total reward: {total_reward}")


if __name__ == '__main__':
    """
    The lines below will load a previously trained model to see how well it performs in a newly initialized environment.
    If desired, it can run multipled episodes by using a loop.
    """

    # Intialize the environment
    env = ObstacleTowerEnv('./ObstacleTower/obstacletower', retro=True, realtime_mode=True)
    # env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    model = PPO.load(model_path)

    run_episode(env, model, render=True)

    env.close()
