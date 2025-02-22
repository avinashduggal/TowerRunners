from datetime import datetime
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gym.spaces import Box
from obstacle_tower_env import ObstacleTowerEnv


class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        return x.view(x.size(0), -1)


def calculate_yellow_reward(observation):
    hsv = cv2.cvtColor(observation, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    yellow_ratio = np.sum(yellow_mask) / (observation.shape[0] * observation.shape[1])

    height, width = observation.shape[:2]
    center_x, center_y = width // 2, height // 2
    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    sigma = max(height, width) / 4
    gaussian_weights = np.exp(-((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2) / (2 * sigma ** 2))

    weighted_yellow_pixels = np.sum(yellow_mask * gaussian_weights)

    max_weighted_value = np.sum(gaussian_weights)
    reward = (weighted_yellow_pixels / max_weighted_value) * 10

    return reward


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def preprocess_observation(observation):
    observation = transform(observation)
    observation = observation.unsqueeze(0)
    features = feature_extractor(observation)
    return features.numpy().flatten()


feature_extractor = ResNetFeatureExtractor()


class ResNetWrappedEnv(gym.Wrapper):
    def __init__(self, env):
        super(ResNetWrappedEnv, self).__init__(env)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(512,), dtype=np.float32)

    def reset(self):
        obs = self.env.reset()
        return preprocess_observation(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        yellow_reward = calculate_yellow_reward(obs)
        reward += yellow_reward
        return preprocess_observation(obs), reward, done, info


policy_name = "MlpPolicy"
experiment_logdir = f"OT_logs/{policy_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
env = ObstacleTowerEnv('./ObstacleTower/obstacletower', retro=True, realtime_mode=False)
env = Monitor(env, filename=experiment_logdir)
env = ResNetWrappedEnv(env)
env = DummyVecEnv([lambda: env])

model = PPO(policy_name, env, verbose=1)


def train_with_fixed_seeds(env, model, seeds, total_timesteps):
    for seed in seeds:
        env.seed(seed)
        model.learn(total_timesteps=total_timesteps // len(seeds))

train_with_fixed_seeds(env, model, seeds=[1001, 1002, 1003, 1004, 1005], total_timesteps=100000)

model.save(f"models/{policy_name}_obstacle_tower_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
