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


# ResNet Feature Extractor
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove final classification layer

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        return x.view(x.size(0), -1)  # Flatten


# Function to calculate additional reward based on yellow pixels
def calculate_yellow_reward(observation):
    # Convert observation to HSV
    hsv = cv2.cvtColor(observation, cv2.COLOR_RGB2HSV)

    # Define yellow color range in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    # Create mask to detect yellow pixels
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Calculate the percentage of yellow pixels
    yellow_ratio = np.sum(yellow_mask) / (observation.shape[0] * observation.shape[1])

    # Create a Gaussian weighting matrix to prioritize center yellow pixels
    height, width = observation.shape[:2]
    center_x, center_y = width // 2, height // 2
    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    # Gaussian weight centered in the middle
    sigma = max(height, width) / 4
    gaussian_weights = np.exp(-((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2) / (2 * sigma ** 2))

    # Apply the weights to yellow pixels
    weighted_yellow_pixels = np.sum(yellow_mask * gaussian_weights)

    # Normalize reward by max possible weighted value
    max_weighted_value = np.sum(gaussian_weights)
    reward = (weighted_yellow_pixels / max_weighted_value) * 10  # Scale appropriately

    return reward


# Preprocess function
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # ResNet requires 224x224 input
    transforms.ToTensor(),
])


def preprocess_observation(observation):
    observation = transform(observation)  # Apply transformations
    observation = observation.unsqueeze(0)  # Add batch dimension
    features = feature_extractor(observation)  # Extract features
    return features.numpy().flatten()


# Initialize feature extractor
feature_extractor = ResNetFeatureExtractor()


# Custom environment wrapper to apply ResNet feature extraction + Reward Shaping
class ResNetWrappedEnv(gym.Wrapper):
    def __init__(self, env):
        super(ResNetWrappedEnv, self).__init__(env)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(512,), dtype=np.float32)  # ResNet18 output size

    def reset(self):
        obs = self.env.reset()
        return preprocess_observation(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Compute additional reward for moving towards yellow regions
        yellow_reward = calculate_yellow_reward(obs)

        # Modify total reward
        reward += yellow_reward

        return preprocess_observation(obs), reward, done, info


# Initialize environment
policy_name = "MlpPolicy"
experiment_logdir = f"OT_logs/{policy_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
env = ObstacleTowerEnv('./ObstacleTower/obstacletower', retro=True, realtime_mode=False)
env = Monitor(env, filename=experiment_logdir)
env = ResNetWrappedEnv(env)  # Apply ResNet preprocessing + reward shaping
env = DummyVecEnv([lambda: env])

# Define PPO model
model = PPO(policy_name, env, verbose=1)


# Function to train model with different seeds
def train_with_fixed_seeds(env, model, seeds, total_timesteps):
    for seed in seeds:
        env.seed(seed)
        model.learn(total_timesteps=total_timesteps // len(seeds))  # Divide training among seeds


# Train the model
train_with_fixed_seeds(env, model, seeds=[1001, 1002, 1003, 1004, 1005], total_timesteps=100000)

# Save the trained model
model.save(f"models/{policy_name}_obstacle_tower_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
