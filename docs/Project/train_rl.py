import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv  # Correct import
from obstacle_tower_env import ObstacleTowerEnv

class ObstacleTowerEnvWrapper(gym.Env):
    def __init__(self, env_path):
        self.env = ObstacleTowerEnv(env_path, retro=False)
        self.observation_space = self.env.observation_space[0]  # Assuming it's an image
        self.action_space = self.env.action_space

    def reset(self):
        obs = self.env.reset()
        return obs[0]  # Return only the visual observation

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs[0], reward, done, info  # Return only the visual observation

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()

# Initialize the environment with the custom wrapper
env = ObstacleTowerEnvWrapper('./ObstacleTower/obstacletower.x86_64')

# Wrap the environment in a DummyVecEnv for stable-baselines
env = DummyVecEnv([lambda: env])

# Now, the PPO model should work with the wrapped environment
model = PPO("MlpPolicy", env, verbose=1)  # Use CnnPolicy
model.learn(total_timesteps=100000)

# Save the model
model.save("ppo_obstacle_tower")
