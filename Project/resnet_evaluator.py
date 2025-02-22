from obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation, UnityGymException
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from resnet_trainer import ResNetWrappedEnv  # Adjust the import path as needed

def run_episode(env, model):
    obs = env.reset()
    done = False
    episode_return = 0.0

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_return += reward[0]
    return episode_return

if __name__ == "__main__":
    # Seeds for evaluation
    eval_seeds = [1001, 1002, 1003, 1004, 1005]

    # Create the base environment
    env = ObstacleTowerEnv("./ObstacleTower/obstacletower", retro=True, realtime_mode=True)

    # Wrap with the evaluation wrapper
    env = ObstacleTowerEvaluation(env, eval_seeds)

    # *** Apply your custom ResNet wrapper to match training ***
    env = ResNetWrappedEnv(env)

    # Wrap with DummyVecEnv to match training
    env = DummyVecEnv([lambda: env])

    # Load the trained model (observation spaces now match)
    model = PPO.load('./models/MlpPolicy_obstacle_tower_2025-02-21_19-58-54', env=env)

    try:
        while not env.envs[0].evaluation_complete:
            episode_rew = run_episode(env, model)
            print("Episode return:", episode_rew)
    except UnityGymException as e:
        if "evaluation has completed" in str(e):
            pass
        else:
            raise

    print("Final evaluation results:", env.envs[0].results)
    env.close()
