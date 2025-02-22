from obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation, UnityGymException
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np


def run_episode(env, model):
    obs = env.reset()
    done = False
    episode_return = 0.0

    while not done:
        # Predict action without converting to int.
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)

        # Since we're using DummyVecEnv, extract the single environment's values.
        episode_return += reward[0]

    return episode_return



if __name__ == "__main__":
    # Seeds for evaluation
    eval_seeds = [1001, 1002, 1003, 1004, 1005]

    # Create the base environment
    env = ObstacleTowerEnv("./ObstacleTower/obstacletower", retro=True, realtime_mode=True)

    # Wrap it with the evaluation wrapper using the seeds
    env = ObstacleTowerEvaluation(env, eval_seeds)

    # Wrap with DummyVecEnv to match training
    env = DummyVecEnv([lambda: env])

    # Load the trained model (ensure the observation spaces match)
    model = PPO.load('./models/ppo_cnn_obstacle_tower', env=env)

    # Run episodes until evaluation is complete
    try:
        while not env.envs[0].evaluation_complete:
            episode_rew = run_episode(env, model)
            print("Episode return:", episode_rew)
    except UnityGymException as e:
        if "evaluation has completed" in str(e):
            pass  # Evaluation is complete; exit the loop.
        else:
            raise

    # Print final evaluation results
    print("Final evaluation results:", env.envs[0].results)
    env.close()
