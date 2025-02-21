import time
from obstacle_tower_env import ObstacleTowerEnv
from stable_baselines3 import PPO

# Create the environment in realtime mode so the Unity window stays visible.
env = ObstacleTowerEnv(retro=False, realtime_mode=True)

# Option 1: Train a new PPO model
# (Note: Training may take a while; you might want to run training on HPC or load a pre-trained model.)
model = PPO("CnnPolicy", env, verbose=1)
# For demonstration purposes, we run a short training.
model.learn(total_timesteps=5000)
model.save("ppo_obstacle_tower")

# Option 2: Load a pre-trained PPO model (uncomment if you have one)
# model = PPO.load("ppo_obstacle_tower", env=env)

# Reset the environment to start a new episode.
obs = env.reset()
done = False

print("Starting continuous simulation. Press Ctrl+C to exit.")

# Continuous simulation loop.
while True:
    # Let the PPO model choose an action based on the current observation.
    action, _ = model.predict(obs)

    # Step the environment with the selected action.
    obs, reward, done, info = env.step(action)

    # Optionally, you can print out some info to the console.
    print("Reward:", reward)

    # If the episode ends, reset the environment.
    if done:
        obs = env.reset()

    # Sleep a little bit to control the simulation speed if necessary.
    time.sleep(0.05)

# When you want to end the simulation, you can call env.close().
