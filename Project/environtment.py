from obstacle_tower_env import ObstacleTowerEnv

env = ObstacleTowerEnv('./ObstacleTower/obstacletower.app', retro=False)
initial_observation = env.reset()

print("Environment initialized successfully!")

env.close()
