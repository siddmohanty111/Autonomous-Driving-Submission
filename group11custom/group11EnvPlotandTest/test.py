from multi_scenario_env import MultiScenarioHighwayEnv
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt
import time

env = MultiScenarioHighwayEnv(render_mode="human")

obs, info = env.reset()


def choose_model(env):
    if env.current_env_id == "highway-v0":
        model = PPO.load("group11/highway/lidar_ppo")
    elif env.current_env_id == "merge-v0":
        model = DQN.load("group11/merge/lidar_DQN")
    else:
        model = DQN.load("group11/intersection/lidar_DQN")
    return model


model = choose_model(env)

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    env.render()
    time.sleep(0.01)

    if done or truncated:
        obs, info = env.reset()
        model = choose_model(env)
