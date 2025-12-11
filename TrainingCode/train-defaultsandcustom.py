import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import gymnasium as gym
import highway_env
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
import os


class LearningCurveCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        if len(self.model.ep_info_buffer) > 0:
            for ep_info in self.model.ep_info_buffer:
                self.episode_rewards.append(ep_info["r"])
                self.episode_lengths.append(ep_info["l"])
        return True


grayscaleconfig = {
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],
        "scaling": 1.75,
    },
}

lidarconfig = {
    "observation": {"type": "LidarObservation"},
}

group11Lidarconfig = {
    "observation": {
        "type": "LidarObservation",
        "cells": 32,
        "maximum_range": 60,
        "normalize": True,
    },
    "simulation_frequency": 15,
    "policy_frequency": 5,
}


model_file = "group11/highway/lidar_DQN"
total_timesteps = 150_000


def make_env():
    return gym.make("highway-v0", config=group11Lidarconfig)


env = make_env()
print(f"Created environment")

model = DQN("MlpPolicy", env, verbose=1)
# model = PPO("MlpPolicy", env, learning_rate=5e-4, gamma=0.8, verbose=1, device="cuda")
callback = LearningCurveCallback()

model.learn(total_timesteps, callback=callback)

# Save to Google Drive
model.save(model_file)
np.save(f"group11/highway/rewardsdqn.npy", callback.episode_rewards)
np.save(f"group11/highway/lengthsdqn.npy", callback.episode_lengths)
env.close()

if len(callback.episode_rewards) > 0:
    plt.figure(figsize=(10, 4))
    plt.plot(callback.episode_rewards, alpha=0.6, color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Part Training")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"group11/highway/learning_curve_partdqn.png", dpi=150)
    plt.show()
