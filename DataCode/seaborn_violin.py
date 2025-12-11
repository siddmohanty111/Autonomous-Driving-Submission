import gymnasium as gym
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, DQN, SAC
from multi_scenario_env import MultiScenarioHighwayEnv

env = MultiScenarioHighwayEnv()


def choose_model(env):
    if env.current_env_id == "highway-v0":
        model = PPO.load("group11/highway/lidar_ppo")
    elif env.current_env_id == "merge-v0":
        model = DQN.load("group11/merge/lidar_DQN")
    else:
        model = DQN.load("group11/intersection/lidar_DQN")
    return model


def run_model_episodes(model, env, n_episodes=500):
    rewards = []

    for _ in range(n_episodes):
        print(f"Running episode {_ + 1}/{n_episodes}", end="\r")
        obs, info = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        rewards.append(ep_reward)

    return rewards


def generate_single_model_violin(model_path, env_id, save_prefix=None):

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
    # Load model

    model = PPO.load(model_path)

    # Create env
    env = gym.make(env_id, config=grayscaleconfig)

    # Evaluate
    rewards = run_model_episodes(model, env)

    # Prepare DataFrame
    import pandas as pd

    df = pd.DataFrame({"reward": rewards, "model": [save_prefix] * len(rewards)})

    # Save name handling
    if save_prefix is None:
        save_prefix = model_path.replace("/", "_")

    save_file = f"{save_prefix}_violin_plot.png"

    # Plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df, x="model", y="reward", inner="box")
    plt.title(f"Reward Distribution for {save_prefix} (500 episodes)")
    plt.ylabel("Episode Reward")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig(save_file, dpi=200)
    plt.show()

    print(f"Saved violin plot → {save_file}")
    return df  # Return DataFrame if user wants to use it later


def generate_two_model_violin(model_path_1, model_path_2, env, names=None):
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

    # Load models
    model1 = DQN.load(model_path_1)
    model2 = PPO.load(model_path_2)

    # pass env
    env = env

    # Run 500 episodes each
    rewards1 = run_model_episodes(model1, env)
    rewards2 = run_model_episodes(model2, env)

    # Choose names
    if names is None:
        names = [model_path_1.replace("/", "_"), model_path_2.replace("/", "_")]

    # Build DataFrame
    import pandas as pd

    df = pd.DataFrame(
        {
            "reward": rewards1 + rewards2,
            "model": [names[0]] * len(rewards1) + [names[1]] * len(rewards2),
        }
    )

    # Output filename
    save_file = f"{names[0]}_{names[1]}_violin_plot.png"

    # Plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="model", y="reward", inner="box", palette="Set2")
    plt.title(f"Reward Distribution: {names[0]} vs {names[1]}")
    plt.ylabel("Episode Reward")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig(save_file, dpi=200)
    plt.show()

    print(f"Saved violin plot → {save_file}")
    return df


generate_single_model_violin(
    model_path="merge/grayscale_ppo",
    env=MultiScenarioHighwayEnv(),
    save_prefix="grayscale_ppo",
)
