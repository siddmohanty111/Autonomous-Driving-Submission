import gymnasium as gym
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, DQN, SAC
from multi_scenario_env import MultiScenarioHighwayEnv


def choose_model(env):
    if env.current_env_id == "highway-v0":
        model = PPO.load("CS272Project/group11custom/group11models/highway/lidar_ppo")
    elif env.current_env_id == "merge-v0":
        model = DQN.load("CS272Project/group11custom/group11models/merge/lidar_DQN")
    else:
        model = DQN.load(
            "CS272Project/group11custom/group11models/intersection/lidar_DQN"
        )
    return model


def run_group11(env, n_episodes=500):
    rewards = []

    for _ in range(n_episodes):
        print(f"Running episode {_ + 1}/{n_episodes}", end="\r")
        obs, info = env.reset()
        done = False
        ep_reward = 0
        model = choose_model(env)
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        rewards.append(ep_reward)

    return rewards


def generate_single_model_violin(env, save_prefix=None):

    # pass env
    env = env

    # Evaluate
    rewards = run_group11(env)

    # Prepare DataFrame
    import pandas as pd

    df = pd.DataFrame({"reward": rewards, "model": [save_prefix] * len(rewards)})

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


generate_single_model_violin(
    env=MultiScenarioHighwayEnv(),
    save_prefix="multi_scenerio",
)
