from src.envs.env import SIRSEnv

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

env_params = {
    "size": 3,
    "encounter_rate": 0.8,
    "recovery_rate": 0.35,
    "resusceptible_rate": 0.15,
    "vaccination_rate": 0.1,
    "cost_infection": 1.5,
    "cost_lockdown": 1,
}

gym.register(
    "SIRSEnv-v0",
    entry_point="src.envs.env:SIRSEnv",
    kwargs=env_params,
)

env = gym.make("SIRSEnv-v0")

model = PPO(MlpPolicy, env, verbose=1)


def evaluate(model, num_episodes=100, deterministic=True):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    vec_env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = vec_env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs, deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            # also note that the step only returns a 4-tuple, as the env that is returned
            # by model.get_env() is an sb3 vecenv that wraps the >v0.26 API
            obs, reward, done, info = vec_env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward


# Use a separate environement for evaluation
eval_env = gym.make("SIRSEnv-v0")

# Random Agent, before training
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# Train the agent for 10000 steps
model.learn(total_timesteps=100_000)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

import matplotlib.pyplot as plt

f, ax = plt.subplots()
states = [
    (m_s, m_i)
    for m_s in range(env_params["size"] + 1)
    for m_i in range(env_params["size"] + 1)
    if m_s + m_i <= env.size
]
for m_s, m_i in states:
    if model.predict([m_s, m_i], deterministic=True)[0] == 0:
        ax.plot(
            m_s,
            m_i,
            "x",
            color="red",
            label="confinement",
        )
    else:
        ax.plot(
            m_s,
            m_i,
            "o",
            color="green",
            label="max exposure",
        )
ax.set_title("Social Optimum Policy")
ax.set_xlabel("Susceptible")
ax.set_ylabel("Infected")
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())
plt.plot()
