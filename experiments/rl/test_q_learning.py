import json
import pathlib

import numpy as np
from loguru import logger
import matplotlib.pyplot as plt

from src.envs.env import SIRSEnv

# plt.ion()

output_dir = "outputs/rl/sirs_q_learning/experiment_6"
pathlib.Path(f"{output_dir}").mkdir(parents=True, exist_ok=False)

# Algorithm parameters
alg_params = {
    "n_episodes": 1e7,
    "alpha": 0.5,
    "epsilon": 0.1,
    "discount_factor": 0.99,
}
logger.info(f"Algorithm parameters: {alg_params}")
with open(f"{output_dir}/alg_params.json", "w") as f:
    json.dump(alg_params, f)

# Environment parameters
env_params = {
    "size": 3,
    "encounter_rate": 0.8,
    "recovery_rate": 0.35,
    "resusceptible_rate": 0.15,
    "vaccination_rate": 0.1,
    "cost_infection": 1.5,
    "cost_lockdown": 1,
}
logger.info(f"Environment parameters: {env_params}")
with open(f"{output_dir}/env_params.json", "w") as f:
    json.dump(env_params, f)

env = SIRSEnv(
    **env_params,
)


def plot(Q):
    policy = Q.argmax(axis=2)
    f, ax = plt.subplots()
    states = [
        (m_s, m_i)
        for m_s in range(policy.shape[0])
        for m_i in range(policy.shape[1])
        if m_s + m_i <= env.size
    ]
    for m_s, m_i in states:
        if policy[m_s, m_i] == 0:
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
    # plt.plot()
    return f, ax


n_episodes = int(alg_params["n_episodes"])
alpha = alg_params["alpha"]
epsilon = alg_params["epsilon"]
discount_factor = alg_params["discount_factor"]
state, info = env.reset()

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

Q = np.zeros(
    (env.observation_space[0].n, env.observation_space[1].n, env.action_space.n)
)
trajectories_fname = f"{output_dir}/trajectories.txt"
with open(trajectories_fname, "w") as trajectories_file:
    trajectories_file.write("episode,step,m_s,m_i,action,reward,m_s_,m_i_\n")
# trajectories_file = open(trajectories_fname, "w", buffering=1)
# trajectories_file.write("episode,step,m_s,m_i,action,reward,m_s_,m_i_\n")
states_count = np.zeros((env.observation_space[0].n, env.observation_space[1].n))
trajectories_tmp = []
for episode in range(n_episodes):
    logger.info(f"Episode {episode}")
    state, info = env.reset()
    done = False
    steps = 0
    while not done:
        states_count[state] += 1
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = Q[state[0], state[1], :].argmax()
        next_state, reward, done, truncated, info = env.step(action)

        Q[state[0], state[1], action] = Q[state[0], state[1], action] + alpha * (
            reward
            + discount_factor * Q[next_state[0], next_state[1], :].max()
            - Q[state[0], state[1], action]
        )
        trajectories_tmp.append(
            (
                episode,
                steps,
                state[0],
                state[1],
                action,
                reward,
                next_state[0],
                next_state[1],
            )
        )
        # trajectories_file.write(
        #    f"{episode},{steps},{state[0]},{state[1]},{action},{reward},{next_state[0]},{next_state[1]}\n"
        # )
        state = next_state
        steps += 1
        if done:
            logger.info(f"Episode {episode} finished after {steps} steps")
    if (episode + 1) % 50000 == 0 or (episode + 1) == n_episodes:
        fig, ax = plot(Q)
        fig.savefig(f"{output_dir}/plot_{episode+1}.png")
        with open(f"{output_dir}/Q_{episode+1}.npy", "wb") as f_q:
            np.save(f_q, Q)
        with open(f"{output_dir}/states_count_{episode+1}.npy", "wb") as f_states_count:
            np.save(f_states_count, states_count)
        with open(trajectories_fname, "a") as trajectories_file:
            trajectories_file.writelines(
                [
                    ",".join(map(str, trajectory)) + "\n"
                    for trajectory in trajectories_tmp
                ]
            )
            trajectories_tmp = []
# trajectories_file.close()
plt.show()
print(0)
