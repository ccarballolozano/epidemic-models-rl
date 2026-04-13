import datetime
import json
import pathlib

import numpy as np
from loguru import logger
import matplotlib.pyplot as plt

from src.envs.env import SIRSEnv
from .q_iteration import compute_social_optimum_policy

# plt.ion()

date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"outputs/rl/sirs_q_learning/{date}"
pathlib.Path(f"{output_dir}").mkdir(parents=True, exist_ok=False)

# Algorithm parameters
alg_params = {
    "n_episodes": 1e7,
    "alpha_max": 0.5,
    "alpha_min": 1e-4,
    "alpha_decay": 1e-4,
    "epsilon": 0.1,
    "discount_factor": 0.99,
}

logger.info(f"Algorithm parameters: {alg_params}")
with open(f"{output_dir}/alg_params.json", "w") as f:
    json.dump(alg_params, f)

# Environment parameters
# env_params = {
#    "size": 3,
#    "encounter_rate": 1,
#    "recovery_rate": 0.5,
#    "resusceptible_rate": 0.2,
#    "vaccination_rate": 0,
#    "cost_infection": 2,
#    "cost_lockdown": 1,
# }
env_params = {
    "size": 3,
    "encounter_rate": 1.1,
    "recovery_rate": 0.6,
    "resusceptible_rate": 0.3,
    "vaccination_rate": 0.2,
    "cost_infection": 2,
    "cost_lockdown": 1,
}
logger.info(f"Environment parameters: {env_params}")
with open(f"{output_dir}/env_params.json", "w") as f:
    json.dump(env_params, f)

env = SIRSEnv(
    **env_params,
)

_, Q_true = compute_social_optimum_policy(
    size=env_params["size"],
    encounter_rate=env_params["encounter_rate"],
    recovery_rate=env_params["recovery_rate"],
    resusceptible_rate=env_params["resusceptible_rate"],
    vaccination_rate=env_params["vaccination_rate"],
    cost_infection=env_params["cost_infection"],
    cost_lockdown=env_params["cost_lockdown"],
    discount_factor=alg_params["discount_factor"],
    theta=1e-12,
    max_iterations=1e6,
)
Q_ = np.zeros((env_params["size"] + 1, env_params["size"] + 1, env.action_space.n))
for k, v in Q_true.items():
    Q_[k] = v
Q_true = Q_
del Q_


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


pathlib.Path(f"{output_dir}/_true").mkdir(parents=True, exist_ok=True)
fig_true, ax_true = plot(Q_true)
fig_true.savefig(f"{output_dir}/_true/plot.png")
with open(f"{output_dir}/_true/Q.npy", "wb") as f:
    np.save(f, Q_true)

n_episodes = int(alg_params["n_episodes"])
alpha_min = alg_params["alpha_min"]
alpha_max = alg_params["alpha_max"]
alpha_decay = alg_params["alpha_decay"]
epsilon = alg_params["epsilon"]
discount_factor = alg_params["discount_factor"]
state, info = env.reset()

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

Q = np.zeros(
    (env.observation_space[0].n, env.observation_space[1].n, env.action_space.n)
)
for s in env.terminal_states:
    Q[s[0], s[1], 0] = Q_true[s[0], s[1], 0]
    Q[s[0], s[1], 1] = Q_true[s[0], s[1], 1]

trajectories_fname = f"{output_dir}/trajectories.txt"
with open(trajectories_fname, "w") as trajectories_file:
    trajectories_file.write("episode,step,m_s,m_i,action,reward,m_s_,m_i_\n")
# trajectories_file = open(trajectories_fname, "w", buffering=1)
# trajectories_file.write("episode,step,m_s,m_i,action,reward,m_s_,m_i_\n")
# states_count = np.zeros((env.observation_space[0].n, env.observation_space[1].n))
trajectories_tmp = []
for episode in range(n_episodes):
    logger.info(f"Episode {episode}")
    state, info = env.reset()
    alpha = alpha_min + (alpha_max - alpha_min) * np.exp(-alpha_decay * episode)
    logger.info(f"Learning rate: {alpha}")
    done = False
    steps = 0
    while not done:
        # states_count[state] += 1
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
            assert tuple(state.tolist()) in env.terminal_states
            assert state[1] == 0
            logger.info(f"Episode {episode} finished after {steps} steps")
        else:
            assert tuple(state.tolist()) not in env.terminal_states
            assert state[1] > 0
    if (episode + 1) % 20000 == 0 or (episode + 1) == n_episodes:
        pathlib.Path(f"{output_dir}/trajectories").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{output_dir}/ckpt-{episode}").mkdir(parents=True, exist_ok=True)

        logger.info(f"Learning rate at episode {episode}: {alpha}")
        fig, ax = plot(Q)
        fig.savefig(f"{output_dir}/ckpt-{episode}/plot.png")
        with open(f"{output_dir}/ckpt-{episode}/Q.npy", "wb") as f_q:
            np.save(f_q, Q)
        # with open(f"{output_dir}/states_count_{episode+1}.npy", "wb") as f_states_count:
        #    np.save(f_states_count, states_count)
        trajectories_fname = f"{output_dir}/trajectories/trajectories_{episode+1}.txt"
        with open(trajectories_fname, "w") as trajectories_file:
            trajectories_file.write("episode,step,m_s,m_i,action,reward,m_s_,m_i_\n")
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
