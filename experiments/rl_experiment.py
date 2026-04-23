import os

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from pathlib import Path

from envs.env import SIRSEnv
from rl.q_iteration import compute_social_optimum_policy

MLFLOW_EXPERIMENT_NAME = "SIRS Q-learning"

size = 10
n_episodes = 1e6

alg_params = {
    "n_episodes": n_episodes,
    "n_steps": 1e6,
    "alpha_max": 0.5,
    "alpha_min": 1e-4,
    "alpha_decay": 1e-4,
    "epsilon": 0.1,
    "discount_factor": 0.99,
    "learn_mode": "complete",  # "complete", "fixed_no_infection", "independent_no_infection"
    "max_steps_episode": 2000,
    "no_infection_learn_steps": n_episodes * (2 / (size + 2)),  # Only for "independent_no_infection" learn mode, number of steps to learn the no infection states
    "state_action_values_initialization": "random",
}

env_params = {
    "size": 5,
    "encounter_rate": 1.1,
    "recovery_rate": 0.6,
    "resusceptible_rate": 0.3,
    "vaccination_rate": 0.2,
    "cost_infection": 2,
    "cost_lockdown": 1.001,
}


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


def Q_dict_to_array(Q: dict) -> np.array:
    indices = list(Q.keys())
    n_dims = len(indices[0])
    shape = []
    for dim in range(n_dims):
        shape.append(max([index[dim] for index in indices]) + 1)
    Q_arr = np.zeros(shape=shape)
    for k, v in Q_true.items():
        Q_arr[k] = v
    return Q_arr


def plot_values(Q: np.array):
    if len(Q.shape) == 3:
        V = np.max(Q, axis=-1)
    elif len(Q.shape) == 2:
        V = Q
    else:
        raise ValueError(f"Invalid Q shape {Q.shape}")
    mask = np.flip(np.tri(V.shape[0], V.shape[0], k=-1), 1)
    V_masked = np.ma.array(V, mask=mask)
    f, ax = plt.subplots()
    im = ax.imshow(V_masked.T)
    bar = plt.colorbar(im)
    ax.set_xticks(np.arange(V.shape[0]))
    ax.set_yticks(np.arange(V.shape[1]))
    ax.invert_yaxis()
    ax.set_xlabel("Susceptible")
    ax.set_ylabel("Infected")
    ax.set_title("Value function")
    return f, ax


def plot_policy(Q: np.array):
    policy = np.argmax(Q, axis=-1)
    f, ax = plt.subplots()
    for m_s in range(policy.shape[0]):
        for m_i in range(policy.shape[1] - m_s):  # same as policy.shape[0]
            if policy[m_s, m_i] == 0:
                ax.plot(m_s, m_i, "x", color="red", label="confinement")
            else:
                ax.plot(m_s, m_i, "o", color="green", label="max exposure")
    ax.set_xticks(np.arange(Q.shape[0]))
    ax.set_yticks(np.arange(Q.shape[1]))
    ax.set_title("Social Optimum Policy")
    ax.set_xlabel("Susceptible")
    ax.set_ylabel("Infected")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    return f, ax


def compute_error(Q_true: np.array, Q_approx: np.array):
    assert Q_true.shape == Q_approx.shape
    mask = np.flip(np.tri(Q_approx.shape[0], Q_approx.shape[0], k=-1), 1)
    Q_true_ = np.ma.array(Q_true, mask=mask)
    Q_approx_ = np.ma.array(Q_approx, mask=mask)
    return (
        np.ma.min(np.abs(Q_true_ - Q_approx_)),
        np.ma.mean(np.abs(Q_true_ - Q_approx_)),
        np.ma.median(np.abs(Q_true_ - Q_approx_)),
        np.ma.max(np.abs(Q_true_ - Q_approx_)),
    )


def compute_relative_error(Q_true: np.array, Q_approx: np.array):
    assert Q_true.shape == Q_approx.shape
    mask = np.flip(np.tri(Q_approx.shape[0], Q_approx.shape[0], k=-1), 1).astype(float)
    mask_zeros = (Q_true == 0).astype(
        float
    )  # TODO: Improve relative error metric when zeros, not remove
    mask = np.ma.mask_or(mask, mask_zeros)
    Q_true_ = np.ma.array(Q_true, mask=mask)
    Q_approx_ = np.ma.array(Q_approx, mask=mask)
    return (
        np.ma.min(np.abs((Q_true_ - Q_approx_) / Q_true_)),
        np.ma.mean(np.abs((Q_true_ - Q_approx_) / Q_true_)),
        np.ma.median(np.abs((Q_true_ - Q_approx_) / Q_true_)),
        np.ma.max(np.abs((Q_true_ - Q_approx_) / Q_true_)),
    )


env = SIRSEnv(
    **env_params,
)

n_episodes = int(alg_params["n_episodes"])
alpha_min = alg_params["alpha_min"]
alpha_max = alg_params["alpha_max"]
alpha_decay = alg_params["alpha_decay"]
epsilon = alg_params["epsilon"]
discount_factor = alg_params["discount_factor"]
learn_mode = alg_params["learn_mode"]
max_steps_episode = int(alg_params["max_steps_episode"])


def initialize_state_action_values(env, type: str = None):
    if type == "random":
        Q = (
            np.random.rand(
                env.observation_space.nvec[0],
                env.observation_space.nvec[1],
                env.action_space.n,
            )
            * 10
        )
        for m_s in range(Q.shape[0]):
            for m_i in range(Q.shape[1]) - m_s:
                Q[m_s, m_i, :] = 0
    elif type == "zeros":
        Q = np.zeros(
            (
                env.observation_space.nvec[0],
                env.observation_space.nvec[1],
                env.action_space.n,
            )
        )
    else:
        raise ValueError(f"Invalid type {type}")
    Q
    return Q


mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
with mlflow.start_run():
    mlflow.log_params(alg_params)
    mlflow.log_params(env_params)
    mlflow.set_tag("Learn mode", alg_params["learn_mode"])

    Q = initialize_state_action_values(
        env, type=alg_params["state_action_values_initialization"]
    )
    Q_true = Q_dict_to_array(Q_true)
    assert learn_mode in [
        "complete",
        "fixed_no_infection",
        "independent_no_infection",
    ], f"Invalid learn mode {learn_mode}"
    if learn_mode == "fixed_no_infection":
        for m_s in range(Q.shape[0]):
            for a in range(Q.shape[2]):
                Q[m_s, 0, a] = Q_true[(m_s, 0, a)]

    f, ax = plot_values(Q_true)
    mlflow.log_figure(f, "value_function_true.png")
    f, ax = plot_policy(Q)
    mlflow.log_figure(f, "policy_true.png")
    f, ax = plot_values(Q)

    for episode in range(n_episodes):
        logger.info(f"Episode {episode}")
        if learn_mode == "complete" or learn_mode == "independent_no_infection":
            state, info = env.reset()
            initial_state = state
        elif learn_mode == "fixed_no_infection":
            initial_states = [
                (m_s, m_i)
                for m_s in range(env.size + 1)
                for m_i in range(1, env.size + 1)
                if m_s + m_i <= env.size
            ]
            initial_state = initial_states[np.random.randint(len(initial_states))]
            state, info = env.reset(options={"initial_state": initial_state})
        alpha = alpha_min + (alpha_max - alpha_min) * np.exp(-alpha_decay * episode)
        logger.info(f"Learning rate: {alpha}")
        done = False
        steps = 0
        while not done:
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
            state = next_state
            steps += 1
            if learn_mode == "complete":
                done = steps >= max_steps_episode
            elif learn_mode == "fixed_no_infection":
                done = state[1] == 0
            else:  # learn_model == "independent_no_infection"
                if initial_state[1] > 0:
                    done = state[1] == 0
                else:
                    done = steps >= max_steps_episode

        errors = compute_error(np.max(Q_true, axis=-1), np.max(Q, axis=-1))
        relative_errors = compute_relative_error(
            np.max(Q_true, axis=-1), np.max(Q, axis=-1)
        )
        metrics = {
            "min_error": errors[0],
            "mean_error": errors[1],
            "median_error": errors[2],
            "max_error": errors[3],
            "relative_min_error": relative_errors[0],
            "relative_mean_error": relative_errors[1],
            "relative_median_error": relative_errors[2],
            "relative_max_error": relative_errors[3],
            "lr": alpha,
        }
        mlflow.log_metrics(metrics, step=episode)

        if episode % 500 == 0:
            V_diff = np.max(Q_true, axis=-1) - np.max(Q, axis=-1)
            ep_dir = f"chkpt_{episode}"
            with open(f"Q_{episode}.npy", "wb") as f:
                np.save(f, Q)
            mlflow.log_artifact(f"Q_{episode}.npy", artifact_path=f"{ep_dir}")
            os.remove(f"Q_{episode}.npy")
            f, ax = plot_values(Q)
            mlflow.log_figure(f, f"{ep_dir}/value_function_{episode}.png")
            f, ax = plot_policy(Q)
            mlflow.log_figure(f, f"{ep_dir}/policy_{episode}.png")
            f, ax = plot_values(V_diff)
            mlflow.log_figure(f, f"{ep_dir}/value_function_diff_{episode}.png")
