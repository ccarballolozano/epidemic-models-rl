import random

from loguru import logger
import mlflow
import numpy as np
from pydantic.dataclasses import dataclass

from envs.env import SIRSEnv
from metrics import (
    compute_state_value_error,
    compute_state_value_relative_error,
    compute_state_action_value_error,
    compute_state_action_value_relative_error,
    compute_proportion_of_states_with_suboptimal_action,
)
from misc import initialize_state_action_values


@dataclass
class QLearningParams:
    n_episodes: int
    n_steps: int
    alpha_max: float
    alpha_min: float
    discount_factor: float
    epsilon: float
    state_action_values_initialization: str
    max_steps_episode: int
    log_every_n_steps: int
    save_every_n_steps: int
    learn_mode: str
    first_stage_steps: int
    alpha_decay: float = 1e-4


def inverse_sqrt_decay(step, n_steps, alpha_max, alpha_min):
    c = (alpha_max**2 / alpha_min**2) - 1
    return alpha_max / np.sqrt(1 + c * (step / n_steps))


def exponential_decay(step, n_steps, alpha_max, alpha_min):
    return alpha_max * (alpha_min / alpha_max) ** (step / n_steps)


def compute_metrics(Q_true, Q, include_subsets=False):
    V = np.max(Q, axis=-1)
    V_true = Q_true.max(axis=-1)
    error_state_values = compute_state_value_error(V_true, V)
    error_state_action_values = compute_state_action_value_error(Q_true, Q)
    rel_error_state_values = compute_state_value_relative_error(V_true, V)
    rel_error_state_action_values = compute_state_action_value_relative_error(Q_true, Q)
    actions_error = compute_proportion_of_states_with_suboptimal_action(Q_true, Q)
    metrics = {
        "mean_absolute_error_V": error_state_values[1],
        "max_absolute_error_V": error_state_values[3],
        "mean_absolute_error": error_state_action_values[1],
        "max_absolute_error": error_state_action_values[3],
        "mean_relative_error_V": rel_error_state_values[1],
        "max_relative_error_V": rel_error_state_values[3],
        "mean_relative_error": rel_error_state_action_values[1],
        "max_relative_error": rel_error_state_action_values[3],
        "suboptimal_action_rate": actions_error,
        "log_mean_relative_error": np.log10(rel_error_state_action_values[1]),
        "log_max_relative_error": np.log10(rel_error_state_action_values[3]),
        "log_mean_relative_error_V": np.log10(rel_error_state_values[1]),
        "log_max_relative_error_V": np.log10(rel_error_state_values[3]),
    }
    if include_subsets:
        rel_error_uninfected = compute_state_value_relative_error(
            V_true, V, "uninfected"
        )
        rel_error_infected = compute_state_value_relative_error(V_true, V, "infected")
        metrics |= {
            "mean_relative_error_uninfected": rel_error_uninfected[1],
            "max_relative_error_uninfected": rel_error_uninfected[3],
            "mean_relative_error_infected": rel_error_infected[1],
            "max_relative_error_infected": rel_error_infected[3],
            "log_mean_relative_error_uninfected": np.log10(rel_error_uninfected[1]),
            "log_mean_relative_error_infected": np.log10(rel_error_infected[1]),
        }
    return metrics


def q_learning(env: SIRSEnv, params: QLearningParams, Q_true: np.array):
    # Initialize Q-values
    Q = initialize_state_action_values(
        env, type=params.state_action_values_initialization
    )

    total_reward = 0
    total_steps = 0
    done_alg = False

    alpha = inverse_sqrt_decay(
        total_steps, params.n_steps, params.alpha_max, params.alpha_min
    )
    metrics = compute_metrics(Q_true, Q)
    metrics |= {"lr": alpha}
    mlflow.log_metrics(metrics, step=total_steps)

    for episode in range(params.n_episodes):
        logger.info(
            f"Episode {episode} - Total steps at the beginning of the episode {episode}: {total_steps}"
        )
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0
        while not done:
            # Choose action
            if np.random.rand() < params.epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state[0], state[1], :])

            # Take action
            next_state, reward, done, _, _ = env.step(action)

            alpha = inverse_sqrt_decay(
                total_steps, params.n_steps, params.alpha_max, params.alpha_min
            )
            # Update Q-value
            Q[state[0], state[1], action] = (1 - alpha) * Q[
                state[0], state[1], action
            ] + alpha * (
                reward
                + params.discount_factor * np.max(Q[next_state[0], next_state[1], :])
            )
            state = next_state
            episode_reward += reward
            step += 1
            total_steps += 1
            if step == params.max_steps_episode:
                done = True
            # Log metrics and checkpoints
            if (
                params.log_every_n_steps
                and (total_steps % params.log_every_n_steps == 0)
            ) or done:
                metrics = compute_metrics(Q_true, Q)
                metrics |= {"lr": alpha}
                mlflow.log_metrics(metrics, step=total_steps)
            if params.save_every_n_steps and (
                total_steps % params.save_every_n_steps == 0
            ):
                outputs_ckpt_dir = f"chkpt_{episode}_{step}_{total_steps}"
                with open(f"Q_{episode}_{step}_{total_steps}.npy", "wb") as f:
                    np.save(f, Q)
                mlflow.log_artifact(
                    f"Q_{episode}_{step}_{total_steps}.npy",
                    artifact_path=outputs_ckpt_dir,
                )
            if total_steps >= params.n_steps:
                done_alg = True
                logger.info(f"Training finished after {total_steps} steps")
                break
        total_reward += episode_reward
        if done_alg:
            break
    return Q


def two_stage_q_learning(env: SIRSEnv, params: QLearningParams, Q_true: np.array):
    # Initialize Q-values
    Q = initialize_state_action_values(
        env, type=params.state_action_values_initialization
    )
    state = env.reset()
    episode = 0
    total_reward = 0
    total_steps = 0
    done_alg = False

    infected_states = [
        (m_s, m_i)
        for m_s in range(env.size + 1)
        for m_i in range(1, env.size + 1)
        if m_s + m_i <= env.size
    ]
    no_infected_states = [(m_s, 0) for m_s in range(env.size + 1)]

    # First stage
    logger.info("First stage")
    for episode in range(params.n_episodes):
        logger.info(
            f"First Stage - Episode {episode} - Total steps at the beginning of the episode {episode}: {total_steps}"
        )
        state, _ = env.reset(
            options={"initial_state": random.choice(no_infected_states)}
        )
        done = False
        episode_reward = 0
        step = 0
        while not done:
            # Choose action
            if np.random.rand() < params.epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state[0], state[1], :])

            # Take action
            next_state, reward, done, _, _ = env.step(action)

            alpha = inverse_sqrt_decay(
                total_steps,
                params.first_stage_steps,
                params.alpha_max,
                params.alpha_min,
            )
            # Update Q-value
            Q[state[0], state[1], action] = (1 - alpha) * Q[
                state[0], state[1], action
            ] + alpha * (
                reward
                + params.discount_factor * np.max(Q[next_state[0], next_state[1], :])
            )
            state = next_state
            episode_reward += reward
            step += 1
            total_steps += 1
            if step == params.max_steps_episode:
                done = True
            # Log metrics and checkpoints
            if (
                params.log_every_n_steps
                and (total_steps % params.log_every_n_steps == 0)
            ) or done:
                metrics = compute_metrics(Q_true, Q)
                metrics |= {"lr": alpha}
                mlflow.log_metrics(metrics, step=total_steps)
            if params.save_every_n_steps and (
                total_steps % params.save_every_n_steps == 0
            ):
                outputs_ckpt_dir = f"chkpt_{episode}_{step}_{total_steps}"
                with open(f"Q_{episode}_{step}_{total_steps}.npy", "wb") as f:
                    np.save(f, Q)
                mlflow.log_artifact(
                    f"Q_{episode}_{step}_{total_steps}.npy",
                    artifact_path=outputs_ckpt_dir,
                )
            if total_steps >= params.first_stage_steps:
                done_alg = True
                logger.info(f"Training first stage finished after {total_steps} steps")
                break
        total_reward += episode_reward
        if done_alg:
            break
    logger.info("Second stage")
    done_alg = False
    for episode in range(episode + 1, params.n_episodes):
        logger.info(
            f"Second Stage - Episode {episode} - Total steps at the beginning of the episode {episode}: {total_steps}"
        )
        state, _ = env.reset(options={"initial_state": random.choice(infected_states)})
        done = False
        episode_reward = 0
        step = 0
        while not done:
            # Choose action
            if np.random.rand() < params.epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state[0], state[1], :])

            # Take action
            next_state, reward, done, _, _ = env.step(action)

            alpha = inverse_sqrt_decay(
                total_steps - params.first_stage_steps,
                params.n_steps - params.first_stage_steps,
                params.alpha_max,
                params.alpha_min,
            )
            # Update Q-value
            Q[state[0], state[1], action] = (1 - alpha) * Q[
                state[0], state[1], action
            ] + alpha * (
                reward
                + params.discount_factor * np.max(Q[next_state[0], next_state[1], :])
            )
            state = next_state
            episode_reward += reward
            step += 1
            total_steps += 1
            if state[1] == 0 or step == params.max_steps_episode:
                done = True
            # Log metrics and checkpoints
            if (
                params.log_every_n_steps
                and (total_steps % params.log_every_n_steps == 0)
            ) or done:
                metrics = compute_metrics(Q_true, Q)
                metrics |= {"lr": alpha}
                mlflow.log_metrics(metrics, step=total_steps)
            if params.save_every_n_steps and (
                total_steps % params.save_every_n_steps == 0
            ):
                outputs_ckpt_dir = f"chkpt_{episode}_{step}_{total_steps}"
                with open(f"Q_{episode}_{step}_{total_steps}.npy", "wb") as f:
                    np.save(f, Q)
                mlflow.log_artifact(
                    f"Q_{episode}_{step}_{total_steps}.npy",
                    artifact_path=outputs_ckpt_dir,
                )
            if total_steps >= params.n_steps:
                done_alg = True
                logger.info(f"Training second stage finished after {total_steps} steps")
                break
        total_reward += episode_reward
        if done_alg:
            break
    return Q


def sirs_q_learning(env: SIRSEnv, params: QLearningParams, Q_true: np.array):
    Q = initialize_state_action_values(
        env, type=params.state_action_values_initialization
    )

    learn_mode = params.learn_mode
    assert learn_mode in [
        "complete",
        "fixed_no_infection",
        "independent_no_infection",
        "two_stages",
    ], f"Invalid learn mode {learn_mode}"

    if learn_mode == "fixed_no_infection":
        for m_s in range(Q.shape[0]):
            for a in range(Q.shape[2]):
                Q[m_s, 0, a] = Q_true[m_s, 0, a]

    infected_states = [
        (m_s, m_i)
        for m_s in range(env.size + 1)
        for m_i in range(1, env.size + 1)
        if m_s + m_i <= env.size
    ]
    no_infected_states = [(m_s, 0) for m_s in range(env.size + 1)]

    total_steps = 0
    n_episodes = int(params.n_episodes)
    n_steps = int(params.n_steps)

    # Log initial metrics at step 0
    metrics = compute_metrics(Q_true, Q, include_subsets=True)
    metrics |= {"lr": params.alpha_max}
    mlflow.log_metrics(metrics, step=total_steps)

    for episode in range(n_episodes):
        logger.info(f"Episode {episode}")
        if learn_mode == "complete" or learn_mode == "independent_no_infection":
            state, info = env.reset()
            initial_state = state
        elif learn_mode == "fixed_no_infection":
            initial_state = infected_states[np.random.randint(len(infected_states))]
            state, info = env.reset(options={"initial_state": initial_state})
        elif learn_mode == "two_stages":
            if total_steps < params.first_stage_steps:
                initial_state = no_infected_states[
                    np.random.randint(len(no_infected_states))
                ]
                state, info = env.reset(options={"initial_state": initial_state})
            else:
                initial_state = infected_states[np.random.randint(len(infected_states))]
                state, info = env.reset(options={"initial_state": initial_state})

        alpha = params.alpha_min + (params.alpha_max - params.alpha_min) * np.exp(
            -params.alpha_decay * episode
        )
        logger.info(f"Learning rate: {alpha}")
        done = False
        steps = 0
        while not done:
            if np.random.rand() < params.epsilon:
                action = env.action_space.sample()
            else:
                action = Q[state[0], state[1], :].argmax()
            next_state, reward, done, truncated, info = env.step(action)

            Q[state[0], state[1], action] = Q[state[0], state[1], action] + alpha * (
                reward
                + params.discount_factor * Q[next_state[0], next_state[1], :].max()
                - Q[state[0], state[1], action]
            )
            state = next_state
            steps += 1
            total_steps += 1
            if learn_mode == "complete":
                done = steps >= params.max_steps_episode
            elif learn_mode == "fixed_no_infection":
                done = state[1] == 0
            else:  # "independent_no_infection" or "two_stages"
                if initial_state[1] > 0:
                    done = state[1] == 0
                else:
                    done = steps >= params.max_steps_episode

            # Log metrics and artifacts every n steps
            if params.log_every_n_steps and (
                total_steps % params.log_every_n_steps == 0
            ):
                metrics = compute_metrics(Q_true, Q, include_subsets=True)
                metrics |= {"lr": alpha}
                mlflow.log_metrics(metrics, step=total_steps)
                # Save checkpoints every n steps
                outputs_ckpt_dir = f"chkpt_{episode}_{steps}_{total_steps}"
                with open(f"Q_{episode}_{steps}_{total_steps}.npy", "wb") as fh:
                    np.save(fh, Q)
                mlflow.log_artifact(
                    f"Q_{episode}_{steps}_{total_steps}.npy",
                    artifact_path=outputs_ckpt_dir,
                )

            if total_steps >= n_steps:
                break

        logger.info(f"Episode {episode} finished after {steps} steps")

        if total_steps >= n_steps:
            break
    return Q
