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


def inverse_sqrt_decay(step, n_steps, alpha_max, alpha_min):
    c = (alpha_max**2 / alpha_min**2) - 1
    return alpha_max / np.sqrt(1 + c * (step / n_steps))


def exponential_decay(step, n_steps, alpha_max, alpha_min):
    return alpha_max * (alpha_min / alpha_max) ^ (step / n_steps)


def compute_metrics(Q_true, Q):
    V = np.max(Q, axis=-1)
    V_true = Q_true.max(axis=-1)
    error_state_values = compute_state_value_error(V_true, V)
    error_state_action_values = compute_state_action_value_error(Q_true, Q)
    rel_error_state_values = compute_state_value_relative_error(V_true, V)
    rel_error_state_action_values = compute_state_action_value_relative_error(Q_true, Q)
    actions_error = compute_proportion_of_states_with_suboptimal_action(Q_true, Q)
    metrics = {
        "error_V_mean": error_state_values[1],
        "error_V_max": error_state_values[3],
        "error_mean": error_state_action_values[1],
        "error_max": error_state_action_values[3],
        "relative_error_V_mean": rel_error_state_values[1],
        "relative_error_V_max": rel_error_state_values[3],
        "relative_error_mean": rel_error_state_action_values[1],
        "relative_error_max": rel_error_state_action_values[3],
        "actions_error": actions_error,
        "log_relative_error_mean": np.log10(rel_error_state_action_values[1]),
        "log_relative_error_max": np.log10(rel_error_state_action_values[3]),
        "log_relative_error_V_mean": np.log10(rel_error_state_values[1]),
        "log_relative_error_V_max": np.log10(rel_error_state_values[3]),
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

    
    alpha = inverse_sqrt_decay(total_steps, params.n_steps, params.alpha_max, params.alpha_min)
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
