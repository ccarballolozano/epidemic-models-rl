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
    # Two-stage customisation (ignored for other learn modes)
    alpha_restart_on_stage_change: bool = False
    stage1_initial_states: list[list[int]] | None = None
    stage1_end_states: list[list[int]] | None = None


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


_VALID_LEARN_MODES = frozenset(
    ["complete", "fixed_no_infection", "independent_no_infection", "two_stages"]
)


def _reset_episode(
    env,
    learn_mode: str,
    total_steps: int,
    first_stage_steps: int,
    absorbing_group_states: list,
    transient_group_states: list,
    stage1_initial_states: list | None = None,
):
    """Reset the environment and choose the initial state for the episode."""
    if learn_mode in ("complete", "independent_no_infection"):
        state, info = env.reset()
        return state, state, info

    if learn_mode == "fixed_no_infection":
        pool = transient_group_states
    elif learn_mode == "two_stages":
        if total_steps < first_stage_steps:
            pool = (
                [tuple(s) for s in stage1_initial_states]
                if stage1_initial_states is not None
                else absorbing_group_states
            )
        else:
            pool = transient_group_states

    initial_state = pool[np.random.randint(len(pool))]
    state, info = env.reset(options={"initial_state": initial_state})
    return initial_state, state, info


def _is_episode_done(
    learn_mode: str,
    total_steps: int,
    first_stage_steps: int,
    steps: int,
    max_steps_episode: int,
    state,
    initial_state,
    absorbing_group_states: list,
    stage1_end_states_set: set | None = None,
) -> bool:
    """Return True when the episode should terminate under the given learn mode."""
    if learn_mode == "complete":
        return steps >= max_steps_episode

    if learn_mode == "fixed_no_infection":
        return state[1] == 0

    if learn_mode == "two_stages":
        if total_steps < first_stage_steps:
            if stage1_end_states_set is not None:
                return (
                    tuple(state) in stage1_end_states_set or steps >= max_steps_episode
                )
            return steps >= max_steps_episode
        return tuple(state) in absorbing_group_states

    # independent_no_infection
    if initial_state[1] > 0:
        return state[1] == 0
    return steps >= max_steps_episode


def sirs_q_learning(env: SIRSEnv, params: QLearningParams, Q_true: np.array):
    learn_mode = params.learn_mode
    assert learn_mode in _VALID_LEARN_MODES, f"Invalid learn mode {learn_mode}"

    Q = initialize_state_action_values(
        env, type=params.state_action_values_initialization
    )
    # Tracks how many times each state's Q-value was updated.
    # int32 supports up to ~2 billion updates per state at negligible memory cost.
    state_update_counts = np.zeros((Q.shape[0], Q.shape[1]), dtype=np.int32)

    if learn_mode == "fixed_no_infection":
        for m_s in range(Q.shape[0]):
            for a in range(Q.shape[2]):
                Q[m_s, 0, a] = Q_true[m_s, 0, a]

    # When reinfection rate is zero, (0,0) is absorbing and stage 1 of two_stages
    # never visits it, so its Q-value is never updated. Force it to 0 regardless
    # of the initialization type.
    if learn_mode == "two_stages" and env.resusceptible_rate == 0:
        Q[0, 0, :] = 0

    # Pre-compute state partitions used by several learn modes
    states = [
        (m_s, m_i)
        for m_s in range(env.size + 1)
        for m_i in range(env.size + 1)
        if m_s + m_i <= env.size
    ]
    states = list(set(states))
    absorbing_group_states = [(m_s, 0) for m_s in range(env.size + 1)]
    if env.resusceptible_rate == 0:
        absorbing_group_states += [(0, m_i) for m_i in range(1, env.size + 1)]
    transient_group_states = [s for s in states if s not in absorbing_group_states]

    total_steps = 0
    n_episodes = int(params.n_episodes)
    n_steps = int(params.n_steps)

    # Track when stage 2 begins for optional alpha restart
    stage2_start_episode: int | None = None

    # Pre-build a set for O(1) end-state look-up (two_stages stage 1 only)
    stage1_end_states_set: set | None = (
        {tuple(s) for s in params.stage1_end_states}
        if params.stage1_end_states is not None
        else None
    )

    # Log initial metrics at step 0
    metrics = compute_metrics(Q_true, Q, include_subsets=True)
    metrics |= {"lr": params.alpha_max}
    mlflow.log_metrics(metrics, step=total_steps)

    for episode in range(n_episodes):
        logger.info(f"Episode {episode}")

        # ---- alpha schedule ------------------------------------------------
        in_stage2 = (
            learn_mode == "two_stages" and total_steps >= params.first_stage_steps
        )
        if params.alpha_restart_on_stage_change and learn_mode == "two_stages":
            if in_stage2 and stage2_start_episode is None:
                stage2_start_episode = episode
                logger.info("Entering stage 2 — resetting alpha decay")
            effective_episode = (
                episode - stage2_start_episode
                if stage2_start_episode is not None
                else episode
            )
        else:
            effective_episode = episode
        alpha = params.alpha_min + (params.alpha_max - params.alpha_min) * np.exp(
            -params.alpha_decay * effective_episode
        )
        logger.info(f"Learning rate: {alpha}")

        # ---- episode reset -------------------------------------------------
        initial_state, state, info = _reset_episode(
            env,
            learn_mode,
            total_steps,
            params.first_stage_steps,
            absorbing_group_states,
            transient_group_states,
            stage1_initial_states=params.stage1_initial_states,
        )

        # ---- inner step loop -----------------------------------------------
        steps = 0
        done = False
        while not done:
            if np.random.rand() < params.epsilon:
                action = env.action_space.sample()
            else:
                action = Q[state[0], state[1], :].argmax()

            next_state, reward, done, truncated, info = env.step(action)

            Q[state[0], state[1], action] += alpha * (
                reward
                + params.discount_factor * Q[next_state[0], next_state[1], :].max()
                - Q[state[0], state[1], action]
            )
            state_update_counts[state[0], state[1]] += 1

            state = next_state
            steps += 1
            total_steps += 1

            done = _is_episode_done(
                learn_mode,
                total_steps,
                params.first_stage_steps,
                steps,
                params.max_steps_episode,
                state,
                initial_state,
                absorbing_group_states,
                stage1_end_states_set=stage1_end_states_set,
            )

            if params.log_every_n_steps and (
                total_steps % params.log_every_n_steps == 0
            ):
                metrics = compute_metrics(Q_true, Q, include_subsets=True)
                metrics |= {"lr": alpha}
                mlflow.log_metrics(metrics, step=total_steps)
                outputs_ckpt_dir = f"chkpt_{episode}_{steps}_{total_steps}"
                with open(f"Q_{episode}_{steps}_{total_steps}.npy", "wb") as fh:
                    np.save(fh, Q)
                mlflow.log_artifact(
                    f"Q_{episode}_{steps}_{total_steps}.npy",
                    artifact_path=outputs_ckpt_dir,
                )
                counts_fname = (
                    f"state_update_counts_{episode}_{steps}_{total_steps}.npy"
                )
                with open(counts_fname, "wb") as fh:
                    np.save(fh, state_update_counts)
                mlflow.log_artifact(counts_fname, artifact_path=outputs_ckpt_dir)

            if total_steps >= n_steps:
                break

        logger.info(
            f"Episode {episode} finished after {steps} steps from {initial_state} to {state}"
        )

        if total_steps >= n_steps:
            break

    return Q, state_update_counts
