import os

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from pathlib import Path

from algorithms import two_stage_q_learning, q_learning, QLearningParams
from envs.env import SIRSEnv
from misc import Q_dict_to_array
from rl.q_iteration import compute_social_optimum_policy

MLFLOW_EXPERIMENT_NAME = "Smart Q-learning for SIRS"


# env_params = {
#    "size": 30,
#    "encounter_rate": 0.7,
#    "recovery_rate": 0.5,
#    "resusceptible_rate": 0.2,
#    "vaccination_rate": 0.3,
#    "cost_infection": 5.5,
#    "cost_lockdown": 1.5,
# }
env_params = {
    "size": 5,
    "encounter_rate": 1.1,
    "recovery_rate": 0.6,
    "resusceptible_rate": 0.2,
    "vaccination_rate": 0.3,
    "cost_infection": 2,
    "cost_lockdown": 1.001,
}

env = SIRSEnv(
    **env_params,
)


params = QLearningParams(
    n_episodes=1e6,
    n_steps=1e6,
    epsilon=0.1,
    alpha_max=0.5,
    alpha_min=1e-3,
    alpha_decay=1e-4,
    discount_factor=0.99,
    state_action_values_initialization="random",
    max_steps_episode=2_000,
    first_stage_steps=35_000,
    log_every_n_steps=1_000,
    save_every_n_steps=10_000,
    learn_mode="two_stages",  # "complete" or "two_stages"
)

_, Q_true = compute_social_optimum_policy(
    size=env_params["size"],
    encounter_rate=env_params["encounter_rate"],
    recovery_rate=env_params["recovery_rate"],
    resusceptible_rate=env_params["resusceptible_rate"],
    vaccination_rate=env_params["vaccination_rate"],
    cost_infection=env_params["cost_infection"],
    cost_lockdown=env_params["cost_lockdown"],
    discount_factor=params.discount_factor,
    theta=1e-12,
    max_iterations=1e6,
)
Q_true = Q_dict_to_array(Q_true)

with mlflow.start_run():
    mlflow.log_params(params.__dict__)
    mlflow.log_params(env_params)
    if params.learn_mode == "two_stages":
        Q = two_stage_q_learning(env, params, Q_true)
    elif params.learn_mode == "complete":
        Q = q_learning(env, params, Q_true)
    else:
        raise ValueError(f"Invalid learn_mode {params.learn_mode}")
