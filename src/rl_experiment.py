import argparse

import mlflow
import numpy as np

from algorithms import sirs_q_learning, QLearningParams
from envs.env import SIRSEnv
from misc import Q_dict_to_array
from plot import plot_values, plot_policy
from rl.q_iteration import compute_social_optimum_policy

MLFLOW_EXPERIMENT_NAME = "Smart Q-learning for SIRS"


def main(args):
    env_params = {
        "size": args.size,
        "encounter_rate": args.encounter_rate,
        "recovery_rate": args.recovery_rate,
        "resusceptible_rate": args.resusceptible_rate,
        "vaccination_rate": args.vaccination_rate,
        "cost_infection": args.cost_infection,
        "cost_lockdown": args.cost_lockdown,
    }

    stage1_initial_states = None
    stage1_end_states = None
    if env_params["resusceptible_rate"] == 0.0:
        stage1_initial_states = [(0, args.size), (args.size, 0)]
        stage1_end_states = [(0, 0)]

    params = QLearningParams(
        n_episodes=args.n_episodes,
        n_steps=args.n_steps,
        alpha_max=args.alpha_max,
        alpha_min=args.alpha_min,
        alpha_decay=args.alpha_decay,
        epsilon=args.epsilon,
        discount_factor=args.discount_factor,
        learn_mode=args.learn_mode,
        max_steps_episode=args.max_steps_episode,
        first_stage_steps=args.first_stage_steps,
        state_action_values_initialization=args.state_action_values_initialization,
        log_every_n_steps=args.log_every_n_steps,
        save_every_n_steps=args.save_every_n_steps,
        alpha_restart_on_stage_change=args.alpha_restart_on_stage_change,
        stage1_end_states=stage1_end_states,
        stage1_initial_states=stage1_initial_states,
    )

    env = SIRSEnv(**env_params)

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
        mlflow.set_tag("Learn mode", params.learn_mode)
        mlflow.set_tag("size", env_params["size"])
        mlflow.set_tag("run_group", args.tag_run_group)

        f, ax = plot_values(Q_true)
        mlflow.log_figure(f, "value_function_true.png")
        f, ax = plot_policy(Q_true)
        mlflow.log_figure(f, "policy_true.png")
        # log Q_true as artifact
        with open("Q_true.npy", "wb") as fh:
            np.save(fh, Q_true)
        mlflow.log_artifact("Q_true.npy", artifact_path="Q_true")

        Q, state_update_counts = sirs_q_learning(env, params, Q_true)
        with open("state_update_counts_final.npy", "wb") as fh:
            np.save(fh, state_update_counts)
        mlflow.log_artifact("state_update_counts_final.npy")
        with open("Q_final.npy", "wb") as fh:
            np.save(fh, Q)
        mlflow.log_artifact("Q_final.npy", artifact_path="Q_final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Environment parameters
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--encounter_rate", type=float, default=1.1)
    parser.add_argument("--recovery_rate", type=float, default=0.6)
    parser.add_argument("--resusceptible_rate", type=float, default=0.3)
    parser.add_argument("--vaccination_rate", type=float, default=0.2)
    parser.add_argument("--cost_infection", type=float, default=2)
    parser.add_argument("--cost_lockdown", type=float, default=1.001)
    # Algorithm parameters
    parser.add_argument("--n_episodes", type=int, default=int(1e6))
    parser.add_argument("--n_steps", type=int, default=int(1e6))
    parser.add_argument("--alpha_max", type=float, default=0.5)
    parser.add_argument("--alpha_min", type=float, default=1e-4)
    parser.add_argument("--alpha_decay", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument(
        "--learn_mode",
        type=str,
        default="complete",
        choices=[
            "complete",
            "fixed_no_infection",
            "independent_no_infection",
            "two_stages",
        ],
    )
    parser.add_argument("--max_steps_episode", type=int, default=2_000)
    parser.add_argument("--first_stage_steps", type=int, default=35_000)
    parser.add_argument(
        "--state_action_values_initialization",
        type=str,
        default="random",
        choices=["random", "zeros"],
    )
    parser.add_argument("--log_every_n_steps", type=int, default=1_000)
    parser.add_argument("--save_every_n_steps", type=int, default=100_000)
    parser.add_argument(
        "--alpha_restart_on_stage_change",
        type=lambda x: str(x).lower() == "true",
        default=True,
    )
    parser.add_argument(
        "--tag-run-group",
        type=str,
        help="Value of the tag 'run_group' to identify the runs of this experiment",
        required=False,
    )
    args = parser.parse_args()
    main(args)
