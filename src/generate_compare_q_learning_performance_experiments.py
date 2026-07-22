# Send experiments to Azure ML (running rl_experiment.py for fixed parameters multiple times, except "complete" and "two_stages")
import argparse
import os

from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from dotenv import load_dotenv

load_dotenv()

BASE_PARAMS = {
    "alpha_decay": 0.0001,
    "alpha_max": 0.5,
    "alpha_min": 0.0001,
    "cost_infection": 2,
    "cost_lockdown": 1.001,
    "discount_factor": 0.99,
    "encounter_rate": 1.1,
    "epsilon": 0.1,
    "max_steps_episode": 2000,
    "n_episodes": 1_000_000,
    "n_steps": 1_000_000,
    "first_stage_steps": 100_000,  # 5_000, 10_000, 35_000, 50_000, 100_000, 105_000, 150_000, 300_000
    "recovery_rate": 0.6,
    "resusceptible_rate": 0.0,  # 0.0, 0.3
    "size": 15,  # 5, 15, 50
    "state_action_values_initialization": "random",
    "vaccination_rate": 0.2,
    "log_every_n_steps": 1_000,
    "save_every_n_steps": 100_000,
    "alpha_restart_on_stage_change": True,
    "stage2_absorbing_extra_steps": 1,
}


def build_command_str(learn_mode: str, tag_run_group: str = None) -> str:
    args = " ".join(f"--{k} {v}" for k, v in BASE_PARAMS.items())
    if tag_run_group:
        args += f" --tag-run-group {tag_run_group}"
    return f"python rl_experiment.py {args} --learn_mode {learn_mode}"


def main(n: int, tag_runs_group: str = None):
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception:
        credential = InteractiveBrowserCredential(tenant_id=os.environ["TENANT_ID"])

    ml_client = MLClient(
        credential=credential,
        subscription_id=os.environ["SUBSCRIPTION_ID"],
        resource_group_name=os.environ["RESOURCE_GROUP_NAME"],
        workspace_name=os.environ["WORKSPACE_NAME"],
    )

    #    for learn_mode in ["complete", "two_stages"]:
    for learn_mode in ["two_stages"]:
        for i in range(n):
            cmd = command(
                experiment_name="SIRS-Q-Learning",
                code="./src",
                command=build_command_str(learn_mode, tag_run_group=tag_runs_group),
                compute=os.environ["COMPUTE_NAME"],
                environment=f"{os.environ['ENVIRONMENT_NAME']}:{os.environ['ENVIRONMENT_VERSION']}",
                tags={"learn_mode": learn_mode},
            )
            job = ml_client.jobs.create_or_update(cmd)
            print(f"Submitted {learn_mode} run {i}: {job.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n", type=int, default=5, help="Number of runs per learn mode"
    )
    parser.add_argument(
        "--tag-runs-group",
        type=str,
        help="Identifier to add to the runs tags",
        required=False,
    )
    args = parser.parse_args()
    main(args.n, args.tag_runs_group)
