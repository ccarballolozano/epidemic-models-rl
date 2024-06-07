import argparse
from datetime import datetime
import os

from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

from ..azureml.config import *


def main(args):
    try:
        # Attempt to use default Azure Credential
        credential = DefaultAzureCredential()

        # Check if the given credential can get a token successfully
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential if Service Principal credentials fail
        # This will open a browser page for authentication
        credential = InteractiveBrowserCredential()

        # Get environment variables
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP_NAME,
        workspace_name=WORKSPACE_NAME,
    )

    cmd = command(
        experiment_name="SIRS_generate_social_optimum_results",
        code="./",
        command="python -m experiments.sirs_with_confinements.generate_social_optimum_results --output_dir ./outputs",
        compute=COMPUTE_NAME,
        environment=f"{ENVIRONMENT_NAME}:{ENVIRONMENT_VERSION}",
    )

    ml_client.jobs.create_or_update(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
