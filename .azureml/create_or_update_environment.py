from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential


from env import *

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

env_docker_conda = Environment(
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    conda_file="../environment.yml",
    name="research-env",
    description="Environment created for Research Experiments",
    tags={},
)

environment_created = ml_client.environments.create_or_update(env_docker_conda)


print("Environment version created: ", environment_created.version)
