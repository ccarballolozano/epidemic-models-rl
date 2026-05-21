from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
import matplotlib.pyplot as plt
import mlflow


def main():
    print("Hello from epidemic-models-rl!")
    # Connect to Azure ML Workspace using SDK v2 and setup MLflow
    # The workspace will be loaded from config.json file if available, or you can specify parameters
    try:
        # Using DefaultAzureCredential for authentication
        # credential = DefaultAzureCredential()
        import os
        credential = InteractiveBrowserCredential(
            tenant_id=os.environ["TENANT_ID"],
        )

        # Create ML Client from config.json
        ml_client = MLClient.from_config(
            credential=credential, path="./experiments/config.json"
        )
        print(f"Connected to workspace: {ml_client.workspace_name}")
        print(f"Subscription: {ml_client.subscription_id}")
        print(f"Resource group: {ml_client.resource_group_name}")

    except Exception as e:
        print(f"Error connecting to workspace: {e}")
        raise

    tracking_uri = ml_client.workspaces.get(
        ml_client.workspace_name
    ).mlflow_tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    print(f"✓ MLflow tracking URI set successfully to {mlflow.get_tracking_uri()}")
    with mlflow.start_run(run_name="test-connection"):
        mlflow.log_param("test_param", "test_value")
        mlflow.log_metric("test_metric", 0.123)


if __name__ == "__main__":
    main()
