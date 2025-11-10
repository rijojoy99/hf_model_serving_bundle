import mlflow
from mlflow import MlflowClient
import logging

logger = logging.getLogger("ModelRegistrar")

class ModelRegistrar:
    """
    Handles MLflow model registration into Unity Catalog.
    """

    def __init__(self, model_cfg: dict):
        self.model_cfg = model_cfg
        self.client = MlflowClient()
        mlflow.set_registry_uri("databricks-uc")

    def register_model(self) -> int:
        model_name = self.model_cfg["name"]
        model_path = self.model_cfg["local_path"]
        catalog_path = self.model_cfg["catalog_path"]

        logger.info(f"🚀 Registering model {model_name} from {model_path} to {catalog_path}")

        with mlflow.start_run():
            mlflow.pyfunc.log_model(
                artifact_path=model_name,
                artifacts={"model_dir": model_path},
                registered_model_name=catalog_path,
            )

        version = self.client.get_latest_versions(catalog_path, stages=["None"])[0].version
        logger.info(f"✅ Model registered: {catalog_path} (v{version})")
        return version
