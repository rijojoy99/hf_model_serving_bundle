import logging
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    AutoCaptureConfigInput,
    ServingModelWorkloadType,
)

logger = logging.getLogger("ModelServingDeployer")

class ModelServingDeployer:
    """
    Deploys or updates a Databricks model serving endpoint in an idempotent way.
    """

    def __init__(self, model_cfg: dict, serving_cfg: dict, model_version: str):
        self.model_cfg = model_cfg
        self.serving_cfg = serving_cfg
        self.model_version = str(model_version)
        self.ws = WorkspaceClient()

    def deploy(self):
        endpoint_name = self.serving_cfg["endpoint_name"]
        catalog_path = self.model_cfg["catalog_path"]

        logger.info(f"🚀 Starting deployment for endpoint '{endpoint_name}' (model: {catalog_path} v{self.model_version})")

        # --- Build served entity config ---
        served_entity = ServedEntityInput(
            entity_name=catalog_path,
            entity_version=self.model_version,
            workload_type=ServingModelWorkloadType(self.serving_cfg["workload_type"]),
            workload_size=self.serving_cfg["workload_size"],
            scale_to_zero_enabled=self.serving_cfg.get("scale_to_zero", True),
            environment_vars={
                "ENABLE_FEATURE_TRACING": str(self.model_cfg.get("enable_tracing", False)).lower()
            },
        )

        # --- Configure inference logging ---
        auto_capture = None
        if self.serving_cfg.get("inference_logging", False):
            auto_capture = AutoCaptureConfigInput(
                catalog_name=self.serving_cfg["inference_catalog"],
                schema_name=self.serving_cfg["inference_schema"],
                table_name_prefix=self.serving_cfg["table_prefix"],
                enabled=True,
            )

        endpoint_config = EndpointCoreConfigInput(
            served_entities=[served_entity],
            auto_capture_config=auto_capture
        )

        # --- Check if endpoint exists ---
        try:
            existing = self.ws.serving_endpoints.get(endpoint_name)
            logger.info(f"🔄 Endpoint '{endpoint_name}' already exists. Updating configuration...")
            self.ws.serving_endpoints.update_config(
                name=endpoint_name,
                served_entities=endpoint_config.served_entities,
                auto_capture_config=endpoint_config.auto_capture_config,
            )
        except Exception as e:
            if "RESOURCE_DOES_NOT_EXIST" in str(e):
                logger.info(f"🆕 Endpoint '{endpoint_name}' does not exist. Creating new endpoint...")
                self.ws.serving_endpoints.create(
                    name=endpoint_name,
                    config=endpoint_config,
                    description=self.model_cfg.get("description", ""),
                )
            else:
                logger.error(f"❌ Unexpected error during deployment: {e}")
                raise

        # --- Wait for endpoint readiness ---
        self._wait_for_ready(endpoint_name)

    def _wait_for_ready(self, endpoint_name: str, timeout: int = 900, interval: int = 15):
        """
        Waits for the endpoint to reach READY state, polling periodically.
        """
        logger.info(f"⏳ Waiting for endpoint '{endpoint_name}' to reach READY state...")
        start = time.time()

        while True:
            endpoint = self.ws.serving_endpoints.get(endpoint_name)
            state = getattr(endpoint, "state", None)

            if state and getattr(state, "ready", False):
                logger.info(f"✅ Endpoint '{endpoint_name}' is READY.")
                return

            if time.time() - start > timeout:
                raise TimeoutError(f"⏰ Timeout: Endpoint '{endpoint_name}' not ready after {timeout}s")

            logger.info(f"   ↳ Current state: {getattr(state, 'ready', None)} ... retrying in {interval}s")
            time.sleep(interval)
