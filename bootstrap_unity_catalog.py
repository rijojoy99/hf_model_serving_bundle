import logging
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import Privilege
from hf_model_serving.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("UC_Bootstrap")

def bootstrap_unity_catalog(config_path="config.yaml"):
    cfg = load_config(config_path)
    bootstrap_cfg = cfg.get("bootstrap", {})
    if not bootstrap_cfg.get("enabled", False):
        logger.info("⏭️ Bootstrap disabled in config.yaml — skipping UC initialization.")
        return

    catalog_name = bootstrap_cfg["catalog_name"]
    schema_name = bootstrap_cfg["schema_name"]
    log_schema_name = bootstrap_cfg["log_schema_name"]
    volume_name = bootstrap_cfg["volume_name"]
    principal = bootstrap_cfg["principal"]

    ws = WorkspaceClient()

    def ensure_catalog(name):
        try:
            ws.catalogs.get(name)
            logger.info(f"✅ Catalog '{name}' already exists.")
        except Exception:
            ws.catalogs.create(name=name, comment="Bootstrap UC catalog for HF models")
            logger.info(f"🆕 Created catalog '{name}'.")

    def ensure_schema(catalog, schema):
        try:
            ws.schemas.get(full_name=f"{catalog}.{schema}")
            logger.info(f"✅ Schema '{catalog}.{schema}' already exists.")
        except Exception:
            ws.schemas.create(catalog_name=catalog, name=schema)
            logger.info(f"🆕 Created schema '{catalog}.{schema}'.")

    def ensure_volume(catalog, schema, volume):
        try:
            ws.volumes.get(full_name=f"{catalog}.{schema}.{volume}")
            logger.info(f"✅ Volume '{catalog}.{schema}.{volume}' already exists.")
        except Exception:
            ws.volumes.create(catalog_name=catalog, schema_name=schema, name=volume)
            logger.info(f"🆕 Created volume '{catalog}.{schema}.{volume}'.")

    def grant_privileges():
        try:
            ws.grants.update(
                securable_type="catalog",
                securable_name=catalog_name,
                changes=[{"principal": principal, "add": [Privilege.USE_CATALOG, Privilege.CREATE_SCHEMA]}],
            )
            ws.grants.update(
                securable_type="schema",
                securable_name=f"{catalog_name}.{schema_name}",
                changes=[{"principal": principal, "add": [Privilege.USE_SCHEMA, Privilege.CREATE_TABLE, Privilege.CREATE_VOLUME]}],
            )
            ws.grants.update(
                securable_type="schema",
                securable_name=f"{catalog_name}.{log_schema_name}",
                changes=[{"principal": principal, "add": [Privilege.USE_SCHEMA, Privilege.CREATE_TABLE]}],
            )
            logger.info(f"🔐 Granted privileges to '{principal}'.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to apply privileges: {e}")

    # Execute bootstrap steps
    ensure_catalog(catalog_name)
    ensure_schema(catalog_name, schema_name)
    ensure_schema(catalog_name, log_schema_name)
    ensure_volume(catalog_name, schema_name, volume_name)
    grant_privileges()
    logger.info("🎉 Unity Catalog bootstrap complete.")


if __name__ == "__main__":
    bootstrap_unity_catalog()
