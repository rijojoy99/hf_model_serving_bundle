import yaml
import logging
from pathlib import Path

logger = logging.getLogger("ConfigLoader")

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Loads configuration from YAML file into a dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
        logger.info(f"✅ Loaded config from {config_path}")
        return cfg
