import os
import logging
from datetime import datetime

def setup_logger(name="vllm_giskard", log_dir="."):
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info("Logger initialized. %s", logfile)
    return logger