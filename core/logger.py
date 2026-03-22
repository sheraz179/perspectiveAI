import logging
import os
from core.config_loader import ConfigLoader

LOG_DIR =  ConfigLoader("config/pipeline_config.yaml").get('paths')['logs']
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("genai_pipeline")
logger.setLevel(logging.INFO)

# Prevent duplicate handlers
if not logger.handlers:

    file_handler = logging.FileHandler(f"{LOG_DIR}/pipeline.log")
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)