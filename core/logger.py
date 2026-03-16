import os, logging

def setup_logger(log_dir="logs"):

    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        filename=f"{log_dir}/pipeline.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    return logging.getLogger("pipeline")

# logger = setup_logger()
# logger.info("Starting generation")
# logger.info(
#     f"Prompt: {prompt} | Labels: {labels}"
# )