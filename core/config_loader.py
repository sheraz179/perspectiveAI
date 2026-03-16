import yaml

class ConfigLoader:

    def __init__(self, config_path):

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def get(self, key):

        return self.config[key]


# config = ConfigLoader("config/pipeline_config.yaml")
# threshold = config.get("pipeline")["quality_threshold"]