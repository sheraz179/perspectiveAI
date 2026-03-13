import yaml
import os

class PromptManager:
    def __init__(self, prompt_dir="configs/prompts"):
        self.prompt_dir = prompt_dir

    def load(self, version):
        path = os.path.join(self.prompt_dir, f"{version}.yaml")
        with open(path, "r") as f:
            return yaml.safe_load(f)
