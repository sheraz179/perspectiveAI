import yaml, os
from core.config_loader import ConfigLoader

def get_system_prompt():
        
    config = ConfigLoader("config/pipeline_config.yaml")
    prompt_version = config.get('pipeline')['system_prompt_version']
    prompt_path = os.path.join('config/prompts', f'{prompt_version}.yaml')
    with open(prompt_path, "r") as f:
        configs = yaml.safe_load(f)


    model_config = ConfigLoader("config/model_config.yaml")
    objects_list = model_config.get('models')['detector']['detectable_objects']
    
    return configs["system_prompt"].format(
            DETECTABLE_OBJECTS=objects_list
        )

    return configs["system_prompt"].replace("{detectable_objects}", objects_list)
        # Inject the dynamic detectable objects list into the template
