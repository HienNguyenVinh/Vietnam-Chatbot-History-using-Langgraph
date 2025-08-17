import uuid
import yaml
from pathlib import Path

def new_uuid():
    return str(uuid.uuid4())

def load_config(file_path= "./config.yaml"):
    """
    Loads the configuration from the YAML file.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()
