import os
import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if config is None:
                raise ValueError(f"Config file is empty or invalid: {path}")
            return config
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML config file {path}: {e}") from e