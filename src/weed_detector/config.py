"""
Config file loader
"""

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(
        path: Path | str = DEFAULT_CONFIG_PATH,
        root: Path | str = PROJECT_ROOT
        ) -> dict:
    """
    Load the hyperparameter config from given path.
    """
    
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()

    root = Path(root).resolve()

    with path.open() as f:
        config = yaml.safe_load(f)

    # Make config paths absolute
    for key, value in config["paths"].items():
        configured_path = Path(value)
        if not configured_path.is_absolute():
            configured_path = root / configured_path
        config["paths"][key] = str(configured_path.resolve())

    return config