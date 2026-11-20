"""
Config file loader
"""

from pathlib import Path

import yaml

from weed_detector.utils import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def apply_overrides(config: dict, overrides: list[str]) -> None:
    """
    Apply 'dotted.key=value' overrides in place, e.g. 'train.epochs=50'.

    Values are parsed as YAML (numbers, booleans, strings). Keys must already
    exist in the config, which catches typos.
    """
    for override in overrides:
        key, sep, raw_value = override.partition("=")
        if not sep or not key.strip():
            raise ValueError(
                f"Override must look like 'section.key=value': '{override}'"
            )

        *parents, leaf = key.strip().split(".")
        target = config
        for part in parents:
            if not isinstance(target, dict) or part not in target:
                raise ValueError(f"Unknown config key in override: '{key}'")
            target = target[part]
        if not isinstance(target, dict) or leaf not in target:
            raise ValueError(f"Unknown config key in override: '{key}'")

        target[leaf] = yaml.safe_load(raw_value)
        logger.info(f"Config override applied: {key.strip()}={target[leaf]!r}")


def load_config(
    path: Path | str = DEFAULT_CONFIG_PATH,
    root: Path | str = PROJECT_ROOT,
    overrides: list[str] | None = None,
) -> dict:
    """
    Load the hyperparameter config from given path.

    Optional 'dotted.key=value' overrides are applied before paths are made absolute.
    """

    logger.info(f"Loading config from '{path}'")

    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()

    root = Path(root).resolve()

    with path.open() as f:
        config = yaml.safe_load(f)

    if overrides:
        apply_overrides(config, overrides)

    # Make config paths absolute
    for key, value in config["paths"].items():
        configured_path = Path(value)
        if not configured_path.is_absolute():
            configured_path = root / configured_path
        config["paths"][key] = str(configured_path.resolve())

    logger.info("Config loaded successfully")
    return config
