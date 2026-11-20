"""
Unit tests for the config loader.
"""

from pathlib import Path

import yaml

from weed_detector.config import load_config


def test_load_config_makes_paths_absolute(tmp_path):
    """All paths in config['paths'] should be resolved to absolute paths."""
    config_content = {
        "seed": 99,
        "data": {"test_split": 0.15, "val_split": 0.15},
        "model": {
            "model_name": "modelv2",
            "pretrained_weights": "yolov8n.pt",
            "imgsz": 512,
        },
        "train": {"epochs": 2, "batch_size": 16, "resume_blocking": False},
        "paths": {
            "raw_dir": "data/raw",
            "raw_data_dir": "data/raw/agri_data/data",
            "processed_data_dir": "data/processed",
            "yolo_yaml_path": "configs/yolo.yaml",
            "models_pretrained_dir": "models/pretrained",
            "models_trained_dir": "models/trained",
        },
    }

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config_content))

    config = load_config(path=config_file, root=tmp_path)

    for key, value in config["paths"].items():
        assert Path(value).is_absolute(), (
            f"Path '{key}' should be absolute, got '{value}'"
        )
        assert str(value).startswith(str(tmp_path))


def test_load_config_returns_dict(tmp_path):
    """load_config should return a dictionary with expected top-level keys."""
    config_content = {
        "seed": 99,
        "data": {"test_split": 0.15, "val_split": 0.15},
        "model": {
            "model_name": "modelv2",
            "pretrained_weights": "yolov8n.pt",
            "imgsz": 512,
        },
        "train": {"epochs": 2, "batch_size": 16, "resume_blocking": False},
        "paths": {
            "raw_dir": "data/raw",
            "processed_data_dir": "data/processed",
            "yolo_yaml_path": "configs/yolo.yaml",
            "models_pretrained_dir": "models/pretrained",
            "models_trained_dir": "models/trained",
        },
    }

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config_content))

    config = load_config(path=config_file, root=tmp_path)

    assert isinstance(config, dict)
    assert config["seed"] == 99
    assert config["model"]["model_name"] == "modelv2"
    assert config["train"]["epochs"] == 2


def _write_config(tmp_path):
    config_content = {
        "seed": 99,
        "train": {"epochs": 2, "resume_blocking": False},
        "paths": {"raw_dir": "data/raw"},
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config_content))
    return config_file


def test_load_config_applies_overrides_with_yaml_types(tmp_path):
    """Overrides should be parsed as YAML so numbers and booleans keep their type."""
    config = load_config(
        path=_write_config(tmp_path),
        root=tmp_path,
        overrides=["train.epochs=50", "train.resume_blocking=true", "seed=7"],
    )

    assert config["train"]["epochs"] == 50
    assert config["train"]["resume_blocking"] is True
    assert config["seed"] == 7


def test_load_config_overrides_paths_before_absolutizing(tmp_path):
    """Path overrides should end up absolute like every other path."""
    config = load_config(
        path=_write_config(tmp_path),
        root=tmp_path,
        overrides=["paths.raw_dir=other/raw"],
    )

    assert config["paths"]["raw_dir"] == str((tmp_path / "other" / "raw").resolve())


def test_load_config_rejects_unknown_override_key(tmp_path):
    """Typos in override keys must fail loudly instead of being ignored."""
    import pytest

    for bad in ("train.epochz=5", "nope.epochs=5", "train.epochs"):
        with pytest.raises(ValueError):
            load_config(path=_write_config(tmp_path), root=tmp_path, overrides=[bad])
