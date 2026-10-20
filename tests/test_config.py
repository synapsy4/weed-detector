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
