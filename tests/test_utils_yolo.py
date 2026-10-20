"""
Unit tests for YOLO utility functions.
"""

import yaml

from weed_detector.utils.yolo import build_train_kwargs, get_device, write_yolo_yaml


def test_get_device_returns_cpu_or_zero():
    """get_device should return either 0 (GPU) or 'cpu' (CPU)."""
    device = get_device()
    assert device == 0 or device == "cpu"


def test_build_train_kwargs(tmp_path):
    """build_train_kwargs should produce the expected keyword arguments."""
    config = {
        "paths": {"yolo_yaml_path": "/tmp/configs/yolo.yaml"},
        "train": {"epochs": 10, "batch_size": 16},
        "model": {"imgsz": 512},
        "seed": 99,
    }
    model_out_dir = tmp_path / "train_output"

    kwargs = build_train_kwargs(
        config=config, model_out_dir=model_out_dir, resume=False
    )

    assert kwargs["data"] == "/tmp/configs/yolo.yaml"
    assert kwargs["epochs"] == 10
    assert kwargs["batch"] == 16
    assert kwargs["imgsz"] == 512
    assert kwargs["seed"] == 99
    assert kwargs["save_dir"] == model_out_dir
    assert kwargs["resume"] is False


def test_write_yolo_yaml(tmp_path):
    """write_yolo_yaml should write a valid YAML file with path, train/val/test, names."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "classes.txt").write_text("crop\nweed\n")

    processed_dir = tmp_path / "processed"
    yaml_path = tmp_path / "configs" / "yolo.yaml"

    write_yolo_yaml(
        raw_dir=str(raw_dir),
        processed_data_dir=str(processed_dir),
        yolo_yaml_path=str(yaml_path),
    )

    assert yaml_path.exists()
    content = yaml.safe_load(yaml_path.read_text())
    assert content["path"] == str(processed_dir)
    assert content["train"] == "images/train"
    assert content["val"] == "images/val"
    assert content["test"] == "images/test"
    assert content["names"] == {0: "crop", 1: "weed"}
