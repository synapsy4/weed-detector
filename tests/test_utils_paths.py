"""
Unit tests for path resolution utilities.
"""

import pytest

from weed_detector.utils.paths import (
    get_model_out_dir,
    get_pretrained_weights_path,
    get_weights_path,
)


def test_get_weights_path_valid_config():
    """get_weights_path should resolve to a Path under the train output dir."""
    config = {
        "paths": {"models_trained_dir": "/tmp/models"},
        "model": {"model_name": "modelv2"},
    }
    path = get_weights_path(config, cp="best")
    assert str(path) == "/tmp/models/modelv2/train/weights/best.pt"


def test_get_weights_path_invalid_cp():
    """get_weights_path should raise ValueError for invalid cp values."""
    config = {
        "paths": {"models_trained_dir": "/tmp/models"},
        "model": {"model_name": "modelv2"},
    }
    with pytest.raises(ValueError, match="cp must be one of"):
        get_weights_path(config, cp="invalid")


def test_get_pretrained_weights_path():
    """get_pretrained_weights_path should combine pretrained dir and weights filename."""
    config = {
        "paths": {"models_pretrained_dir": "/tmp/pretrained"},
        "model": {"pretrained_weights": "yolov8n.pt"},
    }
    path = get_pretrained_weights_path(config)
    assert str(path) == "/tmp/pretrained/yolov8n.pt"


def test_get_model_out_dir_train():
    """get_model_out_dir with mode='train' should resolve correctly."""
    config = {
        "paths": {"models_trained_dir": "/tmp/models"},
        "model": {"model_name": "modelv2"},
    }
    path = get_model_out_dir(config, mode="train")
    assert str(path) == "/tmp/models/modelv2/train"


def test_get_model_out_dir_eval():
    """get_model_out_dir with mode='eval' should resolve correctly."""
    config = {
        "paths": {"models_trained_dir": "/tmp/models"},
        "model": {"model_name": "modelv2"},
    }
    path = get_model_out_dir(config, mode="eval")
    assert str(path) == "/tmp/models/modelv2/eval"


def test_get_model_out_dir_invalid_mode():
    """get_model_out_dir should raise ValueError for invalid mode values."""
    config = {
        "paths": {"models_trained_dir": "/tmp/models"},
        "model": {"model_name": "modelv2"},
    }
    with pytest.raises(ValueError, match="mode must be one of"):
        get_model_out_dir(config, mode="invalid")
