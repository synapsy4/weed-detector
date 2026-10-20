"""
Unit tests for the training module.
"""

from unittest.mock import MagicMock, patch

from weed_detector.training.train import train


def test_train_resumes_when_weights_exist_and_not_blocking(tmp_path):
    """train should resume when 'last.pt' exists and resume_blocking is False."""
    config = {
        "train": {"resume_blocking": False, "epochs": 2, "batch_size": 16},
        "paths": {
            "models_trained_dir": str(tmp_path),
            "yolo_yaml_path": "/tmp/yolo.yaml",
        },
        "model": {"model_name": "test_model", "imgsz": 512},
        "seed": 99,
    }

    # Create existing last.pt to trigger resume path
    weights_dir = tmp_path / "test_model" / "train" / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "last.pt").write_text("dummy")

    mock_model = MagicMock()
    mock_metrics = MagicMock()
    mock_model.train.return_value = mock_metrics

    with patch("weed_detector.training.train.load_model", return_value=mock_model):
        metrics = train(config=config)

    assert metrics is mock_metrics
    mock_model.train.assert_called_once()
    # Check resume=True was passed
    call_kwargs = mock_model.train.call_args.kwargs
    assert call_kwargs["resume"] is True


def test_train_aborts_when_weights_exist_and_blocking(tmp_path):
    """train should return None when weights exist and resume_blocking is True."""
    config = {
        "train": {"resume_blocking": True},
        "paths": {
            "models_trained_dir": str(tmp_path),
            "yolo_yaml_path": "/tmp/yolo.yaml",
        },
        "model": {"model_name": "test_model", "imgsz": 512},
        "seed": 99,
    }

    weights_dir = tmp_path / "test_model" / "train" / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "last.pt").write_text("dummy")

    with patch("weed_detector.training.train.load_model") as mock_load_model:
        metrics = train(config=config)

    assert metrics is None
    mock_load_model.assert_not_called()


def test_train_new_run_loads_pretrained_weights(tmp_path):
    """train should load pretrained weights when no existing weights exist."""
    config = {
        "train": {"resume_blocking": False, "epochs": 2, "batch_size": 16},
        "paths": {
            "models_trained_dir": str(tmp_path),
            "models_pretrained_dir": str(tmp_path / "pretrained"),
            "yolo_yaml_path": "/tmp/yolo.yaml",
        },
        "model": {
            "model_name": "test_model",
            "pretrained_weights": "yolov8n.pt",
            "imgsz": 512,
        },
        "seed": 99,
    }

    # Create pretrained weights file
    (tmp_path / "pretrained").mkdir(parents=True)
    (tmp_path / "pretrained" / "yolov8n.pt").write_text("dummy")

    mock_model = MagicMock()
    mock_metrics = MagicMock()
    mock_model.train.return_value = mock_metrics

    with patch("weed_detector.training.train.load_model", return_value=mock_model):
        metrics = train(config=config)

    assert metrics is mock_metrics
    mock_model.train.assert_called_once()
    call_kwargs = mock_model.train.call_args.kwargs
    assert call_kwargs["resume"] is False
