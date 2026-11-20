"""
Unit tests for the training module.
"""

from unittest.mock import MagicMock, patch

from weed_detector.training.train import train


def _continue_config(tmp_path):
    return {
        "train": {"resume_blocking": False, "epochs": 2, "batch_size": 16},
        "paths": {
            "models_trained_dir": str(tmp_path),
            "yolo_yaml_path": "/tmp/yolo.yaml",
        },
        "model": {"model_name": "test_model", "imgsz": 512},
        "seed": 99,
    }


def _write_last_pt(tmp_path):
    weights_dir = tmp_path / "test_model" / "train" / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "last.pt").write_text("dummy")
    return weights_dir / "last.pt"


def test_train_warm_starts_from_last_pt(tmp_path):
    """train should warm start from 'last.pt' when weights exist and not blocking."""
    config = _continue_config(tmp_path)
    last_pt = _write_last_pt(tmp_path)

    mock_model = MagicMock()
    mock_metrics = MagicMock()
    mock_model.train.return_value = mock_metrics

    with patch(
        "weed_detector.training.train.load_model", return_value=mock_model
    ) as mock_load:
        metrics = train(config=config)

    assert metrics is mock_metrics
    # Weights come from last.pt (never best.pt), without resuming
    mock_load.assert_called_once_with(last_pt)
    assert mock_model.train.call_args.kwargs["resume"] is False


def test_warm_start_archives_previous_results_csv(tmp_path):
    """A warm start must not let the previous run's results.csv be overwritten."""
    config = _continue_config(tmp_path)
    _write_last_pt(tmp_path)
    out_dir = tmp_path / "test_model" / "train"
    (out_dir / "results.csv").write_text("epoch\n1\n")

    with patch("weed_detector.training.train.load_model", return_value=MagicMock()):
        train(config=config)

    assert not (out_dir / "results.csv").exists()
    archived = list(out_dir.glob("results_*.csv"))
    assert len(archived) == 1
    assert archived[0].read_text() == "epoch\n1\n"


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


def test_train_stores_run_id_and_links_previous_run(tmp_path):
    """A warm start tags the new run with the previous one and stores the new id."""
    config = _continue_config(tmp_path)
    _write_last_pt(tmp_path)
    out_dir = tmp_path / "test_model" / "train"
    (out_dir / "mlflow_run_id.txt").write_text("run-old")

    with (
        patch("weed_detector.training.train.load_model", return_value=MagicMock()),
        patch(
            "weed_detector.training.train.log_training_run", return_value="run-new"
        ) as mock_log,
    ):
        train(config=config)

    assert mock_log.call_args.kwargs["continued_from"] == "run-old"
    assert (out_dir / "mlflow_run_id.txt").read_text() == "run-new"


def test_train_removes_stale_run_id_when_logging_fails(tmp_path):
    """If MLflow logging fails, eval must not log to an outdated run."""
    config = _continue_config(tmp_path)
    _write_last_pt(tmp_path)
    out_dir = tmp_path / "test_model" / "train"
    (out_dir / "mlflow_run_id.txt").write_text("run-old")

    with (
        patch("weed_detector.training.train.load_model", return_value=MagicMock()),
        patch("weed_detector.training.train.log_training_run", return_value=None),
    ):
        train(config=config)

    assert not (out_dir / "mlflow_run_id.txt").exists()
