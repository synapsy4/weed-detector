"""
Unit tests for the evaluation module.
"""

import json
from unittest.mock import MagicMock, patch

from weed_detector.training.evaluate import eval as evaluate


def test_eval_calls_model_val_with_test_split():
    """eval should call model.val with split='test' and save_dir set."""
    config = {
        "paths": {"yolo_yaml_path": "/tmp/yolo.yaml"},
        "paths_eval_override": {},
        "model": {"model_name": "test_model"},
        "paths_models_trained_dir": "/tmp/models",
    }

    mock_model = MagicMock()
    mock_metrics = MagicMock()
    mock_model.val.return_value = mock_metrics

    with patch("weed_detector.training.evaluate.get_model_out_dir") as mock_out_dir:
        mock_out_dir.return_value = "/tmp/models/test_model/eval"
        with patch("weed_detector.training.evaluate.get_device", return_value="cpu"):
            metrics = evaluate(model=mock_model, config=config)

    assert metrics is mock_metrics
    mock_model.val.assert_called_once()
    call_kwargs = mock_model.val.call_args.kwargs
    assert call_kwargs["split"] == "test"
    assert call_kwargs["save_dir"] == "/tmp/models/test_model/eval"
    assert call_kwargs["device"] == "cpu"


def test_eval_writes_metrics_json_to_eval_dir(tmp_path):
    """Evaluation should write a DVC-safe metrics.json file outside the training output tree."""
    metrics_path = tmp_path / "models" / "test_model" / "eval" / "metrics.json"
    config = {
        "paths": {
            "yolo_yaml_path": "/tmp/yolo.yaml",
            "models_trained_dir": str(tmp_path / "models"),
        },
        "model": {"model_name": "test_model"},
    }

    mock_model = MagicMock()
    mock_metrics = MagicMock()
    mock_metrics.results_dict = {"metrics/precision(B)": 0.42, "fitness": 0.81}
    mock_model.val.return_value = mock_metrics

    with patch("weed_detector.training.evaluate.get_device", return_value="cpu"):
        evaluate(model=mock_model, config=config)

    assert metrics_path.exists()
    assert json.loads(metrics_path.read_text()) == {
        "metrics_precision_B": 0.42,
        "fitness": 0.81,
    }


def _eval_config(tmp_path):
    return {
        "paths": {
            "yolo_yaml_path": "/tmp/yolo.yaml",
            "models_trained_dir": str(tmp_path / "models"),
        },
        "model": {"model_name": "test_model"},
    }


def test_eval_writes_per_class_metrics(tmp_path):
    """Per-class precision/recall/AP50 should be written as '<class>_<metric>'."""
    config = _eval_config(tmp_path)

    mock_metrics = MagicMock()
    mock_metrics.results_dict = {"fitness": 0.5}
    mock_metrics.names = {0: "crop", 1: "weed"}
    mock_metrics.box.ap_class_index = [0, 1]
    mock_metrics.box.p = [0.9, 0.7]
    mock_metrics.box.r = [0.8, 0.6]
    mock_metrics.box.ap50 = [0.85, 0.65]
    mock_model = MagicMock()
    mock_model.val.return_value = mock_metrics

    with patch("weed_detector.training.evaluate.get_device", return_value="cpu"):
        evaluate(model=mock_model, config=config)

    metrics_path = tmp_path / "models" / "test_model" / "eval" / "metrics.json"
    payload = json.loads(metrics_path.read_text())
    assert payload["weed_recall"] == 0.6
    assert payload["crop_precision"] == 0.9
    assert payload["weed_ap50"] == 0.65
    assert payload["fitness"] == 0.5


def test_eval_logs_to_training_run_when_run_id_exists(tmp_path):
    """Eval metrics should be logged to the MLflow run stored with the model."""
    config = _eval_config(tmp_path)
    train_dir = tmp_path / "models" / "test_model" / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "mlflow_run_id.txt").write_text("run-abc")

    mock_metrics = MagicMock()
    mock_metrics.results_dict = {"fitness": 0.5}
    mock_model = MagicMock()
    mock_model.val.return_value = mock_metrics

    with (
        patch("weed_detector.training.evaluate.get_device", return_value="cpu"),
        patch("weed_detector.training.evaluate.log_eval_metrics") as mock_log,
    ):
        evaluate(model=mock_model, config=config)

    mock_log.assert_called_once()
    assert mock_log.call_args.kwargs["run_id"] == "run-abc"
    assert mock_log.call_args.kwargs["eval_metrics"]["fitness"] == 0.5


def test_eval_skips_mlflow_without_run_id(tmp_path):
    """Without a stored run id nothing is logged to MLflow."""
    config = _eval_config(tmp_path)
    mock_metrics = MagicMock()
    mock_metrics.results_dict = {"fitness": 0.5}
    mock_model = MagicMock()
    mock_model.val.return_value = mock_metrics

    with (
        patch("weed_detector.training.evaluate.get_device", return_value="cpu"),
        patch("weed_detector.training.evaluate.log_eval_metrics") as mock_log,
    ):
        evaluate(model=mock_model, config=config)

    mock_log.assert_not_called()
