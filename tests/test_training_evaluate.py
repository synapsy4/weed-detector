"""
Unit tests for the evaluation module.
"""

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
