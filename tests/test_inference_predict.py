"""
Unit tests for the prediction/inference module.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from weed_detector.inference.predict import predict


def test_predict_returns_expected_dict_structure():
    """predict should return a dict with bbxs, classes, class_names, scores."""
    config = {"model": {"imgsz": 512}}

    # Mock the model and its predict output
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.boxes.xyxy.numpy.return_value = np.array([[10.0, 20.0, 50.0, 60.0]])
    mock_result.boxes.cls.numpy.return_value = np.array([1])
    mock_result.names = {0: "crop", 1: "weed"}
    mock_result.boxes.conf.numpy.return_value = np.array([0.85])
    mock_model.predict.return_value = [mock_result]

    with patch("weed_detector.inference.predict.plot_yolo_pred"):
        result = predict(
            model=mock_model,
            config=config,
            image_path="/tmp/test.jpg",
            conf_threshold=0.25,
            visualize_result=False,
        )

    assert "bbxs" in result
    assert "classes" in result
    assert "class_names" in result
    assert "scores" in result
    assert result["class_names"] == ["weed"]
    assert result["scores"][0] == 0.85


def test_predict_calls_model_with_correct_args():
    """predict should call model.predict with source, imgsz, and conf."""
    config = {"model": {"imgsz": 512}}

    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.boxes.xyxy.numpy.return_value = np.array([])
    mock_result.boxes.cls.numpy.return_value = np.array([])
    mock_result.names = {}
    mock_result.boxes.conf.numpy.return_value = np.array([])
    mock_model.predict.return_value = [mock_result]

    with patch("weed_detector.inference.predict.plot_yolo_pred"):
        predict(
            model=mock_model,
            config=config,
            image_path="/tmp/image.jpg",
            conf_threshold=0.5,
            visualize_result=False,
        )

    mock_model.predict.assert_called_once_with(
        source="/tmp/image.jpg", imgsz=512, conf=0.5
    )
