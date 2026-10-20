"""
Unit tests for the YOLO model detector.
"""

from unittest.mock import MagicMock, patch

from weed_detector.model.detector import load_model


def test_load_model_calls_yolo_with_weights_path(tmp_path):
    """load_model should instantiate YOLO with the given weights path."""
    weights_file = tmp_path / "best.pt"
    weights_file.write_text("dummy weights")

    with patch("weed_detector.model.detector.YOLO") as mock_yolo:
        mock_instance = MagicMock()
        mock_yolo.return_value = mock_instance

        model = load_model(weights_path=str(weights_file))

        mock_yolo.assert_called_once_with(str(weights_file))
        assert model is mock_instance
