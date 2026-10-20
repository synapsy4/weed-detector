"""
Pytest configuration and fixtures.
"""

import pytest
from fastapi.testclient import TestClient

from weed_detector.api.main import create_app
from weed_detector.api.service import PredictionService


@pytest.fixture
def app():
    """Create a FastAPI app instance without triggering model loading."""
    return create_app()


@pytest.fixture
def client(app):
    """Create a TestClient for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_service(monkeypatch):
    """
    Replace PredictionService with a mock that doesn't load a model.

    This allows testing route handlers without downloading YOLO weights.
    """
    from unittest.mock import MagicMock

    service = MagicMock(spec=PredictionService)
    service.model = MagicMock()
    service.model_name = "test-model"

    # Mock predict_from_bytes to return a valid response
    from weed_detector.api.schemas import (
        BBox,
        Prediction,
        PredictionResponse,
    )

    service.predict_from_bytes.return_value = PredictionResponse(
        predictions=[
            Prediction(
                bbox=BBox(x1=10.0, y1=10.0, x2=50.0, y2=50.0),
                class_id=1,
                class_name="weed",
                confidence=0.9,
            )
        ],
        image_size=[512, 512],
        model_name="test-model",
    )

    # Store on app state via monkeypatch of create_app is not straightforward,
    # so we patch the routes module's get_service directly.
    import weed_detector.api.routes.predict as predict_routes

    original_get_service = predict_routes.get_service

    def fake_get_service(request):
        return service

    predict_routes.get_service = fake_get_service
    yield service

    predict_routes.get_service = original_get_service


@pytest.fixture
def sample_jpeg_bytes():
    """Return valid JPEG bytes for testing image validation."""
    import io

    from PIL import Image

    img = Image.new("RGB", (10, 10), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_png_bytes():
    """Return valid PNG bytes for testing image validation."""
    import io

    from PIL import Image

    img = Image.new("RGB", (10, 10), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def invalid_file_bytes():
    """Return bytes that are not a valid image."""
    return b"this is not an image file"
