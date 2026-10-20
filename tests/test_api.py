"""
Unit tests for the FastAPI prediction API.
"""

import io

import pytest

from weed_detector.api.schemas import (
    BBox,
    HealthResponse,
    Prediction,
    PredictionResponse,
)
from weed_detector.api.service import PredictionService

# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_bbox_validation():
    """BBox should reject negative coordinates."""
    with pytest.raises(ValueError):
        BBox(x1=-1.0, y1=0.0, x2=10.0, y2=10.0)

    bbox = BBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
    assert bbox.x1 == 0.0


def test_prediction_confidence_range():
    """Confidence must be between 0 and 1."""
    with pytest.raises(ValueError):
        Prediction(
            bbox=BBox(x1=0, y1=0, x2=10, y2=10),
            class_id=0,
            class_name="crop",
            confidence=1.5,
        )

    pred = Prediction(
        bbox=BBox(x1=0, y1=0, x2=10, y2=10),
        class_id=0,
        class_name="crop",
        confidence=0.5,
    )
    assert pred.confidence == 0.5


def test_prediction_response_serialization():
    """PredictionResponse should serialize to JSON correctly."""
    response = PredictionResponse(
        predictions=[
            Prediction(
                bbox=BBox(x1=1.0, y1=2.0, x2=3.0, y2=4.0),
                class_id=1,
                class_name="weed",
                confidence=0.85,
            )
        ],
        image_size=[512, 512],
        model_name="modelv0",
    )
    data = response.model_dump()
    assert data["image_size"] == [512, 512]
    assert data["model_name"] == "modelv0"
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["class_name"] == "weed"


def test_health_response():
    """HealthResponse should have expected fields."""
    health = HealthResponse(status="ok", model_loaded=True)
    assert health.status == "ok"
    assert health.model_loaded is True


# ---------------------------------------------------------------------------
# Service validation tests
# ---------------------------------------------------------------------------


def test_validate_image_unsupported_mime():
    """Reject non-image MIME types."""
    service = PredictionService.__new__(PredictionService)
    service.config = {}
    with pytest.raises(ValueError, match="Unsupported image type"):
        service.validate_image(b"fake", "text/plain")


def test_validate_image_file_too_large():
    """Reject images larger than MAX_FILE_SIZE_MB."""
    service = PredictionService.__new__(PredictionService)
    service.config = {}
    oversized = b"\x00" * (11 * 1024 * 1024)  # 11 MB
    with pytest.raises(ValueError, match="File too large"):
        service.validate_image(oversized, "image/jpeg")


def test_validate_image_invalid_content():
    """Reject bytes that are not a valid image."""
    service = PredictionService.__new__(PredictionService)
    service.config = {}
    with pytest.raises(ValueError, match="Invalid image file"):
        service.validate_image(b"not an image", "image/png")


def test_validate_image_valid_jpeg(sample_jpeg_bytes):
    """Accept valid JPEG bytes."""
    service = PredictionService.__new__(PredictionService)
    service.config = {}
    # Should not raise
    service.validate_image(sample_jpeg_bytes, "image/jpeg")


def test_validate_image_valid_png(sample_png_bytes):
    """Accept valid PNG bytes."""
    service = PredictionService.__new__(PredictionService)
    service.config = {}
    # Should not raise
    service.validate_image(sample_png_bytes, "image/png")


# ---------------------------------------------------------------------------
# App / route tests
# ---------------------------------------------------------------------------


def test_create_app_has_routes(app):
    """The FastAPI app should expose /predict, /health, /metrics."""
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/predict" in paths
    assert "/health" in paths
    assert "/metrics" in paths


def test_health_endpoint(client, mock_service):
    """GET /health should return ok with model_loaded=True."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_endpoint(client, mock_service):
    """POST /predict should return predictions from the mocked service."""
    files = {"file": ("test.jpg", io.BytesIO(b"\xff\xd8\xff\xd9"), "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["class_name"] == "weed"
    assert data["image_size"] == [512, 512]


def test_predict_endpoint_missing_file(client):
    """POST /predict without a file should return 422."""
    response = client.post("/predict")
    assert response.status_code == 422


def test_metrics_endpoint(client):
    """GET /metrics should return Prometheus text format."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "weed_detector_predictions_total" in response.text
    assert response.headers["content-type"].startswith("text/plain")
