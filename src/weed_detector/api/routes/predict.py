"""
Prediction routes: POST /predict, GET /health, GET /metrics.
"""

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from weed_detector.api.schemas import HealthResponse, PredictionResponse
from weed_detector.api.service import PredictionService
from weed_detector.utils import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Prometheus metrics
PREDICTIONS_TOTAL = Counter(
    "weed_detector_predictions_total",
    "Total number of predictions made",
)
PREDICTION_CONFIDENCE = Histogram(
    "weed_detector_prediction_confidence",
    "Confidence scores of predictions",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
)
PREDICTION_CLASS_TOTAL = Counter(
    "weed_detector_prediction_class_total",
    "Total predictions per class",
    ["class_name"],
)
HTTP_REQUESTS_TOTAL = Counter(
    "weed_detector_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)
HTTP_REQUEST_DURATION = Histogram(
    "weed_detector_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)


def get_service(request: Request) -> PredictionService:
    """Retrieve the PredictionService from the app state."""
    service = request.app.state.service
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service not initialized",
        )
    return service


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict crop/weed bounding boxes in an uploaded image",
)
async def predict_endpoint(
    request: Request,
    file: UploadFile = File(..., description="Image file (JPEG or PNG)"),  # noqa: B008
    conf_threshold: float = 0.25,
):
    """
    Accepts an image upload and returns detected crop/weed bounding boxes
    with class labels and confidence scores.
    """
    service = get_service(request)

    try:
        file_content = await file.read()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to read uploaded file: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read uploaded file",
        )

    try:
        response = service.predict_from_bytes(
            file_content=file_content,
            content_type=file.content_type or "application/octet-stream",
            conf_threshold=conf_threshold,
        )
    except ValueError as exc:
        logger.warning(f"Prediction validation error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed",
        )

    # Record metrics
    PREDICTIONS_TOTAL.inc()
    for pred in response.predictions:
        PREDICTION_CONFIDENCE.observe(pred.confidence)
        PREDICTION_CLASS_TOTAL.labels(class_name=pred.class_name).inc()

    logger.info(
        f"Prediction complete: {len(response.predictions)} box(es), "
        f"classes={[p.class_name for p in response.predictions]}"
    )

    return response


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health_endpoint(request: Request):
    """Returns the health status of the API and whether the model is loaded."""
    service = get_service(request)
    return HealthResponse(
        status="ok",
        model_loaded=service.model is not None,
    )


@router.get(
    "/metrics",
    summary="Prometheus metrics",
)
async def metrics_endpoint():
    """Exposes Prometheus metrics for scraping."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
