"""
Prediction routes: POST /predict, GET /health, GET /metrics.
"""

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from weed_detector.api.model_provider import LoadedModel
from weed_detector.api.schemas import HealthResponse, PredictionResponse, ReadyResponse
from weed_detector.api.service import ModelNotAvailableError, PredictionService
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
MODEL_LOADED = Gauge(
    "weed_detector_model_loaded",
    "1 if a model is loaded and the API can serve predictions, else 0",
)
MODEL_INFO = Gauge(
    "weed_detector_model_info",
    "Model version currently served (value is always 1)",
    ["model_name", "version", "alias"],
)


def update_model_metrics(loaded: LoadedModel | None) -> None:
    """Expose the served model version to Prometheus."""
    MODEL_INFO.clear()
    MODEL_LOADED.set(0 if loaded is None else 1)
    if loaded is not None:
        MODEL_INFO.labels(
            model_name=loaded.registry_name, version=loaded.version, alias=loaded.alias
        ).set(1)


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
    except ModelNotAvailableError as exc:
        logger.error(f"Prediction requested but no model is available: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No model available: {exc}",
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


@router.get("/")
async def root():
    return {"message": "Weed Detector API", "health": "/health"}


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
)
async def health_endpoint(request: Request):
    """
    Liveness: the process is up. Always 200; model_loaded tells whether a model
    is available (use /ready to gate traffic on that).
    """
    service = get_service(request)
    loaded = service.loaded
    return HealthResponse(
        status="ok",
        model_loaded=loaded is not None,
        model_version=loaded.version if loaded else None,
        model_alias=loaded.alias if loaded else None,
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness check",
)
async def ready_endpoint(request: Request):
    """Readiness: 200 only if a model is loaded, otherwise 503."""
    service = get_service(request)
    loaded = service.loaded
    if loaded is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=service.provider.last_error or "No model available in the registry",
        )
    return ReadyResponse(
        ready=True,
        model_name=loaded.registry_name,
        model_version=loaded.version,
        model_alias=loaded.alias,
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
