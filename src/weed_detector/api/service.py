"""
Prediction service: wraps YOLO inference with validation.
"""

import io

from PIL import Image

from weed_detector.api.model_provider import LoadedModel, ModelProvider
from weed_detector.api.schemas import (
    BBox,
    Prediction,
    PredictionResponse,
)
from weed_detector.config import load_config
from weed_detector.inference import predict
from weed_detector.utils import get_logger

logger = get_logger(__name__)

MAX_FILE_SIZE_MB = 10
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}


class ModelNotAvailableError(RuntimeError):
    """Raised when no model could be loaded from the registry or the local cache."""


class PredictionService:
    """Service that serves predictions with the model chosen by the provider."""

    def __init__(
        self, config: dict | None = None, provider: ModelProvider | None = None
    ):
        self.config = config or load_config()
        if provider is None:
            raise ValueError("PredictionService requires a ModelProvider")
        self.provider = provider

    @property
    def loaded(self) -> LoadedModel | None:
        """The model version currently served, or None if none is available."""
        return self.provider.current

    def validate_image(self, file_content: bytes, content_type: str) -> None:
        """Validate uploaded image MIME type and size."""
        if content_type not in ALLOWED_MIME_TYPES:
            raise ValueError(
                f"Unsupported image type '{content_type}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_MIME_TYPES))}"
            )

        size_mb = len(file_content) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(
                f"File too large: {size_mb:.1f} MB. Maximum: {MAX_FILE_SIZE_MB} MB."
            )

        # Verify it's a valid image
        try:
            with Image.open(io.BytesIO(file_content)) as img:
                img.verify()
        except Exception as exc:
            raise ValueError(f"Invalid image file: {exc}") from exc

    def predict_from_bytes(
        self,
        file_content: bytes,
        content_type: str,
        conf_threshold: float = 0.25,
    ) -> PredictionResponse:
        """Run prediction on raw image bytes."""
        loaded = self.loaded
        if loaded is None:
            raise ModelNotAvailableError(
                self.provider.last_error or "No model available in the registry"
            )

        self.validate_image(file_content, content_type)

        # Determine image size from bytes
        with Image.open(io.BytesIO(file_content)) as img:
            width, height = img.size

        pil_image = Image.open(io.BytesIO(file_content))
        # Predict with the image size this model version was trained with
        model_config = {
            **self.config,
            "model": {**self.config["model"], "imgsz": loaded.imgsz},
        }
        result = predict(
            model=loaded.model,
            config=model_config,
            image_source=pil_image,
            conf_threshold=conf_threshold,
            visualize_result=False,
        )

        # Build response
        predictions = []
        for box, cls_id, cls_name, score in zip(
            result["bbxs"],
            result["classes"],
            result["class_names"],
            result["scores"],
        ):
            x1, y1, x2, y2 = [float(v) for v in box]
            predictions.append(
                Prediction(
                    bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    class_id=int(cls_id),
                    class_name=cls_name,
                    confidence=float(score),
                )
            )

        return PredictionResponse(
            predictions=predictions,
            image_size=[width, height],
            model_name=loaded.registry_name,
            model_version=loaded.version,
        )
