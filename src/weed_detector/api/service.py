"""
Prediction service: wraps YOLO inference with validation.
"""

import io
import tempfile
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from weed_detector.api.schemas import (
    BBox,
    Prediction,
    PredictionResponse,
)
from weed_detector.config import load_config
from weed_detector.inference import predict
from weed_detector.model import load_model
from weed_detector.utils import (
    get_logger,
    get_pretrained_weights_path,
    get_weights_path,
)

logger = get_logger(__name__)

MAX_FILE_SIZE_MB = 10
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}


class PredictionService:
    """Service that loads a YOLO model once and serves predictions."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.model: YOLO | None = None
        self.model_name = self.config["model"]["model_name"]
        self._load_model()

    def _load_model(self) -> None:
        """Load the best trained weights, falling back to pretrained."""
        weights_path = get_weights_path(self.config, cp="best")
        if not weights_path.exists():
            logger.info(
                f"No trained weights found at '{weights_path}', "
                f"falling back to pretrained weights"
            )
            weights_path = get_pretrained_weights_path(self.config)

        logger.info(f"Loading model weights from '{weights_path}'")
        self.model = load_model(weights_path)
        logger.info("Model loaded successfully")

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

    def predict_from_path(
        self,
        image_path: str | Path,
        conf_threshold: float = 0.25,
    ) -> PredictionResponse:
        """Run prediction on an image file path."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        result = predict(
            model=self.model,
            config=self.config,
            image_path=image_path,
            conf_threshold=conf_threshold,
            visualize_result=False,  # never plot in the API service
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
            image_size=[0, 0],
            model_name=self.model_name,
        )

    def predict_from_bytes(
        self,
        file_content: bytes,
        content_type: str,
        conf_threshold: float = 0.25,
    ) -> PredictionResponse:
        """Run prediction on raw image bytes (saves to temp file)."""
        self.validate_image(file_content, content_type)

        # Determine image size from bytes
        with Image.open(io.BytesIO(file_content)) as img:
            width, height = img.size

        # Save to temp file for the predict() function (which expects a path)
        suffix = ".jpg" if content_type == "image/jpeg" else ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        try:
            response = self.predict_from_path(
                image_path=tmp_path,
                conf_threshold=conf_threshold,
            )
            response.image_size = [width, height]
            return response
        finally:
            Path(tmp_path).unlink(missing_ok=True)
