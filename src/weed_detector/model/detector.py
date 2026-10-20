"""
Init detector (YOLO model)
"""

from pathlib import Path

from ultralytics import YOLO

from weed_detector.utils.logging import get_logger

logger = get_logger(__name__)


def load_model(weights_path: str | Path) -> YOLO:
    """
    Load YOLO model with given weights.
    """
    logger.info(f"Loading YOLO model from '{weights_path}'")
    model = YOLO(str(weights_path))
    logger.info("YOLO model loaded successfully")
    return model
