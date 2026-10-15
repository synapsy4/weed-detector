"""
Init detector (YOLO model)
"""

from pathlib import Path

from ultralytics import YOLO


def load_model(
        weights_path: str | Path
        ) -> YOLO:
    """
    Load YOLO model with given weights.
    """
    return YOLO(str(weights_path))