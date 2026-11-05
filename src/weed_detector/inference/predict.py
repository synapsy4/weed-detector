"""
YOLO prediction function
"""

import io
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

from weed_detector.utils import get_logger
from weed_detector.visualization import plot_yolo_pred

logger = get_logger(__name__)

# Any source YOLO can run inference on: a file path, raw bytes, a PIL image,
# or a numpy array.
ImageSource = str | Path | bytes | Image.Image | np.ndarray


def predict(
    model: YOLO,
    config: dict,
    image_source: ImageSource,
    conf_threshold: float = 0.25,
    visualize_result: bool = True,
) -> dict:
    """
    Make prediction on a single image.

    ``image_source`` may be a file path, raw image bytes, a PIL Image, or a
    numpy array
    """

    logger.info(f"Running prediction on image source of type {type(image_source)}")

    # YOLO does not consume raw bytes directly; normalise to a PIL Image.
    if isinstance(image_source, (bytes, bytearray)):
        image_source = Image.open(io.BytesIO(image_source))

    # Make prediction
    results = model.predict(
        source=image_source, imgsz=config["model"]["imgsz"], conf=conf_threshold
    )
    result = results[0]

    # Unpack predictions
    bbxs = result.boxes.xyxy.numpy()  # type: ignore[union-attr]
    classes = result.boxes.cls.numpy()  # type: ignore[union-attr]
    class_names = [result.names[c] for c in classes]
    scores = result.boxes.conf.numpy()  # type: ignore[union-attr]

    logger.info(
        f"Prediction complete: {len(bbxs)} box(es) detected, classes={class_names}"
    )

    # Show prediction
    if visualize_result:
        plot_yolo_pred(image_bgr=result.plot())

    return {
        "bbxs": bbxs,
        "classes": classes,
        "class_names": class_names,
        "scores": scores,
    }
