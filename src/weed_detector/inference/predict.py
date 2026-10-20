"""
YOLO prediction function
"""

from pathlib import Path

from ultralytics import YOLO

from weed_detector.utils import get_logger
from weed_detector.visualization import plot_yolo_pred

logger = get_logger(__name__)


def predict(
    model: YOLO,
    config: dict,
    image_path: str | Path,
    conf_threshold: float = 0.25,
    visualize_result: bool = True,
) -> dict:
    """
    Make prediction on single image.
    """
    logger.info(f"Running prediction on '{image_path}'")

    # Make prediction
    results = model.predict(
        source=str(image_path), imgsz=config["model"]["imgsz"], conf=conf_threshold
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
