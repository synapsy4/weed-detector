"""
YOLO prediction function
"""

from pathlib import Path

from ultralytics import YOLO

from weed_detector.visualization.plots import plot_yolo_pred


def predict(
        model: YOLO,
        config: dict,
        image_path: str | Path,
        conf_threshold: float = 0.25,
        visualize_result: bool = True
        ) -> dict:
    """
    Make prediction on single image.
    """

    # Make prediction
    results = model.predict(
        source=str(image_path),
        imgsz=config["model"]["imgsz"],
        conf=conf_threshold
    )
    result = results[0] 

    # Unpack predictions
    bbxs = result.boxes.xyxy.numpy()
    classes = result.boxes.cls.numpy()
    class_names = [result.names[c] for c in classes]
    scores = result.boxes.conf.numpy()

    # Show prediction
    if visualize_result:
        plot_yolo_pred(image_bgr=result.plot())

    return {"bbxs": bbxs, 
            "classes": classes, 
            "class_names": class_names, 
            "scores": scores}