"""
YOLO evaluation function
"""

from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics

from weed_detector.utils.paths import get_model_out_dir
from weed_detector.utils.yolo import get_device


def eval(
        model: YOLO,
        config: dict
        ) -> DetMetrics:
    """
    Call preimplemented YOLO val function.
    """
    # Specify output dir
    model_out_dir = get_model_out_dir(
        config=config,
        mode="eval"
        )
    
    device = get_device()

    # Run validation on test set
    metrics = model.val(
        data=config["paths"]["yolo_yaml_path"],
        split="test",
        save_dir=model_out_dir,
        device=device
    )

    return metrics