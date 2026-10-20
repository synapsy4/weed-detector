"""
YOLO evaluation function
"""

from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics

from weed_detector.utils import get_device, get_logger, get_model_out_dir

logger = get_logger(__name__)


def eval(model: YOLO, config: dict) -> DetMetrics:
    """
    Call preimplemented YOLO val function.
    """
    # Specify output dir
    model_out_dir = get_model_out_dir(config=config, mode="eval")

    device = get_device()

    logger.info("Starting evaluation on test split")

    # Run validation on test set
    metrics = model.val(
        data=config["paths"]["yolo_yaml_path"],
        split="test",
        save_dir=model_out_dir,
        device=device,
    )

    logger.info("Evaluation completed")
    return metrics
