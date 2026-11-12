"""
YOLO evaluation function
"""

import json
import math
import re
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics

from weed_detector.utils import get_device, get_logger, get_model_out_dir

logger = get_logger(__name__)


def _serialize_metrics(metrics: DetMetrics) -> dict[str, float]:
    """Convert YOLO metrics to a JSON-serializable dict with DVC-safe keys."""
    raw_metrics = getattr(metrics, "results_dict", {})
    if not isinstance(raw_metrics, dict):
        return {}

    sanitized: dict[str, float] = {}
    for key, value in raw_metrics.items():
        if not isinstance(key, str) or value is None:
            continue
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            continue
        safe_key = re.sub(r"[^A-Za-z0-9_]+", "_", key).strip("_")
        sanitized[safe_key] = numeric_value

    return sanitized


def eval(model: YOLO, config: dict) -> DetMetrics:
    """
    Call preimplemented YOLO val function.
    """
    # Specify output dir
    model_out_dir = Path(get_model_out_dir(config=config, mode="eval"))
    metrics_path = model_out_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    device = get_device()

    logger.info("Starting evaluation on test split")

    # Run validation on test set
    metrics = model.val(
        data=config["paths"]["yolo_yaml_path"],
        split="test",
        save_dir=str(model_out_dir),
        device=device,
    )

    payload = _serialize_metrics(metrics)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logger.info("Evaluation completed")
    return metrics
