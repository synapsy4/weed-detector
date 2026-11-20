"""
YOLO evaluation function
"""

import json
import math
import re
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics

from weed_detector.utils import (
    get_device,
    get_logger,
    get_model_out_dir,
    load_run_id,
    log_eval_metrics,
)

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


def _per_class_metrics(metrics: DetMetrics) -> dict[str, float]:
    """
    Extract per-class precision, recall and AP50 as '<class>_<metric>' entries.

    Classes without any ground truth in the evaluated split are not reported by
    Ultralytics and therefore missing here.
    """
    try:
        names = metrics.names
        box = metrics.box
        result: dict[str, float] = {}
        for pos, class_idx in enumerate(box.ap_class_index):
            class_name = re.sub(r"[^A-Za-z0-9_]+", "_", str(names[int(class_idx)]))
            for suffix, values in (
                ("precision", box.p),
                ("recall", box.r),
                ("ap50", box.ap50),
            ):
                value = float(values[pos])
                if math.isfinite(value):
                    result[f"{class_name}_{suffix}"] = value
    except (AttributeError, TypeError, KeyError, IndexError, ValueError):
        logger.warning("Per-class metrics not available", exc_info=True)
        return {}

    return result


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
    payload.update(_per_class_metrics(metrics))
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Attach test metrics to the MLflow run that trained this model
    run_id = load_run_id(Path(get_model_out_dir(config=config, mode="train")))
    if run_id:
        log_eval_metrics(run_id=run_id, eval_metrics=payload)
    else:
        logger.info("No MLflow run id found for this model; skipping MLflow logging")

    logger.info("Evaluation completed")
    return metrics
