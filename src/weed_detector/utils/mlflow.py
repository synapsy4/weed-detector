"""MLflow helpers for experiment tracking and artifact logging."""

from __future__ import annotations

import importlib
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import mlflow

from weed_detector.utils.env import (
    get_dagshub_repo_url,
    get_mlflow_experiment_name,
    get_mlflow_tracking_uri,
    load_project_env,
)
from weed_detector.utils.logging import get_logger

logger = get_logger(__name__)


def _sanitize_mlflow_metrics(results_dict: dict) -> dict[str, float]:
    """Normalize metric names and values to the subset MLflow accepts."""
    sanitized: dict[str, float] = {}
    for key, value in results_dict.items():
        if value is None or isinstance(value, bool):
            continue
        try:
            metric_value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(metric_value):
            continue

        metric_name = str(key).strip()
        metric_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", metric_name)
        metric_name = re.sub(r"_+", "_", metric_name).strip("_.-")
        if not metric_name:
            continue

        sanitized[metric_name] = metric_value
    return sanitized


def init_dagshub_tracking() -> bool:
    """Initialize DagsHub-backed MLflow tracking when repo auth is available."""
    try:
        dagshub = importlib.import_module("dagshub")
    except ImportError:
        logger.debug("dagshub package not installed; skipping DagsHub MLflow init.")
        return False

    repo_url = get_dagshub_repo_url()
    parsed = urlparse(repo_url)
    path = parsed.path.strip("/")
    if "/" not in path:
        logger.warning("Invalid DagsHub repo URL '%s'; skipping repo init.", repo_url)
        return False

    repo_owner, repo_name = path.split("/", 1)
    try:
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    except Exception:
        logger.warning(
            "DagsHub repo init failed; continuing with direct MLflow tracking settings.",
            exc_info=True,
        )
        return False

    logger.info("Initialized DagsHub MLflow tracking for %s/%s", repo_owner, repo_name)
    return True


def log_training_run(config: dict, model_out_dir: Path, metrics: object | None) -> None:
    """Log a training run to the configured MLflow tracking backend."""
    try:
        load_project_env()
        init_dagshub_tracking()
        mlflow.set_tracking_uri(get_mlflow_tracking_uri())
        mlflow.set_experiment(get_mlflow_experiment_name())

        with mlflow.start_run(
            run_name=f"{config['model']['model_name']}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        ) as run:
            mlflow.set_tags(
                {
                    "project": "weed-detector",
                    "framework": "ultralytics-yolov8",
                    "model_name": config["model"]["model_name"],
                    "dagshub_repo": get_dagshub_repo_url(),
                }
            )

            params = {
                "seed": config.get("seed"),
                "epochs": config["train"].get("epochs"),
                "batch_size": config["train"].get("batch_size"),
                "imgsz": config["model"].get("imgsz"),
                "model_name": config["model"].get("model_name"),
                "pretrained_weights": config["model"].get("pretrained_weights"),
                "resume_blocking": config["train"].get("resume_blocking"),
            }
            mlflow.log_params(
                {key: value for key, value in params.items() if value is not None}
            )

            results_dict = getattr(metrics, "results_dict", None)
            if isinstance(results_dict, dict):
                mlflow.log_metrics(_sanitize_mlflow_metrics(results_dict))

            for checkpoint_name in ["last.pt", "best.pt"]:
                checkpoint_path = model_out_dir / "weights" / checkpoint_name
                if checkpoint_path.exists():
                    mlflow.log_artifact(str(checkpoint_path), artifact_path="weights")

            logger.info(
                "MLflow run %s logged to tracking URI '%s'",
                run.info.run_id,
                get_mlflow_tracking_uri(),
            )
    except Exception:
        logger.warning(
            "MLflow logging failed; continuing without tracking. "
            "Set MLFLOW_TRACKING_URI and DAGSHUB_TOKEN to enable DagsHub experiment tracking.",
            exc_info=True,
        )
