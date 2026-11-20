"""MLflow helpers for experiment tracking and artifact logging."""

from __future__ import annotations

import importlib
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import yaml

from weed_detector.utils.env import (
    PROJECT_ROOT,
    get_dagshub_repo_url,
    get_mlflow_experiment_name,
    get_mlflow_tracking_uri,
    load_project_env,
)
from weed_detector.utils.logging import get_logger

logger = get_logger(__name__)

RUN_ID_FILENAME = "mlflow_run_id.txt"
DATA_STAGE = "preprocess"
DATA_OUT_PATH = "data/processed"


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


def get_data_version(lock_path: Path | None = None) -> str | None:
    """Return the DVC hash of the processed dataset from dvc.lock, if available."""
    lock_path = lock_path or PROJECT_ROOT / "dvc.lock"
    try:
        lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
        outs = lock["stages"][DATA_STAGE]["outs"]
    except (OSError, yaml.YAMLError, KeyError, TypeError):
        logger.debug("No DVC data version found in '%s'", lock_path)
        return None

    for out in outs:
        if out.get("path") == DATA_OUT_PATH:
            return out.get("md5")
    return None


def save_run_id(model_out_dir: Path, run_id: str | None) -> None:
    """Persist the MLflow run id next to the trained model (or remove a stale one)."""
    run_id_path = Path(model_out_dir) / RUN_ID_FILENAME
    if run_id is None:
        run_id_path.unlink(missing_ok=True)
        return
    run_id_path.parent.mkdir(parents=True, exist_ok=True)
    run_id_path.write_text(run_id, encoding="utf-8")


def load_run_id(model_out_dir: Path) -> str | None:
    """Return the MLflow run id stored next to the trained model, if any."""
    run_id_path = Path(model_out_dir) / RUN_ID_FILENAME
    if not run_id_path.exists():
        return None
    return run_id_path.read_text(encoding="utf-8").strip() or None


def log_training_run(
    config: dict,
    model_out_dir: Path,
    metrics: object | None,
    continued_from: str | None = None,
) -> str | None:
    """
    Log a training run to the configured MLflow tracking backend.

    Returns the MLflow run id, or None if logging failed. ``continued_from`` is the
    run id of the previous training run when a finished model is trained further.
    """
    try:
        load_project_env()
        init_dagshub_tracking()
        mlflow.set_tracking_uri(get_mlflow_tracking_uri())
        mlflow.set_experiment(get_mlflow_experiment_name())

        with mlflow.start_run(
            run_name=f"{config['model']['model_name']}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        ) as run:
            tags = {
                "project": "weed-detector",
                "framework": "ultralytics-yolov8",
                "model_name": config["model"]["model_name"],
                "dagshub_repo": get_dagshub_repo_url(),
            }
            data_version = get_data_version()
            if data_version:
                tags["data_version"] = data_version
            if continued_from:
                tags["continued_from"] = continued_from
            mlflow.set_tags(tags)

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
            return str(run.info.run_id)
    except Exception:
        logger.warning(
            "MLflow logging failed; continuing without tracking. "
            "Set MLFLOW_TRACKING_URI and DAGSHUB_TOKEN to enable DagsHub experiment tracking.",
            exc_info=True,
        )
    return None


def log_eval_metrics(run_id: str, eval_metrics: dict[str, float]) -> bool:
    """Log test-split evaluation metrics (prefixed ``test_``) to an existing run."""
    try:
        load_project_env()
        init_dagshub_tracking()
        mlflow.set_tracking_uri(get_mlflow_tracking_uri())

        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(
                {f"test_{name}": value for name, value in eval_metrics.items()}
            )
            data_version = get_data_version()
            if data_version:
                mlflow.set_tag("eval_data_version", data_version)

        logger.info(
            "Logged %d test metrics to MLflow run %s", len(eval_metrics), run_id
        )
    except Exception:
        logger.warning(
            "Logging evaluation metrics to MLflow failed; continuing.", exc_info=True
        )
        return False
    return True
