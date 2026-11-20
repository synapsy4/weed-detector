"""MLflow helpers for experiment tracking and artifact logging."""

from __future__ import annotations

import importlib
import math
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import yaml
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from weed_detector.utils.env import (
    PROJECT_ROOT,
    get_dagshub_repo_url,
    get_env,
    get_mlflow_experiment_name,
    get_mlflow_tracking_uri,
    load_project_env,
)
from weed_detector.utils.logging import get_logger

logger = get_logger(__name__)

RUN_ID_FILENAME = "mlflow_run_id.txt"
PRODUCTION_ALIAS = "production"
FALLBACK_ALIAS = "fallback"
WEIGHTS_ARTIFACT = "weights/best.pt"
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


def init_dagshub_tracking(interactive: bool = True) -> bool:
    """
    Initialize DagsHub-backed MLflow tracking when repo auth is available.

    With ``interactive=False`` (servers, containers) DagsHub is only initialized if
    a token is set in ``DAGSHUB_USER_TOKEN``; otherwise dagshub.init would wait
    for a browser-based OAuth login that can never complete.
    """
    if not interactive and not os.getenv("DAGSHUB_USER_TOKEN"):
        logger.info(
            "No DAGSHUB_USER_TOKEN set; skipping dagshub.init and using the MLflow "
            "tracking settings from the environment."
        )
        return False

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


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


def get_git_commit() -> str | None:
    """Return the current git commit hash, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def get_registry_client(interactive: bool = True) -> MlflowClient:
    """Set up DagsHub/MLflow tracking and return a client for the model registry."""
    load_project_env()
    dagshub_token = get_env("DAGSHUB_TOKEN")
    if dagshub_token:
        # dagshub reads its token from DAGSHUB_USER_TOKEN
        os.environ.setdefault("DAGSHUB_USER_TOKEN", dagshub_token)
    init_dagshub_tracking(interactive=interactive)
    tracking_uri = get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)


def get_model_version_by_alias(
    client: MlflowClient, model_name: str, alias: str
) -> ModelVersion | None:
    """Return the model version an alias points to, or None if it is not set."""
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except MlflowException as e:
        if getattr(e, "error_code", "") == "RESOURCE_DOES_NOT_EXIST":
            return None
        raise


def register_model_version(
    run_id: str, model_name: str, tags: dict[str, str]
) -> ModelVersion:
    """Register the best.pt artifact of a training run as a new model version."""
    return mlflow.register_model(
        model_uri=f"runs:/{run_id}/{WEIGHTS_ARTIFACT}",
        name=model_name,
        tags=tags,
    )


def set_model_alias(
    client: MlflowClient, model_name: str, alias: str, version: str
) -> None:
    """Point an alias at a model version (moves it if it was set elsewhere)."""
    client.set_registered_model_alias(model_name, alias, str(version))
    logger.info(
        "Alias '%s' of '%s' now points to version %s", alias, model_name, version
    )


def download_model_weights(
    client: MlflowClient, model_name: str, version: str, dst_dir: Path
) -> Path:
    """Download the weights file of a registered model version into dst_dir."""
    uri = client.get_model_version_download_uri(model_name, str(version))
    local = Path(
        mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(dst_dir))
    )
    if local.is_file():
        return local

    candidates = sorted(local.rglob("*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No .pt weights found in model version artifact '{uri}'"
        )
    return next((c for c in candidates if c.name == "best.pt"), candidates[0])
