"""
Gated promotion of trained models in the MLflow model registry.

A candidate (the model trained and evaluated locally) is registered and gets the
'production' alias only if it passes two checks on the test split:
  1. the gate metric (default: weed recall) reaches an absolute floor, and
  2. it beats the current production model by a minimum improvement.
The previous production version keeps the 'fallback' alias.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from weed_detector.model import load_model
from weed_detector.training.evaluate import run_validation
from weed_detector.utils import (
    FALLBACK_ALIAS,
    PRODUCTION_ALIAS,
    download_model_weights,
    get_data_version,
    get_git_commit,
    get_logger,
    get_model_out_dir,
    get_model_version_by_alias,
    get_registry_client,
    load_run_id,
    register_model_version,
    set_model_alias,
)

logger = get_logger(__name__)

PROMOTED_FIRST = "promoted_first"
PROMOTED = "promoted"
REJECTED_BELOW_THRESHOLD = "rejected_below_threshold"
REJECTED_NOT_BETTER = "rejected_not_better"
ALREADY_PRODUCTION = "already_production"
ROLLED_BACK = "rolled_back"


class PromotionError(RuntimeError):
    """Raised when promotion or rollback cannot be carried out."""


@dataclass(frozen=True)
class PromotionResult:
    """Outcome of a promotion or rollback attempt."""

    decision: str
    reason: str
    version: str | None = None
    candidate_value: float | None = None
    production_value: float | None = None

    @property
    def promoted(self) -> bool:
        return self.decision in (PROMOTED_FIRST, PROMOTED)


def decide(
    candidate: float,
    production: float | None,
    min_threshold: float,
    min_improvement: float,
) -> tuple[str, str]:
    """Apply the promotion gate to the candidate's metric value."""
    if candidate < min_threshold:
        return (
            REJECTED_BELOW_THRESHOLD,
            f"{candidate:.4f} is below the minimum of {min_threshold:.4f}",
        )
    if production is None:
        return PROMOTED_FIRST, "no production model exists yet"
    if candidate >= production + min_improvement:
        reason = (
            f"{candidate:.4f} beats production {production:.4f} "
            f"by at least {min_improvement:.4f}"
        )
        return PROMOTED, reason
    reason = (
        f"{candidate:.4f} does not beat production {production:.4f} "
        f"by at least {min_improvement:.4f}"
    )
    return REJECTED_NOT_BETTER, reason


def _load_candidate_metrics(config: dict, metric: str) -> tuple[str, dict[str, float]]:
    """Return the MLflow run id and test metrics of the locally trained model."""
    train_dir = Path(get_model_out_dir(config=config, mode="train"))
    eval_dir = Path(get_model_out_dir(config=config, mode="eval"))

    run_id = load_run_id(train_dir)
    if run_id is None:
        raise PromotionError(
            f"No MLflow run id found in '{train_dir}'. Train with MLflow tracking enabled."
        )

    metrics_path = eval_dir / "metrics.json"
    if not metrics_path.exists():
        raise PromotionError(f"'{metrics_path}' not found. Run eval before promote.")

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metric not in payload:
        raise PromotionError(f"Gate metric '{metric}' missing in '{metrics_path}'")

    return run_id, payload


def _evaluate_production(
    client: MlflowClient, model_name: str, version: str, config: dict, metric: str
) -> float:
    """Re-evaluate the production model on the *current* test set."""
    with tempfile.TemporaryDirectory() as tmp:
        weights = download_model_weights(client, model_name, version, Path(tmp))
        model = load_model(weights)
        _, payload = run_validation(model, config, save_dir=Path(tmp) / "eval")

    if metric not in payload:
        raise PromotionError(
            f"Gate metric '{metric}' not available for production version {version}"
        )
    return float(payload[metric])


def _record_decision(
    client: MlflowClient, run_id: str, result: PromotionResult
) -> None:
    """Store the decision as tags on the candidate's MLflow run."""
    try:
        client.set_tag(run_id, "promotion_decision", result.decision)
        client.set_tag(run_id, "promotion_reason", result.reason)
    except MlflowException:
        logger.warning(
            "Could not record promotion decision on run %s", run_id, exc_info=True
        )


def _version_tags(
    config: dict, run_id: str, metric: str, payload: dict
) -> dict[str, str]:
    """Tags attached to the registered model version (also used by the API)."""
    tags = {
        "train_run_id": run_id,
        "local_model_name": str(config["model"]["model_name"]),
        # The API must predict with the image size the model was trained with
        "imgsz": str(config["model"]["imgsz"]),
        "gate_metric": metric,
    }
    tags.update({f"test_{key}": f"{value:.4f}" for key, value in payload.items()})
    for key, value in (
        ("data_version", get_data_version()),
        ("git_commit", get_git_commit()),
    ):
        if value:
            tags[key] = value
    return tags


def promote(config: dict, client: MlflowClient | None = None) -> PromotionResult:
    """Register the locally trained model and promote it if it passes the gate."""
    registry = config["registry"]
    model_name = registry["model_name"]
    metric = registry["metric"]

    run_id, payload = _load_candidate_metrics(config, metric)
    candidate_value = float(payload[metric])
    client = client or get_registry_client()

    production = get_model_version_by_alias(client, model_name, PRODUCTION_ALIAS)

    if production is not None and production.run_id == run_id:
        result = PromotionResult(
            ALREADY_PRODUCTION,
            f"version {production.version} of this run is already in production",
            version=str(production.version),
            candidate_value=candidate_value,
        )
        logger.info("Promotion skipped: %s", result.reason)
        return result

    # Cheap absolute check first so a hopeless candidate never triggers a download
    production_value: float | None = None
    if candidate_value >= registry["min_threshold"] and production is not None:
        production_value = _evaluate_production(
            client, model_name, production.version, config, metric
        )

    decision, reason = decide(
        candidate_value,
        production_value,
        registry["min_threshold"],
        registry["min_improvement"],
    )
    result = PromotionResult(
        decision,
        f"{metric}: {reason}",
        candidate_value=candidate_value,
        production_value=production_value,
    )

    if not result.promoted:
        logger.info("Model not promoted (%s): %s", decision, result.reason)
        _record_decision(client, run_id, result)
        return result

    new_version = register_model_version(
        run_id, model_name, _version_tags(config, run_id, metric, payload)
    )
    # Order matters: the old production model becomes the fallback first, so
    # there is never a moment without a valid production alias.
    if production is not None:
        set_model_alias(client, model_name, FALLBACK_ALIAS, str(production.version))
    set_model_alias(client, model_name, PRODUCTION_ALIAS, str(new_version.version))

    result = PromotionResult(
        decision,
        result.reason,
        version=str(new_version.version),
        candidate_value=candidate_value,
        production_value=production_value,
    )
    logger.info(
        "Model promoted to production as version %s: %s", result.version, result.reason
    )
    _record_decision(client, run_id, result)
    return result


def rollback(config: dict, client: MlflowClient | None = None) -> PromotionResult:
    """Swap the production and fallback aliases."""
    model_name = config["registry"]["model_name"]
    client = client or get_registry_client()

    production = get_model_version_by_alias(client, model_name, PRODUCTION_ALIAS)
    fallback = get_model_version_by_alias(client, model_name, FALLBACK_ALIAS)
    if fallback is None:
        raise PromotionError(f"Model '{model_name}' has no '{FALLBACK_ALIAS}' version")

    # Production first, so serving never loses its primary model
    set_model_alias(client, model_name, PRODUCTION_ALIAS, str(fallback.version))
    if production is not None:
        set_model_alias(client, model_name, FALLBACK_ALIAS, str(production.version))
    else:
        client.delete_registered_model_alias(model_name, FALLBACK_ALIAS)

    reason = f"production is now version {fallback.version}" + (
        f", fallback is version {production.version}" if production is not None else ""
    )
    logger.info("Rolled back: %s", reason)
    return PromotionResult(ROLLED_BACK, reason, version=str(fallback.version))
