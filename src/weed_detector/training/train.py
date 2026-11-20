"""
YOLO training function
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ultralytics.utils.metrics import DetMetrics

from weed_detector.model import load_model
from weed_detector.utils import (
    build_train_kwargs,
    get_logger,
    get_model_out_dir,
    get_pretrained_weights_path,
    get_weights_path,
    log_training_run,
)

logger = get_logger(__name__)


def _archive_results_csv(model_out_dir: Path) -> Path | None:
    """
    Keep the per-epoch history of the previous run before a warm start.

    A warm start restarts epoch numbering and Ultralytics would overwrite
    'results.csv' in the shared output dir.
    """
    results_csv = model_out_dir / "results.csv"
    if not results_csv.exists():
        return None

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    archived = model_out_dir / f"results_{timestamp}.csv"
    results_csv.rename(archived)
    logger.info(f"Archived previous results to '{archived}'")

    return archived


def train(config: dict) -> DetMetrics | None:
    """
    Call preimplemented YOLO train function.
    """
    # Specify + create output dir
    model_out_dir = get_model_out_dir(config=config, mode="train")

    # Check for already trained weights
    weights_path = get_weights_path(config=config, cp="last")

    if weights_path.exists():
        # If existing model found, resume training or abort
        if config["train"]["resume_blocking"]:
            logger.info(
                "Training aborted: existing weights found at "
                f"'{weights_path}' and resume_blocking is True"
            )
            return None

        # Ultralytics strips optimizer and epoch state from finished runs, so only
        # the weights can be reused (fresh optimizer, epoch counter restarts at 0)
        logger.info(
            f"Continuing training with warm start from weights '{weights_path}' "
            "(fresh optimizer)"
        )
        _archive_results_csv(model_out_dir)

        kwargs = build_train_kwargs(
            config=config, model_out_dir=model_out_dir, resume=False
        )
    else:
        # For new model, create out dir
        model_out_dir.mkdir(parents=True, exist_ok=True)

        # Load pretrained model
        weights_path = get_pretrained_weights_path(config)

        logger.info(f"Starting new training run -> '{model_out_dir}'")

        kwargs = build_train_kwargs(
            config=config, model_out_dir=model_out_dir, resume=False
        )

    # Load model
    model = load_model(weights_path)

    # Run training
    logger.info("Training started")
    metrics = model.train(**kwargs)
    logger.info("Training completed")

    log_training_run(config=config, model_out_dir=model_out_dir, metrics=metrics)

    return metrics
