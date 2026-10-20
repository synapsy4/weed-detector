"""
YOLO training function
"""

from ultralytics.utils.metrics import DetMetrics

from weed_detector.model import load_model
from weed_detector.utils import (
    build_train_kwargs,
    get_logger,
    get_model_out_dir,
    get_pretrained_weights_path,
    get_weights_path,
)

logger = get_logger(__name__)


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

        logger.info(f"Resuming training from existing weights: '{weights_path}'")
        kwargs = build_train_kwargs(
            config=config, model_out_dir=model_out_dir, resume=True
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

    return metrics
