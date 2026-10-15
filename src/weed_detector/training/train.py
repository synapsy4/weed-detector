"""
YOLO training function
"""

from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics

from weed_detector.utils.paths import get_weights_path, get_model_out_dir
from weed_detector.utils.yolo import build_train_kwargs
from weed_detector.model.detector import load_model


def train(
        model: YOLO,
        config: dict
        ) -> DetMetrics | None:
    """
    Call preimplemented YOLO train function.
    """
    # Specify + create output dir
    model_out_dir = get_model_out_dir(
        config=config, 
        mode="train"
        )

    # Check for already trained weights
    weights_path = get_weights_path(
        config=config,
        cp="last")
    
    if weights_path.exists():
        # If existing model found, resume training or abort
        if config["train"]["resume_blocking"]:
            return None

        model = load_model(weights_path)
        kwargs = build_train_kwargs(
            config=config,
            model_out_dir=model_out_dir,
            resume=True
        )
    else:
        # For new model, create out dir
        model_out_dir.mkdir(parents=True, exist_ok=True)
        
        kwargs = build_train_kwargs(
            config=config,
            model_out_dir=model_out_dir,
        )

    # Run training
    metrics = model.train(**kwargs)     
    
    return metrics