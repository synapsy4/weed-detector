"""
Init YOLO model, training, evaluation and prediction functions
"""

from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics

from src.utils import get_weights_path, get_model_out_dir, get_device, build_train_kwargs, ask_resume
from src.visualization import plot_yolo_pred


def load_model(
        weights_path: str | Path
        ) -> YOLO:
    """
    Load YOLO model with given weights.
    """
    return YOLO(str(weights_path))

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
        choice = ask_resume(weights_path)
        if choice == "n":
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
    

def eval(
        model: YOLO,
        config: dict
        ) -> DetMetrics:
    """
    Call preimplemented YOLO val function.
    """
    # Specify output dir
    model_out_dir = get_model_out_dir(
        config=config,
        mode="eval"
        )
    
    device = get_device()

    # Run validation on test set
    metrics = model.val(
        data=config["data"]["yolo_yaml_path"],
        split="test",
        save_dir=model_out_dir,
        device=device
    )

    return metrics


def predict(
        model: YOLO,
        config: dict,
        image_path: str | Path,
        conf_threshold: float = 0.25,
        visualize_result: bool = True
        ) -> dict:
    """
    Make prediction on single image.
    """

    # Make prediction
    results = model.predict(
        source=str(image_path),
        imgsz=config["model"]["imgsz"],
        conf=conf_threshold
    )
    result = results[0] 

    # Unpack predictions
    bbxs = result.boxes.xyxy.numpy()
    classes = result.boxes.cls.numpy()
    class_names = [result.names[c] for c in classes]
    scores = result.boxes.conf.numpy()

    # Show prediction
    if visualize_result:
        plot_yolo_pred(image_bgr=result.plot())

    return {"bbxs": bbxs, 
            "classes": classes, 
            "class_names": class_names, 
            "scores": scores}