"""
Init YOLO model, call train & eval
"""

from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics


def load_model(
        weights: str,
        weights_dir: str) -> YOLO:
    """
    Load YOLO model with pretrained weights.
    """
    return YOLO(weights_dir + "/" + weights)

def train(
        model: YOLO,
        config: dict
        ) -> None:
    """
    Call preimplemented YOLO train function.
    """

    model_out_dir = Path(config["model"]["out_dir"]) / config["model"]["model_name"] / config["model"]["train_subdir"]
            
    model_last_cp_path = model_out_dir / "weights" / "last.pt"
    
    if model_last_cp_path.exists():
        resume_train(
            weights=str(model_last_cp_path.name),
            weights_dir=str(model_last_cp_path.parent)
            )
    else:
        model_out_dir.mkdir(parents=True, exist_ok=True)
        model.train(
            data=config["data"]["yolo_yaml_path"],
            epochs=config["train"]["epochs"],
            batch=config["train"]["batch_size"],
            imgsz=config["model"]["imgsz"],
            seed=config["seed"],
            save_dir=model_out_dir
            ) 
    

def eval(
        model: YOLO,
        config: dict
        ) -> DetMetrics:
    """
    Call preimplemented YOLO val function.
    """
    model_out_dir = Path(config["model"]["out_dir"]) / config["model"]["model_name"] / config["model"]["eval_subdir"]

    metrics = model.val(
        data=config["data"]["yolo_yaml_path"],
        save_dir=model_out_dir
    )

    return metrics

def resume_train(
        weights: str,
        weights_dir: str) -> None:
    """
    Ask user to confirm resume training with exit and train option.
    """
    user_out = input(f"Found a model under '{weights_dir}'.\n \
                        Continue training (y) or exit (n)?")
    while user_out not in ["y", "n"]:
        print("Choose a valid option ('y' or 'n').")
    if user_out == "n":
        return
    else:
        model = load_model(weights=weights,
                           weights_dir=weights_dir)
        model.train(resume=True)