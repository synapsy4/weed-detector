"""
Init YOLO model, call train & eval
"""

from ultralytics import YOLO


def load_model(weights: str) -> YOLO:
    """
    Load YOLO model with pretrained weights.
    """
    return YOLO(weights)

def train(
        model: YOLO,
        config: dict
        ) -> None:
    """
    Call preimplemented YOLO train function.
    """
    model.train(
        data=config["data"]["yolo_yaml_path"],
        epochs=config["train"]["epochs"],
        batch=config["train"]["batch_size"],
        imgsz=config["model"]["imgsz"],
        seed=config["seed"])
    

def eval(
        model: YOLO,
        config: dict
        ) -> None:
    """
    Call preimplemented YOLO val function.
    """
    metrics = model.val(
        data=config["data"]["yolo_yaml_path"],
        imgsz=config["train"]["imgsz"]
    )

    print(metrics)