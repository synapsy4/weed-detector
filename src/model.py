"""
Init YOLO model, training, evaluation and prediction functions
"""

from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics

from src.utils import get_weights_path, get_model_out_dir


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

    # Check for already trained weights
    weights_path = get_weights_path(
        config=config,
        cp="last")
    
    if weights_path.exists():
        # Resume training if trained weights found
        metrics = resume_train(
            config=config,
            weights_path=weights_path
            )
    else:
        # Specify + create output dir
        model_out_dir = get_model_out_dir(
            config=config, 
            mode="train"
            )
        model_out_dir.mkdir(parents=True, exist_ok=True)

        device = 0 if torch.cuda.is_available() else "cpu"

        # Run training
        metrics = model.train(
            data=config["data"]["yolo_yaml_path"],
            epochs=config["train"]["epochs"],
            batch=config["train"]["batch_size"],
            imgsz=config["model"]["imgsz"],
            seed=config["seed"],
            save_dir=model_out_dir,
            device=device
            ) 
        
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
    
    device = 0 if torch.cuda.is_available() else "cpu"

    # Run validation on test set
    metrics = model.val(
        data=config["data"]["yolo_yaml_path"],
        split="test",
        save_dir=model_out_dir,
        device=device
    )

    return metrics

def resume_train(
        config: dict,
        weights_path: str | Path,
        ) -> DetMetrics | None:
    """
    Ask user to confirm resume training with exit and train option.
    """
    user_out = input(f"Found a model under '{str(weights_path)}'.\n \
                        Continue training (y) or exit (n)?")
    while user_out not in ["y", "n"]:
        print("Choose a valid option ('y' or 'n').")
    if user_out == "n":
        return
    else:
        # Init trained model
        model = load_model(weights_path=weights_path)

        # Specify output dir
        model_out_dir = get_model_out_dir(
            config=config, 
            mode="train"
            )
        
        device = 0 if torch.cuda.is_available() else "cpu"

        # Run training
        metrics = model.train(
            resume=True,
            data=config["data"]["yolo_yaml_path"],
            epochs=config["train"]["epochs"],
            batch=config["train"]["batch_size"],
            imgsz=config["model"]["imgsz"],
            seed=config["seed"],
            save_dir=model_out_dir,
            device=device
            )
        return metrics
    
def predict(
        model: YOLO,
        config: dict,
        source: str | Path
        ) -> None:
    """
    Make prediction on single image.
    TODO: Complete + test function logic (setting ouput dir, predict args etc.)
    """

    # Make prediction
    results = model.predict(
        source=str(source), # jpg path
        imgsz=config["model"]["imgsz"],
    )
    result = results[0]

    '''
    # Absolute pixel coords [x1, y1, x2, y2]
    boxes = result.boxes.xyxy  
    # Confidence scores + predictied class indices    
    confs = result.boxes.conf      
    classes = result.boxes.cls     
    '''

    # Display annotaed image
    result.show()  

    # Save annotated image to a specific path                
    result.save("prediction.jpg")      