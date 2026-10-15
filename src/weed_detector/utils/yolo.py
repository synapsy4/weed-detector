"""
YOLO utils
"""

from pathlib import Path

import torch
import yaml

from weed_detector.utils.labels import get_idx_to_class


def write_yolo_yaml(
        raw_dir: Path | str,
        processed_data_dir: Path | str,
        yolo_yaml_path: Path | str
        ) -> None:
    """
    Writes YOLO data config yaml.
    """
    # Get idx to class mapping
    idx_to_class = get_idx_to_class(raw_dir=raw_dir)
    
    # Define file content
    content = {
        "path": processed_data_dir,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": idx_to_class
    }

    # Write content to file
    Path(yolo_yaml_path).parent.mkdir(parents=True, exist_ok=True)
    with open(yolo_yaml_path, "w") as f:
        yaml.dump(content, f, sort_keys=False)

    print(f"[INFO] YOLO data config written to '{yolo_yaml_path}'")


def build_train_kwargs(
        config: dict,
        model_out_dir: Path,
        resume: bool = False
        ) -> dict:
    """
    Define keyword arguments for YOLO's train call.
    """
    kwargs = dict(
            data=config["paths"]["yolo_yaml_path"],
            epochs=config["train"]["epochs"],
            batch=config["train"]["batch_size"],
            imgsz=config["model"]["imgsz"],
            seed=config["seed"],
            save_dir=model_out_dir,
            device=get_device(),
            resume=resume
    )

    return kwargs


def get_device() -> int | str:
    """
    Return GPU index 0 if available, otherwise CPU.
    """
    device = 0 if torch.cuda.is_available() else "cpu"

    return device
