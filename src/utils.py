"""
Utility + helper functions
"""

import yaml
from pathlib import Path

import torch
import numpy as np


def load_config(
        path: Path | str = "configs/config.yaml"
        ) -> dict:
    """
    Load the hyperparameter config from given path.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

def get_idx_to_class(
        raw_dir: Path | str = "data/raw"
        ) -> dict:
    """
    Creates index to class mapping dict.
    """
    # Open classes.txt from raw data dir
    classes_path = Path(raw_dir) / "classes.txt"
    classes = open(classes_path, "r").read().split()

    return {i: c for i,c in enumerate(classes)}


def write_yolo_yaml(
        raw_dir: Path | str = "data/raw",
        processed_dir: Path | str = "data/processed",
        yolo_yaml_path: Path | str = "configs/yolo.yaml"
        ) -> None:
    """
    Writes YOLO data config yaml.
    """
    # Get idx to class mapping
    idx_to_class = get_idx_to_class(raw_dir=raw_dir)
    
    # Define file content
    content = {
        "path": processed_dir,
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


def get_weights_path(
        config: dict,
        cp: str = "last"
        ) -> Path:
    """
    Get the path to trained model weights for model specified in config.
    """
    if cp not in ["last", "best"]:
        raise ValueError(f"cp must be one of {'last', 'best'}, you chose '{cp}'.")

    model_out_dir = get_model_out_dir(config=config, mode="train")
    weights_path = model_out_dir / "weights" / f"{cp}.pt"

    return weights_path

def get_model_out_dir(
        config: dict,
        mode: str
        ) -> Path:
    """
    Get the output directory for the model specified in config.
    """
    if mode not in ["train", "eval"]:
        raise ValueError(f"mode must be one of {'train', 'eval'}, you chose '{mode}'")
    
    model_out_dir = Path(config["model"]["out_dir"]) / config["model"]["model_name"] 
    if mode == "train":
        model_out_dir = model_out_dir / config["model"]["train_subdir"]
    else:
        model_out_dir = model_out_dir / config["model"]["eval_subdir"]

    return model_out_dir
    

def get_lbl_arrays(lbl_path):
    """ 
    Get class (int) + bbox (float) arrays from txt-label file
    """
    lbl_data = Path(lbl_path).read_text().split()
    
    if len(lbl_data) % 5 != 0:
        raise ValueError(f"Inconsistent label format for '{lbl_path}'")
    
    lbl_arr = np.array(lbl_data, dtype=np.float64).reshape(-1, 5)
    classes = lbl_arr[:, :1].astype(int).flatten()
    bbxs = lbl_arr[:, 1:]
    
    return classes, bbxs


def get_device() -> int | str:
    """
    Return GPU index 0 if available, otherwise CPU.
    """
    device = 0 if torch.cuda.is_available() else "cpu"

    return device


def build_train_kwargs(
        config: dict,
        model_out_dir: Path,
        resume: bool = False
        ) -> dict:
    """
    Define keyword arguments for YOLO's train call.
    """
    kwargs = dict(
            data=config["data"]["yolo_yaml_path"],
            epochs=config["train"]["epochs"],
            batch=config["train"]["batch_size"],
            imgsz=config["model"]["imgsz"],
            seed=config["seed"],
            save_dir=model_out_dir,
            device=get_device(),
            resume=resume
    )

    return kwargs


def ask_resume(weights_path: Path | str) -> str:
    """
    Ask user whether to resume or abort training.
    """

    while True:
        choice = input(
                f"Weights found at '{weights_path}'.\n"
                "Resume training (y) or exit (n)? "
                ).lower()
        if choice in ("y", "n"):
            break
        print("Choose a valid option ('y' or 'n').")

    return choice


