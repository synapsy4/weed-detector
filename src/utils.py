"""
Utility + helper functions
"""

import yaml
from pathlib import Path

def load_config(
        path: Path | str = "configs/config.yaml"
        ) -> dict:
    """
    Load the hyperparameter config from given path.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

def get_idx_to_class(raw_dir: Path | str = "data/raw") -> dict:
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