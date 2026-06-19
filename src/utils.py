"""
Utility + helper functions
"""

import yaml
import random
from pathlib import Path
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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


def visualize_sample(img_path, lbl_path, idx_to_class, ax=None):
    """
    Visualize a sample and its labeled bbxs.
    """

    if ax is None:
        _, ax = plt.subplots(1,1,figsize=(10,10))
    
    colors = {0: "green", 1: "red"}
    counts = {0: 0, 1: 0}
    
    img = Image.open(img_path)
    W,H = img.size

    classes, bbxs = get_lbl_arrays(lbl_path)

    for c,(x_c, y_c, w, h) in zip(classes, bbxs):

        w, h = w * W, h * H
        x_c, y_c = x_c * W, y_c * H
        x_min, y_min = x_c - w/2 , y_c - h/2

        counts[c] += 1

        ax.imshow(img)
        ax.add_patch(Rectangle((x_min,y_min), w, h, fc ='none', ec=colors[c], lw=2))

        ax.text(
                x_min, y_min, idx_to_class[c],
                color="white", fontsize=9, va="bottom", ha="left",
                bbox=dict(facecolor=colors[c], edgecolor="none", pad=1.5),
            )

    ax.axis("off")


def inspect_rnd_samples(k=4):
    """
    Visualize k random data-label pairs
    """
    
    # Get random image path
    data_dir = Path("data/raw/agri_data/data")
    img_paths = list(data_dir.glob("*.jpeg"))
    rnd_img_paths = random.sample(img_paths, k)

    # Get class mapping
    classes_path = Path("data/raw/classes.txt")
    classes = open(classes_path, "r").read().split()
    idx_to_class = {i: c for i,c in enumerate(classes)}
 
    _, axs = plt.subplots(1,k,figsize=(5*k,5))

    for img_path, ax in zip(rnd_img_paths, axs):
        # Get label path
        img_id = img_path.stem
        lbl_path = data_dir / (img_id +".txt")

        visualize_sample(img_path, lbl_path, idx_to_class, ax)