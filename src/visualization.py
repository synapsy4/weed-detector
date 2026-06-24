"""
Functions to make visualizations
"""

import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from src.utils import get_lbl_arrays


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


def plot_yolo_pred(image_bgr: np.ndarray) -> None:
    """
    Show the annotated yolo prediction.
    """
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.show()