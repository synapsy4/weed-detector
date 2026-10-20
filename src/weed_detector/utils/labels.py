"""
Label utility functions
"""

from pathlib import Path

import numpy as np


def get_idx_to_class(raw_dir: Path | str = "data/raw") -> dict:
    """
    Creates index to class mapping dict.
    """
    # Open classes.txt from raw data dir
    classes_path = Path(raw_dir) / "classes.txt"
    with open(classes_path) as f:
        classes = f.read().split()

    return {i: c for i, c in enumerate(classes)}


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
