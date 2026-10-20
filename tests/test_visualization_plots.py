"""
Unit tests for the visualization module.
"""

from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from weed_detector.visualization.plots import plot_yolo_pred, visualize_sample


def test_visualize_sample_runs_without_error(tmp_path):
    """visualize_sample should run without error given valid image and label files."""
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend

    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (100, 100), color="red").save(img_path)

    lbl_path = tmp_path / "test.txt"
    lbl_path.write_text("0 0.5 0.5 0.2 0.3\n")

    idx_to_class = {0: "crop", 1: "weed"}

    with patch("matplotlib.pyplot.show"):
        ax = MagicMock()
        visualize_sample(
            img_path=str(img_path),
            lbl_path=str(lbl_path),
            idx_to_class=idx_to_class,
            ax=ax,
        )

    # Verify that ax methods were called (rectangles, text, axis off)
    ax.add_patch.assert_called()
    ax.text.assert_called()
    ax.axis.assert_called_with("off")


def test_plot_yolo_pred_runs_without_error():
    """plot_yolo_pred should run without error given a valid numpy array."""
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend

    image_bgr = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch("matplotlib.pyplot.show"):
        plot_yolo_pred(image_bgr=image_bgr)
