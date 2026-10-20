"""
Unit tests for label utility functions.
"""

import numpy as np
import pytest

from weed_detector.utils.labels import get_idx_to_class, get_lbl_arrays


def test_get_idx_to_class(tmp_path):
    """get_idx_to_class should read classes.txt and return index->name mapping."""
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("crop\nweed\n")

    mapping = get_idx_to_class(raw_dir=str(tmp_path))

    assert mapping == {0: "crop", 1: "weed"}


def test_get_lbl_arrays_valid(tmp_path):
    """get_lbl_arrays should parse a valid YOLO label file."""
    label_content = "0 0.5 0.5 0.2 0.3\n1 0.3 0.7 0.1 0.2\n"
    label_file = tmp_path / "image.txt"
    label_file.write_text(label_content)

    classes, bbxs = get_lbl_arrays(str(label_file))

    assert classes.shape == (2,)
    assert classes[0] == 0
    assert classes[1] == 1
    assert bbxs.shape == (2, 4)
    np.testing.assert_array_almost_equal(bbxs[0], [0.5, 0.5, 0.2, 0.3])
    np.testing.assert_array_almost_equal(bbxs[1], [0.3, 0.7, 0.1, 0.2])


def test_get_lbl_arrays_invalid_format(tmp_path):
    """get_lbl_arrays should raise ValueError for label content not divisible by 5."""
    label_content = "0 0.5 0.5 0.2\n"  # only 4 values
    label_file = tmp_path / "image.txt"
    label_file.write_text(label_content)

    with pytest.raises(ValueError, match="Inconsistent label format"):
        get_lbl_arrays(str(label_file))
