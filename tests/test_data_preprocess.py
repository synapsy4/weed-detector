"""
Unit tests for data preprocessing / splitting.
"""

from pathlib import Path

from PIL import Image

from weed_detector.data.preprocess import create_data_splits


def _create_fake_dataset(raw_dir: Path, num_images: int = 10):
    """Helper: create fake JPEG images and corresponding YOLO label files."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_images):
        img_path = raw_dir / f"image_{i:04d}.jpeg"
        Image.new("RGB", (512, 512), color="red").save(img_path, format="JPEG")
        label_path = raw_dir / f"image_{i:04d}.txt"
        label_path.write_text("0 0.5 0.5 0.2 0.3\n")


def test_create_data_splits_creates_train_val_test(tmp_path):
    """create_data_splits should produce train/val/test dirs with images and labels."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _create_fake_dataset(raw_dir, num_images=20)

    create_data_splits(
        raw_data_dir=str(raw_dir),
        processed_data_dir=str(processed_dir),
        val_split_ratio=0.2,
        test_split_ratio=0.2,
        seed=99,
    )

    for split in ["train", "val", "test"]:
        img_dir = processed_dir / "images" / split
        lbl_dir = processed_dir / "labels" / split
        assert img_dir.is_dir(), f"Missing {img_dir}"
        assert lbl_dir.is_dir(), f"Missing {lbl_dir}"
        assert len(list(img_dir.glob("*.jpeg"))) > 0
        assert len(list(lbl_dir.glob("*.txt"))) > 0


def test_create_data_splits_skips_when_splits_exist(tmp_path):
    """create_data_splits should return early if split dirs already contain files."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _create_fake_dataset(raw_dir, num_images=10)

    # Pre-create split dirs with dummy content to simulate existing splits
    for split in ["train", "val", "test"]:
        (processed_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (processed_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (processed_dir / "images" / split / "dummy.jpeg").write_bytes(b"dummy")
        (processed_dir / "labels" / split / "dummy.txt").write_text("dummy")

    # Should not raise or copy anything new
    create_data_splits(
        raw_data_dir=str(raw_dir),
        processed_data_dir=str(processed_dir),
        val_split_ratio=0.15,
        test_split_ratio=0.15,
        seed=99,
    )

    # Verify dummy files are still there (nothing overwritten)
    assert (processed_dir / "images" / "train" / "dummy.jpeg").exists()


def test_create_data_splits_copies_labels_correctly(tmp_path):
    """Each copied image should have a corresponding label file with matching stem."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _create_fake_dataset(raw_dir, num_images=5)

    create_data_splits(
        raw_data_dir=str(raw_dir),
        processed_data_dir=str(processed_dir),
        val_split_ratio=0.2,
        test_split_ratio=0.2,
        seed=99,
    )

    for img_path in (processed_dir / "images" / "train").glob("*.jpeg"):
        label_path = processed_dir / "labels" / "train" / f"{img_path.stem}.txt"
        assert label_path.exists(), f"Missing label for {img_path.name}"
