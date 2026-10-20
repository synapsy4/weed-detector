"""
Data preprocessing functions
"""

import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from weed_detector.utils.logging import get_logger

logger = get_logger(__name__)


def create_data_splits(
    raw_data_dir: Path | str,
    processed_data_dir: Path | str,
    val_split_ratio: float = 0.15,
    test_split_ratio: float = 0.15,
    seed: int = 99,
) -> None:
    """
    Copys data from data/raw to data/processed into split directories as expected by YOLO.
    """

    logger.info("Creating datasplits...")

    # Data leakage prevention
    splits_exist = False

    # Define parent dirs
    raw_data_dir = Path(raw_data_dir).resolve()
    processed_data_dir = Path(processed_data_dir).resolve()
    images_dir = processed_data_dir / "images"
    labels_dir = processed_data_dir / "labels"

    # Make target folders
    splits = ["train", "val", "test"]
    for split in splits:
        image_split_dir = images_dir / split
        label_split_dir = labels_dir / split
        image_split_dir.mkdir(parents=True, exist_ok=True)
        label_split_dir.mkdir(parents=True, exist_ok=True)

        if len(list(image_split_dir.iterdir()) + list(label_split_dir.iterdir())) > 0:
            splits_exist = True

    if splits_exist:
        logger.info("Data splits already exist - no new splits created")
        return

    # Collect image paths
    all_image_paths = list(raw_data_dir.glob("*.jpeg"))

    # Create data splits
    valtest_split_ratio = val_split_ratio + test_split_ratio
    train_image_paths, valtest_image_paths = train_test_split(
        all_image_paths, test_size=valtest_split_ratio, random_state=seed
    )
    val_image_paths, test_image_paths = train_test_split(
        valtest_image_paths,
        test_size=test_split_ratio / valtest_split_ratio,
        random_state=seed,
    )

    # Move data according to split
    image_paths_by_split = [train_image_paths, val_image_paths, test_image_paths]

    for split, split_image_paths in zip(splits, image_paths_by_split):
        for image_path in split_image_paths:
            label_path = raw_data_dir / (image_path.stem + ".txt")
            # Move image
            target_image_path = images_dir / split / image_path.name
            shutil.copy(image_path, target_image_path)
            # Move label
            target_label_path = labels_dir / split / label_path.name
            shutil.copy(label_path, target_label_path)

    logger.info("Data splits created")
