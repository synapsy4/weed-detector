"""
Data setup functions
"""

import shutil
from pathlib import Path

import kagglehub
from sklearn.model_selection import train_test_split


def download_dataset(raw_dir: str | Path = "data/raw") -> None:
    """
    Downloads data from kaggle + moves it into local data dir
    if data dir is empty
    """
    
    # Make sure data dir exists
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # If data dir empty: Download data, move it, flaten dir
    if not any(raw_dir.iterdir()):
        # Download
        cache_path = kagglehub.dataset_download("ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes")
        # Move data from local kagglehub cache to data dir
        shutil.move(cache_path, raw_dir)
        # Flatten version dir (from data_dir/1/... to data_dir/...)
        version_dir = raw_dir / "1"
        for item in version_dir.iterdir():
            shutil.move(str(item), str(raw_dir / item.name))
        version_dir.rmdir()
        print(f"[INFO] Dataset downloaded successfully")
    else:
        print(f"[INFO] Dataset found at '{raw_dir}'") 


def create_data_splits(
        raw_dir: Path | str = "data/raw",
        raw_subdir: Path | str = "agri_data/data",
        processed_dir: Path | str = "data/processed",
        val_split_ratio: float = 0.15,
        test_split_ratio: float = 0.15,
        seed: int = 99
        ) -> None:
    """
    Copys data from data/raw to data/processed into split directories as expected by YOLO.
    """

    # Data leakage prevention
    splits_exist = False

    # Define parent dirs
    raw_data_dir = Path(raw_dir) / raw_subdir
    images_dir = Path(processed_dir) / "images"
    labels_dir = Path(processed_dir) / "labels"

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
        print("[INFO] Data splits already exist.")
        return
    
    # Collect image paths
    all_image_paths = list(raw_data_dir.glob("*.jpeg"))

    # Create data splits
    valtest_split_ratio = val_split_ratio + test_split_ratio
    train_image_paths, valtest_image_paths = train_test_split(all_image_paths, 
                                                            test_size=valtest_split_ratio,
                                                            random_state=seed)
    val_image_paths, test_image_paths = train_test_split(valtest_image_paths, 
                                                        test_size=test_split_ratio/valtest_split_ratio,
                                                        random_state=seed)

    # Move data according to split
    image_paths_by_split = [train_image_paths, val_image_paths, test_image_paths]

    for split, split_image_paths  in zip(splits, image_paths_by_split):
        for image_path in split_image_paths:
            label_path = raw_data_dir / (image_path.stem + ".txt")
            # Move image
            target_image_path = images_dir / split / image_path.name
            shutil.copy(image_path, target_image_path)
            # Move label
            target_label_path = labels_dir / split / label_path.name
            shutil.copy(label_path, target_label_path)

    print("[INFO] Data splits created.")