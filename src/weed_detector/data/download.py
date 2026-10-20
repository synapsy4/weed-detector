"""
Data downloader
"""

import shutil
from pathlib import Path

import kagglehub

from weed_detector.utils.logging import get_logger

logger = get_logger(__name__)


def download_dataset(raw_dir: str | Path) -> None:
    """
    Downloads data from kaggle + moves it into local data dir
    if data dir is empty
    """

    logger.info("Downloading dataset...")

    # Make sure data dir exists
    raw_dir = Path(raw_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    # If data dir empty: Download data, move it, flaten dir
    if not any(raw_dir.iterdir()):
        # Download
        cache_path = kagglehub.dataset_download(
            "ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes"
        )
        # Move data from local kagglehub cache to data dir
        shutil.move(cache_path, raw_dir)
        # Flatten version dir (from data_dir/1/... to data_dir/...)
        version_dir = raw_dir / "1"
        for item in version_dir.iterdir():
            shutil.move(str(item), str(raw_dir / item.name))
        version_dir.rmdir()
        logger.info("Dataset downloaded successfully")
    else:
        logger.info(f"No download invoked - dataset found at '{raw_dir}'")
