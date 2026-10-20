"""
Unit tests for the data downloader.
"""

from pathlib import Path
from unittest.mock import patch

from weed_detector.data.download import download_dataset


def test_download_dataset_skips_when_data_exists(tmp_path):
    """download_dataset should not download if the raw dir already contains files."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "existing_file.txt").write_text("already here")

    with patch("weed_detector.data.download.kagglehub") as mock_kagglehub:
        download_dataset(raw_dir=str(raw_dir))
        mock_kagglehub.dataset_download.assert_not_called()


def test_download_dataset_downloads_when_empty(tmp_path):
    """download_dataset should call kagglehub when the raw dir is empty."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    mock_cache = tmp_path / "cache"
    mock_cache.mkdir()

    def fake_move(src, dst):
        """Create the version dir structure that download_dataset expects."""
        version_dir = Path(dst) / "1"
        version_dir.mkdir(parents=True)

    with patch("weed_detector.data.download.kagglehub") as mock_kagglehub:
        mock_kagglehub.dataset_download.return_value = str(mock_cache)

        with patch("weed_detector.data.download.shutil.move", side_effect=fake_move):
            download_dataset(raw_dir=str(raw_dir))

    mock_kagglehub.dataset_download.assert_called_once_with(
        "ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes"
    )
