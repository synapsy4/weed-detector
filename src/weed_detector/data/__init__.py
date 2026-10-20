"""
Data module: downloading and preprocessing.
"""

from weed_detector.data.download import download_dataset
from weed_detector.data.preprocess import create_data_splits

__all__ = ["create_data_splits", "download_dataset"]
