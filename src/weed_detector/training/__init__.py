"""
Training module: YOLO training and evaluation.
"""

from weed_detector.training.evaluate import eval as evaluate
from weed_detector.training.train import train

__all__ = ["evaluate", "train"]
