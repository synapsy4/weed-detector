"""
Unit tests for logging utilities.
"""

import logging

from weed_detector.utils.logging import get_logger, setup_logger


def test_setup_logger_does_not_crash():
    """setup_logger should configure logging without raising."""
    setup_logger(level="INFO")


def test_get_logger_returns_logger():
    """get_logger should return a logging.Logger instance."""
    logger = get_logger("test.module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test.module"
