
"""
CineMatch - Centralized Logging System

Provides consistent logging across all project modules.
"""

import logging
from pathlib import Path
from datetime import datetime

from config.settings import PROJECT_ROOT


# ==========================================
# LOGGING CONFIGURATION
# ==========================================

LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | "
    "%(name)s | %(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

DEFAULT_LOG_LEVEL = logging.INFO


# ==========================================
# LOGGER FACTORY
# ==========================================

def get_logger(
    name,
    log_level=DEFAULT_LOG_LEVEL,
    console=True,
    file_logging=True,
):
    """
    Create or retrieve a configured logger.

    Args:
        name: Logger name, usually __name__.
        log_level: Minimum logging level.
        console: Enable console output.
        file_logging: Enable file output.

    Returns:
        logging.Logger: Configured logger.
    """

    logger = logging.getLogger(name)

    logger.setLevel(log_level)

    # Prevent duplicate output through root handlers.
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )

    # Console handler
    if console:

        console_handler = logging.StreamHandler()

        console_handler.setLevel(log_level)

        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    # File handler
    if file_logging:

        log_file = (
            LOG_DIR
            / f"cinematch_{datetime.now():%Y%m%d}.log"
        )

        file_handler = logging.FileHandler(
            log_file,
            encoding="utf-8",
        )

        file_handler.setLevel(log_level)

        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger


# ==========================================
# LOGGING UTILITIES
# ==========================================

def log_exception(logger, message):
    """
    Log an exception with its traceback.

    Call this method inside an exception handler.
    """

    logger.exception(message)


def set_log_level(logger, level):
    """Change the logging level of a logger."""

    if isinstance(level, str):

        level = getattr(
            logging,
            level.upper(),
            None,
        )

    if not isinstance(level, int):
        raise ValueError(
            "Invalid logging level."
        )

    logger.setLevel(level)

    for handler in logger.handlers:
        handler.setLevel(level)


def close_logger(logger):
    """Close and remove all handlers from a logger."""

    for handler in logger.handlers[:]:

        handler.close()

        logger.removeHandler(handler)


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    logger = get_logger(
        "CineMatch",
        log_level=logging.DEBUG,
    )

    logger.debug("Debug logging test.")

    logger.info(
        "CineMatch logging system initialized."
    )

    logger.warning(
        "This is a sample warning."
    )

    logger.error(
        "This is a sample error."
    )

    try:

        result = 10 / 0

    except ZeroDivisionError:

        log_exception(
            logger,
            "Sample exception captured.",
        )

    print("\nLogging system test completed.")

    print(f"Log directory: {LOG_DIR}")

    close_logger(logger)
