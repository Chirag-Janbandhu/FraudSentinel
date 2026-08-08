"""
Centralized Logging Configuration Module for FraudSentinel.
Logs messages to both console (stdout) and persistent log files (`logs/running_logs.log`).
"""

import logging
import sys
from pathlib import Path


LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE_PATH = LOG_DIR / "running_logs.log"

LOG_FORMAT = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("FraudSentinel")


def get_logger(name: str = "FraudSentinel") -> logging.Logger:
    """
    Returns a named logger instance inheriting the global configuration.
    
    Args:
        name: Name of the logger (e.g., 'FraudSentinel.GraphConstruction').
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)
