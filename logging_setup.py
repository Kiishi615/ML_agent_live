import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(level=logging.INFO, log_dir="logs"):
    """Call once at the start of your program."""
    Path(log_dir).mkdir(exist_ok=True)

    script_name = Path(sys.argv[0]).stem
    log_file = Path(log_dir) / f"{script_name}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers = []

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(funcName)s | %(message)s"
    )

    file_handler = RotatingFileHandler(
        log_file, maxBytes=10_000_000, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(
        "%(levelname)s - %(message)s"
    ))

    root.addHandler(file_handler)
    root.addHandler(console_handler)

    return str(log_file)