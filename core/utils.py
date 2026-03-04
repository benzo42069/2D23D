from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path


logger = logging.getLogger("pic2mesh")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


@contextmanager
def timed_step(name: str):
    start = time.perf_counter()
    logger.info("Starting: %s", name)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("Finished: %s (%.2fs)", name, elapsed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_windows() -> bool:
    return os.name == "nt"
