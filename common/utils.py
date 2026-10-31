"""Small helpers shared across the pipeline."""
from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import numpy as np


def setup_logger(name: str = "sentiment", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(h)
    return logger


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def dump_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Timer:
    """Tiny context timer printing wall time on exit."""

    def __init__(self, name: str, log=None):
        self.name = name
        self.log = log
        self.t0 = None

    def __enter__(self):
        self.t0 = time.time()
        if self.log:
            self.log.info(f"[{self.name}] start")
        return self

    def __exit__(self, *exc):
        dt = time.time() - self.t0
        if self.log:
            self.log.info(f"[{self.name}] done in {dt:.1f}s")
        else:
            print(f"[{self.name}] {dt:.1f}s")
        return False
