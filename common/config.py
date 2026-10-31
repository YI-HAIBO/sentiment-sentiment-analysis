"""Project-wide configuration loader and paths.

Usage:
    from common.config import CFG, ROOT
    print(CFG['data']['source'])
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Project root — one level above this file's parent
ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG: dict[str, Any] = load_config()


def ensure_dir(p: str | os.PathLike[str]) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path
