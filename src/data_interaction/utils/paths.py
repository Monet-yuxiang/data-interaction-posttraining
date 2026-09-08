"""Path helpers."""

from __future__ import annotations

from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    value = Path(path)
    value.mkdir(parents=True, exist_ok=True)
    return value

