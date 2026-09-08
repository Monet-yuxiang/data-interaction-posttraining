"""Small YAML configuration helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def deep_merge(*configs: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for config in configs:
        for key, value in config.items():
            if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
                result[key] = deep_merge(result[key], value)  # type: ignore[arg-type]
            else:
                result[key] = deepcopy(value)
    return result

