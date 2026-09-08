"""Helpers for extracting task scores from lm-eval JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_interaction.eval.run_eval import _primary_score


def parse_lm_eval_file(path: str | Path) -> dict[str, float]:
    raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        task: score
        for task, values in (raw.get("results") or {}).items()
        if (score := _primary_score(values)) is not None
    }

