"""JSONL readers and writers with useful validation errors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from data_interaction.data.schema import TrainingExample


def load_jsonl(path: str | Path) -> list[TrainingExample]:
    path = Path(path)
    examples: list[TrainingExample] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = TrainingExample.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if item.id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate id {item.id!r}")
            seen_ids.add(item.id)
            examples.append(item)
    if not examples:
        raise ValueError(f"{path} contains no examples")
    return examples


def write_jsonl(examples: Iterable[TrainingExample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")

