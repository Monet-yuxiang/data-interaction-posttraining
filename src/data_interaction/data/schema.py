"""Schema for one post-training example."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TrainingExample:
    id: str
    instruction: str
    output: str
    data_group: str
    input: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingExample":
        required = ("id", "instruction", "output", "data_group")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")
        metadata = value.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        return cls(
            id=str(value["id"]),
            instruction=str(value["instruction"]),
            input=str(value.get("input") or ""),
            output=str(value["output"]),
            data_group=str(value["data_group"]),
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

