"""Build all core mixtures for a list of scales."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from data_interaction.data.load_jsonl import load_jsonl, write_jsonl
from data_interaction.data.mix_units import MODES, build_mixture


def build_scale_datasets(
    a_path: str | Path,
    b_path: str | Path,
    out_dir: str | Path,
    scales: Iterable[int],
    seed: int = 42,
) -> dict[str, object]:
    a_items = load_jsonl(a_path)
    b_items = load_jsonl(b_path)
    overlapping_ids = {item.id for item in a_items} & {item.id for item in b_items}
    if overlapping_ids:
        example = sorted(overlapping_ids)[0]
        raise ValueError(f"A and B must have globally unique ids; both contain {example!r}")
    out_dir = Path(out_dir)
    scales = [int(scale) for scale in scales]
    if not scales:
        raise ValueError("at least one scale is required")

    manifest: dict[str, object] = {
        "a_path": str(Path(a_path)),
        "b_path": str(Path(b_path)),
        "seed": seed,
        "sampling": "example_count",
        "scales": {},
    }
    for scale in scales:
        scale_dir = out_dir / f"scale_{scale}"
        scale_files: dict[str, dict[str, object]] = {}
        for mode in MODES:
            examples = build_mixture(a_items, b_items, scale, mode, seed)
            destination = scale_dir / f"{mode}.jsonl"
            write_jsonl(examples, destination)
            scale_files[mode] = {"path": str(destination), "num_examples": len(examples)}
        manifest["scales"][str(scale)] = scale_files  # type: ignore[index]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
