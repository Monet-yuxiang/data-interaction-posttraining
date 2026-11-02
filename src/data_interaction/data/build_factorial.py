"""Construct a three-source fixed-example replacement factorial design."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from data_interaction.data.load_jsonl import load_jsonl, write_jsonl

SETTINGS = ("000", "100", "010", "001", "110", "101", "011", "111")


def build(reference_path: Path, paths: list[Path], out_dir: Path, n: int, seed: int) -> dict:
    if n <= 0 or len(paths) != 3:
        raise ValueError("n must be positive and three sources are required")
    if (out_dir / "manifest.json").exists():
        raise FileExistsError(f"{out_dir} already has a manifest; choose a new output directory")
    sources = [load_jsonl(reference_path)] + [load_jsonl(p) for p in paths]
    if len(sources[0]) < 3 * n or any(len(items) < n for items in sources[1:]):
        raise ValueError("need at least 3*n reference examples and n per source")
    all_ids = [item.id for source in sources for item in source]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("IDs must be unique across all sources")
    references = random.Random(seed).sample(sources[0], 3 * n)
    slots = [random.Random(seed + i + 1).sample(sources[i + 1], n) for i in range(3)]
    manifest = {"n_per_slot": n, "seed": seed, "budget": "3*n examples (NOT tokens)",
                "reference": str(reference_path), "sources": [str(p) for p in paths], "settings": {}}
    for setting in SETTINGS:
        examples = []
        for i, present in enumerate(setting):
            examples.extend(slots[i] if present == "1" else references[i*n:(i+1)*n])
        random.Random(seed + 9).shuffle(examples)
        path = out_dir / f"{setting}.jsonl"
        write_jsonl(examples, path)
        manifest["settings"][setting] = {"path": str(path), "count": len(examples)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--c", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build(args.reference, [args.a, args.b, args.c], args.out_dir, args.n, args.seed), indent=2))


if __name__ == "__main__":
    main()
