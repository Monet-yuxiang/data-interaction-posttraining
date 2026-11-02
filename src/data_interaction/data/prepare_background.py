"""Prepare a GSM8K training background and a synthetic reference pool."""
from __future__ import annotations

import argparse
from pathlib import Path

from data_interaction.data.load_jsonl import write_jsonl
from data_interaction.data.schema import TrainingExample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--count", type=int, default=900)
    args = parser.parse_args()
    if not 1 <= args.count <= 2000:
        parser.error("--count must be between 1 and 2000")
    targets = [args.out_dir / "background.jsonl", args.out_dir / "reference.jsonl"]
    if any(path.exists() for path in targets):
        raise FileExistsError("background/reference already exists; choose another output directory")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install data dependencies: pip install -e '.[data]'") from exc
    # Official training split only. Do not use GSM8K test as training data.
    revision = "740312add88f781978c0658806c59bc2815b9866"
    gsm = load_dataset("openai/gsm8k", "main", revision=revision, split=f"train[:{args.count}]")
    background = [TrainingExample(id=f"gsm_train_{i:05d}", instruction=row["question"],
                  output=row["answer"], data_group="background",
                  metadata={"source": "openai/gsm8k", "revision": revision, "split": "train", "row_index": i})
                  for i, row in enumerate(gsm)]
    # Three disjoint reference slots; not length- or topic-matched to SFT sources.
    reference = [TrainingExample(id=f"arithmetic_ref_{i:05d}",
                 instruction=f"What is {i + 101} plus {2 * i + 307}?",
                 output=str(3 * i + 408), data_group="reference",
                 metadata={"source": "deterministic_synthetic_addition", "index": i})
                 for i in range(args.count * 3)]
    write_jsonl(background, targets[0])
    write_jsonl(reference, targets[1])
    print(f"Wrote {len(background)} GSM8K training rows and {len(reference)} synthetic references")


if __name__ == "__main__":
    main()
