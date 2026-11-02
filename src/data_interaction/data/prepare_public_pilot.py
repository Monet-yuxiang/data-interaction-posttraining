"""Download and convert the published s1K and r1k data for the pilot.

This is not a verified paired or contamination-free benchmark dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from data_interaction.data.load_jsonl import write_jsonl
from data_interaction.data.schema import TrainingExample

S1K_REVISION = "96c411f1fe4c49d20f0e2a1565f61e1a28b0b84d"
R1K_COMMIT = "702bab5910d44506f4f60d738b58512bab8650df"
R1K_SHA256 = "7ec23ed775b4a1752e20a8d058682f549fb04e529e79a46de1504ed4a7508f54"
R1K_URL = f"https://raw.githubusercontent.com/Demi-deng2/ReasonAlign-analysis/{R1K_COMMIT}/r1k.parquet"


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _output(reasoning: object, answer: object) -> str:
    reasoning, answer = _text(reasoning), _text(answer)
    if not answer:
        raise ValueError("source row is missing its answer")
    return f"{reasoning}\n\n{answer}" if reasoning else answer


def convert_forward(row: dict, index: int) -> TrainingExample:
    question = _text(row.get("question"))
    if not question:
        raise ValueError(f"s1K row {index} has no question")
    return TrainingExample(
        id=f"s1k_{index:04d}", instruction=question,
        output=_output(row.get("deepseek_thinking_trajectory"), row.get("deepseek_attempt")),
        data_group="forward_reasoning",
        metadata={"source": "simplescaling/s1K-1.1", "revision": S1K_REVISION,
                  "row_index": index, "source_type": _text(row.get("source_type")),
                  "deepseek_grade": _text(row.get("deepseek_grade"))},
    )


def convert_reverse(row: dict, index: int) -> TrainingExample:
    question = _text(row.get("question"))
    if not question:
        raise ValueError(f"r1k row {index} has no question")
    original_index = row.get("original_index")
    return TrainingExample(
        id=f"r1k_{index:04d}", instruction=question,
        output=_output(row.get("thinking_trajectories"), row.get("solution")),
        data_group="reverse_reasoning",
        metadata={"source": "ReasonAlign-analysis/r1k.parquet", "revision": R1K_COMMIT,
                  "row_index": index, "original_index_unverified": original_index},
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(out_dir: Path, r1k_file: Path | None = None, force: bool = False) -> dict:
    try:
        from datasets import load_dataset
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise SystemExit("Install data dependencies: pip install -e '.[data]'") from exc

    destinations = [out_dir / "forward.jsonl", out_dir / "reverse.jsonl", out_dir / "source_manifest.json"]
    if not force and any(path.exists() for path in destinations):
        raise FileExistsError("Converted files already exist; choose another --out-dir or pass --force")
    out_dir.mkdir(parents=True, exist_ok=True)
    if r1k_file is None:
        r1k_file = out_dir / "r1k.parquet"
        if not r1k_file.exists():
            temporary = r1k_file.with_suffix(".download")
            try:
                urllib.request.urlretrieve(R1K_URL, temporary)
                if sha256(temporary) != R1K_SHA256:
                    raise ValueError("r1k download checksum mismatch")
                temporary.replace(r1k_file)
            finally:
                temporary.unlink(missing_ok=True)
    if sha256(r1k_file) != R1K_SHA256:
        raise ValueError(f"r1k checksum mismatch: {r1k_file}")
    forward_raw = load_dataset("simplescaling/s1K-1.1", revision=S1K_REVISION, split="train")
    reverse_raw = parquet.read_table(r1k_file).to_pylist()
    forward = [convert_forward(row, i) for i, row in enumerate(forward_raw)]
    reverse = [convert_reverse(row, i) for i, row in enumerate(reverse_raw)]
    write_jsonl(forward, destinations[0])
    write_jsonl(reverse, destinations[1])
    manifest = {
        "forward": {"dataset": "simplescaling/s1K-1.1", "revision": S1K_REVISION,
                    "count": len(forward), "jsonl_sha256": sha256(destinations[0])},
        "reverse": {"url": R1K_URL, "sha256": R1K_SHA256, "count": len(reverse),
                    "jsonl_sha256": sha256(destinations[1])},
        "warnings": ["Exploratory only: original_index pairing not verified against s1K-1.1.",
                     "Source benchmark contamination and generated answer correctness not audited."],
    }
    destinations[2].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--r1k-file", type=Path, help="Previously downloaded, checksum-verified parquet")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.out_dir, args.r1k_file, args.force), indent=2))


if __name__ == "__main__":
    main()
