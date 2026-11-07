"""Prepare fixed-size packed SFT blocks for the shared-background factorial.

Only complete user/assistant pairs are selected from the published Tulu mixture.
The same source blocks are reused across settings. Two disjoint reference slots
require slot-specific single replacements, hence 15 rather than 11 settings.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

DATASET = "allenai/tulu-3-sft-mixture"
REVISION = "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be"
SOURCES = {
    "a": "ai2-adapt-dev/flan_v2_converted",
    "b": "ai2-adapt-dev/personahub_math_v5_regen_149960",
    "c": "ai2-adapt-dev/evol_codealpaca_heval_decontaminated",
    "d": "ai2-adapt-dev/tulu_v3.9_wildchat_100k",
    "r": "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980",
}
BACKGROUND = (
    "ai2-adapt-dev/tulu_v3.9_sciriff_10k",
    "ai2-adapt-dev/tulu_v3.9_aya_100k",
)
SETTINGS = {"baseline": ("r1", "r2")}
SETTINGS.update({f"{x}1": (x, "r2") for x in "abcd"})
SETTINGS.update({f"{x}2": ("r1", x) for x in "abcd"})
SETTINGS.update({a+b: (a,b) for a,b in itertools.combinations("abcd", 2)})


def encode_pair(row, tokenizer):
    """Encode a two-turn record; mask prompt tokens, not assistant output."""
    msgs = row.get("messages") or []
    if len(msgs) != 2 or [m.get("role") for m in msgs] != ["user", "assistant"]:
        return None
    if not all(isinstance(m.get("content"), str) and m["content"].strip() for m in msgs):
        return None
    # Tokenizing the prompt separately is unsafe: BPE can merge its final
    # newline with the first character(s) of an assistant response.
    prefix = tokenizer.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    if not full.startswith(prefix) or len(full) <= len(prefix):
        raise ValueError(f"chat template text is not a prefix for source={row.get('source')!r}, id={row.get('id')!r}")
    try:
        encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    except (NotImplementedError, ValueError) as exc:
        raise ValueError("A fast tokenizer with offset_mapping support is required") from exc
    ids = list(encoded["input_ids"])
    offsets = encoded["offset_mapping"]
    # A boundary-crossing token contains part of the prompt, so mask it in its
    # entirety; response tokens and the assistant end marker remain targets.
    labels = [token if start >= len(prefix) and end > start else -100
              for token, (start, end) in zip(ids, offsets)]
    if not any(label != -100 for label in labels):
        raise ValueError(f"No assistant tokens after masking source={row.get('source')!r}, id={row.get('id')!r}")
    return ids, labels


def collect_blocks(rows, tokenizer, blocks, seq_length):
    """Fill pools in deterministic stream order, preserving real source records."""
    slot = blocks // 10
    background_total = 8*slot
    bg_counts = [background_total // len(BACKGROUND) + (i < background_total % len(BACKGROUND))
                 for i in range(len(BACKGROUND))]
    required = {**{f"bg{i}": n for i, n in enumerate(bg_counts)},
                "r": 2*slot, **{x: slot for x in "abcd"}}
    inverse = {source: key for key, source in SOURCES.items()}
    inverse.update({source: f"bg{i}" for i, source in enumerate(BACKGROUND)})
    pools = {key: [] for key in required}
    buffers = {key: ([], []) for key in required}
    counts = {key: 0 for key in required}
    discarded = {key: 0 for key in required}
    observed = set()
    for row in rows:
        source = row.get("source")
        key = inverse.get(source)
        if key is None:
            continue
        observed.add(source)
        if len(pools[key]) >= required[key]:
            continue
        pair = encode_pair(row, tokenizer)
        if pair is None:
            continue
        counts[key] += 1
        ids, labels = buffers[key]
        ids.extend(pair[0]); labels.extend(pair[1])
        while len(ids) >= seq_length and len(pools[key]) < required[key]:
            candidate = {"input_ids": ids[:seq_length], "attention_mask": [1]*seq_length,
                         "labels": labels[:seq_length]}
            del ids[:seq_length]; del labels[:seq_length]
            if not any(label != -100 for label in candidate["labels"][1:]):
                discarded[key] += 1
                continue
            pools[key].append(candidate)
            if key == "r" and len(pools[key]) == slot:
                # Start the second slot at the next source record, so R1 and R2
                # cannot share a record at the packing boundary.
                ids.clear(); labels.clear()
                break
        if all(len(pools[k]) >= required[k] for k in required):
            break
    missing = {k: required[k]-len(pools[k]) for k in required if len(pools[k]) < required[k]}
    if missing:
        raise ValueError(f"Insufficient eligible data for requested blocks: {missing}; observed sources: {sorted(observed)}")
    reference = pools.pop("r")
    pools["r1"], pools["r2"] = reference[:slot], reference[slot:]
    pools["background"] = []
    for i in range(len(BACKGROUND)):
        pools["background"].extend(pools.pop(f"bg{i}"))
    return pools, counts, discarded, sorted(observed)


def prepare(rows, tokenizer, out_dir, blocks, seq_length, seed, model):
    if blocks < 80 or blocks % 80:
        raise ValueError("--blocks must be a positive multiple of 80 (10% slots and gradient accumulation 8)")
    if seq_length < 2:
        raise ValueError("--seq-length must be at least 2")
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_dir}")
    pools, counts, discarded, observed = collect_blocks(rows, tokenizer, blocks, seq_length)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for setting, (left, right) in SETTINGS.items():
        data = list(pools["background"]) + list(pools[left]) + list(pools[right])
        if len(data) != blocks:
            raise ValueError(f"incorrect block count for {setting}: {len(data)}")
        random.Random(seed).shuffle(data)
        path = out_dir / f"{setting}.jsonl"
        digest = hashlib.sha256()
        supervised = 0
        with path.open("wb") as handle:
            for block in data:
                line = (json.dumps(block, separators=(",", ":")) + "\n").encode("utf-8")
                handle.write(line); digest.update(line)
                supervised += sum(x != -100 for x in block["labels"][1:])
        files[setting] = {"sha256": digest.hexdigest(), "blocks": len(data),
                          "input_tokens": len(data)*seq_length, "supervised_tokens_after_shift": supervised,
                          "slots": [left,right]}
    manifest = {"dataset": DATASET, "revision": REVISION, "model_tokenizer": model,
                "seed": seed, "sequence_length": seq_length, "blocks_per_setting": blocks,
                "sources": SOURCES, "background_sources": list(BACKGROUND), "observed_sources": observed,
                "eligible_records_consumed": counts, "discarded_all_masked_blocks": discarded, "files": files,
                "limitations": ["Only user/assistant two-turn records; multilingual and source-specific filtering affects scope.",
                                "Packed blocks may combine examples and attend across their boundaries; some records cross block boundaries.",
                                "Equal input token counts and optimizer steps do not guarantee equal supervised target counts or GPU time.",
                                "Background is a selected subset, not the original Tulu mixture; source licenses differ.",
                                "Boundary-crossing prompt/response tokens are masked to avoid training on prompt text.",
                                "Benchmark contamination is not established as absent."]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/tulu_factorial/seed_42"))
    parser.add_argument("--model", required=True, help="Tokenizer used by the model for training")
    parser.add_argument("--blocks", type=int, default=4800, help="Must be multiple of 80; 4800 x 2048 = 9,830,400 input tokens")
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if not tokenizer.chat_template or not tokenizer.is_fast:
        raise ValueError("Tokenizer must be fast and have a chat template")
    rows = load_dataset(DATASET, revision=REVISION, split="train", streaming=True).shuffle(seed=args.seed, buffer_size=10000)
    manifest = prepare(rows, tokenizer, args.out_dir, args.blocks, args.seq_length, args.seed, args.model)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
