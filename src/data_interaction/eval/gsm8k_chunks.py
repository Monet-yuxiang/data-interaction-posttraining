"""Evaluate resumable GSM8K chunks and merge only complete matching coverage."""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

TOTAL = 1319


def fingerprint(checkpoint: str) -> str:
    path = Path(checkpoint)
    digest = hashlib.sha256(checkpoint.encode())
    if path.is_dir():
        for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin", "config.json"):
            file = path / name
            if file.is_file():
                digest.update(name.encode())
                with file.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
    return digest.hexdigest()


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_chunk(args) -> None:
    if args.start < 0 or args.end <= args.start or args.end > TOTAL:
        raise ValueError(f"Require 0 <= start < end <= {TOTAL}")
    max_gen_toks = getattr(args, "max_gen_toks", None)
    if max_gen_toks is not None and max_gen_toks <= 0:
        raise ValueError("max_gen_toks must be positive")
    checkpoint = Path(args.checkpoint)
    if args.setting != "baseline" and not (checkpoint / "TRAINING_COMPLETE.json").is_file():
        raise ValueError("Adapter training is not complete; refusing evaluation")
    identity = {"checkpoint": args.checkpoint, "checkpoint_hash": fingerprint(args.checkpoint),
                "model_name": args.model_name, "pair_name": args.pair_name, "setting": args.setting,
                "seed": args.seed, "scale": args.scale, "task": "gsm8k", "num_fewshot": 0,
                "chat_template": True, "fewshot_as_multiturn": False, "metric": "exact_match,flexible-extract",
                "lm_eval_version": version("lm-eval")}
    # An omitted limit preserves the legacy identity and lm-eval default (256
    # in 0.4.13); explicit limits must never reuse an old chunk silently.
    if max_gen_toks is not None:
        identity["max_gen_toks"] = max_gen_toks
    output = args.out_dir / f"chunk_{args.start:04d}_{args.end:04d}.json"
    if output.exists():
        current = json.loads(output.read_text(encoding="utf-8"))
        if current["identity"] != identity or current["indices"] != list(range(args.start, args.end)):
            raise ValueError("Existing chunk has different settings; use a new output directory")
        print(f"Already complete: {output}")
        return
    from lm_eval import evaluator
    adapter = checkpoint / "adapter_config.json"
    model_args = f"pretrained={args.checkpoint},trust_remote_code=True"
    if adapter.is_file():
        base = json.loads(adapter.read_text(encoding="utf-8"))["base_model_name_or_path"]
        model_args = f"pretrained={base},peft={args.checkpoint},trust_remote_code=True"
    indices = list(range(args.start, args.end))
    generation = {"gen_kwargs": {"max_gen_toks": max_gen_toks}} if max_gen_toks is not None else {}
    raw = evaluator.simple_evaluate(model="hf", model_args=model_args, tasks=["gsm8k"], batch_size="1",
             samples={"gsm8k": indices}, apply_chat_template=True, fewshot_as_multiturn=False,
             num_fewshot=0, log_samples=True, bootstrap_iters=0, device=args.device, **generation)
    samples = raw["samples"]["gsm8k"]
    filtered = [s for s in samples if s.get("filter") == "flexible-extract"]
    by_id = {int(s["doc_id"]): s for s in filtered}
    if len(filtered) != len(indices) or set(by_id) != set(indices):
        raise ValueError("Returned sample IDs/flexible filter do not match requested range; no chunk saved")
    rows = [{"doc_id": i, "exact_match": float(by_id[i]["exact_match"]),
             "doc_hash": by_id[i].get("doc_hash"), "prompt_hash": by_id[i].get("prompt_hash"),
             "responses": by_id[i].get("resps"), "filtered_responses": by_id[i].get("filtered_resps")}
            for i in indices]
    write_atomic(output, {"identity": identity, "indices": indices, "rows": rows})
    print(f"Saved {output}: {len(rows)} examples")


def merge_chunks(paths: list[Path], total: int = TOTAL) -> dict:
    if not paths:
        raise ValueError("No chunks found")
    identity = None
    scores = {}
    for path in sorted(paths):
        chunk = json.loads(path.read_text(encoding="utf-8"))
        if identity is None:
            identity = chunk["identity"]
        if chunk["identity"] != identity:
            raise ValueError(f"Chunk identity mismatch: {path}")
        rows = chunk["rows"]
        if [row["doc_id"] for row in rows] != chunk["indices"]:
            raise ValueError(f"Chunk row IDs mismatch: {path}")
        for row in rows:
            i = int(row["doc_id"])
            if i in scores:
                raise ValueError(f"Duplicate sample {i}")
            scores[i] = float(row["exact_match"])
    if set(scores) != set(range(total)):
        raise ValueError(f"Incomplete or invalid coverage: {len(scores)}/{total} unique IDs")
    return {"project": "data_interaction_posttraining", "pair_name": identity["pair_name"],
            "model_name": identity["model_name"], "scale": identity["scale"], "seed": identity["seed"],
            "setting": identity["setting"], "checkpoint_path": identity["checkpoint"],
            "evaluation_protocol": identity, "n_samples": total,
            "eval_results": {"gsm8k": sum(scores.values()) / total},
            "average_score": sum(scores.values()) / total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    for name in ("checkpoint", "model-name", "setting", "out-dir"):
        run.add_argument("--" + name, required=True, type=Path if name == "out-dir" else str)
    run.add_argument("--pair-name", default="public_forward_reverse_windowed")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--scale", type=int, default=500)
    run.add_argument("--start", type=int, required=True)
    run.add_argument("--end", type=int, required=True)
    run.add_argument("--device", help="Optional lm-eval device override, e.g. cpu")
    run.add_argument("--max-gen-toks", type=int, help="Maximum generated tokens; omit for legacy lm-eval default")
    merge = sub.add_parser("merge")
    merge.add_argument("--chunks", type=Path, required=True)
    merge.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "run":
        run_chunk(args)
    else:
        payload = merge_chunks(list(args.chunks.glob("chunk_*.json")))
        if args.output_json.exists():
            raise FileExistsError("Merged output already exists; choose a new path")
        write_atomic(args.output_json, payload)
        print(f"Merged {payload['n_samples']} examples: {args.output_json}")


if __name__ == "__main__":
    main()
