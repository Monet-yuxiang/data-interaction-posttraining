"""Audit full-example truncation under the actual training chat template."""
from __future__ import annotations

import argparse
import json

from data_interaction.data.load_jsonl import load_jsonl
from data_interaction.train.train_lora_sft import _format_example


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=100, help="0 means all examples")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    for path in args.paths:
        examples = load_jsonl(path)
        if args.limit:
            examples = examples[:args.limit]
        lengths = []
        for item in examples:
            text = _format_example(item.to_dict(), tokenizer)
            lengths.append(len(tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)))
        print(json.dumps({"path": path, "count": len(lengths), "max_length": args.max_length,
                          "truncated": sum(length > args.max_length for length in lengths),
                          "largest_tokens": max(lengths), "mean_tokens": sum(lengths) / len(lengths)}))


if __name__ == "__main__":
    main()
