"""Score pre-generated JSONL answers with EM, normalized EM, and token F1."""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any


def normalize_answer(value: str) -> str:
    value = value.lower()
    value = "".join(character for character in value if character not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def token_f1(prediction: str, gold: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(gold).split()
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not predicted or not expected:
        return float(predicted == expected)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def score_file(path: str | Path, include_f1: bool = False) -> dict[str, Any]:
    totals = {"exact_match": 0.0, "normalized_exact_match": 0.0, "f1": 0.0}
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            try:
                str(item["question"])
                prediction = str(item["model_output"])
                gold = str(item["gold_answer"])
            except KeyError as exc:
                raise ValueError(f"{path}:{line_number}: missing {exc.args[0]}") from exc
            totals["exact_match"] += float(prediction.strip() == gold.strip())
            totals["normalized_exact_match"] += float(normalize_answer(prediction) == normalize_answer(gold))
            totals["f1"] += token_f1(prediction, gold)
            count += 1
    if count == 0:
        raise ValueError(f"{path} contains no examples")
    result = {"count": count, "exact_match": totals["exact_match"] / count,
              "normalized_exact_match": totals["normalized_exact_match"] / count}
    if include_f1:
        result["f1"] = totals["f1"] / count
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--f1", action="store_true")
    args = parser.parse_args()
    result = score_file(args.input_jsonl, args.f1)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
