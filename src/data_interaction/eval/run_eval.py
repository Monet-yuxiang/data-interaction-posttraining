"""Run arbitrary lm-evaluation-harness tasks and save a normalized result record."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from data_interaction.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)
PREFERRED_METRICS = ("exact_match,strict-match", "exact_match", "acc_norm,none", "acc,none", "acc")


def _primary_score(metrics: dict[str, Any]) -> float | None:
    for name in PREFERRED_METRICS:
        value = metrics.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    for name, value in metrics.items():
        if "stderr" not in name and isinstance(value, (int, float)):
            return float(value)
    return None


def evaluate(checkpoint: str, tasks: list[str], batch_size: str, limit: float | None) -> dict[str, float]:
    try:
        from lm_eval import evaluator
    except ImportError as exc:
        raise SystemExit("lm-eval is missing. Install with: pip install -e '.[eval]'") from exc

    raw = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={checkpoint},trust_remote_code=True",
        tasks=tasks,
        batch_size=batch_size,
        limit=limit,
        log_samples=False,
    )
    scores: dict[str, float] = {}
    for task, metrics in (raw.get("results") or {}).items():
        score = _primary_score(metrics)
        if score is not None:
            scores[task] = score
        else:
            LOGGER.warning("No numeric primary metric found for task %s", task)
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", nargs="+", required=True, help="Any task names installed in lm-eval")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--batch_size", default="auto")
    parser.add_argument("--limit", type=float, help="Debug-only lm-eval sample limit")
    parser.add_argument("--project", default="data_interaction_posttraining")
    parser.add_argument("--pair_name", default="forward_reverse")
    parser.add_argument("--model_name")
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument(
        "--setting",
        required=True,
        choices=("baseline", "a_only", "b_only", "fixed_mix", "expanded_mix"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_data_path", default="")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    scores = evaluate(args.checkpoint, args.tasks, args.batch_size, args.limit)
    payload = {
        "project": args.project,
        "pair_name": args.pair_name,
        "model_name": args.model_name or Path(args.checkpoint).name,
        "scale": args.scale,
        "setting": args.setting,
        "seed": args.seed,
        "train_data_path": args.train_data_path,
        "checkpoint_path": args.checkpoint,
        "eval_results": scores,
        "average_score": sum(scores.values()) / len(scores) if scores else None,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", output)


if __name__ == "__main__":
    main()

