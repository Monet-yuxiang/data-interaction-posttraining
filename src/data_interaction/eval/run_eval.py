"""Run arbitrary lm-evaluation-harness tasks and save a normalized result record."""

from __future__ import annotations

import argparse
import json
import logging
from importlib.metadata import version
from pathlib import Path
from typing import Any

from data_interaction.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)
PREFERRED_METRICS = ("exact_match,flexible-extract", "exact_match,strict-match", "exact_match", "acc_norm,none", "acc,none", "acc")


def _primary_score(metrics: dict[str, Any]) -> float | None:
    for name in PREFERRED_METRICS:
        value = metrics.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    for name, value in metrics.items():
        if "stderr" not in name and isinstance(value, (int, float)):
            return float(value)
    return None


def evaluate_raw(checkpoint: str, tasks: list[str], batch_size: str, limit: float | None,
                 max_gen_toks: int | None = None) -> dict[str, Any]:
    try:
        from lm_eval import evaluator
    except ImportError as exc:
        raise SystemExit("lm-eval is missing. Install with: pip install -e '.[eval]'") from exc

    adapter_config = Path(checkpoint) / "adapter_config.json"
    if adapter_config.exists():
        base_model = json.loads(adapter_config.read_text(encoding="utf-8")).get("base_model_name_or_path")
        if not base_model:
            raise ValueError(f"adapter has no base_model_name_or_path: {adapter_config}")
        model_args = f"pretrained={base_model},peft={checkpoint},trust_remote_code=True"
    else:
        model_args = f"pretrained={checkpoint},trust_remote_code=True"
    if max_gen_toks is not None and max_gen_toks <= 0:
        raise ValueError("max_gen_toks must be positive")
    generation = {"gen_kwargs": {"max_gen_toks": max_gen_toks}} if max_gen_toks is not None else {}
    return evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        batch_size=batch_size,
        limit=limit,
        apply_chat_template=True,
        fewshot_as_multiturn=False,
        num_fewshot=0,
        log_samples=True,
        **generation,
    )


def evaluate(checkpoint: str, tasks: list[str], batch_size: str, limit: float | None,
             max_gen_toks: int | None = None) -> dict[str, float]:
    raw = evaluate_raw(checkpoint, tasks, batch_size, limit, max_gen_toks)
    return scores_from_raw(raw)


def scores_from_raw(raw: dict[str, Any]) -> dict[str, float]:
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
    parser.add_argument("--max_gen_toks", type=int, required=True, help="Maximum generated tokens for generation tasks")
    parser.add_argument("--metric", required=True, help="Exact lm-eval metric key, e.g. exact_match,flexible-extract")
    parser.add_argument("--project", default="data_interaction_posttraining")
    parser.add_argument("--pair_name", default="forward_reverse")
    parser.add_argument("--model_name")
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument(
        "--setting",
        required=True,
        help="Setting name, e.g. a_only, fixed_mix, or factorial 000/111",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_data_path", default="")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    output = Path(args.output_json)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a new path")
    raw = evaluate_raw(args.checkpoint, args.tasks, args.batch_size, args.limit, args.max_gen_toks)
    metrics_by_task = raw.get("results") or {}
    scores = {}
    for task, metrics in metrics_by_task.items():
        value = metrics.get(args.metric)
        if not isinstance(value, (int, float)):
            raise ValueError(f"Task {task} has no numeric metric {args.metric!r}; available: {list(metrics)}")
        scores[task] = float(value)
    if not scores:
        raise ValueError("Evaluation returned no task results")
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
        "evaluation_protocol": {"tasks": args.tasks, "limit": args.limit, "batch_size": args.batch_size,
                                "max_gen_toks": args.max_gen_toks, "num_fewshot": 0,
                                "chat_template": True, "lm_eval_version": version("lm-eval"), "metric": args.metric},
        "task_metrics": raw.get("results", {}),
        "samples": raw.get("samples", {}),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", output)


if __name__ == "__main__":
    main()

