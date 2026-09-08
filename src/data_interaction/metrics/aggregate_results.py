"""Aggregate normalized result JSON files into three CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from data_interaction.metrics.interaction_metrics import compute_interaction_metrics, sign_is_persistent

CORE_SETTINGS = ("baseline", "a_only", "b_only", "fixed_mix", "expanded_mix")


def _read_results(results_dir: str | Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(Path(results_dir).rglob("*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if item.get("setting") in CORE_SETTINGS and "eval_results" in item:
            item["_source"] = str(path)
            records.append(item)
    if not records:
        raise ValueError(f"no normalized result JSON files found under {results_dir}")
    return records


def _score(record: dict[str, Any]) -> float:
    value = record.get("average_score")
    if isinstance(value, (int, float)):
        return float(value)
    scores = [float(v) for v in record.get("eval_results", {}).values() if isinstance(v, (int, float))]
    if not scores:
        raise ValueError(f"result has no numeric scores: {record.get('_source')}")
    return mean(scores)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        key = (str(record.get("pair_name", "unknown")), str(record.get("model_name", "unknown")), int(record["scale"]))
        grouped[key][record["setting"]].append(record)

    global_baselines: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (pair_name, model_name, _), settings in grouped.items():
        global_baselines[(pair_name, model_name)].extend(_score(item) for item in settings.get("baseline", []))

    summaries: list[dict[str, Any]] = []
    trends: list[dict[str, Any]] = []
    breakdown: list[dict[str, Any]] = []
    for (pair_name, model_name, scale), settings in sorted(grouped.items()):
        averages = {setting: mean(map(_score, values)) for setting, values in settings.items()}
        baseline = averages.get("baseline")
        if baseline is None and global_baselines[(pair_name, model_name)]:
            baseline = mean(global_baselines[(pair_name, model_name)])
        a_score, b_score = averages.get("a_only"), averages.get("b_only")
        summary: dict[str, Any] = {
            "pair_name": pair_name,
            "model_name": model_name,
            "scale": scale,
            "baseline_avg": baseline,
            "forward_avg": a_score,
            "reverse_avg": b_score,
            "fixed_mix_avg": averages.get("fixed_mix"),
            "expanded_mix_avg": averages.get("expanded_mix"),
        }
        labels: list[str] = []
        for mix_setting, prefix in (("fixed_mix", "fixed"), ("expanded_mix", "expanded")):
            mixed = averages.get(mix_setting)
            if a_score is None or b_score is None or mixed is None:
                summary[f"{prefix}_mix_vs_best"] = None
                summary[f"{prefix}_gamma"] = None
                labels.append(f"{prefix}:incomplete")
                continue
            gap = mixed - max(a_score, b_score)
            summary[f"{prefix}_mix_vs_best"] = gap
            if baseline is None:
                summary[f"{prefix}_gamma"] = None
                label = "baseline_missing"
            else:
                metrics = compute_interaction_metrics(baseline, a_score, b_score, mixed)
                summary[f"{prefix}_gamma"] = metrics.gamma
                label = metrics.interaction_type
            labels.append(f"{prefix}:{label}")
            trends.append({
                "pair_name": pair_name,
                "model_name": model_name,
                "scale": scale,
                "mix_setting": mix_setting,
                "gap_vs_best": gap,
                "gamma": summary[f"{prefix}_gamma"],
                "interaction_type": label,
            })
        summary["interaction_type"] = ";".join(labels)
        summaries.append(summary)

        for setting, setting_records in sorted(settings.items()):
            tasks = sorted({task for record in setting_records for task in record.get("eval_results", {})})
            row: dict[str, Any] = {"pair_name": pair_name, "model_name": model_name, "scale": scale, "setting": setting}
            for task in tasks:
                values = [record["eval_results"].get(task) for record in setting_records]
                numeric = [float(value) for value in values if isinstance(value, (int, float))]
                row[task] = mean(numeric) if numeric else None
            row["average"] = mean(map(_score, setting_records))
            breakdown.append(row)

    for key in {(row["pair_name"], row["model_name"], row["mix_setting"]) for row in trends}:
        relevant = [row for row in trends if (row["pair_name"], row["model_name"], row["mix_setting"]) == key]
        gap_persistent = sign_is_persistent(row["gap_vs_best"] for row in relevant)
        gammas = [row["gamma"] for row in relevant if isinstance(row["gamma"], (int, float))]
        gamma_persistent = len(gammas) == len(relevant) and sign_is_persistent(gammas)
        for row in relevant:
            row["gap_sign_persistent"] = gap_persistent
            row["gamma_sign_persistent"] = gamma_persistent
    return summaries, trends, breakdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--scale_trend_csv")
    parser.add_argument("--benchmark_csv")
    args = parser.parse_args()
    summary_path = Path(args.output_csv)
    trend_path = Path(args.scale_trend_csv) if args.scale_trend_csv else summary_path.with_name(summary_path.stem + "_scale_trend.csv")
    benchmark_path = Path(args.benchmark_csv) if args.benchmark_csv else summary_path.with_name(summary_path.stem + "_benchmark_breakdown.csv")
    summaries, trends, breakdown = aggregate(_read_results(args.results_dir))
    summary_fields = ["pair_name", "model_name", "scale", "baseline_avg", "forward_avg", "reverse_avg", "fixed_mix_avg", "expanded_mix_avg", "fixed_mix_vs_best", "expanded_mix_vs_best", "fixed_gamma", "expanded_gamma", "interaction_type"]
    trend_fields = ["pair_name", "model_name", "scale", "mix_setting", "gap_vs_best", "gamma", "interaction_type", "gap_sign_persistent", "gamma_sign_persistent"]
    dynamic_tasks = sorted({key for row in breakdown for key in row} - {"pair_name", "model_name", "scale", "setting", "average"})
    _write_csv(summary_path, summaries, summary_fields)
    _write_csv(trend_path, trends, trend_fields)
    _write_csv(benchmark_path, breakdown, ["pair_name", "model_name", "scale", "setting", *dynamic_tasks, "average"])
    print(f"Wrote {summary_path}, {trend_path}, and {benchmark_path}")


if __name__ == "__main__":
    main()

