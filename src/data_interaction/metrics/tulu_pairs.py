"""Pairwise replacement contrasts for the shared-background two-slot design."""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path


SOURCES = "abcd"
SETTINGS = ("baseline", *(f"{s}{slot}" for slot in (1, 2) for s in SOURCES),
            *(a + b for a, b in combinations(SOURCES, 2)))


def pair_contrasts(scores: dict[str, float]) -> dict[str, float]:
    missing = set(SETTINGS) - scores.keys()
    if missing:
        raise ValueError(f"Missing settings: {sorted(missing)}")
    return {a + b: scores[a + b] - scores[a + "1"] - scores[b + "2"] + scores["baseline"]
            for a, b in combinations(SOURCES, 2)}


def sample_identity(record: dict, task: str) -> list[tuple]:
    samples = record.get("samples", {}).get(task)
    if not samples or any("doc_id" not in row for row in samples):
        raise ValueError(f"Missing per-example identities for task {task}")
    return sorted((str(r["doc_id"]), str(r.get("doc_hash")), str(r.get("prompt_hash")), str(r.get("filter")))
                  for r in samples)


def validate_records(records: dict[str, dict]) -> None:
    baseline = records["baseline"]
    for key in ("model_name", "pair_name", "scale", "seed", "evaluation_protocol"):
        if key not in baseline or any(record.get(key) != baseline[key] for record in records.values()):
            raise ValueError(f"Missing or inconsistent metadata: {key}")
    protocol = baseline["evaluation_protocol"]
    required = {"tasks", "limit", "batch_size", "max_gen_toks", "num_fewshot", "chat_template", "lm_eval_version", "metric"}
    if not isinstance(protocol, dict) or not required.issubset(protocol):
        raise ValueError("Incomplete evaluation protocol")
    tasks = set(baseline["eval_results"])
    if not tasks or any(set(r["eval_results"]) != tasks for r in records.values()):
        raise ValueError("Each setting must have the same evaluation tasks")
    for task in tasks:
        metric = protocol["metric"]
        samples = sample_identity(baseline, task)
        for record in records.values():
            if record["task_metrics"][task].get(metric) != record["eval_results"][task]:
                raise ValueError(f"Selected score does not match raw metric for {task}")
            if sample_identity(record, task) != samples:
                raise ValueError(f"Different evaluation samples for {task}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory containing normalized SETTING.json files from the same protocol")
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    records = {setting: json.loads((args.results_dir / f"{setting}.json").read_text(encoding="utf-8"))
               for setting in SETTINGS}
    for setting, record in records.items():
        if record.get("setting") != setting:
            raise ValueError(f"Setting mismatch: {setting}")
    validate_records(records)
    baseline = records["baseline"]
    tasks = set(baseline["eval_results"])
    rows = []
    for task in sorted(tasks):
        scores = {setting: float(record["eval_results"][task]) for setting, record in records.items()}
        for pair, value in pair_contrasts(scores).items():
            rows.append({"model": baseline["model_name"], "scale": baseline["scale"], "seed": baseline["seed"],
                         "task": task, "pair": pair, "contrast": value})
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} pair/task rows to {args.output_csv}")


if __name__ == "__main__":
    main()
