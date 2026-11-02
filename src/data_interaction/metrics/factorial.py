"""Compute matched 2x2x2 replacement contrasts from normalized eval JSON."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SETTINGS = ("000", "100", "010", "001", "110", "101", "011", "111")


def contrasts(y: dict[str, float]) -> dict[str, float]:
    missing = set(SETTINGS) - y.keys()
    if missing:
        raise ValueError(f"missing settings: {sorted(missing)}")
    pair = y["110"] - y["100"] - y["010"] + y["000"]
    pair_c = y["111"] - y["101"] - y["011"] + y["001"]
    predicted = y["110"] + y["101"] + y["011"] - y["100"] - y["010"] - y["001"] + y["000"]
    return {"pair_ab": pair, "pair_ab_given_c": pair_c, "triple": pair_c - pair,
            "predicted_111_pairwise": predicted, "observed_111": y["111"],
            "prediction_error": y["111"] - predicted}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    groups: dict[tuple[str, str, int, int, str], dict[str, float]] = {}
    for path in sorted(args.results_dir.rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        setting = record.get("setting")
        if setting not in SETTINGS:
            continue
        for task, score in record.get("eval_results", {}).items():
            if isinstance(score, (int, float)):
                key = (str(record["pair_name"]), str(record["model_name"]),
                       int(record["scale"]), int(record["seed"]), task)
                cell = groups.setdefault(key, {})
                if setting in cell:
                    raise ValueError(f"duplicate result for {key}, setting {setting}")
                cell[setting] = float(score)
    if not groups:
        raise ValueError("no factorial result records found")
    rows = []
    for key, values in sorted(groups.items()):
        row = dict(zip(("experiment", "model", "scale", "seed", "task"), key))
        row.update(contrasts(values))
        rows.append(row)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} seed/task rows to {args.output_csv}")


if __name__ == "__main__":
    main()
