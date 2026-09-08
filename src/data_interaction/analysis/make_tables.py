"""Create collaborator-friendly CSV and Markdown tables from the summary."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _markdown(rows: list[dict[str, str]], columns: list[str]) -> str:
    def clean(value: str) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(clean(row.get(column, "")) for column in columns) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    with Path(args.summary_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "interaction_summary.md").write_text(_markdown(rows, columns), encoding="utf-8")
    with (out_dir / "interaction_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote tables to {out_dir}")


if __name__ == "__main__":
    main()

