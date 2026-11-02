"""Generate the two basic plots required by the pilot."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _load(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is missing. Install with: pip install -e '.[analysis]'") from exc

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    rows = _load(args.summary_csv)
    if not rows:
        raise ValueError("summary CSV is empty")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        for prefix in ("fixed", "expanded"):
            value = row.get(f"{prefix}_mix_vs_best", "")
            if value != "":
                lines[(row["model_name"], prefix)].append((int(row["scale"]), float(value)))
    fig, axis = plt.subplots(figsize=(8, 5))
    for (model, mix), points in sorted(lines.items()):
        points.sort()
        axis.plot([x for x, _ in points], [y for _, y in points], marker="o", label=f"{model} / {mix}")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(xlabel="Training scale (examples per group)", ylabel="Mixed vs best single", title="Mixture performance gap")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "scale_trends.png", dpi=180)
    plt.close(fig)

    labels = [f"{row['model_name']}\nN={row['scale']}" for row in rows]
    settings = [
        ("forward_avg", "A only"), ("reverse_avg", "B only"),
        ("fixed_mix_avg", "Fixed mix"), ("expanded_mix_avg", "Expanded mix"),
    ]
    width = 0.2
    x_positions = list(range(len(rows)))
    fig, axis = plt.subplots(figsize=(max(8, len(rows) * 1.4), 5))
    for index, (column, label) in enumerate(settings):
        offset = (index - 1.5) * width
        values = [float(row[column]) if row.get(column, "") != "" else 0.0 for row in rows]
        axis.bar([x + offset for x in x_positions], values, width=width, label=label)
    axis.set_xticks(x_positions, labels, rotation=20, ha="right")
    axis.set(ylabel="Average score", title="Single-group and mixture comparison")
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "setting_comparison.png", dpi=180)
    plt.close(fig)
    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()

