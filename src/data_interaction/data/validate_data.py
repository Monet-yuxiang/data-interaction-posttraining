"""Validate one or more source JSONL files."""

from __future__ import annotations

import argparse

from data_interaction.data.load_jsonl import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    for path in args.paths:
        items = load_jsonl(path)
        print(f"{path}: OK ({len(items)} examples)")


if __name__ == "__main__":
    main()

