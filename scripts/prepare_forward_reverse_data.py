"""Prepare all A/B mixtures for the forward/reverse pilot."""

from __future__ import annotations

import argparse
import json

from data_interaction.data.build_units import build_scale_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a_path", required=True)
    parser.add_argument("--b_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scales", nargs="+", required=True, type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build_scale_datasets(args.a_path, args.b_path, args.out_dir, args.scales, args.seed)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

