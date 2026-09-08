#!/usr/bin/env bash
set -euo pipefail
python -m data_interaction.analysis.make_tables \
  --summary_csv "${1:-analysis/interaction_summary.csv}" \
  --out_dir "${2:-paper_tables}"

