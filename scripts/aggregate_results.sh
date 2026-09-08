#!/usr/bin/env bash
set -euo pipefail
python -m data_interaction.metrics.aggregate_results \
  --results_dir "${1:-results}" \
  --output_csv "${2:-analysis/interaction_summary.csv}"

