#!/usr/bin/env bash
set -euo pipefail

# Assumes prepare_forward_reverse_data.py has already been run.
EXPERIMENT_CONFIG="${1:-configs/experiments/forward_reverse_scale_qwen.yaml}"
DATA_DIR="${2:-data/processed/forward_reverse}"
OUTPUT_ROOT="${3:-outputs}"
SCALES="${SCALES:-1000 5000 10000}"
MODELS="${MODELS:-qwen25_15b qwen25_7b}"

for model in $MODELS; do
  for scale in $SCALES; do
    for setting in a_only b_only fixed_mix expanded_mix; do
      scripts/launch_single_run.sh \
        "configs/models/${model}.yaml" \
        "${DATA_DIR}/scale_${scale}/${setting}.jsonl" \
        "${OUTPUT_ROOT}/${model}/forward_reverse/scale_${scale}/${setting}" \
        "$EXPERIMENT_CONFIG"
    done
  done
done
