#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 MODEL_CONFIG TRAIN_DATA OUTPUT_DIR EXPERIMENT_CONFIG [LORA_CONFIG]" >&2
  exit 2
fi

MODEL_CONFIG="$1"
TRAIN_DATA="$2"
OUTPUT_DIR="$3"
EXPERIMENT_CONFIG="$4"
LORA_CONFIG="${5:-configs/lora/default_lora.yaml}"

accelerate launch --num_processes "${NUM_PROCESSES:-8}" \
  src/data_interaction/train/train_lora_sft.py \
  --config "$EXPERIMENT_CONFIG" \
  --model_config "$MODEL_CONFIG" \
  --lora_config "$LORA_CONFIG" \
  --train_data "$TRAIN_DATA" \
  --output_dir "$OUTPUT_DIR"

