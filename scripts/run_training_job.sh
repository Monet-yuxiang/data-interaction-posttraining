#!/usr/bin/env bash
set -euo pipefail

# One single-GPU allocation. Rerun the identical command until complete.
SETTING="${1:?usage: bash scripts/run_training_job.sh SETTING [SCALE] [SEED]}"
SCALE="${2:-500}"
SEED="${3:-42}"
case "$SETTING" in a_only|b_only|fixed_mix|expanded_mix) ;; *) echo 'Invalid setting' >&2; exit 2;; esac
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
DATA="data/processed/public_pilot/seed_${SEED}/scale_${SCALE}/${SETTING}.jsonl"
OUT="outputs/windowed/seed_${SEED}/scale_${SCALE}/${SETTING}"
TIME_LIMIT_SECONDS="${TIME_LIMIT_SECONDS:-0}"
if [[ ! -f "$DATA" ]]; then echo "Missing $DATA; prepare data first (see docs/JOBS.md)" >&2; exit 2; fi
python src/data_interaction/train/train_lora_sft.py \
  --config configs/experiments/public_pilot.yaml --model_config configs/models/qwen25_15b.yaml \
  --lora_config configs/lora/default_lora.yaml --model_name "$MODEL" \
  --train_data "$DATA" --output_dir "$OUT" --seed "$SEED" --resume_auto \
  --time_limit_seconds "$TIME_LIMIT_SECONDS"
if [[ -f "$OUT/TRAINING_COMPLETE.json" ]]; then
  echo "TRAINING COMPLETE: $OUT"
else
  echo "PARTIAL: rerun this exact command in a new allocation to resume $OUT"
fi
