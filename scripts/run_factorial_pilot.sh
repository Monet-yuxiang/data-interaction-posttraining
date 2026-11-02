#!/usr/bin/env bash
set -euo pipefail

# Exploratory fixed-example replacement design, not a token-matched experiment.
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/models/qwen25_15b.yaml}"
N="${N:-50}"
SEEDS="${SEEDS:-42}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
TASK="${TASK:-gsm8k}"
MAX_STEPS="${MAX_STEPS:--1}"
eval_args=()
if [[ -n "${EVAL_LIMIT:-}" ]]; then eval_args+=(--limit "$EVAL_LIMIT"); fi
if [[ "$MAX_STEPS" == "-1" && "${ACK_TRUNCATION:-0}" != "1" ]]; then
  echo 'Full run blocked: audit token lengths and decide sequence length first. If understood, set ACK_TRUNCATION=1. See docs/COLLABORATOR.md.' >&2
  exit 2
fi
for seed in $SEEDS; do
  data_dir="data/processed/factorial/n_${N}_seed_${seed}"
  python -m data_interaction.data.build_factorial \
    --reference data/raw/reference.jsonl --a data/raw/forward.jsonl \
    --b data/raw/reverse.jsonl --c data/raw/background.jsonl \
    --out-dir "$data_dir" --n "$N" --seed "$seed"
  for setting in 000 100 010 001 110 101 011 111; do
    train_data="$data_dir/${setting}.jsonl"
    out_dir="outputs/factorial/n_${N}_seed_${seed}/${setting}"
    result="results/factorial/n_${N}_seed_${seed}/${setting}.json"
    accelerate launch --num_processes "$NUM_PROCESSES" src/data_interaction/train/train_lora_sft.py \
      --config configs/experiments/public_pilot.yaml --model_config "$MODEL_CONFIG" --model_name "$MODEL" \
      --lora_config configs/lora/default_lora.yaml --train_data "$train_data" \
      --output_dir "$out_dir" --seed "$seed" --max_steps "$MAX_STEPS"
    python -m data_interaction.eval.run_eval --checkpoint "$out_dir" --model_name "$MODEL" \
      --pair_name reference_replacement_exploratory --tasks "$TASK" --scale "$N" \
      --setting "$setting" --seed "$seed" --train_data_path "$train_data" --output_json "$result" "${eval_args[@]}"
  done
done
python -m data_interaction.metrics.factorial --results-dir results/factorial \
  --output-csv analysis/factorial_by_seed_and_task.csv
