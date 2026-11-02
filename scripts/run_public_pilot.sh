#!/usr/bin/env bash
set -euo pipefail

# Invoke from the project root. Run smoke test first; see docs/COLLABORATOR.md.
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/models/qwen25_15b.yaml}"
SCALES="${SCALES:-500}"
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
  data_dir="data/processed/public_pilot/seed_${seed}"
  python scripts/prepare_forward_reverse_data.py \
    --a_path data/raw/forward.jsonl --b_path data/raw/reverse.jsonl \
    --out_dir "$data_dir" --scales $SCALES --seed "$seed"
  for scale in $SCALES; do
    result_dir="results/public_pilot/seed_${seed}/scale_${scale}"
    python -m data_interaction.eval.run_eval --checkpoint "$MODEL" --model_name "$MODEL" \
      --pair_name public_forward_reverse_exploratory --tasks "$TASK" --scale "$scale" \
      --setting baseline --seed "$seed" --output_json "$result_dir/baseline.json" "${eval_args[@]}"
    for setting in a_only b_only fixed_mix expanded_mix; do
      train_data="$data_dir/scale_${scale}/${setting}.jsonl"
      out_dir="outputs/public_pilot/seed_${seed}/scale_${scale}/${setting}"
      accelerate launch --num_processes "$NUM_PROCESSES" src/data_interaction/train/train_lora_sft.py \
        --config configs/experiments/public_pilot.yaml --model_config "$MODEL_CONFIG" --model_name "$MODEL" \
        --lora_config configs/lora/default_lora.yaml --train_data "$train_data" \
        --output_dir "$out_dir" --seed "$seed" --max_steps "$MAX_STEPS"
      python -m data_interaction.eval.run_eval --checkpoint "$out_dir" --model_name "$MODEL" \
        --pair_name public_forward_reverse_exploratory --tasks "$TASK" --scale "$scale" \
        --setting "$setting" --seed "$seed" --train_data_path "$train_data" \
        --output_json "$result_dir/${setting}.json" "${eval_args[@]}"
    done
  done
done
python -m data_interaction.metrics.aggregate_results --results_dir results/public_pilot \
  --output_csv analysis/public_pilot_summary.csv
