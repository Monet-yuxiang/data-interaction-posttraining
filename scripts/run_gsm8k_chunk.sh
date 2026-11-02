#!/usr/bin/env bash
set -euo pipefail
SETTING="${1:?usage: bash scripts/run_gsm8k_chunk.sh SETTING START [SCALE] [SEED]}"
START="${2:?missing zero-based START}"
SCALE="${3:-500}"
SEED="${4:-42}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
case "$SETTING" in baseline) CHECKPOINT="$MODEL";; a_only|b_only|fixed_mix|expanded_mix) CHECKPOINT="outputs/windowed/seed_${SEED}/scale_${SCALE}/${SETTING}";; *) echo 'Invalid setting' >&2; exit 2;; esac
if [[ "$START" -lt 0 || "$START" -ge 1319 || "$CHUNK_SIZE" -lt 1 || "$CHUNK_SIZE" -gt 1319 ]]; then
  echo 'Require START in 0..1318 and CHUNK_SIZE in 1..1319' >&2; exit 2
fi
END=$((START + CHUNK_SIZE))
if [[ "$END" -gt 1319 ]]; then END=1319; fi
OUT="results/windowed_chunks/seed_${SEED}/scale_${SCALE}/${SETTING}"
python -m data_interaction.eval.gsm8k_chunks run \
  --checkpoint "$CHECKPOINT" --model-name "$MODEL" --setting "$SETTING" --seed "$SEED" --scale "$SCALE" \
  --start "$START" --end "$END" --out-dir "$OUT"
