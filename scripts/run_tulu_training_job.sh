#!/usr/bin/env bash
set -euo pipefail

# One independently scheduled single-GPU job. Rerun exactly to resume.
SETTING="${1:?usage: bash scripts/run_tulu_training_job.sh SETTING [SEED]}"
SEED="${2:-42}"
case "$SETTING" in baseline|a1|b1|c1|d1|a2|b2|c2|d2|ab|ac|ad|bc|bd|cd) ;; *) echo "Invalid setting: $SETTING" >&2; exit 2;; esac
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
DATA_DIR="${DATA_DIR:-data/processed/tulu_factorial/seed_${SEED}}"
OUT_DIR="${OUT_DIR:-outputs/tulu_factorial/seed_${SEED}}"
DATA="${DATA_DIR}/${SETTING}.jsonl"
OUT="${OUT_DIR}/${SETTING}"
[[ -f "$DATA" ]] || { echo "Missing $DATA; prepare packed data first" >&2; exit 2; }
python - "$DATA_DIR/manifest.json" "$DATA" "$SETTING" "$MODEL" "$SEED" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, data_path, setting, model, seed = sys.argv[1:]
manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
if manifest["model_tokenizer"] != model or manifest["seed"] != int(seed):
    raise SystemExit("Prepared data tokenizer or seed does not match this job")
if manifest["sequence_length"] != 2048:
    raise SystemExit("Prepared sequence length differs from training config (2048)")
digest = hashlib.sha256()
with open(data_path, "rb") as source:
    for block in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(block)
if manifest["files"][setting]["sha256"] != digest.hexdigest():
    raise SystemExit("Prepared data hash does not match manifest")
PY
python src/data_interaction/train/train_lora_sft.py \
  --config configs/experiments/tulu_factorial.yaml --model_config configs/models/qwen25_15b.yaml \
  --lora_config configs/lora/default_lora.yaml --model_name "$MODEL" \
  --train_data "$DATA" --output_dir "$OUT" --seed "$SEED" --resume_auto \
  --time_limit_seconds "${TIME_LIMIT_SECONDS:-0}"
if [[ -f "$OUT/TRAINING_COMPLETE.json" ]]; then
  echo "TRAINING COMPLETE: $OUT"
else
  echo "PARTIAL: rerun this exact command to resume $OUT"
fi
