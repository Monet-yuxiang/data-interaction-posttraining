# Resumable two-source experiment

This workflow focuses on the A/B main line. A two-hour allocation is **not** assumed; train one setting per job, checkpoint every 20 optimizer steps, and evaluate GSM8K in batches of 100. Actual duration depends on GPU, data lengths, network and queue. No GPU or lm-eval shard has yet been validated on the collaborator's server: first run the short checks below and report timings/peak memory. Use the same model, tokenizer, environment and filesystem for all jobs and resumptions.

## What changed

1. Evaluation uses the Qwen chat template, zero-shot GSM8K and `exact_match,flexible-extract` on every setting. Flexible extraction is a proxy and individual predictions should be inspected. Earlier five-shot/no-chat/strict smoke scores are a different protocol and **must not be combined**.
2. SFT splits each assistant response into windows of at most 2048 tokens. Every assistant target token is used exactly once, with the question and up to 512 prior response tokens repeated as masked context. The prompt and repeated context are not included in loss. This avoids silent target loss but is **not full-context training**: long-range reasoning history is missing, and long examples make more windows and therefore receive more optimizer updates. This is an exploratory training protocol, not a token-budget-matched design. Questions exceeding the window fail clearly.
3. `outputs/windowed/` and `results/windowed*` are new locations; previous smoke artifacts remain untouched. A run signature checks configuration and source JSONL hash on resume. Training saves optimizer, scheduler and RNG with Hugging Face checkpoints; interruption between checkpoints can lose up to 20 steps. Complete output has `TRAINING_COMPLETE.json`.

In one local check, the first 10 forward examples expanded into 70 windows and the first 10 reverse examples into 36 windows. They are not random samples. Expect considerably more training steps than in the earlier truncating smoke test; measure the new workflow on the server.

## First setup

Update your existing clone with `git pull --ff-only` after checking for local changes. Install the updated evaluation version in the existing environment: `pip install -e '.[data,train,eval,analysis,dev]'` (uses `lm-eval==0.4.13`). Existing `data/raw/forward.jsonl` and `reverse.jsonl` can be reused. Otherwise see [COLLABORATOR.md](COLLABORATOR.md) for source download. The GPU machine needs its usual CUDA-enabled PyTorch; keep `deepspeed: null` for one GPU.

Prepare source-example mixtures; 20 is for smoke test and 500 is the first real exploratory scale:

```bash
python scripts/prepare_forward_reverse_data.py \
  --a_path data/raw/forward.jsonl --b_path data/raw/reverse.jsonl \
  --out_dir data/processed/public_pilot/seed_42 --scales 20 500 --seed 42
export MODEL=models/Qwen2.5-1.5B-Instruct  # use your actual local path or HF ID
```

Retain the same `MODEL` value across training and evaluation; do not replace weights or data in place. Sampling seeds 43 and 44 require separate preparation with matching `--seed` and `seed_43`/`seed_44` output paths.

## Verify the revised pipeline

Run a small training job and a baseline evaluation chunk. The evaluation command below uses 5 questions and is a format check only:

```bash
bash scripts/run_training_job.sh a_only 20 42
python -m data_interaction.eval.gsm8k_chunks run \
  --checkpoint "$MODEL" --model-name "$MODEL" --setting baseline \
  --scale 20 --seed 42 --start 0 --end 5 --out-dir results/windowed-smoke
```

Record wall time and peak GPU memory (`nvidia-smi` during training); verify there is no out-of-memory or chat-template warning. Then test one adapter chunk:

```bash
python -m data_interaction.eval.gsm8k_chunks run \
  --checkpoint outputs/windowed/seed_42/scale_20/a_only --model-name "$MODEL" \
  --setting a_only --scale 20 --seed 42 --start 0 --end 5 --out-dir results/windowed-smoke-adapter
```

The smoke output is not suitable for scientific conclusions. Ask the project owner before scaling up if these checks fail or take unexpectedly long.

## Train a single setting

Submit one command per allocation; repeat for `a_only`, `b_only`, `fixed_mix`, and `expanded_mix`:

```bash
bash scripts/run_training_job.sh a_only 500 42
```

If interrupted, re-run the **identical command**; it resumes the latest saved checkpoint or restarts if no checkpoint was saved. A completed run is skipped. Do not run two jobs for the same setting/seed simultaneously. If your scheduler has a wall limit, optionally set `TIME_LIMIT_SECONDS=4200` (soft limit, starts after data preprocessing); the trainer saves and stops at the next completed step. Leave extra time for model loading, tokenization and checkpoint saving. It cannot guarantee a hard scheduler deadline or protect against a kill during the first step/save.

## Evaluate in independent batches

For each of `baseline`, `a_only`, `b_only`, `fixed_mix`, `expanded_mix`, run a separate batch job with start indices `0, 100, 200, ..., 1300`. For example:

```bash
bash scripts/run_gsm8k_chunk.sh baseline 0 500 42
bash scripts/run_gsm8k_chunk.sh a_only 0 500 42
bash scripts/run_gsm8k_chunk.sh a_only 100 500 42
```

The last batch, start 1300, contains 19 examples. Each setting needs 14 batches for the full 1,319-question test set. Once a batch completes, running the exact same command skips it; if interrupted, only the current batch needs to be rerun. Different batches may be submitted separately. `CHUNK_SIZE` changes batch size but must remain consistent for all starts of one evaluation; avoid overlapping jobs. With 100 examples and the previous 10-example timing, roughly 20 minutes is an initial guess only. Scheduler/model loading and answer length may change it; measure on the server.

After **all** 14 batches of a setting finish, merge it:

```bash
python -m data_interaction.eval.gsm8k_chunks merge \
  --chunks results/windowed_chunks/seed_42/scale_500/a_only \
  --output-json results/windowed/seed_42/scale_500/a_only.json
```

Repeat for the other settings, including `baseline`, changing both paths. Merge checks all IDs 0–1318 exactly once and rejects mixed run identities. Per-sample predictions remain in chunk files for inspection. After all five merges:

```bash
python -m data_interaction.metrics.aggregate_results --results_dir results/windowed \
  --output_csv analysis/windowed_summary.csv
```

Do not mix old and new results, different model/tokenizer versions, or different evaluation protocols. Keep generated manifests, signatures and logs. For this pilot, benchmark overlap and incorrect synthetic answers have not been fully audited. Fixed example counts do not imply fixed token or compute budgets. The three-source factorial, matched-token controls, sequential order and mechanistic analysis remain future work; do not start a large matrix before reviewing the first seed.
