# Good Data Does Not Always Add Up

Minimal, config-driven code for measuring whether two individually useful (or harmful) post-training data groups behave differently when trained together.

The first pilot studies forward reasoning (group A) and reverse reasoning (group B) across 1k, 5k, and 10k examples with Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct. The runnable core intentionally stays small: JSONL construction, LoRA SFT, an `lm-evaluation-harness` wrapper, interaction tables, and two basic plots.

## Research question

Data quality is not necessarily additive. We test two primary failure/success modes:

- **Mutual undermining:** A and B each improve over the untrained baseline, but their mixture is worse than at least one single-group run.
- **Rescuing complementarity:** A and B each hurt the baseline, but their mixture improves over it.

For baseline score \(S_0\), A-only score \(S_A\), B-only score \(S_B\), and mixture score \(S_{AB}\), the main continuous metrics are:

```text
gap_vs_best = S_AB - max(S_A, S_B)
gamma       = S_AB - S_A - S_B + S_0
```

Negative `gap_vs_best` means the mixture loses to the better single group. `gamma` compares the observed mixture with an additive counterfactual. The aggregator also records whether the sign of each metric persists across the tested scales.

## What is implemented

- Validated JSONL loading with duplicate-ID checks.
- Reproducible A-only, B-only, fixed-budget, and expanded-budget construction.
- LoRA causal-LM SFT with Hugging Face Transformers, Datasets, and PEFT.
- Accelerate/DeepSpeed-ready configuration for multi-GPU training.
- Arbitrary installed lm-eval task names, without assuming a benchmark exists.
- A lightweight JSONL QA scorer (EM, normalized EM, optional token F1).
- Result aggregation across seeds, per-benchmark output, scale-persistence flags, and basic plots.

Advanced features are explicit TODOs: tokenizer-aware token-budget sampling, sequential A→B/B→A training, mixture-ratio curves, and pair-level heatmaps.

## Hardware

The intended pilot platform is 8 × A100 80GB. The checked-in defaults use bf16, gradient checkpointing, LoRA, per-device batch size 1, gradient accumulation 8, and DeepSpeed ZeRO-2. They are conservative starting points, not universal optima. Effective global batch size is:

```text
per_device_batch × gradient_accumulation × number_of_processes
```

The 1.5B model can be tested on fewer GPUs by lowering `NUM_PROCESSES`. ZeRO-3 is available in `configs/deepspeed/zero3.json`. The model configs default to PyTorch SDPA so `flash-attn` is not required; switch `attn_implementation` only after installing and validating FlashAttention on the cluster.

## Installation

Use Python 3.10+ and install a CUDA-compatible PyTorch build for the cluster first. Then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[train,eval,analysis,dev]"
accelerate config
pytest
```

Editable installation is important because both `python -m ...` commands and scripts import the `data_interaction` package.

## Data format

Each UTF-8 JSONL line must contain `id`, `instruction`, `output`, and `data_group`. `input` and `metadata` are optional.

```json
{
  "id": "forward_000001",
  "instruction": "Solve the problem.",
  "input": "A shop sold 12 of 20 items. How many remain?",
  "output": "20 - 12 = 8. The answer is 8.",
  "data_group": "forward_reasoning",
  "metadata": {
    "source": "s1k",
    "pair_id": "problem_000001",
    "task_type": "math_reasoning"
  }
}
```

IDs must be unique within each source file. Pair IDs are preserved but are not yet used for pair-constrained sampling. Validate inputs with:

```bash
python -m data_interaction.data.validate_data data/raw/forward.jsonl data/raw/reverse.jsonl
```

## Prepare A/B mixtures

The sources must each contain at least the largest requested scale. Fixed mix uses `N` total examples (`floor(N/2)` from A and the remainder from B). Expanded mix uses `N` from each group (`2N` total).

```bash
python scripts/prepare_forward_reverse_data.py \
  --a_path data/raw/forward.jsonl \
  --b_path data/raw/reverse.jsonl \
  --out_dir data/processed/forward_reverse \
  --scales 1000 5000 10000 \
  --seed 42
```

Each `scale_N/` directory receives `a_only.jsonl`, `b_only.jsonl`, `fixed_mix.jsonl`, and `expanded_mix.jsonl`. `manifest.json` records paths, counts, scales, seed, and sampling mode.

## Launch one LoRA run

```bash
accelerate launch --num_processes 8 \
  src/data_interaction/train/train_lora_sft.py \
  --config configs/experiments/forward_reverse_scale_qwen.yaml \
  --model_config configs/models/qwen25_7b.yaml \
  --lora_config configs/lora/default_lora.yaml \
  --train_data data/processed/forward_reverse/scale_1000/expanded_mix.jsonl \
  --output_dir outputs/qwen25_7b/forward_reverse/scale_1000/expanded_mix
```

The output directory contains the LoRA adapter/tokenizer, Trainer checkpoints, and `run_metadata.json` with the exact command, merged config, and Python version. Resume with `--resume_from_checkpoint PATH`.

The equivalent shell helper is:

```bash
NUM_PROCESSES=8 scripts/launch_single_run.sh \
  configs/models/qwen25_7b.yaml \
  data/processed/forward_reverse/scale_1000/expanded_mix.jsonl \
  outputs/qwen25_7b/forward_reverse/scale_1000/expanded_mix \
  configs/experiments/forward_reverse_scale_qwen.yaml
```

## Launch the full scale matrix

After preparing data, this runs 2 models × 3 scales × 4 settings sequentially, with each training job distributed over the configured processes:

```bash
NUM_PROCESSES=8 scripts/launch_scale_experiment.sh
```

Run the untrained base model as a separate evaluation baseline. Sequential ordering is deliberately outside the minimal matrix.

## Evaluation

Task availability and exact names depend on the installed lm-eval version. Pass any names supported by that environment; do not assume that `math500`, `aime`, or `gpqa` aliases exist. Start with a small `--limit` smoke test.

```bash
python -m data_interaction.eval.run_eval \
  --checkpoint outputs/qwen25_7b/forward_reverse/scale_1000/expanded_mix \
  --tasks gsm8k \
  --scale 1000 \
  --setting expanded_mix \
  --model_name Qwen2.5-7B-Instruct \
  --train_data_path data/processed/forward_reverse/scale_1000/expanded_mix.jsonl \
  --output_json results/qwen25_7b_forward_reverse_1k_expanded_mix.json
```

Evaluate the untrained model with `--setting baseline` and the same `pair_name`, `model_name`, and scale. A single baseline result may be reused across scales by the aggregator, although writing one per scale makes provenance clearer. Repeat with all benchmark task names validated in the cluster's lm-eval install.

For already-generated QA answers:

```bash
python -m data_interaction.eval.custom_qa_eval \
  --input_jsonl predictions.jsonl --output_json qa_scores.json --f1
```

Each input row needs `question`, `gold_answer`, and `model_output`; `question` is carried for provenance and is not scored.

## Aggregate results

```bash
python -m data_interaction.metrics.aggregate_results \
  --results_dir results \
  --output_csv analysis/interaction_summary.csv
```

This produces:

- `interaction_summary.csv`: forward/reverse/mix means, gaps, gammas, and interaction labels.
- `interaction_summary_scale_trend.csv`: one row per mixture and scale, including sign-persistence flags.
- `interaction_summary_benchmark_breakdown.csv`: dynamic per-task columns plus the average.

Scores from repeated seeds are averaged inside each pair/model/scale/setting cell. If a baseline is missing, gaps are still computed, while gamma and causal interaction labels are left unavailable instead of silently assuming `S0 = 0`.

## Tables and plots

```bash
python -m data_interaction.analysis.make_tables \
  --summary_csv analysis/interaction_summary.csv \
  --out_dir paper_tables

python -m data_interaction.analysis.plot_scale_trends \
  --summary_csv analysis/interaction_summary.csv \
  --out_dir analysis/plots
```

The plot command creates a scale-trend line plot for `gap_vs_best` and a grouped bar plot for the four training settings.

## Add another data pair

Copy `configs/experiments/common_pairs_template.yaml`, change `pair_name` and the A/B paths, then run the same preparation, training, evaluation, and aggregation commands. Suitable studies include concise QA vs long CoT, refusal/safety vs helpful chat, math vs general instruction, and coding vs tool-use. Keep the data schema and result labels unchanged so different pairs remain comparable.

Before interpreting an interaction, verify that the baseline, all single/mixed runs, benchmark set, prompt format, seeds, and training hyperparameters match. Expanded mix changes both composition and total optimization examples; compare it separately from fixed mix.

## Minimal-version boundaries

This repository intentionally does not yet implement token-budget matching, paired sampling by `metadata.pair_id`, sequential transfer, mixture-ratio sweeps, confidence intervals/significance tests, or multi-pair heatmaps. The relevant modules contain explicit TODO exits so unsupported analyses cannot be mistaken for completed ones.

