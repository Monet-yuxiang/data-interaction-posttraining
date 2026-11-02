# Collaborator runbook: data and experiments

All commands below are run from the cloned repository directory containing `pyproject.toml`, on Linux with a CUDA GPU. Do not launch paid full runs before the smoke test passes. This is an exploratory minimal framework; the GPU training and lm-eval path has not yet been validated on the collaborator's server.

## 1. Data locations, provenance and limits

| Role | Source | Current count | Prepared file |
|---|---|---:|---|
| A, forward | [simplescaling/s1K-1.1](https://huggingface.co/datasets/simplescaling/s1K-1.1), pinned revision `96c411f1fe4c49d20f0e2a1565f61e1a28b0b84d`, DeepSeek trajectory and attempt | 1,000 | `data/raw/forward.jsonl` |
| B, reverse | [ReasonAlign authors' r1k.parquet](https://github.com/Demi-deng2/ReasonAlign-analysis/blob/702bab5910d44506f4f60d738b58512bab8650df/r1k.parquet), SHA256 `7ec23ed775b4a1752e20a8d058682f549fb04e529e79a46de1504ed4a7508f54` | 994 | `data/raw/reverse.jsonl` |
| C, background | [GSM8K](https://huggingface.co/datasets/openai/gsm8k), pinned revision `740312add88f781978c0658806c59bc2815b9866`, **train split only** | default 900 | `data/raw/background.jsonl` |
| R, reference | deterministic generated addition questions | default 2,700 | `data/raw/reference.jsonl` |

Data preparation downloads the pinned s1K-1.1 and r1k sources, converts them to the project schema, and records hashes in `data/raw/source_manifest.json`. The scripts do not require a paid API or generate reverse answers. The source files are **not included in Git**; access to GitHub and Hugging Face is required on the server. Review the source licences and the [r1k paper](https://aclanthology.org/2025.findings-emnlp.346/) before reuse or redistribution.

**Important limitations:**

- The r1k `original_index` has not been verified against this s1K-1.1 revision. Do not call these rows aligned pairs. Generated answers have not been independently verified.
- s1K-1.1 includes AIME-derived and GPQA-derived entries. AIME and GPQA are therefore unsuitable as independent evaluations of these runs; MATH-500 and other benchmarks also need a proper overlap audit. Even GSM8K test requires an overlap check against all A/B sources before publication. Source labels and exact matching do not prove absence of semantic contamination.
- The public B source has 994 examples. The original 1k/5k/10k matrix cannot run from these sources. Start with 500; 900 is possible. Do not duplicate records to manufacture 5k/10k.
- A/B outputs are often long. A local check of the **first 100** rows at 2048 tokens found 99/100 forward and 83/100 reverse examples would be truncated (not a random sample or full corpus audit). This can remove final answers. Run the token audit below, choose and record a feasible sequence length, and assess remaining truncation before any scientific run. Adjust `training.max_seq_length` in `configs/experiments/public_pilot.yaml`; GPU memory and cost increase with length. Full-run scripts require `ACK_TRUNCATION=1` after review. The current trainer learns on prompt **and** response tokens; this is not response-only SFT.
- The factorial reference uses short synthetic addition data. Its topic and length differ greatly from A/B/C; counts are matched but tokens, information content, and training signal are not. Treat it as a pipeline/phenomenon pilot, not a controlled mechanism test. The token-budget sampler, matched controls, gradient/Hessian interventions and statistical confidence intervals remain future work.

## 2. Install and check the machine

Use Python 3.10 or newer in a fresh environment. Install a CUDA-compatible PyTorch for the machine first, following the [official installation instructions](https://pytorch.org/get-started/locally/), then:

```bash
python -m venv .venv
source .venv/bin/activate
# Install the appropriate PyTorch build inside this environment first.
pip install -e '.[data,train,eval,analysis,dev]'
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)'
pytest -q
```

The optional `train` extra installs DeepSpeed; it is disabled in the single-GPU pilot config. If DeepSpeed installation fails on the server, install the training dependencies separately without DeepSpeed. Record exact Python, torch/CUDA, transformers, PEFT, accelerate, datasets and lm-eval versions. The repository's lower-bounded dependencies are not a frozen environment.

Download and validate:

```bash
python -m data_interaction.data.prepare_public_pilot
python -m data_interaction.data.validate_data data/raw/forward.jsonl data/raw/reverse.jsonl
python -m data_interaction.data.prepare_background --count 900
python -m data_interaction.data.token_audit data/raw/forward.jsonl data/raw/reverse.jsonl --limit 100 --max-length 2048
```

The converter refuses to overwrite converted files unless `--force` is passed. If GitHub downloads are restricted, download the pinned `r1k.parquet` from the URL in the source manifest table and pass `--r1k-file /path/to/r1k.parquet`; its checksum is verified. If the Hugging Face source is inaccessible, resolve server connectivity/cache first. The audit reports truncation of full formatted examples, not token-matched sampling.

## 3. Two-source pilot

First check the lm-eval task name available in the installed version (`lm-eval ls tasks` or `lm_eval --tasks list`, depending on version). GSM8K is the default and is **only a pipeline check until contamination is audited**. The model is `Qwen/Qwen2.5-1.5B-Instruct`. Check the model licence, GPU support for bf16, disk space, and cluster/network policy.

Run a small, explicitly non-scientific smoke test (4 training runs, 1 baseline evaluation, 20 examples, 2 optimizer steps, 10 evaluation examples):

```bash
SCALES='20' SEEDS='42' MAX_STEPS=2 EVAL_LIMIT=10 NUM_PROCESSES=1 bash scripts/run_public_pilot.sh
```

After checking logs, output files and truncation, a full exploratory run at one scale with three seeds is:

```bash
ACK_TRUNCATION=1 SCALES='500' SEEDS='42 43 44' NUM_PROCESSES=1 bash scripts/run_public_pilot.sh
```

For 900, set `SCALES='900'`; to run both use `SCALES='500 900'`. Each seed/scale runs four training conditions (A, B, fixed mix, expanded mix), plus evaluation of the base model. Thus 500 × three seeds means **12 training runs**, plus 15 evaluations. Estimating GPU-hours requires timing the smoke run; no training-time estimate is implied. Overrides: `MODEL`, `MODEL_CONFIG`, `TASK`, `NUM_PROCESSES`. Keep `MODEL` consistent with the model config. Scripts write to `data/processed/public_pilot/`, `outputs/public_pilot/`, `results/public_pilot/`, and `analysis/public_pilot_summary*.csv`. A rerun using the same paths can overwrite results; archive them or use a clean workspace first. Do not combine smoke and full results under the same seed/scale paths.

## 4. Three-source background pilot

This is a **fixed-example replacement design**, not a fixed-token experiment. Every condition has three slots of `N` examples. Slot A contains either A or reference R, slot B contains B or R, slot C contains C or R. Reused slots are identical across conditions for the same seed. `000` is R/R/R, **not** the unfine-tuned base model. Bit order is A/B/C:

```text
000  100  010  001  110  101  011  111
```

Minimal pipeline smoke test: eight training runs and eight evaluations.

```bash
N=10 SEEDS='42' MAX_STEPS=2 EVAL_LIMIT=10 NUM_PROCESSES=1 bash scripts/run_factorial_pilot.sh
```

Exploratory pilot after the truncation review: `ACK_TRUNCATION=1 N=50 SEEDS='42 43 44' NUM_PROCESSES=1 bash scripts/run_factorial_pilot.sh` (24 trainings and evaluations). Use a new `N` or archive old outputs when re-running: the builder refuses to overwrite an existing manifest. Source size must satisfy `3*N <= reference count` and `N <= each A/B/C count`.

The per-seed, per-task output `analysis/factorial_by_seed_and_task.csv` reports `pair_ab`, `pair_ab_given_c`, their difference `triple`, and pairwise prediction residual for `111`. The two are algebraically identical for these eight cells. This single triple is a case study, **not** proof of general prediction ability. A genuinely held-out evaluation across many combinations and statistical uncertainty require more sources/cells and additional work.

## 5. Interpretation and planned follow-up

For a fixed-budget 50/50 mixture, report `S_mix - (S_A+S_B)/2` (for odd N, use actual proportions) and `S_mix - max(S_A,S_B)` separately. Below-best performance is not automatically non-additivity. The expanded contrast `S_AB-S_A-S_B+S_0` changes data quantity and often optimizer steps; do not claim a pure mechanism or causal effect from it. Aggregate results by seed **and** task before any average, and report variation. The current legacy summary retains some expanded-budget heuristic labels for compatibility only.

Next research work, **not yet implemented**: tokenizer-aware and optimizer-step matched budgets; source identity and benchmark contamination audit, including fuzzy/manual review; verified A/B pairing and disjoint splits; separate C backgrounds and matched references; checkpoint loss curves and evaluation uncertainty; more source combinations for held-out prediction; training dynamics/gradient-curvature interventions; preregistered theory hypotheses and bounds. Sequential A→B/B→A, ratio sweeps and 5k/10k are also not ready. Do not interpret absence of a statistically significant effect as proof of no interaction.
