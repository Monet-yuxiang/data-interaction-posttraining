# Good Data Does Not Always Add Up

An exploratory framework for studying interactions between data sources in LLM supervised fine-tuning. It provides reproducible public data preparation, LoRA SFT, evaluation, two-source mixture comparisons and a three-source replacement-design pilot.

## Start here

**[Collaborator runbook](docs/COLLABORATOR.md)** contains the actual download sources and checksums, installation, token audit, smoke-test commands, two-source and three-source runs, hardware caveats and interpretation limits. Commands are run from the directory containing `pyproject.toml`.

The public forward source is [s1K-1.1](https://huggingface.co/datasets/simplescaling/s1K-1.1) (1,000 rows); the reverse source is the [r1k dataset released by the authors](https://github.com/Demi-deng2/ReasonAlign-analysis/blob/702bab5910d44506f4f60d738b58512bab8650df/r1k.parquet) (994 rows). **The data is downloaded by the preparation command; it is not stored in this repository.** Their pairing and benchmark decontamination have not been verified. Do not assume 1k/5k/10k can be drawn from these sources.

On a Linux CUDA machine with a fresh environment and a suitable PyTorch build:

```bash
pip install -e '.[data,train,eval,analysis,dev]'
python -m data_interaction.data.prepare_public_pilot
python -m data_interaction.data.validate_data data/raw/forward.jsonl data/raw/reverse.jsonl
python -m data_interaction.data.token_audit data/raw/forward.jsonl data/raw/reverse.jsonl --limit 100 --max-length 2048
SCALES='20' SEEDS='42' MAX_STEPS=2 EVAL_LIMIT=10 NUM_PROCESSES=1 bash scripts/run_public_pilot.sh
```

**Read the runbook before full runs.** In a local check, 99/100 forward and 83/100 reverse examples from the first 100 rows exceeded 2048 tokens. A shortened smoke test is for verifying the pipeline only. The scripts require an explicit truncation acknowledgment before an unrestricted run.

## Experiment status

- The A/B pilot compares single-source, fixed-total-example and expanded-example mixtures. The fixed mixture's deviation from an affine single-source reference and its gap to the better single source are different quantities. The expanded contrast changes exposure and compute.
- The three-source pilot uses fixed **example** slots and a synthetic reference. Its `000` condition is reference-trained, not the untrained model. It supports within-seed, within-task pairwise and third-order replacement contrasts; it is not token matched and does not prove mechanism or generalization.
- The original 1k/5k/10k configuration, multi-GPU scripts and advanced TODO modules remain as legacy templates. They are not the public-data runbook.
- Token-matched sampling, benchmark decontamination, verified paired sampling, confidence intervals, sequential training and mechanism interventions are not implemented. Consult the runbook before using results in research.

Unit tests: `pytest -q`. The code has been exercised with real downloaded data and a one-step CPU run on a random tiny model. The CUDA training and lm-eval execution require a smoke test on the target server.
