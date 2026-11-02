# Good Data Does Not Always Add Up

An exploratory framework for studying interactions between data sources in LLM supervised fine-tuning. It provides reproducible public data preparation, LoRA SFT, evaluation, two-source mixture comparisons and a three-source replacement-design pilot.

## Start here

**[Short-job runbook](docs/JOBS.md)** describes the new resumable two-source workflow, and [data sources](docs/COLLABORATOR.md) lists download sources and scientific limitations. Commands are run from the directory containing `pyproject.toml`.

The public forward source is [s1K-1.1](https://huggingface.co/datasets/simplescaling/s1K-1.1) (1,000 rows); the reverse source is the [r1k dataset released by the authors](https://github.com/Demi-deng2/ReasonAlign-analysis/blob/702bab5910d44506f4f60d738b58512bab8650df/r1k.parquet) (994 rows). **The data is downloaded by the preparation command; it is not stored in this repository.** Their pairing and benchmark decontamination have not been verified. Do not assume 1k/5k/10k can be drawn from these sources.

On a Linux CUDA machine with a fresh environment and a suitable PyTorch build:

```bash
pip install -e '.[data,train,eval,analysis,dev]'
python -m data_interaction.data.prepare_public_pilot
python -m data_interaction.data.validate_data data/raw/forward.jsonl data/raw/reverse.jsonl
python -m data_interaction.data.token_audit data/raw/forward.jsonl data/raw/reverse.jsonl --limit 100 --max-length 2048
# Continue with the windowed smoke test in docs/JOBS.md.
```

**Read the runbook before full runs.** Long assistant answers are now split into short windows so target tokens are not silently dropped; this is not equivalent to full-context training. The previous smoke scores used different training/evaluation formats and must not be mixed with new results.

## Experiment status

- The A/B pilot compares single-source, fixed-total-example and expanded-example mixtures. The fixed mixture's deviation from an affine single-source reference and its gap to the better single source are different quantities. The expanded contrast changes exposure and compute.
- The three-source builder remains exploratory; its previous monolithic runner is disabled until a resumable workflow is validated. Focus first on the two-source main experiment.
- The original 1k/5k/10k configuration, multi-GPU scripts and advanced TODO modules remain as legacy templates. They are not the public-data runbook.
- Token-matched sampling, benchmark decontamination, verified paired sampling, confidence intervals, sequential training and mechanism interventions are not implemented. Consult the runbook before using results in research.

Unit tests: `pytest -q`. The code has been exercised with real downloaded data and a one-step CPU run on a random tiny model. The CUDA training and lm-eval execution require a smoke test on the target server.
