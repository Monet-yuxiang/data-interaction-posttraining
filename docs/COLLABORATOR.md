# Public data and scientific limits

For the current resumable training and evaluation commands, use [JOBS.md](JOBS.md). Commands are run from the repository directory containing `pyproject.toml`.

| Role | Source | Available | Prepared file |
|---|---|---:|---|
| A, forward | [simplescaling/s1K-1.1](https://huggingface.co/datasets/simplescaling/s1K-1.1), revision `96c411f1fe4c49d20f0e2a1565f61e1a28b0b84d`, DeepSeek trajectory and attempt | 1,000 | `data/raw/forward.jsonl` |
| B, reverse | [authors' r1k.parquet](https://github.com/Demi-deng2/ReasonAlign-analysis/blob/702bab5910d44506f4f60d738b58512bab8650df/r1k.parquet), SHA256 `7ec23ed775b4a1752e20a8d058682f549fb04e529e79a46de1504ed4a7508f54` | 994 | `data/raw/reverse.jsonl` |

These public sources are downloaded on the server, not committed to this repository. Review the source licences and the [r1k paper](https://aclanthology.org/2025.findings-emnlp.346/) before reuse or redistribution. Install the dependencies described in [JOBS.md](JOBS.md), then:

```bash
python -m data_interaction.data.prepare_public_pilot
python -m data_interaction.data.validate_data data/raw/forward.jsonl data/raw/reverse.jsonl
```

The converter refuses to overwrite output unless `--force` is supplied. If needed, manually obtain the pinned r1k parquet and pass `--r1k-file /path/to/r1k.parquet`; its checksum is verified. `source_manifest.json` records provenance and output hashes.

Important limits:

- The r1k `original_index` has not been verified against this s1K-1.1 revision; do not describe the two files as verified pairs. Generated answers have not been independently validated.
- s1K-1.1 includes AIME-derived and GPQA-derived entries. AIME and GPQA cannot be treated as independent evaluations for this training set. MATH-500, GSM8K and any other benchmark also require overlap analysis, including semantic matches.
- B has 994 examples. The old 1k/5k/10k settings cannot be run using these files. The current pilot uses 500 or fewer. Do not duplicate records to manufacture larger datasets.
- The old 2048-token truncating trainer discarded parts of many long responses. The current trainer preserves target tokens in windows, but later windows do not see the entire prior reasoning. Long examples yield more windows; counts, tokens and compute therefore differ across groups. This is a new exploratory protocol, not equivalent to the original paper or a matched-token experiment.
- The old `run_public_pilot.sh` and `run_factorial_pilot.sh` monolithic runners are disabled. The three-source builder still exists as a pilot, but there is no validated resumable three-source job workflow. Focus on two-source validation first.
- The original smoke test used different training and evaluation formats and must never be merged with new results. Confidence intervals, paired data verification, matched-token budgets, sequential order, and mechanistic interventions are not yet implemented.
