# Data interactions in LLM post-training

An experimental framework for measuring how SFT data sources interact in a shared, real-data mixture. The previous forward/reverse experiment is an exploratory pilot; it cannot establish prevalence.

**Start here: [实验操作手册](docs/JOBS.md).** It contains both the 1024-token re-evaluation of the existing checkpoints and the first real-data experiment. The datasets are downloaded on the server and are not part of this repository.

The first main experiment uses four preselected sources from the published [Tulu 3 SFT mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture), a common background, and two disjoint reference slots. Fifteen settings provide six pairwise replacement contrasts. Sources, model, background, budget, and evaluation determine the scope of the result; six pairs are not evidence of universal prevalence.

Implementation and tests have been checked locally on CPU. **The new Tulu download, packed training, and multi-task evaluation have not yet been verified end to end on the collaborator's GPU server.** Run the first real reference job and review its output before scheduling the rest. See the runbook for assumptions, costs, and resume instructions.

Run tests with `pytest -q`.
