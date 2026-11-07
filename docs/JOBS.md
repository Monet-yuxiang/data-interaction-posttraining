# 实验操作手册

本手册是同学执行实验的唯一入口。先做第 1 部分，使用已训练的模型重评；第 2 部分是新的真实数据主实验。所有命令在 Linux 服务器、包含 `pyproject.toml` 的项目目录下运行。训练和评测需要已申请到 GPU，并使用与之前相同的 CUDA/Python 环境。

## 1. 已有模型：延长回答后重新评测

旧评测没有显式设置生成上限，`lm-eval 0.4.13` 默认最多生成 256 token，许多微调后的回答可能未写完。旧数据不应删除，也不能与新结果合并。此次先评测基座和 fixed mix 的相同前 100 道 GSM8K 题，不重新训练。

在每个独立 GPU 任务内，先进入项目目录、激活原环境，然后从训练记录取得原基座模型：

```bash
export MODEL="$(python -c 'import json; print(json.load(open("outputs/windowed/seed_42/scale_500/a_only/run_signature.json", encoding="utf-8"))["config"]["model"]["name_or_path"])')"
export CHUNK_SIZE=100
export MAX_GEN_TOKS=1024
echo "MODEL=$MODEL; CHUNK_SIZE=$CHUNK_SIZE; MAX_GEN_TOKS=$MAX_GEN_TOKS"
```

确认 `MODEL` 与训练时的基座路径一致，`outputs/windowed/seed_42/scale_500/fixed_mix/TRAINING_COMPLETE.json` 存在。以下两条命令各为一个 GPU 任务：

```bash
bash scripts/run_gsm8k_chunk.sh baseline 0 500 42
bash scripts/run_gsm8k_chunk.sh fixed_mix 0 500 42
```

将以下文件发回项目负责人：

```text
results/windowed_chunks_gen_1024/seed_42/scale_500/baseline/chunk_0000_0100.json
results/windowed_chunks_gen_1024/seed_42/scale_500/fixed_mix/chunk_0000_0100.json
```

这两个文件已记录生成上限，保存到新目录。重跑同一任务会跳过已完成的相同结果；中断后当前 100 题需重跑。1024 是新的生成上限，不能保证所有回答自然结束。请同时保留运行日志和实际耗时。如果评测失败，把日志发回；不要重训或覆盖旧数据。`merge` 仅用于完整 1319 题，此处不需要。

## 2. 主实验：真实多来源 SFT 数据

**状态：已在本地用真实 Tulu 流式数据生成 15 组各 80 块，并验证每组输入 token 数与优化步数一致；完整 4800 块准备、目标 GPU 训练和多任务评测尚未在同学的服务器上端到端验证。先执行共同参照的一个正式任务并汇报，不要一次性提交全部任务。** 该任务正常完成后可作为正式实验的一部分。

### 实验问题与 15 组

来源 A、B、C、D 分别是 Tulu 3 混合集合中的 FLAN、Persona Math、Evol CodeAlpaca、WildChat；参考来源 R 是 Persona IF；背景为 SciRIFF 和 Aya。实际 source 名称和固定的上游版本记录在数据 manifest。仅使用单轮 user/assistant 对话；背景两个来源按块数均分，并不是 Tulu 原始配比。

共同背景占 80% 的输入 token，两个可替换槽各占 10%。两份参考数据 R1、R2 使用不同样本，避免对照重复训练同一批数据。固定相同 tokenizer、输入 token 数、优化步数和其他设置：

| 组名 | 第一个槽 | 第二个槽 |
|---|---|---|
| baseline | R1 | R2 |
| a1、b1、c1、d1 | A/B/C/D | R2 |
| a2、b2、c2、d2 | R1 | A/B/C/D |
| ab、ac、ad、bc、bd、cd | 字母序前一个来源 | 字母序后一个来源 |

例如 AB 的差分为 `score(ab) - score(a1) - score(b2) + score(baseline)`；它反映相对于此背景、参考数据、槽位、预算和评测任务的非加性效应。15 组比早期的 11 组建议多 4 组，是为了使用不重复的参考数据并正确匹配两槽对照。六对不代表所有真实数据对。不要只根据一组结果称为普遍规律或推断机制。

### 环境和数据准备

使用与服务器 CUDA 驱动匹配的 PyTorch，并安装本仓库：

```bash
pip install -e '.[data,train,eval,analysis,dev]'
```

保持同一 `MODEL` 路径贯穿数据准备、训练和评测。可沿用此前实际训练的 Qwen2.5-1.5B-Instruct 路径，例如：

```bash
export MODEL="$(python -c 'import json; print(json.load(open("outputs/windowed/seed_42/scale_500/a_only/run_signature.json", encoding="utf-8"))["config"]["model"]["name_or_path"])')"
export DATA_DIR=data/processed/tulu_factorial/seed_42
export OUT_DIR=outputs/tulu_factorial/seed_42
```

数据准备需要访问 Hugging Face 并下载公开数据。`4800` 个块，每块 2048 输入 token，即每组 9,830,400 输入 token；是第一档建议预算，不是预测的耗时。准备会生成 15 份文件，占用多份磁盘空间，请预留空间：

```bash
python -m data_interaction.data.prepare_tulu_factorial \
  --model "$MODEL" --blocks 4800 --seed 42 --out-dir "$DATA_DIR"
```

完成后保留 `$DATA_DIR/manifest.json`。如输出目录非空，准备程序会拒绝覆盖；若数据源不足、下载失败或报错，请把日志发回。准备使用流式缓冲区打乱，不是对完整数据集的完全均匀抽样；上游各来源许可证可能不同。

数据以 2048 token 打包，块内不同样本可以互相注意；长样本可跨块。提示与回答边界处若合并为一个 token，则将该 token 掩码以避免训练提示内容。所有组的输入 token 与优化步数相同，但各组实际用于 loss 的监督 token 可能不同，并记录在 manifest。因此不应称为完全相同的监督 token 或 GPU 时间。各来源与评测题目的重合尚未完整核查。

### 训练：每次一个 GPU，支持恢复

首先运行共同参照，记录耗时和显存：

```bash
bash scripts/run_tulu_training_job.sh baseline 42
```

确保 `MODEL`、`DATA_DIR`、`OUT_DIR` 在每个作业中都设置为同一值。可用 `TIME_LIMIT_SECONDS=4200` 设置软时间上限，但要为加载和保存留出额外时间；如任务中断，在相同环境、同一文件系统及相同参数下重新运行同一命令。已经完成的任务会跳过；不要同时运行同一设置的两个任务。

第一组确认正常后，其他设置各提交一次，最多并行使用服务器实际获准的 GPU 数，单个任务只分配 1 张卡：

```text
a1 b1 c1 d1 a2 b2 c2 d2 ab ac ad bc bd cd
```

例如单独提交 `bash scripts/run_tulu_training_job.sh ab 42`。输出是 `$OUT_DIR/ab/`，完成后包含 `TRAINING_COMPLETE.json`。具体耗时由首个任务实测；不能把旧 pilot 耗时直接用于估算。

### 评测与汇总

正式比较时，对所有设置使用相同题目、模型版本、评测设置及生成上限。下面是共同参照在 GSM8K 的示例；其他设置替换 `SETTING`，确保基座路径 `MODEL` 不变，每个任务输出到对应的唯一文件：

```bash
export SETTING=baseline
mkdir -p results/tulu_factorial/gsm8k/seed_42
python -m data_interaction.eval.run_eval \
  --checkpoint "$OUT_DIR/$SETTING" --model_name "$MODEL" \
  --tasks gsm8k --metric 'exact_match,flexible-extract' --max_gen_toks 1024 \
  --pair_name tulu_factorial --scale 4800 --seed 42 --setting "$SETTING" \
  --output_json "results/tulu_factorial/gsm8k/seed_42/$SETTING.json"
```

评测脚本保存完整指标、逐题输出、生成长度和协议；存在同名结果时拒绝覆盖。它目前**不能分批恢复**，中断后该任务需重跑。1024 token 的设定应一致，并检查回答是否到达上限。

为覆盖指令遵循能力，建议额外评测 IFEval。其任务配置和指标键已核对官方 `lm-eval 0.4.13`，但尚未在目标服务器实际运行。先安装任务依赖并确保能下载或缓存 `google/IFEval` 及 NLTK `punkt_tab`：

```bash
pip install 'lm-eval[ifeval]==0.4.13'
export SETTING=baseline
mkdir -p results/tulu_factorial/ifeval/seed_42
python -m data_interaction.eval.run_eval \
  --checkpoint "$OUT_DIR/$SETTING" --model_name "$MODEL" \
  --tasks ifeval --metric 'prompt_level_strict_acc,none' --max_gen_toks 1280 --batch_size 1 \
  --pair_name tulu_factorial --scale 4800 --seed 42 --setting "$SETTING" \
  --output_json "results/tulu_factorial/ifeval/seed_42/$SETTING.json"
```

如首组通过，再逐组使用相同设置；各任务分别汇总。数学和指令遵循也不能代表所有能力，代码能力的安全评测尚未集成。

15 组同一任务评测完成后：

```bash
python -m data_interaction.metrics.tulu_pairs \
  --results-dir results/tulu_factorial/gsm8k/seed_42 \
  --output-csv results/tulu_factorial/gsm8k/seed_42/pair_contrasts.csv
```

脚本检查各组的评测协议与题目是否一致。将 manifest、所有评测 JSON、汇总 CSV 和训练日志交给项目负责人；不要仅依据一次种子和少量来源宣布普遍性。更换模型或 token 预算时需要分别准备新目录、重跑 15 组，并保持同档内部协议一致。不同预算同时改变训练计算量，不能单独解释为数据量效应。
