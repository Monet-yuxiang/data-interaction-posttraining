"""Minimal LoRA supervised fine-tuning entry point."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from data_interaction.utils.config import deep_merge, load_yaml
from data_interaction.utils.logging import configure_logging
from data_interaction.utils.reproducibility import save_run_metadata, seed_everything

LOGGER = logging.getLogger(__name__)


def _require_training_packages() -> tuple[Any, ...]:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install with: pip install -e '.[train]'"
        ) from exc
    return (
        torch,
        load_dataset,
        LoraConfig,
        get_peft_model,
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )


def _detect_target_modules(model: Any) -> list[str]:
    common = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    found = sorted({name.rsplit(".", 1)[-1] for name, _ in model.named_modules()} & common)
    if not found:
        raise ValueError("could not auto-detect LoRA target modules; set lora.target_modules")
    return found


def _format_example(example: dict[str, Any], tokenizer: Any) -> str:
    prompt = example["instruction"]
    if example.get("input"):
        prompt = f"{prompt}\n\n{example['input']}"
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": example["output"]},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    eos = tokenizer.eos_token or ""
    return f"### Instruction:\n{prompt}\n\n### Response:\n{example['output']}{eos}"


def run_training(config: dict[str, Any], train_data: str, output_dir: str, resume: str | None = None) -> None:
    (
        torch,
        load_dataset,
        LoraConfig,
        get_peft_model,
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    ) = _require_training_packages()

    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    train_cfg = config.get("training", {})
    model_name = model_cfg.get("name_or_path")
    if not model_name:
        raise ValueError("model.name_or_path is required")

    seed = int(train_cfg.get("seed", config.get("seed", 42)))
    seed_everything(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        use_fast=bool(model_cfg.get("use_fast_tokenizer", True)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if train_cfg.get("bf16", True) else None
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
        "torch_dtype": dtype,
    }
    if model_cfg.get("attn_implementation"):
        model_kwargs["attn_implementation"] = model_cfg["attn_implementation"]
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if train_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    targets = lora_cfg.get("target_modules") or _detect_target_modules(model)
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 64)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 128)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        target_modules=targets,
        bias=str(lora_cfg.get("bias", "none")),
        task_type=str(lora_cfg.get("task_type", "CAUSAL_LM")),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    raw_dataset = load_dataset("json", data_files=train_data, split="train")
    max_length = int(train_cfg.get("max_seq_length", 2048))

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        texts = [
            _format_example({key: batch[key][index] for key in batch}, tokenizer)
            for index in range(len(batch["instruction"]))
        ]
        return tokenizer(texts, truncation=True, max_length=max_length)

    tokenized = raw_dataset.map(
        tokenize,
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing SFT data",
    )

    arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 2)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        bf16=bool(train_cfg.get("bf16", True)),
        tf32=bool(train_cfg.get("tf32", True)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        save_strategy=str(train_cfg.get("save_strategy", "epoch")),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        report_to=train_cfg.get("report_to", "none"),
        seed=seed,
        data_seed=seed,
        deepspeed=train_cfg.get("deepspeed"),
        ddp_find_unused_parameters=False,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=arguments, train_dataset=tokenized, data_collator=collator)
    LOGGER.info("Starting training on %d examples", len(tokenized))
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_config")
    parser.add_argument("--lora_config")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--deepspeed", help="Override training.deepspeed")
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    parts = [load_yaml(args.config)]
    if args.model_config:
        parts.append(load_yaml(args.model_config))
    if args.lora_config:
        parts.append(load_yaml(args.lora_config))
    config = deep_merge(*parts)
    if args.deepspeed:
        config.setdefault("training", {})["deepspeed"] = args.deepspeed
    config["train_data"] = str(Path(args.train_data))
    if int(os.environ.get("RANK", "0")) == 0:
        save_run_metadata(args.output_dir, config)
    run_training(config, args.train_data, args.output_dir, args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
