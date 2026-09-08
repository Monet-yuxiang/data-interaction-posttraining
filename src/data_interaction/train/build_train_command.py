"""Print an Accelerate command for one training run."""

from __future__ import annotations

import argparse
import shlex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--lora_config", default="configs/lora/default_lora.yaml")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument("--deepspeed")
    args = parser.parse_args()
    command = [
        "accelerate", "launch", "--num_processes", str(args.num_processes),
        "src/data_interaction/train/train_lora_sft.py",
        "--config", args.config,
        "--model_config", args.model_config,
        "--lora_config", args.lora_config,
        "--train_data", args.train_data,
        "--output_dir", args.output_dir,
    ]
    if args.deepspeed:
        command += ["--deepspeed", args.deepspeed]
    print(shlex.join(command))


if __name__ == "__main__":
    main()

