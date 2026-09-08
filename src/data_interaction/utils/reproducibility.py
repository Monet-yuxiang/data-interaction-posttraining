"""Record commands/configuration and initialize random generators."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_run_metadata(output_dir: str | Path, config: dict[str, Any]) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "run_metadata.json"
    payload = {
        "command": [sys.executable, *sys.argv],
        "config": config,
        "python": sys.version,
    }
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination

