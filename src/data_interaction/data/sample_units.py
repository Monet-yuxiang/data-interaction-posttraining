"""Reproducible example-count sampling."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def sample_examples(examples: Sequence[T], count: int, seed: int) -> list[T]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if count > len(examples):
        raise ValueError(f"requested {count} examples but only {len(examples)} are available")
    return random.Random(seed).sample(list(examples), count)


def sample_by_token_budget(*args: object, **kwargs: object) -> list[object]:
    """Reserved for tokenizer-aware sampling in a later version."""
    raise NotImplementedError("token-budget sampling is a TODO; use example-count mode")

