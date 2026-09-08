"""Construct the four core A/B data settings."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeVar

from data_interaction.data.sample_units import sample_examples

T = TypeVar("T")

MODES = ("a_only", "b_only", "fixed_mix", "expanded_mix")


def build_mixture(a_items: Sequence[T], b_items: Sequence[T], scale: int, mode: str, seed: int) -> list[T]:
    if scale <= 0:
        raise ValueError("scale must be positive")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")

    if mode == "a_only":
        result = sample_examples(a_items, scale, seed)
    elif mode == "b_only":
        result = sample_examples(b_items, scale, seed + 1)
    elif mode == "fixed_mix":
        a_count = scale // 2
        b_count = scale - a_count
        result = sample_examples(a_items, a_count, seed) + sample_examples(b_items, b_count, seed + 1)
    else:
        result = sample_examples(a_items, scale, seed) + sample_examples(b_items, scale, seed + 1)

    random.Random(seed + 2).shuffle(result)
    return result

