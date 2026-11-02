"""Research-specific interaction metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isclose
from typing import Iterable


@dataclass(frozen=True)
class InteractionMetrics:
    gap_vs_best: float
    gamma: float
    mutual_undermining: bool
    strong_mutual_undermining: bool
    rescuing_complementarity: bool
    interaction_type: str

    def to_dict(self) -> dict[str, float | bool | str]:
        return asdict(self)


def compute_interaction_metrics(baseline: float, a_only: float, b_only: float, mixed: float) -> InteractionMetrics:
    """Legacy expanded-exposure contrast; not a causal mechanism estimate.

    Do not apply this contrast to fixed-total mixtures, which change dose.
    """
    gap = mixed - max(a_only, b_only)
    gamma = mixed - a_only - b_only + baseline
    strong = a_only > baseline and b_only > baseline and mixed < min(a_only, b_only)
    mutual = a_only > baseline and b_only > baseline and mixed < max(a_only, b_only)
    rescue = a_only < baseline and b_only < baseline and mixed > baseline
    if strong:
        label = "strong_mutual_undermining"
    elif mutual:
        label = "mutual_undermining"
    elif rescue:
        label = "rescuing_complementarity"
    elif gamma > 0:
        label = "positive_interaction"
    elif gamma < 0:
        label = "negative_interaction"
    else:
        label = "additive"
    return InteractionMetrics(gap, gamma, mutual, strong, rescue, label)


def mixture_curvature(a_only: float, b_only: float, mixed: float, a_weight: float = 0.5) -> float:
    """Deviation from the chosen affine single-source reference (higher-is-better)."""
    if not 0 <= a_weight <= 1:
        raise ValueError("a_weight must be between 0 and 1")
    return mixed - (a_weight * a_only + (1 - a_weight) * b_only)


def sign_is_persistent(values: Iterable[float]) -> bool:
    """True when all non-zero values have one sign and no value is zero."""
    signs = []
    for value in values:
        if isclose(value, 0.0, abs_tol=1e-12):
            signs.append(0)
        else:
            signs.append(1 if value > 0 else -1)
    return len(signs) >= 2 and 0 not in signs and len(set(signs)) == 1
