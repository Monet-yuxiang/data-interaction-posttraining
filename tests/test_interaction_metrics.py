import math

from data_interaction.metrics.interaction_metrics import compute_interaction_metrics, mixture_curvature, sign_is_persistent


def test_mutual_undermining_metrics() -> None:
    metrics = compute_interaction_metrics(baseline=0.4, a_only=0.6, b_only=0.55, mixed=0.5)
    assert math.isclose(metrics.gap_vs_best, -0.1)
    assert math.isclose(metrics.gamma, -0.25)
    assert metrics.mutual_undermining
    assert metrics.strong_mutual_undermining
    assert metrics.interaction_type == "strong_mutual_undermining"


def test_rescuing_complementarity() -> None:
    metrics = compute_interaction_metrics(baseline=0.5, a_only=0.4, b_only=0.45, mixed=0.6)
    assert metrics.rescuing_complementarity
    assert metrics.interaction_type == "rescuing_complementarity"


def test_sign_persistence() -> None:
    assert sign_is_persistent([-0.1, -0.2, -0.01])
    assert not sign_is_persistent([-0.1, 0.2])
    assert not sign_is_persistent([0.0, 0.2])
    assert not sign_is_persistent([-0.1])


def test_below_best_does_not_imply_nonadditivity() -> None:
    assert math.isclose(mixture_curvature(0.7, 0.6, 0.65), 0.0, abs_tol=1e-12)
    assert math.isclose(mixture_curvature(0.7, 0.6, 0.62), -0.03, abs_tol=1e-12)
