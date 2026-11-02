from data_interaction.data.prepare_public_pilot import convert_forward, convert_reverse
from data_interaction.metrics.aggregate_results import aggregate, aggregate_by_seed_task


def test_source_columns_convert_without_asserting_pair_identity() -> None:
    a = convert_forward({"question": "q", "deepseek_thinking_trajectory": "think", "deepseek_attempt": "answer"}, 2)
    b = convert_reverse({"question": "reverse q", "thinking_trajectories": "think back", "solution": "reverse answer", "original_index": 2}, 3)
    assert a.output == "think\n\nanswer"
    assert b.output == "think back\n\nreverse answer"
    assert "pair_id" not in a.metadata and "pair_id" not in b.metadata


def test_fixed_budget_gamma_is_not_reported() -> None:
    scores = {"baseline": 0.5, "a_only": 0.7, "b_only": 0.6, "fixed_mix": 0.65, "expanded_mix": 0.6}
    records = [{"setting": setting, "pair_name": "x", "model_name": "m", "scale": 20,
                "seed": 42, "eval_results": {"task": score}, "average_score": score}
               for setting, score in scores.items()]
    summary = aggregate(records)[0][0]
    assert summary["fixed_gamma"] is None
    assert abs(summary["fixed_affine_deviation"]) < 1e-12
    assert abs(aggregate_by_seed_task(records)[0]["fixed_affine_deviation"]) < 1e-12
