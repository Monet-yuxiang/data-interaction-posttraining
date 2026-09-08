from data_interaction.data.mix_units import build_mixture


def test_all_mixture_sizes_and_composition() -> None:
    a_items = [f"a{index}" for index in range(10)]
    b_items = [f"b{index}" for index in range(10)]

    assert len(build_mixture(a_items, b_items, 6, "a_only", 42)) == 6
    assert len(build_mixture(a_items, b_items, 6, "b_only", 42)) == 6

    fixed = build_mixture(a_items, b_items, 6, "fixed_mix", 42)
    assert len(fixed) == 6
    assert sum(value.startswith("a") for value in fixed) == 3
    assert sum(value.startswith("b") for value in fixed) == 3

    expanded = build_mixture(a_items, b_items, 6, "expanded_mix", 42)
    assert len(expanded) == 12
    assert sum(value.startswith("a") for value in expanded) == 6
    assert sum(value.startswith("b") for value in expanded) == 6


def test_sampling_is_reproducible() -> None:
    values = list(range(20))
    assert build_mixture(values, values, 8, "a_only", 7) == build_mixture(values, values, 8, "a_only", 7)


def test_insufficient_data_is_an_error() -> None:
    try:
        build_mixture([1], [2], 2, "a_only", 42)
    except ValueError as exc:
        assert "only 1" in str(exc)
    else:
        raise AssertionError("expected ValueError")

