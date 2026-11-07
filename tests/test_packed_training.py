import pytest

from data_interaction.train.train_lora_sft import validate_packed_records


def row(ids, labels=None, mask=None):
    return {"input_ids": ids, "attention_mask": mask or [1] * len(ids), "labels": labels or ids}


def test_equal_length_packed_budget():
    result = validate_packed_records([row([1, 2], [-100, 2])] * 8, 2, 8, 1)
    assert result == {"blocks": 8, "input_tokens": 16, "supervised_tokens_after_shift": 8, "optimizer_steps": 1}


@pytest.mark.parametrize("rows", [
    [row([1, 2])] * 7,
    [row([1])] * 8,
    [row([1, 2], mask=[1, 0])] * 8,
    [row([1, 2], labels=[1, 5])] * 8,
    [row([1, 2], labels=[-100, -100])] * 8,
])
def test_reject_invalid_budgets(rows):
    with pytest.raises(ValueError):
        validate_packed_records(rows, 2, 8, 1)
