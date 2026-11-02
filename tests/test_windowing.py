import pytest

from data_interaction.train.windowing import response_windows


class FakeTokenizer:
    chat_template = "fake"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        prefix = [1, 2, 3]
        return prefix if add_generation_prompt else prefix + list(range(10, 21)) + [99]


def test_all_response_tokens_once_and_prompt_masked():
    rows = response_windows(FakeTokenizer(), {"instruction": "question", "output": "answer"}, 7)
    assert all(len(row["input_ids"]) <= 7 for row in rows)
    assert all(row["labels"][:3] == [-100, -100, -100] for row in rows)
    assert [x for row in rows for x in row["labels"] if x != -100] == list(range(10, 21)) + [99]
    assert rows[1]["input_ids"][3] == 13
    assert rows[1]["labels"][3] == -100


def test_prompt_that_fills_window_is_rejected():
    with pytest.raises(ValueError, match="no room"):
        response_windows(FakeTokenizer(), {"instruction": "q", "output": "a"}, 3)
