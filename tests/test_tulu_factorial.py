import json

import pytest

from data_interaction.data.prepare_tulu_factorial import BACKGROUND, SETTINGS, SOURCES, prepare


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        if len(messages) == 1:
            return "PQ"
        return "PQ" + messages[1]["content"]

    def __call__(self, text, add_special_tokens, return_offsets_mapping):
        value = int(text[2:])
        return {"input_ids": [1, 2] + [value] * 6,
                "offset_mapping": [(0, 1), (1, 2)] + [(2, len(text))] * 6}


def source_rows():
    rows = []
    for source_index, source in enumerate([*SOURCES.values(), *BACKGROUND]):
        for i in range(100):
            rows.append({"id": f"{source_index}-{i}", "source": source,
                         "messages": [{"role": "user", "content": "question"},
                                      {"role": "assistant", "content": str(1000*source_index+i+10)}]})
    return rows


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_factorial_slots_match_and_reference_disjoint(tmp_path):
    assert SOURCES["r"] not in BACKGROUND
    assert len(set(SOURCES.values()) | set(BACKGROUND)) == len(SOURCES) + len(BACKGROUND)
    manifest = prepare(source_rows(), FakeTokenizer(), tmp_path / "out", 80, 8, 42, "fake")
    assert len(SETTINGS) == 15
    assert all(x["input_tokens"] == 640 for x in manifest["files"].values())
    data = {setting: read(tmp_path / "out" / f"{setting}.jsonl") for setting in SETTINGS}
    assert all(len(block["input_ids"]) == 8 and len(block["labels"]) == 8 for blocks in data.values() for block in blocks)
    # Source-coded tokens make it possible to compare the exact source blocks.
    a1 = {b["input_ids"][2] for b in data["a1"] if 10 <= b["input_ids"][2] < 1010}
    a2 = {b["input_ids"][2] for b in data["a2"] if 10 <= b["input_ids"][2] < 1010}
    assert a1 == a2 and len(a1) == 8
    reference1 = {b["input_ids"][2] for b in data["a2"] if 4010 <= b["input_ids"][2] < 5010}
    reference2 = {b["input_ids"][2] for b in data["a1"] if 4010 <= b["input_ids"][2] < 5010}
    assert len(reference1) == len(reference2) == 8
    assert reference1.isdisjoint(reference2)
    assert {b["input_ids"][2] for b in data["baseline"] if 4010 <= b["input_ids"][2] < 5010} == reference1 | reference2


def test_budget_validation_and_missing_source(tmp_path):
    with pytest.raises(ValueError, match="multiple of 80"):
        prepare([], FakeTokenizer(), tmp_path / "bad", 40, 8, 42, "fake")
    with pytest.raises(ValueError, match="Insufficient eligible data"):
        prepare([], FakeTokenizer(), tmp_path / "missing", 80, 8, 42, "fake")


def test_drops_blocks_without_supervised_target(tmp_path):
    class LongPromptTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            prefix = "P" * 10
            return prefix if len(messages) == 1 else prefix + messages[1]["content"]

        def __call__(self, text, add_special_tokens, return_offsets_mapping):
            return {"input_ids": [1] * 10 + [2] * 6,
                    "offset_mapping": [(i, i + 1) for i in range(10)] + [(10, len(text))] * 6}

    manifest = prepare(source_rows(), LongPromptTokenizer(), tmp_path / "out", 80, 8, 42, "fake")
    assert sum(manifest["discarded_all_masked_blocks"].values()) > 0
    for block in read(tmp_path / "out" / "baseline.jsonl"):
        assert any(value != -100 for value in block["labels"][1:])


def test_masks_token_that_crosses_prompt_response_boundary():
    from data_interaction.data.prepare_tulu_factorial import encode_pair

    class BoundaryTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return "prompt\n" if len(messages) == 1 else "prompt\n\nanswer"

        def __call__(self, text, add_special_tokens, return_offsets_mapping):
            return {"input_ids": [1, 2, 3], "offset_mapping": [(0, 6), (6, 8), (8, 14)]}

    row = {"messages": [{"role": "user", "content": "question"},
                        {"role": "assistant", "content": "\nanswer"}]}
    assert encode_pair(row, BoundaryTokenizer()) == ([1, 2, 3], [-100, -100, 3])
