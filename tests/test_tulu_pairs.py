import copy
import pytest
from data_interaction.metrics.tulu_pairs import SETTINGS, pair_contrasts, validate_records


def test_pair_contrast_uses_correct_slots():
    values = dict.fromkeys(SETTINGS, 0.0)
    values.update(baseline=0.2, a1=0.3, b2=0.4, ab=0.7, a2=0.9, b1=0.8)
    assert pair_contrasts(values)["ab"] == pytest.approx(0.2)


def test_missing_cell_fails():
    with pytest.raises(ValueError):
        pair_contrasts({"baseline": 0.1})


def test_refuse_mixed_protocol_or_examples():
    record = {"model_name": "m", "pair_name": "p", "scale": 4, "seed": 42,
              "evaluation_protocol": {"tasks": ["gsm8k"], "limit": 100, "batch_size": "1",
                  "max_gen_toks": 1024, "num_fewshot": 0, "chat_template": True, "lm_eval_version": "0.4.13",
                  "metric": "exact_match,flexible-extract"},
              "eval_results": {"gsm8k": .3}, "task_metrics": {"gsm8k": {"exact_match,flexible-extract": .3}},
              "samples": {"gsm8k": [{"doc_id": 0, "doc_hash": "a", "prompt_hash": "p", "filter": "flexible-extract"}]}}
    records = {setting: copy.deepcopy(record) for setting in SETTINGS}
    validate_records(records)
    records["a1"]["evaluation_protocol"]["max_gen_toks"] = 256
    with pytest.raises(ValueError):
        validate_records(records)
    records["a1"] = copy.deepcopy(record)
    records["a1"]["samples"]["gsm8k"][0]["doc_hash"] = "different"
    with pytest.raises(ValueError):
        validate_records(records)
