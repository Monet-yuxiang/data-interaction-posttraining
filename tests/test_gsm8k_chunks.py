import json
import sys
import types
from pathlib import Path

import pytest

from data_interaction.eval.gsm8k_chunks import merge_chunks, run_chunk


def make_chunk(path: Path, indices, identity):
    path.write_text(json.dumps({"identity": identity, "indices": indices,
                    "rows": [{"doc_id": i, "exact_match": i % 2} for i in indices]}), encoding="utf-8")


def test_merge_checks_completeness_overlap_and_identity(tmp_path):
    identity = {"pair_name": "p", "model_name": "m", "scale": 1, "seed": 42,
                "setting": "baseline", "checkpoint": "m"}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    make_chunk(a, [0, 1], identity)
    make_chunk(b, [2], identity)
    result = merge_chunks([a, b], total=3)
    assert result["n_samples"] == 3
    assert result["eval_results"]["gsm8k"] == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="Incomplete"):
        merge_chunks([a], total=3)
    with pytest.raises(ValueError, match="Duplicate"):
        merge_chunks([a, a, b], total=3)
    make_chunk(b, [2], {**identity, "setting": "a_only"})
    with pytest.raises(ValueError, match="identity mismatch"):
        merge_chunks([a, b], total=3)


def test_chunk_uses_exact_ids_and_chat_template(tmp_path, monkeypatch):
    calls = []
    def evaluate(**kwargs):
        calls.append(kwargs)
        return {"samples": {"gsm8k": [{"doc_id": i, "filter": "flexible-extract", "exact_match": 1.0}
                                     for i in kwargs["samples"]["gsm8k"]]}}
    monkeypatch.setitem(sys.modules, "lm_eval", types.SimpleNamespace(evaluator=types.SimpleNamespace(simple_evaluate=evaluate)))
    monkeypatch.setattr("data_interaction.eval.gsm8k_chunks.version", lambda _: "0.4.13")
    args = types.SimpleNamespace(start=3, end=5, checkpoint="base", setting="baseline", model_name="base",
                                 pair_name="p", seed=42, scale=500, out_dir=tmp_path, device="cpu")
    run_chunk(args)
    run_chunk(args)
    assert len(calls) == 1
    assert calls[0]["samples"] == {"gsm8k": [3, 4]}
    assert calls[0]["apply_chat_template"] is True
    assert calls[0]["num_fewshot"] == 0
