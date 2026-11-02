import json
import sys
import types
from pathlib import Path

from data_interaction.eval.run_eval import evaluate


def test_adapter_evaluation_includes_base_model(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "example/base"}), encoding="utf-8")
    captured = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"results": {"gsm8k": {"exact_match,strict-match": 0.4}}}

    monkeypatch.setitem(sys.modules, "lm_eval", types.SimpleNamespace(evaluator=types.SimpleNamespace(simple_evaluate=fake_evaluate)))
    assert evaluate(str(tmp_path), ["gsm8k"], "auto", 10) == {"gsm8k": 0.4}
    assert "pretrained=example/base" in captured["model_args"]
    assert f"peft={tmp_path}" in captured["model_args"]
    assert captured["apply_chat_template"] is True
    assert captured["num_fewshot"] == 0
