import json
import sys
import types
from pathlib import Path

from data_interaction.eval.run_eval import evaluate, main


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


def test_cli_records_generation_protocol_metrics_and_samples(tmp_path, monkeypatch):
    captured = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"results": {"gsm8k": {"exact_match,flexible-extract": 0.5, "other": 0.75}},
                "samples": {"gsm8k": [{"doc_id": 0, "exact_match": 1}]}}

    monkeypatch.setitem(sys.modules, "lm_eval", types.SimpleNamespace(evaluator=types.SimpleNamespace(simple_evaluate=fake_evaluate)))
    monkeypatch.setattr("data_interaction.eval.run_eval.version", lambda _: "0.4.13")
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["run_eval", "--checkpoint", "base", "--tasks", "gsm8k", "--output_json", str(output),
                                     "--max_gen_toks", "1024", "--metric", "exact_match,flexible-extract",
                                     "--scale", "500", "--setting", "baseline"])
    main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert captured["gen_kwargs"] == {"max_gen_toks": 1024}
    assert captured["log_samples"] is True
    assert result["evaluation_protocol"]["max_gen_toks"] == 1024
    assert result["evaluation_protocol"]["metric"] == "exact_match,flexible-extract"
    assert result["task_metrics"]["gsm8k"]["other"] == 0.75
    assert result["samples"]["gsm8k"][0]["doc_id"] == 0
