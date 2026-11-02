import json

from data_interaction.train.train_lora_sft import latest_complete_checkpoint


def test_resume_skips_an_incomplete_later_checkpoint(tmp_path):
    valid = tmp_path / "checkpoint-20"
    valid.mkdir()
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_config.json", "adapter_model.safetensors"):
        (valid / name).write_text("test")
    (valid / "trainer_state.json").write_text(json.dumps({"global_step": 20}))
    incomplete = tmp_path / "checkpoint-40"
    incomplete.mkdir()
    assert latest_complete_checkpoint(tmp_path) == str(valid)
