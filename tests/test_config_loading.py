from pathlib import Path

from data_interaction.utils.config import deep_merge, load_yaml


def test_yaml_loading_and_deep_merge(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("training:\n  learning_rate: 0.001\nseed: 7\n", encoding="utf-8")
    loaded = load_yaml(path)
    merged = deep_merge(loaded, {"training": {"num_train_epochs": 2}})
    assert merged["training"] == {"learning_rate": 0.001, "num_train_epochs": 2}
    assert merged["seed"] == 7

