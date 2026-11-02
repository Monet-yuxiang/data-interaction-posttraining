import math
from pathlib import Path

from data_interaction.data.build_factorial import build, SETTINGS
from data_interaction.data.load_jsonl import load_jsonl, write_jsonl
from data_interaction.data.schema import TrainingExample
from data_interaction.metrics.factorial import contrasts


def test_factorial_uses_identical_slots_and_counts(tmp_path: Path) -> None:
    paths = []
    for label, count in (("ref", 9), ("a", 3), ("b", 3), ("c", 3)):
        path = tmp_path / f"{label}.jsonl"
        write_jsonl([TrainingExample(id=f"{label}{i}", instruction="q", output="a", data_group=label)
                     for i in range(count)], path)
        paths.append(path)
    out = tmp_path / "out"
    build(paths[0], paths[1:], out, 3, 42)
    cells = {s: {x.id for x in load_jsonl(out / f"{s}.jsonl")} for s in SETTINGS}
    assert all(len(x) == 9 for x in cells.values())
    assert len(cells["000"] & cells["100"]) == 6
    assert cells["100"] & cells["111"] == cells["100"] - cells["000"]


def test_three_way_is_context_difference_and_prediction_residual() -> None:
    values = {"000": 0.5, "100": 0.6, "010": 0.55, "001": 0.52,
              "110": 0.7, "101": 0.62, "011": 0.57, "111": 0.8}
    result = contrasts(values)
    assert math.isclose(result["triple"], result["prediction_error"], abs_tol=1e-12)


def test_missing_cell_is_rejected() -> None:
    try:
        contrasts({"000": 0.0})
    except ValueError as exc:
        assert "missing settings" in str(exc)
    else:
        raise AssertionError("missing cell was accepted")
