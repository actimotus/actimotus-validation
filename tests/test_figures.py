import altair as alt
import numpy as np
import pandas as pd

from actimotus_validation.figures import get_confusion_matrix
from actimotus_validation.metrics import get_scores, summarize_values

LABELS = ["sit", "walk"]


def _averaged(df: pd.DataFrame, labels: list[str] = LABELS) -> pd.DataFrame:
    scores = get_scores(df, "ground_truth", "activity", "id", labels)
    return summarize_values(scores, ["true", "pred"])


def _one(subject: str, true: list[str], pred: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"ground_truth": true, "activity": pred, "id": subject})


def test_returns_a_layered_chart():
    matrix = _averaged(_one("a", ["sit", "sit", "walk"], ["sit", "walk", "walk"]))
    assert isinstance(get_confusion_matrix(matrix, LABELS, title="T"), alt.LayerChart)


def test_rows_are_normalized_over_true_class():
    matrix = _averaged(_one("a", ["sit"] * 4, ["sit", "sit", "sit", "walk"]))
    chart = get_confusion_matrix(matrix, LABELS, title="T")
    values = chart.data.set_index(["True", "Predicted"])["Value"]
    assert values[("Sit", "Sit")] == 0.75
    assert values[("Sit", "Walk")] == 0.25


def test_draws_the_average_over_people_not_the_pooled_seconds():
    """The reason this function changed. Pooled here is 0.98; averaged is 0.75."""
    df = pd.concat([
        _one("a", ["sit"] * 100, ["sit"] * 100),
        _one("b", ["sit"] * 4, ["sit", "sit", "walk", "walk"]),
    ])
    chart = get_confusion_matrix(_averaged(df), LABELS, title="T")
    values = chart.data.set_index(["True", "Predicted"])["Value"]
    assert values[("Sit", "Sit")] == 0.75


def test_each_cell_carries_its_own_interval():
    df = pd.concat([
        _one("a", ["sit"] * 4, ["sit", "sit", "sit", "walk"]),
        _one("b", ["sit"] * 4, ["sit", "sit", "walk", "walk"]),
        _one("c", ["sit"] * 4, ["sit", "walk", "walk", "walk"]),
    ])
    chart = get_confusion_matrix(_averaged(df), LABELS, title="T")
    row = chart.data.set_index(["True", "Predicted"]).loc[("Sit", "Sit")]
    assert row["Lower"] < row["Value"] < row["Upper"]


def test_a_behaviour_nobody_performed_draws_blank_not_nan():
    """Shuffle and stairs are empty rows in the laboratory panels. They stay blank."""
    chart = get_confusion_matrix(_averaged(_one("a", ["sit"], ["sit"])), LABELS, "T")
    walk = chart.data.set_index(["True", "Predicted"]).loc[("Walk", "Sit")]
    assert walk["Value"] == 0.0
    assert not np.isnan(walk["Value"])


def test_labels_are_capitalised_for_display():
    matrix = _averaged(_one("a", ["sit"], ["sit"]))
    chart = get_confusion_matrix(matrix, LABELS, title="T")
    assert set(chart.data["True"]) == {"Sit", "Walk"}


def test_renders_to_png(tmp_path):
    matrix = _averaged(_one("a", ["sit", "walk"], ["sit", "walk"]))
    chart = get_confusion_matrix(matrix, LABELS, title="T")
    out = tmp_path / "cm.png"
    chart.save(str(out), scale_factor=1)
    assert out.stat().st_size > 0


def test_a_dataset_with_none_of_its_reported_classes_draws_an_empty_grid():
    """`summarize_values` returns a (0, 0) frame when every group is skipped, and
    merging on columns that do not exist raised KeyError rather than drawing.
    """
    matrix = _averaged(_one("a", ["cartwheel"], ["cartwheel"]))
    chart = get_confusion_matrix(matrix, LABELS, title="T")
    assert isinstance(chart, alt.LayerChart)
    assert (chart.data["Value"] == 0.0).all()
