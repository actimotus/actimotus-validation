import altair as alt
import pandas as pd

from actimotus_validation.figures import get_confusion_matrix

LABELS = ["sit", "walk"]


def test_returns_a_layered_chart():
    true = pd.Series(["sit", "sit", "walk"])
    pred = pd.Series(["sit", "walk", "walk"])
    chart = get_confusion_matrix(true, pred, LABELS, title="T")
    assert isinstance(chart, alt.LayerChart)


def test_rows_are_normalized_over_true_class():
    true = pd.Series(["sit"] * 4)
    pred = pd.Series(["sit", "sit", "sit", "walk"])
    chart = get_confusion_matrix(true, pred, LABELS, title="T")
    values = chart.data.set_index(["True", "Predicted"])["Value"]
    assert values[("Sit", "Sit")] == 0.75
    assert values[("Sit", "Walk")] == 0.25


def test_labels_are_capitalised_for_display():
    true = pd.Series(["sit"])
    pred = pd.Series(["sit"])
    chart = get_confusion_matrix(true, pred, LABELS, title="T")
    assert set(chart.data["True"]) == {"Sit", "Walk"}


def test_renders_to_png(tmp_path):
    true = pd.Series(["sit", "walk"])
    pred = pd.Series(["sit", "walk"])
    chart = get_confusion_matrix(true, pred, LABELS, title="T")
    out = tmp_path / "cm.png"
    chart.save(str(out), scale_factor=1)
    assert out.stat().st_size > 0
