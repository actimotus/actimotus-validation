import altair as alt
import pandas as pd

from actimotus_validation.labels import LABELS_FUSED
from actimotus_validation.reports import build_report, to_fused

LABELS = ["sit", "walk"]


def _predictions() -> pd.DataFrame:
    rows = []
    for subject in ("a", "b"):
        rows.append(pd.DataFrame({
            "ground_truth": ["sit", "sit", "walk", "walk"],
            "activity": ["sit", "sit", "walk", "sit"],
            "id": subject,
        }))
    return pd.concat(rows)


def test_build_report_returns_chart_and_table():
    chart, table = build_report(_predictions(), title="Test", labels=LABELS)
    assert isinstance(chart, alt.LayerChart)
    assert list(table.columns) == LABELS


def test_table_rows_cover_the_reported_metrics():
    _, table = build_report(_predictions(), title="Test", labels=LABELS)
    for metric in ("precision", "recall", "fscore", "accuracy", "support"):
        assert metric in table.index


def test_to_fused_collapses_both_columns():
    df = pd.DataFrame({
        "ground_truth": ["lie", "sit", "stairs"],
        "activity": ["sit", "lie", "fast-walk"],
        "id": "a",
    })
    out = to_fused(df)
    assert list(out["ground_truth"]) == ["sedentary", "sedentary", "walk"]
    assert list(out["activity"]) == ["sedentary", "sedentary", "walk"]


def test_fused_report_uses_five_classes():
    df = pd.concat([
        pd.DataFrame({
            "ground_truth": ["lie", "sit", "walk", "run"],
            "activity": ["sit", "sit", "walk", "run"],
            "id": s,
        })
        for s in ("a", "b")
    ])
    _, table = build_report(to_fused(df), title="Fused", labels=LABELS_FUSED)
    assert list(table.columns) == LABELS_FUSED
