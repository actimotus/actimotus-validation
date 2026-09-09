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


def test_kappa_is_one_of_the_reported_metrics():
    """Regression guard, not a discovery: kappa already reaches the table.

    It is here because `test_table_rows_cover_the_reported_metrics` listed the
    four metrics kappa was added beside on 2026-09-09 and was never extended, so
    kappa could have been dropped from `metrics.py` without a single test
    failing. The thesis reports it in 2.5 and 2.6.
    """
    _, table = build_report(_predictions(), title="Test", labels=LABELS)
    assert "kappa" in table.index
    assert table.loc["kappa", "sit"] == table.loc["kappa", "walk"]


def test_prediction_outside_the_reported_labels_counts_as_an_error():
    """A predicted class the panel does not report is a mistake, not a gap.

    `ntnu_children` holds 21 seconds predicted `non-wear`, which is not in
    LABELS. Kappa is deliberately computed WITHOUT `labels=`, so those seconds
    stay in as a ninth class and score as errors; passing `labels=` would drop
    the rows and quietly raise the score. The effect is 0.0001 on that dataset,
    but the direction is the point.
    """
    df = pd.concat([
        pd.DataFrame({
            "ground_truth": ["sit", "sit", "walk", "walk"],
            "activity": ["sit", "sit", "walk", "non-wear"],
            "id": s,
        })
        for s in ("a", "b")
    ])
    _, table = build_report(df, title="Test", labels=LABELS)
    assert float(table.loc["accuracy", "sit"].split()[0]) == 0.75
    assert float(table.loc["kappa", "sit"].split()[0]) < 1.0
