import pytest
import altair as alt
import pandas as pd

from actimotus_validation.labels import LABELS_FUSED
from actimotus_validation.reports import build_report, to_activities, to_fused

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


def test_build_report_returns_chart_table_and_matrix():
    chart, table, matrix = build_report(_predictions(), title="Test", labels=LABELS)
    assert isinstance(chart, alt.LayerChart)
    assert list(table.columns) == LABELS
    assert {"true", "pred", "mean", "lower", "upper"} <= set(matrix.columns)


def test_reported_matrix_is_averaged_over_people_not_pooled():
    """One long recording must not outvote a short one. 2.4 beat 6."""
    df = pd.concat([
        pd.DataFrame({
            "ground_truth": ["sit"] * 100, "activity": ["sit"] * 100, "id": "a",
        }),
        pd.DataFrame({
            "ground_truth": ["sit"] * 4,
            "activity": ["sit", "sit", "walk", "walk"],
            "id": "b",
        }),
    ])
    _, _, matrix = build_report(df, title="Test", labels=LABELS)
    cell = matrix[(matrix["true"] == "sit") & (matrix["pred"] == "sit")].iloc[0]
    assert cell["mean"] == 0.75


def test_matrix_diagonal_matches_the_tables_recall():
    """The thesis drops the sensitivity column because it is the diagonal."""
    _, table, matrix = build_report(_predictions(), title="Test", labels=LABELS)
    for label in LABELS:
        cell = matrix[(matrix["true"] == label) & (matrix["pred"] == label)].iloc[0]
        assert table.loc["recall", label].startswith(f"{cell['mean']:.2f} ")


def test_table_rows_cover_the_reported_metrics():
    _, table, _ = build_report(_predictions(), title="Test", labels=LABELS)
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
    _, table, _ = build_report(to_fused(df), title="Fused", labels=LABELS_FUSED)
    assert list(table.columns) == LABELS_FUSED


def test_kappa_is_one_of_the_reported_metrics():
    """Regression guard, not a discovery: kappa already reaches the table.

    It is here because `test_table_rows_cover_the_reported_metrics` listed the
    four metrics kappa was added beside on 2026-09-09 and was never extended, so
    kappa could have been dropped from `metrics.py` without a single test
    failing. The thesis reports it in 2.5 and 2.6.
    """
    _, table, _ = build_report(_predictions(), title="Test", labels=LABELS)
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
    _, table, _ = build_report(df, title="Test", labels=LABELS)
    assert float(table.loc["accuracy", "sit"].split()[0]) == 0.75
    assert float(table.loc["kappa", "sit"].split()[0]) < 1.0


# ------------------------------------------------------- sensor comparison ----
# 2.5.2's second block: what a second sensor on the back buys on the lie/sit
# split. The README used to assemble this by hand from two workbooks.


def _posture(subject: str, activity: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "ground_truth": ["lie", "lie", "sit", "sit"],
        "activity": activity,
        "id": subject,
    })


POSTURES = ["lie", "sit"]


def test_comparison_covers_only_the_focus_labels():
    from actimotus_validation.reports import build_comparison

    thigh = pd.concat([_posture(s, ["sit"] * 2 + ["sit"] * 2) for s in "ab"])
    trunk = pd.concat([_posture(s, ["lie"] * 2 + ["sit"] * 2) for s in "ab"])
    out = build_comparison(thigh, trunk, labels=POSTURES, focus=["lie"])
    assert set(out["label"]) == {"lie"}


def test_delta_is_what_the_back_sensor_adds():
    """Thigh alone calls all the lying sitting; the back sensor recovers it."""
    from actimotus_validation.reports import build_comparison

    thigh = pd.concat([_posture(s, ["sit"] * 4) for s in "abc"])
    trunk = pd.concat([_posture(s, ["lie", "lie", "sit", "sit"]) for s in "abc"])
    out = build_comparison(thigh, trunk, labels=POSTURES, focus=POSTURES)
    lie = out[(out["metric"] == "recall") & (out["label"] == "lie")].iloc[0]
    assert lie["thigh"] == 0.0
    assert lie["thigh_back"] == 1.0
    assert lie["delta"] == 1.0


def test_comparison_refuses_a_different_set_of_people():
    """A delta between two different cohorts is not a sensor effect."""
    from actimotus_validation.reports import build_comparison

    thigh = pd.concat([_posture(s, ["sit"] * 4) for s in "abc"])
    trunk = pd.concat([_posture(s, ["sit"] * 4) for s in "ab"])
    with pytest.raises(ValueError, match="same participants"):
        build_comparison(thigh, trunk, labels=POSTURES, focus=POSTURES)


def test_comparison_reports_how_many_people_are_behind_each_number():
    """`precision` is NaN when the classifier never emitted the class, so its mean
    is taken over a self-selected subset. In `ntnu_older_adults`, thigh-only
    precision for lying is ONE participant out of eighteen. Without n beside it,
    that reads as a cohort mean.
    """
    from actimotus_validation.reports import build_comparison

    thigh = pd.concat([_posture(s, ["sit"] * 4) for s in "abc"])
    trunk = pd.concat([_posture(s, ["lie", "lie", "sit", "sit"]) for s in "abc"])
    out = build_comparison(thigh, trunk, labels=POSTURES, focus=POSTURES)
    assert {"thigh_n", "thigh_back_n", "n_total"} <= set(out.columns)
    lie = out[(out["metric"] == "recall") & (out["label"] == "lie")].iloc[0]
    assert lie["thigh_n"] == 3
    assert lie["n_total"] == 3


def test_comparison_drops_metrics_that_are_not_per_behaviour():
    """`support` is a count, so its clipped [0, 1] interval is meaningless, and
    `accuracy` and `kappa` are whole-recording numbers broadcast across every
    label. In a workbook about lie and sit they read as lie-specific.
    """
    from actimotus_validation.reports import build_comparison

    thigh = pd.concat([_posture(s, ["sit"] * 4) for s in "abc"])
    trunk = pd.concat([_posture(s, ["lie", "lie", "sit", "sit"]) for s in "abc"])
    out = build_comparison(thigh, trunk, labels=POSTURES, focus=POSTURES)
    assert set(out["metric"]) == {"precision", "recall", "fscore"}


def test_to_activities_folds_paces_in_both_columns():
    """A pace error is not an activity error: slow walking read as walk is right."""
    df = pd.DataFrame({
        "ground_truth": ["slow-walk", "walk", "fast-walk", "sit"],
        "activity": ["walk", "fast-walk", "slow-walk", "sit"],
        "id": "a",
    })
    out = to_activities(df)
    assert list(out["ground_truth"]) == ["walk", "walk", "walk", "sit"]
    assert list(out["activity"]) == ["walk", "walk", "walk", "sit"]
