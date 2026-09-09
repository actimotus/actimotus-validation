import numpy as np
import pandas as pd

from actimotus_validation.metrics import get_metrics, get_table, summarize_values

LABELS = ["sit", "walk"]


def _perfect(subject: str) -> pd.DataFrame:
    return pd.DataFrame({
        "ground_truth": ["sit", "sit", "walk", "walk"],
        "activity": ["sit", "sit", "walk", "walk"],
        "id": subject,
    })


def test_perfect_prediction_gives_unit_scores():
    df = pd.concat([_perfect("a"), _perfect("b")])
    m = get_metrics(df, "ground_truth", "activity", "id", LABELS)
    recall = m[(m["metric"] == "recall") & (m["label"] == "walk")]["value"]
    assert (recall == 1.0).all()


def test_support_counts_seconds_per_label():
    df = _perfect("a")
    m = get_metrics(df, "ground_truth", "activity", "id", LABELS)
    support = m[(m["metric"] == "support") & (m["label"] == "sit")]["value"].iloc[0]
    assert support == 2


def test_absent_label_is_nan_not_zero():
    """A label the subject never performed must not drag the mean toward zero."""
    df = pd.DataFrame({
        "ground_truth": ["sit", "sit"], "activity": ["sit", "sit"], "id": "a",
    })
    m = get_metrics(df, "ground_truth", "activity", "id", LABELS)
    walk = m[(m["metric"] == "recall") & (m["label"] == "walk")]["value"].iloc[0]
    assert np.isnan(walk)


def test_summarize_reports_n_of_contributing_subjects():
    df = pd.concat([_perfect("a"), _perfect("b"), _perfect("c")])
    m = get_metrics(df, "ground_truth", "activity", "id", LABELS)
    s = summarize_values(m, ["metric", "label"])
    row = s[(s["metric"] == "recall") & (s["label"] == "sit")].iloc[0]
    assert row["n"] == 3
    assert row["mean"] == 1.0


def test_confidence_bounds_are_clipped_to_zero_one():
    df = pd.concat([_perfect(s) for s in "abcd"])
    m = get_metrics(df, "ground_truth", "activity", "id", LABELS)
    s = summarize_values(m, ["metric", "label"])
    assert (s["lower"] >= 0).all()
    assert (s["upper"] <= 1).all()


def test_get_table_formats_mean_and_interval():
    df = pd.concat([_perfect(s) for s in "abc"])
    m = get_metrics(df, "ground_truth", "activity", "id", LABELS)
    table = get_table(summarize_values(m, ["metric", "label"]))
    assert table.loc["sit", "recall"] == "1.00 [1.00, 1.00]"


def _mirrored(subject: str) -> pd.DataFrame:
    """Every second called as the other class. Cohen's kappa is exactly -1."""
    return pd.DataFrame({
        "ground_truth": ["sit", "sit", "walk", "walk"],
        "activity": ["walk", "walk", "sit", "sit"],
        "id": subject,
    })


def test_kappa_interval_is_not_clipped_at_zero():
    """Kappa runs -1 to 1, and worse than chance is a finding, not a floor.

    Every other metric here is bounded below by zero, so the interval was
    clipped to [0, 1] for all of them. That silently reports a classifier
    performing worse than chance as though it merely scored zero. One
    participant in `ntnu_walking_speeds` scores -0.084.
    """
    df = pd.concat([_mirrored(s) for s in "abc"])
    m = get_metrics(df, "ground_truth", "activity", "id", LABELS)
    s = summarize_values(m, ["metric", "label"])
    kappa = s[s["metric"] == "kappa"].iloc[0]
    assert kappa["mean"] == -1.0
    assert kappa["lower"] == -1.0
    assert kappa["upper"] == -1.0
