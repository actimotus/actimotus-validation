import numpy as np
import pandas as pd

from actimotus_validation.metrics import (
    get_metrics,
    get_scores,
    get_table,
    summarize_values,
)

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


# --------------------------------------------------------------- averaged ----
# The thesis normalises every confusion matrix PER PARTICIPANT and then averages
# over people, so that one long recording cannot outvote nine short ones. The
# published papers pool all seconds instead. writing/structure.md, 2.4 beat 6.


def _long_and_correct(subject: str) -> pd.DataFrame:
    """100 s of sitting, every second right."""
    return pd.DataFrame({
        "ground_truth": ["sit"] * 100,
        "activity": ["sit"] * 100,
        "id": subject,
    })


def _short_and_half_wrong(subject: str) -> pd.DataFrame:
    """4 s of sitting, half of it called walking."""
    return pd.DataFrame({
        "ground_truth": ["sit"] * 4,
        "activity": ["sit", "sit", "walk", "walk"],
        "id": subject,
    })


def test_scores_normalise_per_participant_before_averaging():
    """The whole point. Pooled would be 0.98; each person weighing once is 0.75."""
    df = pd.concat([_long_and_correct("a"), _short_and_half_wrong("b")])
    scores = get_scores(df, "ground_truth", "activity", "id", LABELS)
    averaged = summarize_values(scores, ["true", "pred"])
    cell = averaged[(averaged["true"] == "sit") & (averaged["pred"] == "sit")].iloc[0]
    assert cell["mean"] == 0.75
    assert cell["n"] == 2


def test_scores_cover_the_full_label_grid():
    """Every cell exists for every participant, or the average is over a moving n."""
    df = _long_and_correct("a")
    scores = get_scores(df, "ground_truth", "activity", "id", LABELS)
    pairs = set(zip(scores["true"], scores["pred"], strict=True))
    assert pairs == {(t, p) for t in LABELS for p in LABELS}


def test_never_predicted_class_scores_zero_not_missing():
    """The participant did sit and was never called walk. That is a real zero."""
    df = _long_and_correct("a")
    scores = get_scores(df, "ground_truth", "activity", "id", LABELS)
    cell = scores[(scores["true"] == "sit") & (scores["pred"] == "walk")]["value"]
    assert cell.iloc[0] == 0.0


def test_unperformed_class_is_nan_so_it_leaves_the_average_alone():
    """The participant never walked, so their walk row says nothing about walking."""
    df = _long_and_correct("a")
    scores = get_scores(df, "ground_truth", "activity", "id", LABELS)
    row = scores[scores["true"] == "walk"]["value"]
    assert row.isna().all()


def test_each_participants_rows_sum_to_one():
    df = pd.concat([_long_and_correct("a"), _short_and_half_wrong("b")])
    scores = get_scores(df, "ground_truth", "activity", "id", LABELS)
    sums = scores.dropna(subset=["value"]).groupby(["id", "true"])["value"].sum()
    assert np.allclose(sums.to_numpy(), 1.0)


def test_averaged_cells_carry_a_95_interval():
    df = pd.concat([
        _long_and_correct("a"), _short_and_half_wrong("b"), _short_and_half_wrong("c"),
    ])
    scores = get_scores(df, "ground_truth", "activity", "id", LABELS)
    averaged = summarize_values(scores, ["true", "pred"])
    cell = averaged[(averaged["true"] == "sit") & (averaged["pred"] == "sit")].iloc[0]
    assert cell["lower"] < cell["mean"] < cell["upper"]
    assert cell["lower"] >= 0.0
    assert cell["upper"] <= 1.0


def test_support_is_not_smuggled_into_the_grid():
    """Support is a count. Averaged with proportions it would be clipped to 1."""
    df = _long_and_correct("a")
    scores = get_scores(df, "ground_truth", "activity", "id", LABELS)
    assert "support" not in set(scores["pred"])


def _with_offlabel(subject: str) -> pd.DataFrame:
    """One second of true sitting predicted as something nobody reports."""
    return pd.DataFrame({
        "ground_truth": ["sit", "sit", "sit", "sit"],
        "activity": ["sit", "sit", "sit", "non-wear"],
        "id": subject,
    })


def test_diagonal_equals_the_tables_recall():
    """The invariant the thesis leans on when it drops the sensitivity column.

    2.4 beat 6: "Sensitivity is not repeated there, being already the diagonal."
    If these two ever disagree, dropping the column loses information rather
    than saving a duplicate.
    """
    df = pd.concat([_with_offlabel(s) for s in "abc"])
    diagonal = summarize_values(
        get_scores(df, "ground_truth", "activity", "id", LABELS), ["true", "pred"]
    )
    cell = diagonal[(diagonal["true"] == "sit") & (diagonal["pred"] == "sit")].iloc[0]

    metrics = summarize_values(
        get_metrics(df, "ground_truth", "activity", "id", LABELS), ["metric", "label"]
    )
    recall = metrics[
        (metrics["metric"] == "recall") & (metrics["label"] == "sit")
    ].iloc[0]

    assert cell["mean"] == recall["mean"]
    assert cell["mean"] == 0.75


def test_offlabel_predictions_are_counted_against_the_row():
    """A row falls short of 1 by whatever was predicted outside the reported set.

    `ntnu_children` holds 21 seconds predicted `non-wear`. Normalising by the
    reported columns alone would quietly divide that error away and lift the
    diagonal above the recall in the table beside it.
    """
    scores = get_scores(_with_offlabel("a"), "ground_truth", "activity", "id", LABELS)
    row = scores[scores["true"] == "sit"]["value"].sum()
    assert row == 0.75


# ------------------------------------------------------- nothing prints nan ----


def _solo(subject: str, true: list[str], pred: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"ground_truth": true, "activity": pred, "id": subject})


def test_a_single_contributor_prints_the_mean_without_an_interval():
    """t.ppf(0.975, df=0) is NaN, and "0.78 [nan, nan]" must never reach a page.

    Josef 2026-09-10: "We shold not get nan, nan numbers ... maybe we could
    kind of write it as empty, and in the note write reason". The mean is real
    and stays; the interval is not computable from one person and is dropped,
    which is what makes the cell visibly different from an ordinary one.
    """
    df = pd.concat([
        _solo("a", ["sit", "walk"], ["sit", "walk"]),
        _solo("b", ["sit", "sit"], ["sit", "sit"]),
    ])
    m = get_metrics(df, "ground_truth", "activity", "id", LABELS)
    table = get_table(summarize_values(m, ["metric", "label"]))
    assert table.loc["walk", "recall"] == "1.00"


def test_a_cell_with_no_contributors_at_all_is_empty():
    df = pd.concat([_solo(s, ["sit", "sit"], ["sit", "sit"]) for s in "ab"])
    m = get_metrics(df, "ground_truth", "activity", "id", LABELS)
    table = get_table(summarize_values(m, ["metric", "label"]))
    assert table.loc["walk", "recall"] == ""


def test_no_rendered_cell_ever_contains_the_word_nan():
    """The guard, not the case. Any NaN anywhere used to render as the string."""
    df = pd.concat([
        _solo("a", ["sit", "walk"], ["sit", "walk"]),
        _solo("b", ["sit", "sit"], ["sit", "sit"]),
    ])
    m = get_metrics(df, "ground_truth", "activity", "id", LABELS)
    table = get_table(summarize_values(m, ["metric", "label"]))
    rendered = table.select_dtypes(include="object").to_numpy().ravel()
    assert not any("nan" in str(cell).lower() for cell in rendered)


def test_a_behaviour_nobody_performed_counts_zero_people_not_nan():
    """`n` is the column the thesis tables print beside each behaviour, so a NaN
    in it reaches the page. Nobody performing a behaviour is a count of zero,
    which is a fact; NaN is a hole. `n_total` is the cohort and is never absent.
    """
    df = pd.concat([_solo(s, ["sit", "sit"], ["sit", "sit"]) for s in "abc"])
    m = get_metrics(df, "ground_truth", "activity", "id", LABELS)
    table = get_table(summarize_values(m, ["metric", "label"]))
    assert table.loc["walk", "n"] == 0
    assert table.loc["walk", "n_total"] == 3
    assert table.loc["sit", "n"] == 3


def test_a_behaviour_nobody_performed_renders_empty_in_every_metric():
    """The row still appears -- absence is information -- but no cell says NaN."""
    df = pd.concat([_solo(s, ["sit", "sit"], ["sit", "sit"]) for s in "abc"])
    m = get_metrics(df, "ground_truth", "activity", "id", LABELS)
    table = get_table(summarize_values(m, ["metric", "label"]))
    for metric in ("precision", "recall", "fscore", "support"):
        assert table.loc["walk", metric] == ""
