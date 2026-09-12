"""Per-subject metrics and across-subject aggregation with confidence intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
)


def _get_metrics(true: pd.Series, pred: pd.Series, labels: list[str]) -> pd.DataFrame:
    metrics = precision_recall_fscore_support(
        true.values,  # type: ignore
        pred.values,  # type: ignore
        average=None,
        labels=labels,
        zero_division=np.nan,  # type: ignore
    )
    metrics = pd.DataFrame(
        metrics,
        columns=labels,
        index=["precision", "recall", "fscore", "support"],
    )
    no_support = metrics.loc["support"] == 0
    metrics.loc[:, no_support] = np.nan  # type: ignore
    accuracy = accuracy_score(
        true.values,  # type: ignore
        pred.values,  # type: ignore
    )
    accuracy = pd.DataFrame(accuracy, columns=metrics.columns, index=["accuracy"])
    # Cohen's kappa, UNWEIGHTED. Activity types have no order, so there is no
    # defensible way to say one confusion is worse than another; the weighted
    # form belongs to the intensity bands, which are ordered. Computed like
    # accuracy -- one number for the whole recording, broadcast across the label
    # columns -- and per participant like everything else here, so it is not the
    # one number in the table aggregated a different way. Settled 2026-09-09;
    # writing/structure.md, "THE METRICS ARE SETTLED".
    kappa = cohen_kappa_score(
        true.values,  # type: ignore
        pred.values,  # type: ignore
    )
    kappa = pd.DataFrame(kappa, columns=metrics.columns, index=["kappa"])
    metrics = pd.concat([metrics, accuracy, kappa], axis=0)
    metrics = metrics.melt(
        var_name="label",
        value_name="value",
        ignore_index=False,
    ).reset_index(names="metric")

    return metrics


def get_metrics(
    df: pd.DataFrame, true: str, pred: str, group: str, labels: list[str]
) -> pd.DataFrame:
    metrics = []

    for id, temp in df.groupby(group):
        results = _get_metrics(temp[true], temp[pred], labels=labels)
        results["id"] = id
        metrics.append(results)

    metrics = pd.concat(metrics, ignore_index=True)

    return metrics


def _get_scores(true: pd.Series, pred: pd.Series, labels: list[str]) -> pd.DataFrame:
    """One participant's confusion matrix, each row divided by its own total.

    Reindexed onto the full `labels` grid so every participant contributes the
    same cells. Without that the average across people would be taken over a
    moving denominator -- a cell would be the mean of however many participants
    happened to produce it.

    Two kinds of empty cell are deliberately different:

    * The participant performed the true class but was never predicted this
      column. That is a real **0.0** and belongs in the mean.
    * The participant never performed the true class at all. The whole row is
      **NaN**, so it says nothing about that behaviour and `summarize_values`
      drops it rather than dragging the mean toward zero. Same rule as
      `_get_metrics`.

    EACH ROW IS DIVIDED BY THE PARTICIPANT'S FULL SUPPORT FOR THAT CLASS, not by
    the part of it that landed in a reported column. A prediction outside
    `labels` -- `ntnu_children` holds 21 seconds of `non-wear` -- therefore
    counts against the row, which then sums to less than 1 by exactly that
    share. Normalising by the reported columns alone divides the error away and
    lifts the diagonal above the recall in the table beside it. That matters
    because the thesis prints no sensitivity column, on the ground that it is
    already the diagonal: the two have to be the same number or dropping the
    column loses information rather than saving a duplicate.
    `tests/test_metrics.py::test_diagonal_equals_the_tables_recall` holds it.

    This is where the function differs from
    `sklearn.metrics.confusion_matrix(labels=..., normalize="true")`, which
    drops those seconds entirely.
    """
    matrix = pd.crosstab(index=true, columns=pred, dropna=False)

    # Before restricting the columns, so out-of-label predictions stay in the
    # denominator. Reindexing gives NaN for a class this participant never
    # performed, and NaN/NaN leaves the whole row missing without a special case.
    total = matrix.sum(axis=1).reindex(labels)

    matrix = matrix.reindex(index=labels, columns=labels, fill_value=0)
    matrix = matrix.div(total, axis=0)

    df = matrix.melt(var_name="pred", value_name="value", ignore_index=False)

    return df.reset_index(names="true")


def get_scores(
    df: pd.DataFrame,
    true: str,
    pred: str,
    group: str,
    labels: list[str],
) -> pd.DataFrame:
    """Per-participant row-normalised confusion matrices, stacked long.

    Feed to `summarize_values(scores, ["true", "pred"])` for the averaged matrix
    the thesis reports: each person's grid normalised by its own rows, then
    averaged over people, so the diagonal reads as mean sensitivity and every
    cell carries a 95% interval. The published papers pool all seconds instead,
    which lets one long recording outvote several short ones.
    """
    scores = []

    for id, temp in df.groupby(group):
        results = _get_scores(temp[true], temp[pred], labels)
        results["id"] = id
        scores.append(results)

    return pd.concat(scores, axis=0, ignore_index=True)


def _mean_ci(row: pd.Series) -> str:
    """Render one cell, and never render the string "nan".

    Three states, and the page has to tell them apart:

    * No participant contributed -- nobody performed the behaviour, or the
      classifier never emitted it. The cell is EMPTY.
    * One participant contributed. The mean is real and stays; the interval is
      `t.ppf(0.975, df=0)`, which is NaN, and is dropped. A bare number is
      therefore the visible mark of a cell resting on a single person, and the
      table's note owes the reason.
    * Two or more. Mean and interval, as always.

    Josef 2026-09-10: "We shold not get nan, nan numbers ... maybe we could kind
    of write it as empty, and in the note write reason maybe?"
    """
    if pd.isna(row["mean"]):
        return ""

    if pd.isna(row["lower"]) or pd.isna(row["upper"]):
        return f"{row['mean']:.2f}"

    return f"{row['mean']:.2f} [{row['lower']:.2f}, {row['upper']:.2f}]"


def get_mean_ci(df: pd.DataFrame) -> pd.Series:
    return df.apply(_mean_ci, axis=1)


def get_mean_std(df: pd.DataFrame) -> pd.Series:
    return df.apply(
        lambda x: ""
        if pd.isna(x["mean"])
        else f"{x['mean']:.2f} ± {x['std']:.2f}"
        if pd.notna(x["std"])
        else f"{x['mean']:.2f}",
        axis=1,
    )


def summarize_values(df: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    metrics = []

    for id, temp in df.groupby(group):
        n_total = len(temp)

        values = temp.loc[temp["value"].notna(), "value"]

        n = len(values)

        if values.empty:
            continue

        sum = values.sum()
        mean = values.mean()
        std = values.std()
        # Two-sided 95%: ppf takes the ONE-SIDED quantile, so 0.975 is what a 95%
        # interval needs. Was 0.95, which is a 90% interval, and Paper II published
        # those brackets labelled 95%. Josef's ruling 2026-09-08: report a true 95%.
        t = stats.t.ppf(0.975, df=n - 1)
        e = t * (std / np.sqrt(n))
        lower, upper = mean - e, mean + e

        results = {}
        for col in group:
            results[col] = id[group.index(col)]

        # KAPPA RUNS -1 TO 1, EVERY OTHER METRIC HERE RUNS 0 TO 1. The clip was
        # written when only the bounded ones existed; applied to kappa it reports
        # a participant performing worse than chance as though they had merely
        # scored zero, and one participant in ntnu_walking_speeds scores -0.084.
        floor = -1.0 if results.get("metric") == "kappa" else 0.0
        lower, upper = np.clip(lower, floor, 1), np.clip(upper, floor, 1)

        results.update({
            "n": n,
            "n_total": n_total,
            "sum": sum,
            "mean": mean,
            "std": std,
            "lower": lower,
            "upper": upper,
        })

        metrics.append(results)

    return pd.DataFrame(metrics)


def get_table(df: pd.DataFrame):
    df = df.copy()
    df["table"] = get_mean_ci(df)
    support = df["metric"] == "support"
    df.loc[support, "table"] = get_mean_std(df.loc[support])

    other = df.loc[support, ["label"]]
    other["support_total"] = df.loc[support, "sum"]
    other["n"] = df.loc[support, "n"]
    other["n_total"] = df.loc[support, "n_total"]
    other.set_index("label", inplace=True)

    labels = df["label"].unique()
    # A behaviour NOBODY performed has no support row at all, so it would leave
    # a hole in the one column the thesis prints beside each behaviour. Zero
    # people is a fact and NaN is a hole. `n_total` is the cohort size and is
    # the same for every row, so it is never genuinely missing.
    other = other.reindex(labels)
    other["n"] = other["n"].fillna(0).astype(int)
    other["n_total"] = other["n_total"].ffill().bfill().astype(int)
    other["support_total"] = other["support_total"].fillna(0.0)

    # Reindex BEFORE filling: a behaviour nobody performed has no row in the
    # pivot at all, so concatenating `other` afterwards would put the hole back.
    df = df.pivot(index="label", columns="metric", values="table")
    df = df.reindex(labels).fillna("")
    df = pd.concat([df, other], axis=1)

    return df
