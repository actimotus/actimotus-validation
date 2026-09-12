"""Build one report -- a metric table plus a confusion matrix -- from predictions."""

from __future__ import annotations

import altair as alt
import pandas as pd

from .figures import get_confusion_matrix
from .labels import fuse
from .metrics import get_metrics, get_scores, get_table, summarize_values

TRUE = "ground_truth"
PRED = "activity"
GROUP = "id"

# The metrics that mean something for a single behaviour. `support` is a count
# and `accuracy` and `kappa` describe the whole recording.
PER_BEHAVIOUR = ("precision", "recall", "fscore")


def to_fused(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse both ground truth and predictions to the five fused classes."""
    out = df.copy()
    out[TRUE] = fuse(out[TRUE])
    out[PRED] = fuse(out[PRED])

    return out


def build_report(
    df: pd.DataFrame,
    title: str,
    labels: list[str],
    hide_yaxis: bool = False,
    color: str = "greens",
    size: tuple[int, int] = (300, 300),
) -> tuple[alt.LayerChart, pd.DataFrame, pd.DataFrame]:
    """Metrics table and confusion matrix for one dataset.

    BOTH are computed per subject and then averaged across subjects with 95%
    confidence intervals, so every participant weighs equally regardless of
    recording length. The matrix used to pool all seconds, which let one long
    recording outvote several short ones; the diagonal is now mean sensitivity
    and equals the `recall` row of the table beside it.

    Returns:
        (chart, table, matrix). `table` is metrics-by-label with `labels` as
        columns; `matrix` is the averaged confusion matrix, long, one row per
        (true, pred) with `mean`, `lower` and `upper`.
    """
    metrics = get_metrics(df, TRUE, PRED, GROUP, labels)
    table = get_table(summarize_values(metrics, ["metric", "label"])).T
    table = table[labels]

    scores = get_scores(df, TRUE, PRED, GROUP, labels)
    matrix = summarize_values(scores, ["true", "pred"])

    chart = get_confusion_matrix(
        matrix,
        labels,
        title=title,
        color=color,
        y_title="True",
        x_title="Predicted",
        hide_yaxis=hide_yaxis,
        size=size,
    )

    return chart, table, matrix


def build_comparison(
    thigh: pd.DataFrame,
    trunk: pd.DataFrame,
    labels: list[str],
    focus: list[str],
) -> pd.DataFrame:
    """What a second sensor buys, on the behaviours it was added for.

    ActiMotus tells lying from sitting by the thigh's inclination, which barely
    changes between the two, so daytime lying is largely called sitting. A
    sensor on the back sees the trunk go horizontal and settles it. This is the
    artefact behind 2.5.2's second block; it used to be assembled by hand from
    two separate workbooks.

    Both runs must cover the same people. A delta between two different cohorts
    is a cohort effect wearing a sensor effect's clothes, and the two prediction
    tables can silently disagree -- a subject can drop out of one run because a
    back-sensor file is missing.

    Only datasets 1, 2 and 3 can supply this. The two Lendt cohorts are
    thigh-only and have no trunk prediction table at all.

    Args:
        thigh: Predictions from the thigh sensor alone.
        trunk: Predictions from thigh and back together, same participants.
        labels: The full reported vocabulary, so every metric is scored against
            the same class set in both runs.
        focus: The behaviours to report, normally lie and sit.

    ONLY THE PER-BEHAVIOUR METRICS ARE REPORTED. `support` is a count, and
    `summarize_values` clips every interval to [0, 1], which turns a mean of 208
    seconds into "[1.00, 1.00]"; `get_table` hides that in the panel workbooks by
    re-rendering support as mean +/- SD, and this function does not go through
    `get_table`. `accuracy` and `kappa` are whole-recording numbers over the full
    vocabulary, broadcast across the label columns by `_get_metrics`, so in a
    sheet about lying and sitting they read as lie-specific and are not.

    EVERY NUMBER CARRIES THE COUNT OF PEOPLE BEHIND IT. `precision` is NaN when
    the classifier never emitted the class at all, so its mean is taken over
    whoever it did emit for -- in `ntnu_older_adults` thigh-only precision for
    lying is ONE participant out of eighteen, and thigh-only precision for lying
    in `ntnu_adults` is six out of thirty-one. Without `thigh_n` beside it that
    reads as a cohort mean. It also means `fscore` is not the harmonic mean of
    the `precision` and `recall` printed above it, because the three are averaged
    over different people.

    Returns:
        One row per (metric, label) with `thigh`, `thigh_back` and their 95%
        intervals, `thigh_n` / `thigh_back_n` contributing participants against
        `n_total`, and `delta`, what the second sensor added.
    """
    people = set(thigh[GROUP].unique()), set(trunk[GROUP].unique())
    if people[0] != people[1]:
        missing = sorted(people[0] ^ people[1])
        raise ValueError(
            "thigh and thigh+back runs must cover the same participants; "
            f"{len(missing)} differ: {missing[:5]}"
        )

    frames = {}
    for name, df in (("thigh", thigh), ("thigh_back", trunk)):
        metrics = get_metrics(df, TRUE, PRED, GROUP, labels)
        summary = summarize_values(metrics, ["metric", "label"])
        keep = summary["label"].isin(focus) & summary["metric"].isin(PER_BEHAVIOUR)
        frames[name] = summary[keep][
            ["metric", "label", "mean", "lower", "upper", "n", "n_total"]
        ].rename(columns={
            "mean": name,
            "lower": f"{name}_lower",
            "upper": f"{name}_upper",
            "n": f"{name}_n",
        })

    out = frames["thigh"].merge(
        frames["thigh_back"],
        on=["metric", "label", "n_total"],
        how="outer",
        validate="one_to_one",
    )
    out["delta"] = out["thigh_back"] - out["thigh"]

    columns = [
        "metric", "label", "n_total",
        "thigh", "thigh_lower", "thigh_upper", "thigh_n",
        "thigh_back", "thigh_back_lower", "thigh_back_upper", "thigh_back_n",
        "delta",
    ]

    return out[columns].sort_values(["label", "metric"], ignore_index=True)
