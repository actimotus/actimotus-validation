"""Build one report -- a metric table plus a confusion matrix -- from predictions."""

from __future__ import annotations

import altair as alt
import pandas as pd

from .figures import get_confusion_matrix
from .labels import fuse
from .metrics import get_metrics, get_table, summarize_values

TRUE = "ground_truth"
PRED = "activity"
GROUP = "id"


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
) -> tuple[alt.LayerChart, pd.DataFrame]:
    """Metrics table and confusion matrix for one dataset.

    Metrics are computed per subject and then averaged across subjects with 90%
    confidence intervals, so every participant weighs equally regardless of
    recording length. The confusion matrix pools all seconds.

    Returns:
        (chart, table) where table is metrics-by-label with `labels` as columns.
    """
    metrics = get_metrics(df, TRUE, PRED, GROUP, labels)
    table = get_table(summarize_values(metrics, ["metric", "label"])).T
    table = table[labels]

    chart = get_confusion_matrix(
        df[TRUE],
        df[PRED],
        labels,
        title=title,
        color=color,
        y_title="True",
        x_title="Predicted",
        hide_yaxis=hide_yaxis,
        size=size,
    )

    return chart, table
