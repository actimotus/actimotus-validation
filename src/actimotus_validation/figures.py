"""Confusion matrix charts, averaged over participants.

Takes the frame `summarize_values(get_scores(...), ["true", "pred"])` produces:
each participant's grid normalised by its own rows, then averaged over people,
so the diagonal reads as mean sensitivity. It used to pool every second through
`sklearn.metrics.confusion_matrix`, which lets one long recording outvote
several short ones.

The 95% interval travels in the chart data as `Lower` and `Upper` but is NOT
drawn. An eight-class grid with two lines in every cell is unreadable at this
size, and the thesis redraws these matrices itself in matplotlib from the
workbook. The repository's own figures stay single-line.
"""

from __future__ import annotations

import altair as alt
import pandas as pd


def get_confusion_matrix(
    matrix: pd.DataFrame,
    labels: list[str],
    title: str = "Confusion Matrix",
    color: str = "purples",
    x_title: str | None = "Predicted",
    y_title: str | None = "True",
    hide_yaxis: bool = False,
    size: tuple[int, int] = (250, 250),
) -> alt.LayerChart:
    """Draw one averaged confusion matrix.

    Args:
        matrix: Long frame with `true`, `pred`, `mean`, `lower` and `upper`.
        labels: Reported classes, in display order.
    """
    # Every cell has to exist or the grid grows holes. `summarize_values` drops a
    # group whose value is NaN for every participant, which is what a behaviour
    # nobody performed looks like -- the empty shuffle and stairs rows of the
    # laboratory panels. Those draw as an unlabelled 0 cell, exactly as they did
    # when sklearn returned 0 for an all-zero row.
    grid = pd.MultiIndex.from_product(
        [labels, labels], names=["true", "pred"]
    ).to_frame(index=False)

    # A dataset holding none of its reported classes leaves `summarize_values`
    # with nothing to group, and it returns a (0, 0) frame with no columns at
    # all -- merging on `true` and `pred` then raises rather than drawing an
    # empty grid.
    if matrix.empty:
        matrix = grid.assign(mean=float("nan"), lower=float("nan"), upper=float("nan"))

    df = grid.merge(matrix, on=["true", "pred"], how="left")

    df = pd.DataFrame({
        "True": df["true"].str.capitalize(),
        "Predicted": df["pred"].str.capitalize(),
        "Value": df["mean"].fillna(0.0).round(2),
        "Lower": df["lower"].fillna(0.0).round(2),
        "Upper": df["upper"].fillna(0.0).round(2),
    })

    labels = [label.capitalize() for label in labels]

    title_size = 14
    axes_title_size = 12
    label_size = 12

    font = "Open Sans"
    axis = alt.Axis(
        titleFont=font,
        titleFontSize=axes_title_size,
        labelFont=font,
        labelFontSize=label_size,
    )

    x = alt.X(
        "Predicted:N",
        sort=labels[::-1],
        axis=axis,
        title=x_title,
    )
    y = alt.Y(
        "True:N",
        sort=labels,
        axis=None if hide_yaxis else axis,
        title=y_title,
    )

    # Base heatmap layer
    heatmap = (
        alt
        .Chart(df)
        .mark_rect()
        .encode(
            x=x,
            y=y,
            color=alt.Color("Value:Q", scale=alt.Scale(scheme=color), legend=None),
        )
        .properties(
            title=alt.Title(
                text=title,
                fontSize=title_size,
                fontWeight="bold",
                font=font,
            ),
            width=size[0],
            height=size[1],
        )
    )

    text_mean = (
        alt
        .Chart(df)
        .mark_text(
            align="center",
            baseline="middle",
            fontSize=label_size,
            font=font,
        )
        .encode(
            x=x,
            y=y,
            text=alt.condition(
                alt.datum.Value == 0,
                alt.value(""),  # If the count is 0, display an empty string
                alt.Text("Value:Q", format=".2f"),  # Otherwise, display the count
            ),
            color=alt.condition(
                alt.datum.Value > 0.50, alt.value("white"), alt.value("black")
            ),
        )
    )

    chart = heatmap + text_mean

    return chart
