"""Confusion matrix charts, normalized over the true class."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def get_confusion_matrix(
    true: pd.Series,
    pred: pd.Series,
    labels: list[str],
    title: str = "Confusion Matrix",
    color: str = "purples",
    x_title: str | None = "Predicted",
    y_title: str | None = "True",
    hide_yaxis: bool = False,
    size: tuple[int, int] = (250, 250),
) -> alt.LayerChart:
    matrix = confusion_matrix(true, pred, labels=labels, normalize="true").round(2)
    matrix = np.flip(matrix, axis=1)

    labels = [label.capitalize() for label in labels]
    df = (
        pd
        .DataFrame(matrix, index=labels, columns=labels[::-1])
        .reset_index()
        .melt(id_vars="index")
    )
    df.columns = ["True", "Predicted", "Value"]

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
