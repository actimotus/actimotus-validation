"""Per-subject metrics and across-subject aggregation with confidence intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


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
    metrics = pd.concat([metrics, accuracy], axis=0)
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


def _get_scores(true: pd.Series, pred: pd.Series) -> pd.DataFrame:
    total = pd.crosstab(
        index=true,
        columns=pred,
        dropna=False,
    ).sum(axis=1)

    df = pd.crosstab(
        index=true,
        columns=pred,
        dropna=False,
        normalize="index",
    )
    df["support"] = total
    df = df.melt(var_name="pred", value_name="value", ignore_index=False).reset_index(
        names="true"
    )

    return df


def get_scores(
    df: pd.DataFrame,
    true: str,
    pred: str,
    group: str,
) -> pd.DataFrame:
    scores = []

    for id, temp in df.groupby(group):
        results = _get_scores(temp[true], temp[pred])
        results["id"] = id
        scores.append(results)

    scores = pd.concat(scores, axis=0)

    return scores


def get_mean_ci(df: pd.DataFrame) -> pd.Series:
    return df.apply(
        lambda x: f"{x['mean']:.2f} [{x['lower']:.2f}, {x['upper']:.2f}]", axis=1
    )


def get_mean_std(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda x: f"{x['mean']:.2f} ± {x['std']:.2f}", axis=1)


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
        t = stats.t.ppf(0.95, df=n - 1)
        e = t * (std / np.sqrt(n))
        lower, upper = mean - e, mean + e
        lower, upper = np.clip(lower, 0, 1), np.clip(upper, 0, 1)

        results = {}
        for col in group:
            results[col] = id[group.index(col)]

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
    df = df.pivot(index="label", columns="metric", values="table")
    df = pd.concat([df, other], axis=1)

    return df
