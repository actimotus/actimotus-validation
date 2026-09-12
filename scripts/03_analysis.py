"""Stage 3: turn cached predictions into the paper's tables and figures.

Usage:
    uv run python scripts/03_analysis.py
    uv run python scripts/03_analysis.py --only ntnu_walking_speeds
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from actimotus_validation import provenance  # noqa: E402
from actimotus_validation.labels import LABELS, LABELS_FUSED, LABELS_WALKING_SPEEDS  # noqa: E402
from actimotus_validation.reports import (  # noqa: E402
    build_comparison,
    build_report,
    to_fused,
)

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS = ROOT / "cache" / "predictions"
RESULTS = ROOT / "results"

# Colour carries meaning: purple marks the one laboratory protocol, green the
# free-living and semi-structured ones. Entries are (prediction table, panel
# title, colour scheme).
GREEN = "greens"
PURPLE = "purples"

NTNU = [
    ("ntnu_children", "Children", GREEN),
    ("ntnu_adults", "Adults", GREEN),
    ("ntnu_older_adults", "Older Adults", GREEN),
]
LENDT = [
    ("lendt_laboratory", "Laboratory", PURPLE),
    ("lendt_free_living", "Free-living", GREEN),
]
# The energy-expenditure cohort is laboratory only: treadmill and ergometer
# stages against indirect calorimetry, no free-living arm.
LENDT_ENERGY = [
    ("lendt_energy", "Energy Expenditure", PURPLE),
]
WALKING_SPEEDS = [
    ("ntnu_walking_speeds", "Walking Speeds", GREEN),
]

# Every output stem, as (panel, reported labels, fused). This is the single place
# a dataset becomes results: a dataset absent from here is downloaded, classified
# and written to a prediction table, and then silently produces nothing.
# tests/test_analysis_panels.py holds that invariant.
PANELS: dict[str, tuple[list[tuple[str, str, str]], list[str], bool]] = {
    "ntnu_datasets": (NTNU, LABELS, False),
    "ntnu_datasets_fused": (NTNU, LABELS_FUSED, True),
    "ntnu_datasets_trunk": (
        [(f"{n}_trunk", t, c) for n, t, c in NTNU], LABELS, False
    ),
    "lendt_adults": (LENDT, LABELS, False),
    "lendt_adults_fused": (LENDT, LABELS_FUSED, True),
    "lendt_energy": (LENDT_ENERGY, LABELS, False),
    "lendt_energy_fused": (LENDT_ENERGY, LABELS_FUSED, True),
    "ntnu_walking_speeds": (WALKING_SPEEDS, LABELS_WALKING_SPEEDS, False),
}

# Lying and sitting are what the second sensor was added for: the thigh is at
# much the same angle in both, so daytime lying is largely called sitting, and a
# sensor on the back sees the trunk go horizontal. Only the NTNU cohorts can
# supply this -- both Lendt datasets are thigh-only.
POSTURES = ["lie", "sit"]

# (entries as (thigh table, title), reported labels, behaviours to report).
COMPARISONS: dict[str, tuple[list[tuple[str, str]], list[str], list[str]]] = {
    "ntnu_datasets_postures": ([(n, t) for n, t, _ in NTNU], LABELS, POSTURES),
}


def sheet_name(title: str) -> str:
    """Excel allows 31 characters, and a clash is silently destructive.

    `title[:31]` collides with `(title + " matrix")[:31]` the moment a title
    reaches 31 characters, and openpyxl writes both into one sheet without a
    word. Truncating the stem instead keeps the discriminating suffix.
    """
    if title.endswith(" matrix"):
        return f"{title.removesuffix(' matrix')[:24]} matrix"

    return title[:31]


def load(predictions: Path, name: str) -> pd.DataFrame:
    path = predictions / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/02_activities.py first")

    return pd.read_parquet(path)


def grouped(
    predictions: Path,
    results: Path,
    entries: list[tuple[str, str, str]],
    labels: list[str],
    stem: str,
    fused: bool = False,
) -> None:
    """Build one side-by-side figure and one multi-sheet workbook.

    Each entry carries its own colour scheme, so a figure can mix protocols --
    the Lendt panel pairs a purple laboratory matrix with a green free-living one.
    """
    charts, tables = [], {}

    for i, (name, title, color) in enumerate(entries):
        df = load(predictions, name)
        if fused:
            df = to_fused(df)
        chart, table, matrix = build_report(
            df, title=title, labels=labels, hide_yaxis=i > 0, color=color
        )
        charts.append(chart)
        tables[title] = table
        # The averaged matrix goes in beside its table, indexed rather than
        # drawn. The thesis redraws these grids itself in matplotlib and needs
        # the numbers; without this sheet the only averaged figures leaving the
        # pipeline are pixels.
        tables[f"{title} matrix"] = matrix

    combined = charts[0]
    for chart in charts[1:]:
        combined = combined | chart

    combined.resolve_scale(color="independent").save(
        str(results / f"{stem}.png"), scale_factor=4
    )

    with pd.ExcelWriter(results / f"{stem}.xlsx") as writer:
        for title, table in tables.items():
            # The matrix sheets are flat: writing `true`/`pred` as a MultiIndex
            # makes Excel blank the repeated key, and reading the sheet back
            # then gives NaN for seven rows in eight.
            table.to_excel(
                writer,
                sheet_name=sheet_name(title),
                index=not title.endswith(" matrix"),
            )

    print(f"{stem}.png / {stem}.xlsx", flush=True)


def compare(
    predictions: Path,
    results: Path,
    entries: list[tuple[str, str]],
    labels: list[str],
    focus: list[str],
    stem: str,
) -> None:
    """One workbook: what adding the back sensor does to lie and sit.

    A sheet per dataset, each carrying the thigh-only value, the thigh-and-back
    value, both with 95% intervals, and the difference. No figure -- the finding
    is a handful of numbers and a matrix would spend a half-page saying it.
    """
    tables = {}

    for name, title in entries:
        thigh = load(predictions, name)
        trunk = load(predictions, f"{name}_trunk")
        tables[title] = build_comparison(thigh, trunk, labels=labels, focus=focus)

    with pd.ExcelWriter(results / f"{stem}.xlsx") as writer:
        for title, table in tables.items():
            table.to_excel(writer, sheet_name=sheet_name(title), index=False)

    print(f"{stem}.xlsx", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--only", help="build a single output stem")
    args = parser.parse_args()

    args.results.mkdir(parents=True, exist_ok=True)

    known = list(PANELS) + list(COMPARISONS)
    if args.only and args.only not in known:
        parser.error(f"unknown output {args.only!r}. Known: {known}")

    built = [args.only] if args.only else known
    for stem in built:
        if stem in PANELS:
            panel, labels, fused = PANELS[stem]
            grouped(args.predictions, args.results, panel, labels, stem, fused=fused)
        else:
            entries, labels, focus = COMPARISONS[stem]
            compare(args.predictions, args.results, entries, labels, focus, stem)

    upstream = provenance.read(args.predictions)
    provenance.write(
        args.results,
        stage="analysis",
        # Name what this run actually built. After --only, the other outputs in
        # results/ are from an earlier run and may be stale or absent.
        dataset=",".join(built),
        revision=upstream["revision"],
        extra={"outputs": built, "complete": len(built) == len(known)},
    )


if __name__ == "__main__":
    main()
