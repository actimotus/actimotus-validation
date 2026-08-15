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
from actimotus_validation.reports import build_report, to_fused  # noqa: E402

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
        chart, table = build_report(
            df, title=title, labels=labels, hide_yaxis=i > 0, color=color
        )
        charts.append(chart)
        tables[title] = table

    combined = charts[0]
    for chart in charts[1:]:
        combined = combined | chart

    combined.resolve_scale(color="independent").save(
        str(results / f"{stem}.png"), scale_factor=4
    )

    with pd.ExcelWriter(results / f"{stem}.xlsx") as writer:
        for title, table in tables.items():
            table.to_excel(writer, sheet_name=title[:31])

    print(f"{stem}.png / {stem}.xlsx", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--only", help="build a single output stem")
    args = parser.parse_args()

    args.results.mkdir(parents=True, exist_ok=True)

    builders = {
        "ntnu_datasets": lambda: grouped(
            args.predictions, args.results, NTNU, LABELS, "ntnu_datasets"
        ),
        "ntnu_datasets_fused": lambda: grouped(
            args.predictions, args.results, NTNU, LABELS_FUSED, "ntnu_datasets_fused", fused=True
        ),
        "ntnu_datasets_trunk": lambda: grouped(
            args.predictions,
            args.results,
            [(f"{n}_trunk", t, c) for n, t, c in NTNU],
            LABELS,
            "ntnu_datasets_trunk",
        ),
        "lendt_adults": lambda: grouped(
            args.predictions, args.results, LENDT, LABELS, "lendt_adults"
        ),
        "lendt_adults_fused": lambda: grouped(
            args.predictions, args.results, LENDT, LABELS_FUSED, "lendt_adults_fused", fused=True
        ),
        "ntnu_walking_speeds": lambda: grouped(
            args.predictions,
            args.results,
            [("ntnu_walking_speeds", "Walking Speeds", GREEN)],
            LABELS_WALKING_SPEEDS,
            "ntnu_walking_speeds",
        ),
    }

    if args.only:
        if args.only not in builders:
            parser.error(f"unknown output {args.only!r}. Known: {list(builders)}")
        builders[args.only]()
        built = [args.only]
    else:
        for build in builders.values():
            build()
        built = list(builders)

    upstream = provenance.read(args.predictions)
    provenance.write(
        args.results,
        stage="analysis",
        # Name what this run actually built. After --only, the other outputs in
        # results/ are from an earlier run and may be stale or absent.
        dataset=",".join(built),
        revision=upstream["revision"],
        extra={"outputs": built, "complete": len(built) == len(builders)},
    )


if __name__ == "__main__":
    main()
