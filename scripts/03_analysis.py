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

NTNU = [("ntnu_children", "Children"), ("ntnu_adults", "Adults"), ("ntnu_older_adults", "Older Adults")]


def load(predictions: Path, name: str) -> pd.DataFrame:
    path = predictions / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/02_activities.py first")

    return pd.read_parquet(path)


def grouped(
    predictions: Path,
    results: Path,
    entries: list[tuple[str, str]],
    labels: list[str],
    stem: str,
    fused: bool = False,
    color: str = "greens",
) -> None:
    """Build one side-by-side figure and one multi-sheet workbook."""
    charts, tables = [], {}

    for i, (name, title) in enumerate(entries):
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
            [(f"{n}_trunk", t) for n, t in NTNU],
            LABELS,
            "ntnu_datasets_trunk",
        ),
        "lendt_adults": lambda: grouped(
            args.predictions,
            args.results,
            [("lendt_laboratory", "Laboratory"), ("lendt_free_living", "Free-living")],
            LABELS,
            "lendt_adults",
            color="purples",
        ),
        "lendt_adults_fused": lambda: grouped(
            args.predictions,
            args.results,
            [("lendt_laboratory", "Laboratory"), ("lendt_free_living", "Free-living")],
            LABELS_FUSED,
            "lendt_adults_fused",
            fused=True,
            color="purples",
        ),
        "ntnu_walking_speeds": lambda: grouped(
            args.predictions,
            args.results,
            [("ntnu_walking_speeds", "Walking Speeds")],
            LABELS_WALKING_SPEEDS,
            "ntnu_walking_speeds",
        ),
    }

    if args.only:
        if args.only not in builders:
            parser.error(f"unknown output {args.only!r}. Known: {list(builders)}")
        builders[args.only]()
    else:
        for build in builders.values():
            build()

    provenance.write(
        args.results,
        stage="analysis",
        dataset="all",
        revision=provenance.read(args.predictions)["revision"],
    )


if __name__ == "__main__":
    main()
