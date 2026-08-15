"""Stage 2: classify cached features into activities and join ground truth.

Runs in seconds off the stage 1 cache, so re-running under a different threshold
config is cheap. Refuses to consume a cache built with a different acti-motus
version or dataset revision.

Usage:
    uv run python scripts/02_activities.py
    uv run python scripts/02_activities.py --dataset lendt_adults
    uv run python scripts/02_activities.py --force     # ignore stale-cache checks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from actimotus import Activities

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from actimotus_validation import data, provenance  # noqa: E402
from actimotus_validation.registry import DatasetSpec, load_registry  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "datasets.toml"
FEATURES = ROOT / "cache" / "features"
PREDICTIONS = ROOT / "cache" / "predictions"


def classify(spec: DatasetSpec, features_dir: Path, use_back: bool) -> pd.DataFrame:
    """Predict activities for every cached subject and join ground truth."""
    activities = Activities(vendor=spec.vendor, orientation=True, config="DEFAULT")
    harmonized = data.download(spec)

    rows = []
    for thigh_path in sorted((features_dir / "thigh").glob("*.parquet")):
        subject = thigh_path.stem
        thigh = pd.read_parquet(thigh_path)

        trunk = None
        if use_back:
            back_path = features_dir / "back" / f"{subject}.parquet"
            if not back_path.exists():
                raise ValueError(f"{spec.name}: {subject} has no cached back features")
            trunk = pd.read_parquet(back_path)

        activity, _ = activities.compute(thigh, trunk=trunk)

        raw = data.read_subject(harmonized / f"{subject}.parquet")
        truth = data.ground_truth_1s(raw, spec.labels)

        joined = truth.join(activity, how="left").dropna(subset=["activity"])
        if joined.empty:
            raise ValueError(
                f"{spec.name}: {subject} has no overlap between ground truth and predictions"
            )

        joined["id"] = subject
        if spec.split_by:
            per_second = raw[spec.split_by].resample("1s").first()
            joined[spec.split_by] = per_second.reindex(joined.index)

        rows.append(joined)

    df = pd.concat(rows)
    # Free-living video cannot establish walking speed, so fast-walk predictions
    # are reported as walk everywhere except the walking-speeds protocol.
    if spec.labels != "walking_speeds":
        df["activity"] = df["activity"].astype(str).replace("fast-walk", "walk")
    else:
        df["activity"] = df["activity"].astype(str)

    return df


def write_outputs(
    spec: DatasetSpec, df: pd.DataFrame, suffix: str, out: Path
) -> list[tuple[str, int, int]]:
    """Write one prediction table, splitting by the registry's split_by column.

    Returns:
        One (name, seconds, subjects) per file written -- counted per split part,
        not for the whole frame, so a split dataset reports its real sizes.
    """
    out.mkdir(parents=True, exist_ok=True)
    written = []

    if spec.split_by:
        for value, part in df.groupby(spec.split_by):
            name = f"{spec.name.split('_')[0]}_{str(value).replace('-', '_')}{suffix}"
            part = part.drop(columns=[spec.split_by])
            part.to_parquet(out / f"{name}.parquet")
            written.append((name, len(part), part["id"].nunique()))
    else:
        name = f"{spec.name}{suffix}"
        df.to_parquet(out / f"{name}.parquet")
        written.append((name, len(df), df["id"].nunique()))

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", help="dataset name; repeatable")
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--out", type=Path, default=PREDICTIONS)
    parser.add_argument("--force", action="store_true", help="skip stale-cache checks")
    args = parser.parse_args()

    specs = load_registry(REGISTRY)
    names = args.dataset or list(specs)

    unknown = [n for n in names if n not in specs]
    if unknown:
        parser.error(f"unknown dataset(s): {unknown}. Known: {list(specs)}")

    version = provenance.actimotus_version()

    revisions: dict[str, str] = {}

    for name in names:
        spec = specs[name]
        features_dir = args.features / name

        if not args.force:
            provenance.verify(features_dir, revision=spec.revision, actimotus_version=version)

        for use_back, suffix in ((False, ""), (True, "_trunk")):
            if use_back and spec.back is None:
                continue
            df = classify(spec, features_dir, use_back)
            for name_out, seconds, subjects in write_outputs(spec, df, suffix, args.out):
                print(f"{name_out}: {seconds:,} seconds, {subjects} subjects", flush=True)

        revisions[name] = spec.revision

    provenance.write(
        args.out,
        stage="activities",
        dataset=",".join(sorted(revisions)),
        revision=provenance.digest(revisions),
        extra={"config": "DEFAULT", "orientation": True, "revisions": revisions},
    )


if __name__ == "__main__":
    main()
