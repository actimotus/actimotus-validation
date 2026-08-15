"""Stage 1: download published datasets and extract acti-motus features.

This is the expensive stage -- roughly 2 GB of downloads and feature extraction
over ~30M samples. Its output is cached so stages 2 and 3 run in seconds.

Usage:
    uv run python scripts/01_features.py                       # all datasets
    uv run python scripts/01_features.py --dataset lendt_adults
    uv run python scripts/01_features.py --limit 2             # smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from actimotus import Features

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from actimotus_validation import data, provenance  # noqa: E402
from actimotus_validation.registry import DatasetSpec, load_registry  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "datasets.toml"
CACHE = ROOT / "cache" / "features"


def process(spec: DatasetSpec, limit: int, cache: Path) -> int:
    harmonized = data.download(spec)
    files = data.subject_files(spec, harmonized)
    if limit:
        files = files[:limit]

    features = Features(chunking=False, calibrate=False)
    out = cache / spec.name

    for path in files:
        raw = data.read_subject(path)
        subject = path.stem

        for role, prefix in (("thigh", spec.thigh), ("back", spec.back)):
            if prefix is None:
                continue
            target = out / role
            target.mkdir(parents=True, exist_ok=True)
            features.compute(data.sensor_frame(raw, prefix)).to_parquet(
                target / f"{subject}.parquet"
            )

        print(f"  {spec.name}: {subject}", flush=True)

    provenance.write(
        out,
        stage="features",
        dataset=spec.name,
        revision=spec.revision,
        extra={"hf_repo": spec.hf_repo, "n_subjects": len(files), "limited": bool(limit)},
    )

    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", help="dataset name; repeatable")
    parser.add_argument("--limit", type=int, default=0, help="subjects per dataset (0 = all)")
    parser.add_argument("--cache", type=Path, default=CACHE)
    args = parser.parse_args()

    specs = load_registry(REGISTRY)
    names = args.dataset or list(specs)

    unknown = [n for n in names if n not in specs]
    if unknown:
        parser.error(f"unknown dataset(s): {unknown}. Known: {list(specs)}")

    for name in names:
        print(f"{name} ...", flush=True)
        n = process(specs[name], args.limit, args.cache)
        print(f"{name}: {n} subjects -> {args.cache / name}", flush=True)


if __name__ == "__main__":
    main()
