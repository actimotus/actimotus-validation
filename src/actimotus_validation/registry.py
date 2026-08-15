"""Parse datasets.toml into typed dataset specifications."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset's location, sensor layout and label vocabulary.

    Attributes:
        name: Registry key, also the cache and output directory name.
        hf_repo: HuggingFace dataset repo id.
        revision: Pinned commit SHA (40 hex chars).
        vendor: acti-motus vendor string; 'Sens' applies SENS-specific corrections.
        thigh: Column prefix for the thigh sensor, e.g. 'thigh_acc' -> thigh_acc_x.
        back: Column prefix for the back sensor, or None for thigh-only datasets.
        labels: Key into labels.LABEL_TABLES.
        filter: Optional equality filter on a metadata column, e.g. {'cohort': 'td'}.
        split_by: Optional column whose values split the dataset into sub-reports.
    """

    name: str
    hf_repo: str
    revision: str
    vendor: Literal["Sens", "Other"]
    thigh: str
    back: str | None
    labels: str
    filter: dict[str, str] | None = None
    split_by: str | None = None


def load_registry(path: Path) -> dict[str, DatasetSpec]:
    """Read datasets.toml and return specs keyed by dataset name."""
    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    specs: dict[str, DatasetSpec] = {}
    for name, entry in raw.items():
        back = entry.get("back")
        specs[name] = DatasetSpec(
            name=name,
            hf_repo=entry["hf_repo"],
            revision=entry["revision"],
            vendor=entry["vendor"],
            thigh=entry["thigh"],
            back=back if back else None,
            labels=entry["labels"],
            filter=entry.get("filter"),
            split_by=entry.get("split_by"),
        )

    return specs
