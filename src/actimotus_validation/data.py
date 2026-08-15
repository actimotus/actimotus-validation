"""Fetch published datasets and shape them for acti-motus."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from .frames import hub_to_acti
from .labels import resolve_series
from .registry import DatasetSpec

AXES = ("x", "y", "z")


def download(spec: DatasetSpec) -> Path:
    """Fetch the pinned revision of a dataset and return its harmonized directory."""
    root = snapshot_download(
        repo_id=spec.hf_repo,
        repo_type="dataset",
        revision=spec.revision,
        allow_patterns=["harmonized/*.parquet"],
    )

    return Path(root) / "harmonized"


def subject_files(spec: DatasetSpec, harmonized: Path) -> list[Path]:
    """Subject parquet files for a dataset, honouring the registry's filter."""
    files = sorted(harmonized.glob("*.parquet"))
    if not files:
        raise ValueError(f"no parquet files under {harmonized} for {spec.name}")

    if not spec.filter:
        return files

    column, wanted = next(iter(spec.filter.items()))
    kept = []
    for path in files:
        value = pd.read_parquet(path, columns=[column])[column].iloc[0]
        if str(value) == wanted:
            kept.append(path)

    if not kept:
        raise ValueError(f"filter {spec.filter} matched no subjects in {spec.name}")

    return kept


def read_subject(path: Path) -> pd.DataFrame:
    """Read one subject file, indexed by timestamp with duplicates dropped."""
    df = pd.read_parquet(path)
    df = df.sort_values("timestamp").set_index("timestamp")

    return df[~df.index.duplicated(keep="first")]


def sensor_frame(raw: pd.DataFrame, prefix: str, *, to_acti_frame: bool) -> pd.DataFrame:
    """Extract one sensor as an acti-motus-ready frame.

    Selects the three axis columns for `prefix` and renames them to acc_x/y/z,
    optionally converting from the published hub frame to the frame acti-motus
    expects.

    The two sensors want different frames, which is why this is a required
    argument rather than a default:

    * thigh -- pass True. acti-motus expects z posterior; hub is z anterior.
      Without the conversion Lendt drops from 0.952 to 0.665 accuracy.
    * back -- pass False. acti-motus expects the trunk z anterior, matching hub.
      Converting it collapses lying detection (older adults lie recall 0.992 ->
      0.000 with orientation=False).

    In both cases the correctness check is that Activities(orientation=True) and
    Activities(orientation=False) then agree, leaving flip detection to handle
    genuinely mis-worn sensors rather than our own frame errors.

    Raises:
        ValueError: If the prefix's columns are absent.
    """
    columns = [f"{prefix}_{axis}" for axis in AXES]
    missing = [c for c in columns if c not in raw.columns]
    if missing:
        raise ValueError(f"sensor prefix {prefix!r} not in frame; missing {missing}")

    frame = raw[columns].copy()
    frame.columns = ["acc_x", "acc_y", "acc_z"]

    return hub_to_acti(frame) if to_acti_frame else frame


def ground_truth_1s(raw: pd.DataFrame, table: str) -> pd.DataFrame:
    """Per-second ground truth, resolved to canonical activities.

    acti-motus emits activities at 1 Hz, so labels are reduced to the modal label
    within each second. Rows whose label is unevaluated (jumping, transition) or
    null are dropped.

    Order note: unevaluated labels are dropped *before* the per-second mode is
    taken. The original study took the mode over raw labels first and dropped
    afterwards, which discards a second that is mostly jumping but contains a few
    walk samples. Measured over 10 children and 8 Lendt subjects, the two orders
    differ by +0.97% and +0.01% of seconds respectively, with zero label
    disagreements -- activity bouts are long relative to one second, so mixed
    seconds occur only at bout boundaries. Dropping first is kept because it is
    simpler and evaluates every second that has any evaluable ground truth.
    """
    activities = resolve_series(table, raw["label"], raw["variant"])
    activities = activities.dropna()
    if activities.empty:
        raise ValueError("no evaluable ground-truth labels in this subject")

    per_second = (
        activities.resample("1s")
        .agg(lambda x: x.value_counts().index[0] if len(x) else None)
        .dropna()
    )

    return per_second.to_frame("ground_truth")
