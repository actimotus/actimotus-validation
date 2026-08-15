# tests/test_integration.py
"""End-to-end checks. Marked slow: these download data and extract features."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from actimotus import Activities, Features

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from actimotus_validation import data  # noqa: E402
from actimotus_validation.registry import load_registry  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "datasets.toml"

# Smallest dataset (38 MB) -- keeps the integration test tolerable.
SMOKE = "ntnu_older_adults"


@pytest.mark.slow
def test_three_stages_run_end_to_end(tmp_path):
    features = tmp_path / "features"
    predictions = tmp_path / "predictions"
    results = tmp_path / "results"

    def run(*args: str) -> None:
        proc = subprocess.run(
            [sys.executable, *args], cwd=ROOT, capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stderr

    run("scripts/01_features.py", "--dataset", SMOKE, "--limit", "2",
        "--cache", str(features))
    run("scripts/02_activities.py", "--dataset", SMOKE,
        "--features", str(features), "--out", str(predictions))

    df = pd.read_parquet(predictions / f"{SMOKE}.parquet")
    assert df["id"].nunique() == 2
    assert {"ground_truth", "activity", "id"} <= set(df.columns)
    assert (predictions / f"{SMOKE}_trunk.parquet").exists()

    results.mkdir()
    # Stage 3's grouped outputs need all three NTNU datasets, so assert on the
    # single-dataset report instead of invoking the script.
    from actimotus_validation.labels import LABELS
    from actimotus_validation.reports import build_report

    chart, table = build_report(df, title="Smoke", labels=LABELS)
    chart.save(str(results / "smoke.png"), scale_factor=1)
    assert (results / "smoke.png").stat().st_size > 0
    assert list(table.columns) == LABELS


@pytest.mark.slow
@pytest.mark.parametrize("name", ["ntnu_older_adults", "lendt_adults"])
def test_orientation_flag_is_a_noop_for_the_thigh(name):
    """The invariant that proves the thigh conversion is correct.

    With the conversion applied, acti-motus's flip detector finds nothing to fix,
    so orientation=True and orientation=False must agree exactly. If this fails,
    the frame conversion is wrong -- not the flag.
    """
    spec = load_registry(REGISTRY)[name]
    harmonized = data.download(spec)
    path = data.subject_files(spec, harmonized)[0]

    raw = data.read_subject(path)
    features = Features(chunking=False, calibrate=False).compute(
        data.sensor_frame(raw, spec.thigh, to_acti_frame=True)
    )

    off, _ = Activities(vendor=spec.vendor, orientation=False, config="DEFAULT").compute(features)
    on, _ = Activities(vendor=spec.vendor, orientation=True, config="DEFAULT").compute(features)

    pd.testing.assert_series_equal(
        off["activity"].astype(str), on["activity"].astype(str)
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", ["ntnu_older_adults", "ntnu_adults"])
def test_orientation_flag_is_a_noop_with_the_back_sensor(name):
    """The same invariant with the trunk attached -- the case that caught a bug.

    The back sensor stays in the hub frame while the thigh is converted. An
    earlier version converted both, which collapsed lying detection (older adults
    lie recall 0.992 -> 0.000 at orientation=False) and was invisible because
    orientation=True auto-corrected it.

    ntnu_children is deliberately excluded: its upstream ingest normalised thigh
    orientation per subject but never audited the back, so back mounting genuinely
    varies between children and flip detection does real work there.
    """
    spec = load_registry(REGISTRY)[name]
    harmonized = data.download(spec)
    path = data.subject_files(spec, harmonized)[0]

    raw = data.read_subject(path)
    extract = Features(chunking=False, calibrate=False)
    thigh = extract.compute(data.sensor_frame(raw, spec.thigh, to_acti_frame=True))
    back = extract.compute(data.sensor_frame(raw, spec.back, to_acti_frame=False))

    off, _ = Activities(vendor=spec.vendor, orientation=False, config="DEFAULT").compute(
        thigh, trunk=back
    )
    on, _ = Activities(vendor=spec.vendor, orientation=True, config="DEFAULT").compute(
        thigh, trunk=back
    )

    pd.testing.assert_series_equal(
        off["activity"].astype(str), on["activity"].astype(str)
    )
