"""Stage 3 must consume everything stage 2 writes.

A dataset can be registered, downloaded, classified and written to a prediction
table, and then produce no results at all, because stage 3 builds its outputs
from hand-written panel lists rather than from the registry. Nothing else in the
suite notices; the run simply ends quietly one dataset short.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from actimotus_validation.registry import load_registry

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "datasets.toml"


def _analysis():
    spec = importlib.util.spec_from_file_location(
        "stage3_analysis", ROOT / "scripts" / "03_analysis.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _panelled_tables(module) -> set[str]:
    """Every prediction table named by any panel, across every output stem."""
    return {
        table
        for panel, _labels, _fused in module.PANELS.values()
        for table, _title, _colour in panel
    }


def test_every_unsplit_dataset_reaches_a_panel():
    """Datasets with a split_by are excluded: stage 2 names their tables after
    values found in the data, so the names are not derivable from the registry.
    """
    module = _analysis()
    panelled = _panelled_tables(module)

    missing = sorted(
        name
        for name, spec in load_registry(REGISTRY).items()
        if spec.split_by is None and name not in panelled
    )

    assert missing == [], f"registered but never analysed: {missing}"


def test_main_stamps_provenance_after_building(tmp_path, monkeypatch):
    """main() ran to completion only in a full pipeline run, so a name that does
    not exist there reaches the user as a traceback after all the work is done."""
    from actimotus_validation import provenance

    module = _analysis()
    predictions = tmp_path / "predictions"
    results = tmp_path / "results"
    provenance.write(predictions, stage="activities", dataset="all", revision="x")

    monkeypatch.setattr(module, "grouped", lambda *a, **k: None)
    monkeypatch.setattr(
        "sys.argv",
        ["03_analysis.py", "--predictions", str(predictions),
         "--results", str(results), "--only", "lendt_energy"],
    )

    module.main()

    stamp = provenance.read(results)
    assert stamp["outputs"] == ["lendt_energy"]
    assert stamp["complete"] is False


def test_workbook_carries_the_averaged_matrix_beside_each_table(tmp_path):
    """The thesis redraws these matrices itself and reads the numbers from here.

    Without a matrix sheet the only averaged figures leaving the pipeline are
    pixels in a PNG, and 2.5 would have to be written from the published pooled
    ones.
    """
    import pandas as pd

    module = _analysis()
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    results = tmp_path / "results"
    results.mkdir()

    frame = pd.concat([
        pd.DataFrame({
            "ground_truth": ["sit", "sit", "walk", "walk"],
            "activity": ["sit", "sit", "walk", "sit"],
            "id": subject,
        })
        for subject in ("a", "b")
    ])
    frame.to_parquet(predictions / "toy.parquet")

    module.grouped(
        predictions, results, [("toy", "Toy", "greens")], ["sit", "walk"], "toy"
    )

    sheets = pd.ExcelFile(results / "toy.xlsx").sheet_names
    assert "Toy" in sheets
    assert "Toy matrix" in sheets

    matrix = pd.read_excel(results / "toy.xlsx", sheet_name="Toy matrix")
    assert {"true", "pred", "mean", "lower", "upper"} <= set(matrix.columns)
    # Flat, not a MultiIndex: Excel blanks a repeated index key and the sheet
    # then reads back with NaN in seven rows out of eight.
    assert matrix["true"].notna().all()
    assert len(matrix) == 4


def _posture_frame(activity: list[str]) -> "object":
    import pandas as pd

    return pd.concat([
        pd.DataFrame({
            "ground_truth": ["lie", "lie", "sit", "sit"],
            "activity": activity,
            "id": subject,
        })
        for subject in ("a", "b")
    ])


def test_comparison_output_reports_what_the_back_sensor_adds(tmp_path):
    """2.5.2's second block needs this as an artefact, not a manual cross-read."""
    import pandas as pd

    module = _analysis()
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    results = tmp_path / "results"
    results.mkdir()

    _posture_frame(["sit"] * 4).to_parquet(predictions / "toy.parquet")
    _posture_frame(["lie", "lie", "sit", "sit"]).to_parquet(
        predictions / "toy_trunk.parquet"
    )

    module.compare(
        predictions, results, [("toy", "Toy")], ["lie", "sit"], ["lie"], "postures"
    )

    out = pd.read_excel(results / "postures.xlsx", sheet_name="Toy")
    assert {"thigh", "thigh_back", "delta"} <= set(out.columns)
    recall = out[out["metric"] == "recall"].iloc[0]
    assert recall["delta"] == 1.0


def test_every_comparison_has_a_back_sensor_table():
    """A comparison over a thigh-only dataset fails at read time, after the work.

    This asserted the THIGH name was panelled, which every entry trivially is.
    `compare()` reads `f"{name}_trunk"`, and adding `lendt_laboratory` -- which
    has no back sensor at all -- passed the old test and then raised
    FileNotFoundError. It guarded nothing it claimed to.
    """
    module = _analysis()
    panelled = _panelled_tables(module)
    for entries, _labels, _focus in module.COMPARISONS.values():
        for name, _title in entries:
            trunk = f"{name}_trunk"
            assert trunk in panelled, f"{name} is compared but {trunk} is never built"


def test_sheet_names_stay_unique_when_a_title_is_long():
    """`title[:31]` equals `(title + " matrix")[:31]` once a title reaches 31
    characters, and openpyxl then overlays both sheets in silence: matrix rows
    under metrics-table rows, sharing one header. No title is that long today;
    adding a second sheet per title is what created the hazard.
    """
    module = _analysis()
    long = "Older Adults From The Second Cohort"
    assert module.sheet_name(long) != module.sheet_name(f"{long} matrix")
    assert len(module.sheet_name(f"{long} matrix")) <= 31


def test_provenance_completeness_counts_comparisons_too(tmp_path, monkeypatch):
    """`complete` guards against a partial run masquerading as a full one."""
    from actimotus_validation import provenance

    module = _analysis()
    predictions = tmp_path / "predictions"
    results = tmp_path / "results"
    provenance.write(predictions, stage="activities", dataset="all", revision="x")

    monkeypatch.setattr(module, "grouped", lambda *a, **k: None)
    monkeypatch.setattr(module, "compare", lambda *a, **k: None)
    monkeypatch.setattr(
        "sys.argv",
        ["03_analysis.py", "--predictions", str(predictions), "--results", str(results)],
    )

    module.main()

    stamp = provenance.read(results)
    assert stamp["complete"] is True
    assert set(module.COMPARISONS) <= set(stamp["outputs"])
