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
