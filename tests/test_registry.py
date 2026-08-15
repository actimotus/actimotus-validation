from pathlib import Path

import pytest

from actimotus_validation.registry import DatasetSpec, load_registry

REGISTRY = Path(__file__).parent.parent / "datasets.toml"


def test_loads_all_five_datasets():
    specs = load_registry(REGISTRY)
    assert set(specs) == {
        "ntnu_adults",
        "ntnu_children",
        "ntnu_older_adults",
        "ntnu_walking_speeds",
        "lendt_adults",
    }


def test_children_spec_fields():
    spec = load_registry(REGISTRY)["ntnu_children"]
    assert spec.hf_repo == "josefheidler/har_children_2024-harth"
    assert spec.revision == "fc5c6e4d5fa7e88c56a289e71b5199bba28cad75"
    assert spec.vendor == "Other"
    assert spec.thigh == "thigh_acc"
    assert spec.back == "back_acc"
    assert spec.filter == {"cohort": "td"}
    assert spec.labels == "ntnu"
    assert spec.split_by is None


def test_lendt_is_thigh_only_and_splits_by_condition():
    spec = load_registry(REGISTRY)["lendt_adults"]
    assert spec.vendor == "Sens"
    assert spec.thigh == "acc"
    assert spec.back is None
    assert spec.split_by == "condition"


def test_every_revision_is_a_full_sha():
    for spec in load_registry(REGISTRY).values():
        assert len(spec.revision) == 40, spec.name
        assert all(c in "0123456789abcdef" for c in spec.revision), spec.name


def test_spec_is_frozen():
    spec = load_registry(REGISTRY)["ntnu_adults"]
    with pytest.raises(Exception):
        spec.vendor = "Sens"  # type: ignore[misc]
