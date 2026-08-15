import json

import pytest

from actimotus_validation.provenance import StaleCacheError, read, verify, write


def test_write_then_read_roundtrip(tmp_path):
    write(tmp_path, stage="features", dataset="ntnu_adults", revision="a" * 40, extra={"n": 3})
    stamp = read(tmp_path)
    assert stamp["stage"] == "features"
    assert stamp["dataset"] == "ntnu_adults"
    assert stamp["revision"] == "a" * 40
    assert stamp["n"] == 3
    assert "actimotus_version" in stamp
    assert "created" in stamp


def test_write_records_the_installed_actimotus_version(tmp_path):
    from importlib.metadata import version

    write(tmp_path, stage="features", dataset="d", revision="b" * 40)
    assert read(tmp_path)["actimotus_version"] == version("acti-motus")


def test_verify_passes_when_everything_matches(tmp_path):
    from importlib.metadata import version

    write(tmp_path, stage="features", dataset="d", revision="c" * 40)
    verify(tmp_path, revision="c" * 40, actimotus_version=version("acti-motus"))


def test_verify_raises_on_revision_mismatch(tmp_path):
    from importlib.metadata import version

    write(tmp_path, stage="features", dataset="d", revision="c" * 40)
    with pytest.raises(StaleCacheError, match="revision"):
        verify(tmp_path, revision="d" * 40, actimotus_version=version("acti-motus"))


def test_verify_raises_on_actimotus_version_mismatch(tmp_path):
    write(tmp_path, stage="features", dataset="d", revision="c" * 40)
    with pytest.raises(StaleCacheError, match="actimotus"):
        verify(tmp_path, revision="c" * 40, actimotus_version="0.0.1")


def test_verify_raises_when_stamp_is_absent(tmp_path):
    with pytest.raises(StaleCacheError, match="no provenance"):
        verify(tmp_path, revision="c" * 40, actimotus_version="1.0.0")


def test_stamp_is_readable_json_on_disk(tmp_path):
    write(tmp_path, stage="features", dataset="d", revision="e" * 40)
    payload = json.loads((tmp_path / "provenance.json").read_text())
    assert payload["dataset"] == "d"


def test_digest_is_stable_and_order_independent():
    from actimotus_validation.provenance import digest

    a = digest({"one": "a" * 40, "two": "b" * 40})
    b = digest({"two": "b" * 40, "one": "a" * 40})
    assert a == b
    assert len(a) == 40


def test_digest_changes_when_any_revision_changes():
    from actimotus_validation.provenance import digest

    assert digest({"one": "a" * 40}) != digest({"one": "c" * 40})
