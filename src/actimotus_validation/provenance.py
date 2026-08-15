"""Stage provenance: what produced a cache directory, and is it still valid."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

FILENAME = "provenance.json"


def digest(revisions: dict[str, str]) -> str:
    """One stable 40-char id standing for a set of per-dataset revisions.

    Stage 2 spans several datasets, so its stamp needs a single `revision` value.
    Order-independent, and changes if any member revision changes.
    """
    payload = json.dumps(revisions, sort_keys=True).encode()

    return hashlib.sha1(payload).hexdigest()


class StaleCacheError(RuntimeError):
    """A cached stage was produced under different inputs than the current run."""


def actimotus_version() -> str:
    """The installed acti-motus version."""
    return version("acti-motus")


def write(
    directory: Path,
    *,
    stage: str,
    dataset: str,
    revision: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Stamp `directory` with what produced it."""
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "stage": stage,
        "dataset": dataset,
        "revision": revision,
        "actimotus_version": actimotus_version(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)

    (directory / FILENAME).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read(directory: Path) -> dict[str, Any]:
    """Read a stamp. Raises StaleCacheError if absent."""
    path = directory / FILENAME
    if not path.exists():
        raise StaleCacheError(f"no provenance stamp at {path}; run the previous stage first")

    return json.loads(path.read_text())


def verify(directory: Path, *, revision: str, actimotus_version: str) -> dict[str, Any]:
    """Check a cached stage was built with the expected inputs.

    Raises:
        StaleCacheError: If the stamp is missing, or its dataset revision or
            acti-motus version differs from what this run expects.
    """
    stamp = read(directory)

    if stamp.get("revision") != revision:
        raise StaleCacheError(
            f"{directory} was built from dataset revision {stamp.get('revision')!r} "
            f"but this run expects {revision!r}. Re-run the previous stage, or pass --force."
        )

    if stamp.get("actimotus_version") != actimotus_version:
        raise StaleCacheError(
            f"{directory} was built with actimotus {stamp.get('actimotus_version')!r} "
            f"but {actimotus_version!r} is installed. Re-run the previous stage, or pass --force."
        )

    return stamp
