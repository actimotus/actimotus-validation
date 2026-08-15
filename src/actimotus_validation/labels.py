"""Ground-truth label resolution.

The published datasets carry separate `label` and `variant` columns. Earlier
intermediate layouts used pre-composed strings ("stairs (descending)",
"bicycle (sit)", "slow-walk") and the variant vocabulary has since changed, so
mapping is an explicit reviewable table rather than string manipulation.

Every table below is the complete observed (label, variant) inventory for its
datasets. An unlisted pair raises: silently dropping unknown labels would let an
upstream rename quietly shrink the evaluation set.
"""

from __future__ import annotations

import pandas as pd

# Reported label sets, in display order.
LABELS = ["lie", "sit", "stand", "shuffle", "walk", "stairs", "run", "bicycle"]
LABELS_FUSED = ["sedentary", "stand", "walk", "run", "bicycle"]
LABELS_WALKING_SPEEDS = ["shuffle", "walk", "fast-walk", "run"]

# Collapse to five classes. Unlisted activities pass through unchanged.
FUSED = {
    "lie": "sedentary",
    "sit": "sedentary",
    "shuffle": "stand",
    "stairs": "walk",
    "fast-walk": "walk",
}

# None as a value means "drop this row" -- a deliberately unevaluated label,
# distinct from a pair that is absent from the table entirely (which raises).
_NTNU: dict[tuple[str | None, str | None], str | None] = {
    ("lie", None): "lie",
    ("sit", None): "sit",
    ("stand", None): "stand",
    ("shuffle", None): "shuffle",
    ("walk", None): "walk",
    ("run", None): "run",
    ("bending", None): "stand",
    ("jumping", None): None,
    ("transition", None): None,
    ("stairs", "ascending"): "stairs",
    ("stairs", "descending"): "stairs",
    ("bicycle", "seated"): "bicycle",
    ("bicycle", "standing"): "bicycle",
    ("bicycle", "pedalling-seated"): "bicycle",
    ("bicycle", "pedalling-standing"): "bicycle",
    ("bicycle", "coasting-seated"): "bicycle",
    ("bicycle", "coasting-standing"): "bicycle",
}

_LENDT: dict[tuple[str | None, str | None], str | None] = {
    ("sit", None): "sit",
    ("stand", None): "stand",
    ("shuffle", None): "shuffle",
    ("stairs", None): "stairs",
    ("lie", None): "lie",
    ("lie", "prone"): "lie",
    ("lie", "side"): "lie",
    ("lie", "supine"): "lie",
    ("walk", None): "walk",
    ("walk", "slow"): "walk",
    ("walk", "moderate"): "walk",
    ("walk", "fast"): "walk",
    ("run", None): "run",
    ("run", "slow"): "run",
    ("run", "moderate"): "run",
    ("run", "fast"): "run",
    ("bicycle", "slow"): "bicycle",
    ("bicycle", "moderate"): "bicycle",
    ("bicycle", "fast"): "bicycle",
    ("bicycle", "coasting"): "bicycle",
    ("bicycle", "pedalling-seated"): "bicycle",
    ("bicycle", "pedalling-standing"): "bicycle",
}

# Cohort mean speeds: slow 3.1, moderate 4.9, fast 6.1 km/h. 'fast' is the
# successor of the earlier 'brisk-walk' label, which mapped to fast-walk.
_WALKING_SPEEDS: dict[tuple[str | None, str | None], str | None] = {
    ("walk", "slow"): "walk",
    ("walk", "moderate"): "walk",
    ("walk", "fast"): "fast-walk",
    ("run", None): "run",
}

LABEL_TABLES = {
    "ntnu": _NTNU,
    "lendt": _LENDT,
    "walking_speeds": _WALKING_SPEEDS,
}


class UnknownLabelError(KeyError):
    """A (label, variant) pair not present in the dataset's table."""


def _normalise(value: object) -> str | None:
    """Map every flavour of null to None. Lendt uses the literal string 'None'."""
    if value is None or pd.isna(value):
        return None
    text = str(value)
    return None if text in {"None", "nan", "<NA>", ""} else text


def resolve_series(table: str, label: pd.Series, variant: pd.Series) -> pd.Series:
    """Resolve (label, variant) pairs to canonical activities.

    Args:
        table: Key into LABEL_TABLES -- 'ntnu', 'lendt' or 'walking_speeds'.
        label: Raw label column.
        variant: Raw variant column, aligned with `label`.

    Returns:
        A Series of activity names, NA where the pair is deliberately unevaluated
        (jumping, transition) or the label is null.

    Raises:
        UnknownLabelError: If any pair is absent from the table.
    """
    mapping = LABEL_TABLES[table]

    pairs = [
        (_normalise(a), _normalise(b))
        for a, b in zip(label.to_numpy(), variant.to_numpy(), strict=True)
    ]

    unknown = {p for p in set(pairs) if p != (None, None) and p not in mapping}
    if unknown:
        listed = ", ".join(f"({a!r}, {b!r})" for a, b in sorted(unknown, key=str))
        raise UnknownLabelError(
            f"unknown (label, variant) pair(s) for table {table!r}: {listed}. "
            "The upstream vocabulary changed; update labels.py rather than dropping them."
        )

    resolved = [mapping.get(p) if p != (None, None) else None for p in pairs]

    return pd.Series(resolved, index=label.index, dtype="object")


def fuse(activities: pd.Series) -> pd.Series:
    """Collapse the eight-activity vocabulary to the five fused classes."""
    return activities.astype(str).replace(FUSED)
