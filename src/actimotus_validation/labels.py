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
LABELS_SPLIT = [
    "lie", "sit", "stand", "shuffle", "slow-walk", "walk", "fast-walk", "stairs", "run",
    "bicycle",
]

# acti-motus 2.4.0 splits walking into three paces on its step rate. Walking is
# scored split wherever the protocol measured a speed, and folded into `walk`
# wherever it did not: free-living video cannot establish a pace.
PACES = {"slow-walk": "walk", "fast-walk": "walk"}

# Collapse to five classes. Unlisted activities pass through unchanged.
FUSED = {
    "lie": "sedentary",
    "sit": "sedentary",
    "shuffle": "stand",
    "stairs": "walk",
    "slow-walk": "walk",
    "fast-walk": "walk",
}

# Walking paces follow the hub's `walking_speed_bands_v1`, anchored on the 2024
# Adult Compendium: `stroll` (< 3.2 km/h) and `slow` (3.2-4.0) are light and pool
# into slow-walk; `normal` (4.0-5.5) is walk; `fast` (5.5-7.2) is fast-walk. A walk
# with no variant has no measured speed and stays `walk`.
#
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
    ("walk", "stroll"): "slow-walk",
    ("walk", "slow"): "slow-walk",
    ("walk", "normal"): "walk",
    ("run", None): "run",
    ("bicycle", None): "bicycle",
    ("bicycle", "coasting"): "bicycle",
    ("bicycle", "pedalling-seated"): "bicycle",
    ("bicycle", "pedalling-standing"): "bicycle",
}

# Cohort mean speeds 3.1, 4.9 and 6.1 km/h. v1.1.0 publishes the slowest as the
# compound `stroll/slow`, because 3.1 sits on the 3.2 km/h edge between the two.
_WALKING_SPEEDS: dict[tuple[str | None, str | None], str | None] = {
    ("walk", "stroll/slow"): "slow-walk",
    ("walk", "normal"): "walk",
    ("walk", "fast"): "fast-walk",
    ("run", None): "run",
}

# Treadmill walking at 2, 3, 4 and 5 km/h, plus an outdoor walk at a preferred
# speed that was never recorded. That walk has no band, so it is dropped: the
# dataset is here only for its paces.
_GAIT: dict[tuple[str | None, str | None], str | None] = {
    ("walk", "stroll"): "slow-walk",
    ("walk", "normal"): "walk",
    ("walk", None): None,
}

# The energy-expenditure cohort shares Lendt's activity vocabulary and adds a
# pre-session static calibration block: six sensor orientations on a cube, before
# the protocol starts. It is a sensor procedure, not a behaviour, so it is dropped
# rather than evaluated -- but it is listed, because an absent pair raises.
_LENDT_EE: dict[tuple[str | None, str | None], str | None] = {
    **_LENDT,
    ("calibration", None): None,
}

LABEL_TABLES = {
    "ntnu": _NTNU,
    "lendt": _LENDT,
    "lendt_ee": _LENDT_EE,
    "walking_speeds": _WALKING_SPEEDS,
    "gait": _GAIT,
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
        table: Key into LABEL_TABLES, e.g. 'ntnu', 'lendt' or 'walking_speeds'.
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


def merge_paces(activities: pd.Series) -> pd.Series:
    """Fold the three walking paces into `walk`; everything else is unchanged."""
    return activities.astype(str).replace(PACES)


def fuse(activities: pd.Series) -> pd.Series:
    """Collapse the activity vocabulary to the five fused classes."""
    return activities.astype(str).replace(FUSED)
