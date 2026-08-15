"""Coordinate frame conversion between the published data and acti-motus."""

from __future__ import annotations

import pandas as pd

REQUIRED = ("acc_x", "acc_y", "acc_z")


def hub_to_acti(df: pd.DataFrame) -> pd.DataFrame:
    """Convert hub-frame accelerometry to the frame acti-motus expects.

    The published datasets use the hub frame: x up along the limb, y right,
    z forward (anterior). acti-motus expects z posterior -- its inside-out
    detector flags a sensor when sitting median z > +0.1, so hub-frame data
    trips that check on every correctly-worn subject.

    The conversion is diag(1, -1, -1): negate y and z. It is a proper rotation
    (180 degrees about the long axis) and its own inverse.

    Skipping it costs the Lendt dataset 29 accuracy points. Activities(
    orientation=True) happens to recover the correct frame on the NTNU datasets
    but does not fire on Lendt or walking speeds, so it is not a substitute.

    Args:
        df: Frame with acc_x, acc_y, acc_z columns in g.

    Returns:
        A new frame with y and z negated. The input is not modified.

    Raises:
        ValueError: If any of acc_x, acc_y, acc_z is missing.
    """
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"hub_to_acti requires columns {REQUIRED}; missing {missing}")

    out = df.copy()
    out["acc_y"] = -out["acc_y"]
    out["acc_z"] = -out["acc_z"]

    return out
