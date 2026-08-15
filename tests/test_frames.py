import pandas as pd
import pytest

from actimotus_validation.frames import hub_to_acti


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"acc_x": [1.0, 0.0], "acc_y": [0.5, -0.25], "acc_z": [0.9, -0.9]},
        index=pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:01"]),
    )


def test_negates_y_and_z_leaves_x():
    out = hub_to_acti(_frame())
    assert list(out["acc_x"]) == [1.0, 0.0]
    assert list(out["acc_y"]) == [-0.5, 0.25]
    assert list(out["acc_z"]) == [-0.9, 0.9]


def test_is_an_involution():
    df = _frame()
    pd.testing.assert_frame_equal(hub_to_acti(hub_to_acti(df)), df)


def test_does_not_mutate_input():
    df = _frame()
    before = df.copy()
    hub_to_acti(df)
    pd.testing.assert_frame_equal(df, before)


def test_preserves_index():
    df = _frame()
    pd.testing.assert_index_equal(hub_to_acti(df).index, df.index)


def test_rejects_missing_columns():
    df = pd.DataFrame({"acc_x": [1.0], "acc_y": [1.0]})
    with pytest.raises(ValueError, match="acc_z"):
        hub_to_acti(df)
