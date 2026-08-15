import pandas as pd
import pytest

from actimotus_validation.data import ground_truth_1s, sensor_frame
from actimotus_validation.registry import DatasetSpec

DUAL = DatasetSpec(
    name="d", hf_repo="x/y", revision="a" * 40, vendor="Other",
    thigh="thigh_acc", back="back_acc", labels="ntnu",
)


def _raw() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=4, freq="500ms", tz="UTC")
    return pd.DataFrame(
        {
            "thigh_acc_x": [1.0, 1.0, 1.0, 1.0],
            "thigh_acc_y": [0.2, 0.2, 0.2, 0.2],
            "thigh_acc_z": [0.9, 0.9, 0.9, 0.9],
            "back_acc_x": [0.8, 0.8, 0.8, 0.8],
            "back_acc_y": [0.1, 0.1, 0.1, 0.1],
            "back_acc_z": [0.3, 0.3, 0.3, 0.3],
            "label": ["sit", "sit", "walk", "walk"],
            "variant": [None, None, None, None],
        },
        index=idx,
    )


def test_sensor_frame_converts_thigh_to_acti_frame():
    out = sensor_frame(_raw(), "thigh_acc", to_acti_frame=True)
    assert list(out.columns) == ["acc_x", "acc_y", "acc_z"]
    assert out["acc_x"].iloc[0] == 1.0
    assert out["acc_y"].iloc[0] == -0.2   # negated
    assert out["acc_z"].iloc[0] == -0.9   # negated


def test_sensor_frame_leaves_back_in_the_hub_frame():
    """acti-motus wants the trunk z anterior, which is what hub already is."""
    out = sensor_frame(_raw(), "back_acc", to_acti_frame=False)
    assert out["acc_x"].iloc[0] == 0.8
    assert out["acc_y"].iloc[0] == 0.1    # unchanged
    assert out["acc_z"].iloc[0] == 0.3    # unchanged


def test_sensor_frame_can_convert_the_back_prefix_too():
    """The choice is the caller's; the function is not sensor-role aware."""
    out = sensor_frame(_raw(), "back_acc", to_acti_frame=True)
    assert out["acc_z"].iloc[0] == -0.3


def test_sensor_frame_requires_an_explicit_frame_choice():
    with pytest.raises(TypeError):
        sensor_frame(_raw(), "thigh_acc")  # type: ignore[call-arg]


def test_sensor_frame_raises_when_prefix_absent():
    with pytest.raises(ValueError, match="calf_acc"):
        sensor_frame(_raw(), "calf_acc", to_acti_frame=True)


def test_ground_truth_takes_the_per_second_mode():
    idx = pd.date_range("2024-01-01", periods=5, freq="200ms", tz="UTC")
    raw = pd.DataFrame(
        {"label": ["sit", "sit", "sit", "walk", "walk"], "variant": [None] * 5}, index=idx
    )
    out = ground_truth_1s(raw, "ntnu")
    assert list(out.columns) == ["ground_truth"]
    assert out["ground_truth"].iloc[0] == "sit"


def test_ground_truth_index_is_one_second_resolution():
    out = ground_truth_1s(_raw(), "ntnu")
    assert (out.index.to_series().diff().dropna() == pd.Timedelta("1s")).all()


def test_ground_truth_drops_unevaluated_labels():
    idx = pd.date_range("2024-01-01", periods=2, freq="1s", tz="UTC")
    raw = pd.DataFrame({"label": ["jumping", "walk"], "variant": [None, None]}, index=idx)
    out = ground_truth_1s(raw, "ntnu")
    assert list(out["ground_truth"]) == ["walk"]
