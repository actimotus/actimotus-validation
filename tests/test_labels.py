import pandas as pd
import pytest

from actimotus_validation.labels import (
    FUSED,
    LABELS,
    LABELS_FUSED,
    LABELS_SPLIT,
    UnknownLabelError,
    fuse,
    merge_paces,
    resolve_series,
)


def _series(pairs):
    labels = pd.Series([p[0] for p in pairs], dtype="object")
    variants = pd.Series([p[1] for p in pairs], dtype="object")
    return labels, variants


def test_ntnu_bicycle_variants_all_collapse():
    pairs = [
        ("bicycle", "pedalling-seated"),
        ("bicycle", "pedalling-standing"),
        ("bicycle", "coasting-seated"),
        ("bicycle", "coasting-standing"),
        ("bicycle", "seated"),
        ("bicycle", "standing"),
    ]
    out = resolve_series("ntnu", *_series(pairs))
    assert list(out) == ["bicycle"] * 6


def test_ntnu_stairs_directions_collapse():
    out = resolve_series("ntnu", *_series([("stairs", "ascending"), ("stairs", "descending")]))
    assert list(out) == ["stairs", "stairs"]


def test_ntnu_bending_is_stand_and_jumping_transition_are_dropped():
    out = resolve_series(
        "ntnu", *_series([("bending", None), ("jumping", None), ("transition", None)])
    )
    assert out.iloc[0] == "stand"
    assert pd.isna(out.iloc[1])
    assert pd.isna(out.iloc[2])


def test_walking_speeds_resolves_three_paces():
    """v1.1.0 publishes the compound `stroll/slow`, pooled below 4.0 km/h."""
    pairs = [("walk", "stroll/slow"), ("walk", "normal"), ("walk", "fast"), ("run", None)]
    out = resolve_series("walking_speeds", *_series(pairs))
    assert list(out) == ["slow-walk", "walk", "fast-walk", "run"]


def test_lendt_paces_from_the_speed_bands():
    """`stroll` and `slow` sit below 4.0 km/h, the Compendium's light edge."""
    pairs = [("walk", "stroll"), ("walk", "slow"), ("walk", "normal")]
    out = resolve_series("lendt", *_series(pairs))
    assert list(out) == ["slow-walk", "slow-walk", "walk"]


def test_lendt_walk_without_a_speed_stays_walk():
    """Free-living video cannot establish a pace, so the walk carries none."""
    out = resolve_series("lendt", *_series([("walk", None)]))
    assert list(out) == ["walk"]


def test_lendt_old_protocol_relative_variants_are_gone():
    """The pre-v1.0.3 variants were relative to the protocol, not to a speed."""
    with pytest.raises(UnknownLabelError, match="moderate"):
        resolve_series("lendt", *_series([("walk", "moderate")]))


def test_gait_treadmill_paces_and_the_outdoor_walk_is_dropped():
    """The outdoor walk was at an unrecorded preferred speed, so it has no band."""
    pairs = [("walk", "stroll"), ("walk", "normal"), ("walk", None)]
    out = resolve_series("gait", *_series(pairs))
    assert list(out[:2]) == ["slow-walk", "walk"]
    assert pd.isna(out.iloc[2])


def test_merge_paces_folds_every_pace_into_walk():
    s = pd.Series(["slow-walk", "walk", "fast-walk", "run", "sit"])
    assert list(merge_paces(s)) == ["walk", "walk", "walk", "run", "sit"]


def test_lendt_lie_postures_collapse():
    pairs = [("lie", "prone"), ("lie", "side"), ("lie", "supine"), ("lie", None)]
    out = resolve_series("lendt", *_series(pairs))
    assert list(out) == ["lie"] * 4


def test_lendt_string_none_is_treated_as_null():
    """Lendt encodes nulls inconsistently: the string 'None' and <NA> both occur."""
    out = resolve_series("lendt", *_series([("walk", "None"), ("None", "None")]))
    assert out.iloc[0] == "walk"
    assert pd.isna(out.iloc[1])


def test_null_label_is_dropped():
    out = resolve_series("ntnu", *_series([(None, None)]))
    assert pd.isna(out.iloc[0])


def test_unknown_pair_raises_naming_the_pair():
    with pytest.raises(UnknownLabelError, match=r"swimming.*butterfly"):
        resolve_series("ntnu", *_series([("swimming", "butterfly")]))


def test_known_label_with_unknown_variant_raises():
    with pytest.raises(UnknownLabelError, match="moonwalk"):
        resolve_series("ntnu", *_series([("walk", "moonwalk")]))


def test_fuse_collapses_to_five_classes():
    s = pd.Series(
        ["lie", "sit", "stand", "shuffle", "slow-walk", "walk", "stairs", "fast-walk", "run",
         "bicycle"]
    )
    assert list(fuse(s)) == [
        "sedentary", "sedentary", "stand", "stand", "walk", "walk", "walk", "walk", "run",
        "bicycle",
    ]


def test_label_orders():
    assert LABELS == ["lie", "sit", "stand", "shuffle", "walk", "stairs", "run", "bicycle"]
    assert LABELS_FUSED == ["sedentary", "stand", "walk", "run", "bicycle"]
    assert LABELS_SPLIT == [
        "lie", "sit", "stand", "shuffle", "slow-walk", "walk", "fast-walk", "stairs", "run",
        "bicycle",
    ]
    assert set(FUSED) == {"lie", "sit", "shuffle", "stairs", "slow-walk", "fast-walk"}


def test_lendt_ee_drops_the_calibration_block():
    """har_ee_adults_2024-lendt carries a pre-session static calibration block.

    It is a sensor procedure, not a behaviour, so it is unevaluated -- but it must
    be listed explicitly, or the table raises on it like any unknown label.
    """
    out = resolve_series(
        "lendt_ee",
        pd.Series(["calibration", "walk"]),
        pd.Series([None, "slow"]),
    )

    assert out.isna().tolist() == [True, False]
    assert out.iloc[1] == "slow-walk"
