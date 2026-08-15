import pandas as pd
import pytest

from actimotus_validation.labels import (
    FUSED,
    LABELS,
    LABELS_FUSED,
    LABELS_WALKING_SPEEDS,
    UnknownLabelError,
    fuse,
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


def test_walking_speeds_fast_is_fast_walk_others_are_walk():
    pairs = [("walk", "slow"), ("walk", "moderate"), ("walk", "fast"), ("run", None)]
    out = resolve_series("walking_speeds", *_series(pairs))
    assert list(out) == ["walk", "walk", "fast-walk", "run"]


def test_lendt_fast_walk_stays_walk():
    """Only the walking-speeds protocol resolves speed; free-living video cannot."""
    out = resolve_series("lendt", *_series([("walk", "fast")]))
    assert list(out) == ["walk"]


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
    s = pd.Series(["lie", "sit", "stand", "shuffle", "walk", "stairs", "fast-walk", "run", "bicycle"])
    assert list(fuse(s)) == [
        "sedentary", "sedentary", "stand", "stand", "walk", "walk", "walk", "run", "bicycle",
    ]


def test_label_orders():
    assert LABELS == ["lie", "sit", "stand", "shuffle", "walk", "stairs", "run", "bicycle"]
    assert LABELS_FUSED == ["sedentary", "stand", "walk", "run", "bicycle"]
    assert LABELS_WALKING_SPEEDS == ["shuffle", "walk", "fast-walk", "run"]
    assert set(FUSED) == {"lie", "sit", "shuffle", "stairs", "fast-walk"}
