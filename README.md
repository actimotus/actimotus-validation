# ActiMotus Validation — Reproducibility Package

Reproduces the validation of the [ActiMotus](https://github.com/actimotus/actimotus)
human activity recognition algorithm against video ground truth, on five public
accelerometry datasets covering children, adults and older adults.

Three commands regenerate every table and confusion matrix in the study. Nothing is
precomputed: the data is fetched from HuggingFace at pinned revisions and the
algorithm comes from PyPI, so the package supplies only the pipeline between them.

## Run it

```bash
uv sync
uv run python scripts/01_features.py    # ~2 GB download, tens of minutes
uv run python scripts/02_activities.py  # seconds
uv run python scripts/03_analysis.py    # seconds
```

Results land in `results/` as `.xlsx` tables and `.png` confusion matrices.
Add `--dataset <name>` to restrict stages 1–2 to one dataset, `--limit N` for a quick
smoke run, `--only <stem>` to rebuild a single stage 3 output.

Each stage stamps its output with the ActiMotus version and dataset revisions it
used, and refuses to consume a cache built from different inputs — a stale cache is
an error, not a source of plausible wrong numbers.

## Datasets

| Dataset | Population | Sensors | HuggingFace | License |
|---|---|---|---|---|
| NTNU Adults | 31 adults | thigh + back, AX3 50 Hz | `josefheidler/har_adults_2021-harth` | MIT |
| NTNU Children | 46 typically-developing children | thigh + back, AX3 50 Hz | `josefheidler/har_children_2024-harth` | CC0-1.0 |
| NTNU Older Adults | 18 adults aged 70–95 | thigh + back, AX3 50 Hz | `josefheidler/har_older-adults_2023-harth` | CC-BY-4.0 |
| NTNU Walking Speeds | 24 adults | thigh + back, AX3 50 Hz | `josefheidler/har_ws_adults_2025-harth` | CC-BY-4.0 |
| Lendt Adults | 35 adults | lateral thigh, SENS 12.5 Hz | `josefheidler/har_adults_2024-lendt` | CC-BY-4.0 |

Revisions are pinned in `datasets.toml`. This package redistributes no data; the
dataset licenses bind you at download, and CC-BY-4.0 requires attribution to the
source study listed on each dataset card.

## Results

Overall agreement with video ground truth, ActiMotus 2.3.3 with its built-in
`DEFAULT` thresholds:

| Dataset | Thigh only | Thigh + trunk |
|---|---|---|
| Lendt, laboratory | 0.995 | — |
| Lendt, free-living | 0.889 | — |
| NTNU Adults | 0.824 | 0.880 |
| NTNU Children | 0.831 | 0.848 |
| NTNU Older Adults | 0.791 | 0.824 |
| NTNU Walking Speeds | 0.795 | — |

Per-activity precision, recall and F1 with 90% confidence intervals, computed per
participant and then averaged across participants, are written to the `.xlsx`
tables in `results/`, alongside confusion matrices as `.png`. Both the eight-activity
vocabulary and the fused five-class collapse (sedentary, standing, walking, running,
cycling) are reported.

## Sensor orientation

The published datasets use a hub frame with `x` up along the limb, `y` right and
`z` forward. The two sensors need opposite treatment, so `data.sensor_frame` takes a
required `to_acti_frame` argument rather than guessing:

- **Thigh** — rotated 180° about its long axis (`y` and `z` negated). ActiMotus
  expects the thigh `z` posterior. Without this, Lendt falls from 0.952 to 0.665.
- **Back** — left as published. ActiMotus expects the trunk `z` anterior, which the
  hub frame already provides.

The data itself is correctly worn: roll measured from video-labelled sitting is
−2.4° (adults), +3.1° (children), +0.4° (older adults) and +13.7° (Lendt), with 157
of 161 participants within 30° of zero and none near 180°. The rotation translates
between coordinate conventions; it does not correct a mounting error.

Automatic flip detection stays enabled, but only as a guard against genuinely
mis-worn sensors — never as a substitute for the conversion. With the frames correct
it changes nothing on 153 of 154 thigh recordings, and both `orientation=True` and
`orientation=False` give identical predictions, an invariant enforced by
`tests/test_integration.py`.

## Known limitations

- The Lendt data carries an uncorrected +11–15° roll offset. ActiMotus's sitting
  threshold has its cliff near 41°, so there is margin, but it is not zero. Only a
  per-subject correction would be defensible and none is applied.
- Walking-speeds roll was never measured; that protocol has no static postures, so
  its orientation is verified by gait skew alone.
- Thresholds are ActiMotus's built-in `DEFAULT` configuration, tuned by Bayesian
  optimization outside this package. Re-deriving them is out of scope.
- Lying detection from the thigh alone fails for adults and older adults (recall
  0.06 in both); the back sensor resolves it (0.87 and 0.78). Children are the
  exception, reaching 0.97 from the thigh alone.

## Citing

Cite this package via the Zenodo DOI, and cite the source datasets separately —
see `CITATION.cff` and the individual dataset cards.

## License

MIT. See `LICENSE`.
