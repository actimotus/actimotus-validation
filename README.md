# ActiMotus Validation — Reproducibility Package

Reproduces the validation of the [ActiMotus](https://github.com/actimotus/actimotus)
human activity recognition algorithm against video ground truth on five public
datasets.

## Quick start

```bash
uv sync
uv run python scripts/01_features.py    # slow: ~2 GB download, tens of minutes
uv run python scripts/02_activities.py  # seconds
uv run python scripts/03_analysis.py    # seconds
```

Results land in `results/` as `.xlsx` tables and `.png` confusion matrices.
Nothing is precomputed — the cache is generated locally and gitignored.

Useful flags: `--dataset <name>` restricts stages 1 and 2 to one dataset,
`--limit N` caps subjects per dataset for a quick smoke run, and
`--only <stem>` builds a single stage 3 output.

## Datasets

| Dataset | Population | Sensors | HuggingFace | License |
|---|---|---|---|---|
| NTNU Adults | 31 adults | thigh + back, AX3 50 Hz | `josefheidler/har_adults_2021-harth` | MIT |
| NTNU Children | 46 typically-developing children | thigh + back, AX3 50 Hz | `josefheidler/har_children_2024-harth` | CC0-1.0 |
| NTNU Older Adults | 18 adults 70–95 | thigh + back, AX3 50 Hz | `josefheidler/har_older-adults_2023-harth` | CC-BY-4.0 |
| NTNU Walking Speeds | 24 adults | thigh + back, AX3 50 Hz | `josefheidler/har_ws_adults_2025-harth` | CC-BY-4.0 |
| Lendt Adults | 35 adults | lateral thigh, SENS 12.5 Hz | `josefheidler/har_adults_2024-lendt` | CC-BY-4.0 |

Each is pinned to an exact commit in `datasets.toml`. This package redistributes
no data; dataset licenses bind you at download. CC-BY-4.0 requires attribution to
the source study — cite the papers listed on each dataset card.

## Reference frame

The published datasets use the **hub frame**: `x` up along the limb, `y` right,
`z` forward (anterior). The two sensors need different treatment, and
`data.sensor_frame` therefore takes a required `to_acti_frame` argument rather
than guessing:

- **Thigh** — converted with `diag(1, -1, -1)` (negate `y` and `z`). ActiMotus
  expects the thigh `z` posterior. Without this, Lendt drops from 0.952 to 0.665
  accuracy.
- **Back** — left in the hub frame. ActiMotus expects the trunk `z` anterior,
  which hub already provides. Converting it collapses lying detection: older
  adults' `lie` recall falls from 0.992 to 0.000.

The check that both choices are right: with the frames correct,
`Activities(orientation=True)` and `Activities(orientation=False)` produce
identical predictions, because the flip detector finds nothing to fix. That
invariant is enforced by `tests/test_integration.py`. Flip detection stays
enabled so it can handle genuinely mis-worn sensors — never as a substitute for
getting the frame right.

`ntnu_children` is the one exception: its back sensor shows no such invariant.
The upstream ingest normalised thigh orientation per subject but never audited
the back, so back mounting genuinely varies between children and flip detection
does real work there.

## Differences from the published paper

These results differ from the tables in the manuscript for three independent
reasons. Do not attribute the whole difference to any one of them.

1. **ActiMotus version.** The paper used 2.3.0. Since then: directed rotational
   crossings, a bound on row detection, and a walk-only valid-day flag.
2. **Corrected upstream data.** `har_children_2024-harth` moved to v1.1.0 on
   2026-07-22, correcting per-subject thigh orientation; `har_adults_2024-lendt`
   is at v1.0.1, likewise post-correction. Both landed after the published
   analysis was run.
3. **Reference frame.** The paper consumed an intermediate data layout that no
   longer exists. This package consumes the hub-frame public release with the
   handling described above.

## Pipeline

Three stages, split where the cost is:

| Stage | Reads | Writes | Cost |
|---|---|---|---|
| `01_features.py` | HuggingFace | `cache/features/` | slow |
| `02_activities.py` | `cache/features/` | `cache/predictions/` | seconds |
| `03_analysis.py` | `cache/predictions/` | `results/` | seconds |

Each stage stamps its output with the ActiMotus version and the dataset revisions
it used, and refuses to consume a cache built from different inputs. That makes a
stale cache an error rather than a source of plausible wrong numbers. Override
with `--force` if you know what you are doing.

## Runtime and disk

Roughly 2 GB downloaded to the HuggingFace cache, plus extracted features under
`cache/`, and tens of minutes of CPU for stage 1. Stages 2 and 3 take seconds, so
re-running the classifier or re-cutting tables is cheap.

## Known limitations

- The Lendt data carries an uncorrected +11–15° roll offset. ActiMotus's `sit`
  threshold has its cliff near |roll| ≈ 41°, so there is margin, but it is not
  zero. Only a per-subject correction would be defensible and none is applied.
- `har_ws_adults_2025-harth` roll was never measured — it has no static postures,
  so orientation is verified by gait skew alone.
- Thresholds are ActiMotus's built-in `DEFAULT` config, tuned by Bayesian
  optimization outside this package. Re-deriving them is out of scope.

## Citing

See `CITATION.cff`. Cite both this package (Zenodo DOI) and the source datasets.
