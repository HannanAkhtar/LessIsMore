# Configs

- `wholeframe/`: notes/configs copied from the whole-frame LR-wide project.
- `../spatial/configs/quadrant_equal_lrwide.json`: N=4 spatial LR-wide setting.
- `../spatial/configs/tile16_equal_3m_lrwide.json`: historical N=16 rank-22 setting.
- `../spatial/configs/tile16_rank23_0p198_unverified.json`: manuscript rank-23 candidate, not checkpoint-verified.
- `../spatial/configs/quadrant_dense_3m_reference.json`: dense four-tile target configuration.

The top-level `run_framework.py` selects the historical configs automatically for `spatial4` and `spatial16`. Use `spatial-custom --config` for the dense control or rank-23 candidate.
