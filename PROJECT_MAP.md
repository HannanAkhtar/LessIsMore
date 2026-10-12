# Project Map / What Came From Where

## From `lowrank_quant_pruned.zip`

- `src/`: whole-frame dense and low-rank HNeRV implementation.
- `run_wholeframe_lrwide.py`: renamed copy of `run_all_lowrank_quant_pruned.py`.
- `README_wholeframe_lrwide.md`: original patch notes for quantized/pruned playback evaluation.
- `configs/wholeframe/`: original whole-frame config notes.
- `scripts/lowrank/`: original low-rank helper scripts.

## From `spatial-split.zip`

- `spatial/run_experiment.py`: spatial prepare/train/extract/reconstruct/score pipeline.
- `spatial/configs/`: quadrant, 16-tile, and DWT configs.
- `spatial/HNeRV-main/`: modified HNeRV reference implementation used to train each tile model.
- `spatial/run_batch_splits.py`: batch runner; patched to use the current folder instead of hard-coded absolute paths.

## New unified files

- `README.md`: single project instructions.
- `run_framework.py`: one command-line entry point for N=1, N=4, and N=16 runs.
- `scripts/run_wholeframe_n1_3m_q8e8_p10.sh`
- `scripts/run_spatial_n4_3m_q8e6_p10.sh`
- `scripts/run_spatial_n16_3m_q8e6_p10.sh`

## Main methodological interpretation

- `N = 1`: whole-frame low-rank widened HNeRV.
- `N = 4`: same LR-wide design applied independently to four quadrants.
- `N = 16`: same LR-wide design applied independently to a 4×4 grid.

Spatial splitting is the outer wrapper. The low-rank widened decoder is the core decoder design.
