# HNeRV-DWT-v3: LR-wide DWT, Quadrant, and 16-tile experiments

This regenerated project adds:

- DWT mode: LL/LH/HL/HH subbands.
- Quadrant mode: Q0/Q1/Q2/Q3 spatial split.
- 16-tile mode: 4x4 spatial grid T00...T33.
- Q8/E6 quantized prediction extraction by default.
- P10 global magnitude pruning before final evaluation/quantization.
- Run outputs contained inside `runs/` under the project folder.

## Important metric behavior

Default reconstruction uses `--prediction_variant quant`, so full-frame PSNR/SSIM are computed from `visualize_model_quant` predictions. This matches Q8/E6 + P10 reporting. Use `--prediction_variant orig` only if you want unquantized/unpruned comparison from `visualize_model_orig`.

## Install/use environment

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /shared/tshanableh/conda_envs/hnerv310
unset PYTHONPATH
export PYTHONNOUSERSITE=1
```

## 16-tile debug run

```bash
cd /shared/tshanableh/NeRV/HNeRV-DWT-v3

/shared/tshanableh/conda_envs/hnerv310/bin/python run_experiment.py debug   --config /shared/tshanableh/NeRV/HNeRV-DWT-v3/configs/tile16_equal_3m_lrwide.json   --sequence bunny   --tag tile16_3m_debug_2ep_q8p10   --epochs 2   --eval_freq 1   --gpu 0   --expected_frames 132   --clean
```

## 16-tile full run, 3M target

```bash
cd /shared/tshanableh/NeRV/HNeRV-DWT-v3

/shared/tshanableh/conda_envs/hnerv310/bin/python run_experiment.py debug   --config /shared/tshanableh/NeRV/HNeRV-DWT-v3/configs/tile16_equal_3m_lrwide.json   --sequence bunny   --tag tile16_3m_300ep_q8p10   --epochs 300   --eval_freq 30   --gpu 0   --expected_frames 132   --clean
```

## 16-tile full run, 4M target

```bash
cd /shared/tshanableh/NeRV/HNeRV-DWT-v3

/shared/tshanableh/conda_envs/hnerv310/bin/python run_experiment.py debug   --config /shared/tshanableh/NeRV/HNeRV-DWT-v3/configs/tile16_equal_4m_lrwide.json   --sequence bunny   --tag tile16_4m_300ep_q8p10   --epochs 300   --eval_freq 30   --gpu 0   --expected_frames 132   --clean
```

## Check results

```bash
cat runs/tiles16_bunny_tile16_3m_300ep_q8p10/summary_total_quant.csv
cat runs/tiles16_bunny_tile16_3m_300ep_q8p10/part_metrics_quant.csv
cat runs/tiles16_bunny_tile16_3m_300ep_q8p10/recon_quant/metrics.csv | tail
```

## Configs

- `configs/quadrant_equal_lrwide.json`: previous 4-quadrant equal split, now Q8/E6 + P10.
- `configs/quadrant_unequal_lrwide.json`: stronger Q0/Q3 allocation.
- `configs/dwt_unequal_lrwide.json`: DWT split, now Q8/E6 + P10.
- `configs/tile16_equal_3m_lrwide.json`: 16 tiles, ~3M target, 0.1875M per tile.
- `configs/tile16_equal_4m_lrwide.json`: 16 tiles, ~4M target, 0.25M per tile.
