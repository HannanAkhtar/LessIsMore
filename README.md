# HNeRV LR-Wide + Spatial Split Unified Project

This folder unifies the two code folders into one project:

1. **Whole-frame LR-conv + widening**: the `N = 1` case.
2. **Spatial split + LR-conv + widening**: the `N = 4` and `N = 16` cases.

The intended paper narrative is preserved in the code structure: spatial splitting is an outer wrapper around the same low-rank widened HNeRV idea, not a separate competing method.

## Folder structure

```text
HNeRV_LRWide_Spatial_Unified/
├── run_framework.py                 # unified command-line entry point
├── run_wholeframe_lrwide.py          # N=1 whole-frame LR-wide runner
├── src/                              # whole-frame LR-wide HNeRV implementation
├── spatial/                          # spatial wrapper experiments
│   ├── run_experiment.py             # prepare/train/extract/reconstruct/score spatial tiles
│   ├── configs/
│   │   ├── quadrant_equal_lrwide.json # N=4, 3M total budget
│   │   └── tile16_equal_3m_lrwide.json# N=16, 3M total budget
│   └── HNeRV-main/                   # modified HNeRV reference code used per tile
├── scripts/                          # ready-to-edit example commands
├── requirements-lowrank.txt
├── requirements-spatial-reference.txt
└── runs/                             # created at runtime
```

## Conceptual mapping

| Paper setting | Code mode | Main command | Notes |
|---|---|---|---|
| Dense HNeRV baseline | `wholeframe` | `run_wholeframe_lrwide.py` through `run_framework.py` | Trains/evaluates baseline as part of N=1 sweep |
| LR-wide HNeRV, `N = 1` | `wholeframe` | `python run_framework.py wholeframe ...` | Factorizes blocks 3 and 4, then widens |
| Spatial LR-wide HNeRV, `N = 4` | `spatial4` | `python run_framework.py spatial4 ...` | 2×2 quadrants, one LR-wide decoder per tile |
| Spatial LR-wide HNeRV, `N = 16` | `spatial16` | `python run_framework.py spatial16 ...` | 4×4 tiles, one LR-wide decoder per tile |

## Environment

Use your existing HNeRV conda environment if possible. The two original folders had different requirement files, so I kept them separately:

- `requirements-lowrank.txt`: used by the standalone whole-frame LR-wide implementation.
- `requirements-spatial-reference.txt`: the original HNeRV-main reference requirements for the spatial folder.

Recommended run setup:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /shared/tshanableh/conda_envs/hnerv310
unset PYTHONPATH
export PYTHONNOUSERSITE=1
```

## Data layout

The commands expect this layout:

```text
data/
├── bunny/
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── honeybee/
├── jockey/
├── readysetgo/
└── yachtride/
```

Use `--data_root` to point to the folder that contains the sequence folders.

## Run the whole-frame N=1 experiments

This trains/evaluates dense baseline, same-width LR ablation, and proposed LR-wide model for the listed sequences:

```bash
python run_framework.py wholeframe \
  --data_root ../data \
  --runs_root runs/wholeframe \
  --only bunny honeybee jockey readysetgo yachtride \
  --epochs 300 \
  --batch_size 2 \
  --model_bits 8 \
  --embed_bits 8 \
  --prune_ratio 0.10 \
  --gpu 0
```

The combined results are written to:

```text
runs/wholeframe/all_results.csv
```

To recompute quantized/pruned metrics from existing checkpoints without retraining:

```bash
python run_framework.py wholeframe \
  --data_root ../data \
  --runs_root runs/wholeframe \
  --only bunny honeybee jockey readysetgo yachtride \
  --epochs 300 \
  --skip_existing_models \
  --gpu 0
```

## Run the spatial N=4 experiment

```bash
python run_framework.py spatial4 \
  --data_root ../data \
  --sequence bunny \
  --epochs 300 \
  --eval_freq 30 \
  --expected_frames 132 \
  --gpu 0 \
  --clean
```

Key outputs are under:

```text
runs/spatial/quadrant_bunny_<tag>/
├── summary_total_quant.csv
├── part_metrics_quant.csv
└── recon_quant/metrics.csv
```

## Run the spatial N=16 experiment

```bash
python run_framework.py spatial16 \
  --data_root ../data \
  --sequence bunny \
  --epochs 300 \
  --eval_freq 30 \
  --expected_frames 132 \
  --gpu 0 \
  --clean
```

Key outputs are under:

```text
runs/spatial/tiles16_bunny_<tag>/
├── summary_total_quant.csv
├── part_metrics_quant.csv
└── recon_quant/metrics.csv
```

## Ready-made shell scripts

The `scripts/` folder contains editable examples:

```bash
bash scripts/run_wholeframe_n1_3m_q8e8_p10.sh
bash scripts/run_spatial_n4_3m_q8e6_p10.sh
bash scripts/run_spatial_n16_3m_q8e6_p10.sh
```

## Important notes

- Whole-frame experiments use `Q8/E8 + 10% pruning` by default.
- Spatial experiments use the bit depth stored in the JSON configs, currently `Q8/E6 + 10% pruning`.
- The spatial code was patched so `--work_root` can be moved safely; HNeRV-main output paths now resolve to the selected run folder.
- The same-width low-rank model is kept as an ablation. The main method is the LR-wide decoder and its spatial generalization.
