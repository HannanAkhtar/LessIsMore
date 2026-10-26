# HNeRV-DWT Spatial Split Experiments

This README explains how to run and summarize the spatial-split HNeRV experiments for:

1. **4-quadrant split**: each `640×1280` frame is split into four `320×640` quadrants.
2. **16-tile split**: each `640×1280` frame is split into a `4×4` grid of sixteen `160×320` tiles.

Both methods use low-rank widened HNeRV submodels, Q8/E6 quantization, and 10% pruning.

---

## 1. Project Location

All commands assume the project is located at:

```bash
/shared/tshanableh/NeRV/HNeRV-DWT
```

Use the clean Python environment:

```bash
/shared/tshanableh/conda_envs/hnerv310/bin/python
```

Before running experiments, clear possible conflicting user/system Python paths:

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
```

---

## 2. Main Files

```text
HNeRV-DWT/
  run_experiment.py
  run_batch_splits.py
  configs/
    quadrant_equal_lrwide.json
    tile16_equal_3m_exact_v2_lrwide.json
  HNeRV-main/
  runs/
```

Important configs:

| Method | Config |
|---|---|
| 4-quadrant | `configs/quadrant_equal_lrwide.json` |
| 16-tile ~3M | `configs/tile16_equal_3m_exact_v2_lrwide.json` |

Both configs should contain:

```json
"quant_model_bit": 8,
"quant_embed_bit": 6,
"prune_ratio": 0.10
```

Verify:

```bash
cd /shared/tshanableh/NeRV/HNeRV-DWT

grep -n "quant_model_bit\|quant_embed_bit\|prune_ratio" configs/quadrant_equal_lrwide.json
grep -n "quant_model_bit\|quant_embed_bit\|prune_ratio" configs/tile16_equal_3m_exact_v2_lrwide.json
```

---

## 3. Preprocessing and Evaluation Setup

Input frames are center-cropped:

```text
720×1280 → 640×1280
```

Then the cropped frame is split spatially:

| Method | Split | Part size |
|---|---:|---:|
| 4-quadrant | 2×2 | 320×640 |
| 16-tile | 4×4 | 160×320 |

All models are trained for 300 epochs.

After training:

```text
10% global magnitude pruning
8-bit model weight quantization
6-bit embedding quantization
```

Final PSNR/SSIM values are computed from the reconstructed full `640×1280` RGB frames using the quantized/pruned predictions.

---

## 4. Running 4-Quadrant Experiments

Use GPU 0:

```bash
cd /shared/tshanableh/NeRV/HNeRV-DWT

unset PYTHONPATH
export PYTHONNOUSERSITE=1

/shared/tshanableh/conda_envs/hnerv310/bin/python run_batch_splits.py \
  --method quadrant \
  --gpu 0 \
  --epochs 300 \
  --eval_freq 30 \
  --expected_frames 132 \
  --sequences bunny honeybee jockey readysetgo yachtride bosphorus \
  --skip_existing \
  --aggregate_after
```

Example run folders:

```text
runs/quadrant_bunny_quadrant_3m_300ep_q8p10/
runs/quadrant_honeybee_quadrant_3m_300ep_q8p10/
```

Monitor a log:

```bash
tail -f /shared/tshanableh/NeRV/HNeRV-DWT/runs/logs/bunny_quadrant_3m_300ep_q8p10.log
```

---

## 5. Running 16-Tile Experiments

Use GPU 1:

```bash
cd /shared/tshanableh/NeRV/HNeRV-DWT

unset PYTHONPATH
export PYTHONNOUSERSITE=1

/shared/tshanableh/conda_envs/hnerv310/bin/python run_batch_splits.py \
  --method tile16 \
  --gpu 1 \
  --epochs 300 \
  --eval_freq 30 \
  --expected_frames 132 \
  --sequences bunny honeybee jockey readysetgo yachtride bosphorus \
  --skip_existing \
  --aggregate_after
```

Example run folders:

```text
runs/tiles16_bunny_tile16_3m_300ep_q8p10/
runs/tiles16_honeybee_tile16_3m_300ep_q8p10/
```

Monitor a log:

```bash
tail -f /shared/tshanableh/NeRV/HNeRV-DWT/runs/logs/honeybee_tile16_3m_300ep_q8p10.log
```

---

## 6. Using Two GPUs to Speed Up 16-Tile Runs

If one GPU is already running the 16-tile batch and the other GPU becomes free, run a second batch on later sequences only.

Example: if GPU 1 is currently running `honeybee`, start GPU 0 on later sequences:

```bash
cd /shared/tshanableh/NeRV/HNeRV-DWT

unset PYTHONPATH
export PYTHONNOUSERSITE=1

/shared/tshanableh/conda_envs/hnerv310/bin/python run_batch_splits.py \
  --method tile16 \
  --gpu 0 \
  --epochs 300 \
  --eval_freq 30 \
  --expected_frames 132 \
  --sequences yachtride bosphorus \
  --skip_existing \
  --aggregate_after
```

The `--skip_existing` flag prevents rerunning sequences with completed summaries.

Do not start two jobs on the same sequence at the same time, because they may write to the same run folder.

---

## 7. Monitoring Progress

Check GPU usage:

```bash
watch -n 10 nvidia-smi
```

Check completed quadrant runs:

```bash
find /shared/tshanableh/NeRV/HNeRV-DWT/runs -name "summary_total_quant.csv" \
  | grep "quadrant" \
  | sort
```

Check completed 16-tile runs:

```bash
find /shared/tshanableh/NeRV/HNeRV-DWT/runs -name "summary_total_quant.csv" \
  | grep "tiles16" \
  | sort
```

---

## 8. Output Files Per Run

Each completed run folder contains:

```text
summary_total_quant.csv
part_metrics_quant.csv
recon_quant/metrics.csv
hnerv_output/
predictions_quant/
data/
```

Important files:

| File | Purpose |
|---|---|
| `summary_total_quant.csv` | Main summary: PSNR, SSIM, params, GFLOPs, BPP, quantization, pruning |
| `part_metrics_quant.csv` | Per-quadrant or per-tile PSNR/SSIM and model stats |
| `recon_quant/metrics.csv` | Per-frame full reconstructed RGB PSNR/SSIM |
| `runs/logs/*.log` | Training and extraction logs |

---

## 9. Aggregate Results

After some or all runs finish, regenerate the aggregate CSV:

```bash
cd /shared/tshanableh/NeRV/HNeRV-DWT

/shared/tshanableh/conda_envs/hnerv310/bin/python run_batch_splits.py \
  --method quadrant \
  --gpu 0 \
  --skip_existing \
  --aggregate_after
```

The aggregate file is saved at:

```text
runs/aggregate_4tile_16tile_6seq_q8p10.csv
```

View it:

```bash
cat /shared/tshanableh/NeRV/HNeRV-DWT/runs/aggregate_4tile_16tile_6seq_q8p10.csv
```

---

## 10. Print Paper-Ready Summary Columns

Use this to print per-sequence and average results with conceptual FPS:

```bash
cd /shared/tshanableh/NeRV/HNeRV-DWT

/shared/tshanableh/conda_envs/hnerv310/bin/python - <<'PY'
from pathlib import Path
import csv
from collections import defaultdict

agg = Path("runs/aggregate_4tile_16tile_6seq_q8p10.csv")

rows = []
with agg.open() as f:
    for r in csv.DictReader(f):
        if r["mode"] in {"quadrant", "tiles16"}:
            rows.append(r)

print("Per-sequence results")
print("method,sequence,psnr,ssim,params_m,gflops,conceptual_fps_10t")
for r in rows:
    g = float(r["gflops"])
    fps = 10000.0 / g
    print(f'{r["mode"]},{r["sequence"]},{float(r["psnr"]):.3f},{float(r["ssim"]):.4f},{float(r["params_m"]):.3f},{g:.2f},{fps:.1f}')

groups = defaultdict(list)
for r in rows:
    groups[r["mode"]].append(r)

print()
print("Averages")
print("method,avg_psnr,avg_ssim,avg_params_m,avg_gflops,conceptual_fps_10t")
for mode, rs in groups.items():
    avg_psnr = sum(float(r["psnr"]) for r in rs) / len(rs)
    avg_ssim = sum(float(r["ssim"]) for r in rs) / len(rs)
    avg_params = sum(float(r["params_m"]) for r in rs) / len(rs)
    avg_gflops = sum(float(r["gflops"]) for r in rs) / len(rs)
    fps = 10000.0 / avg_gflops
    print(f"{mode},{avg_psnr:.3f},{avg_ssim:.4f},{avg_params:.3f},{avg_gflops:.2f},{fps:.1f}")
PY
```

Conceptual FPS is computed as:

```text
FPS = 10,000 / GFLOPs_per_frame
```

This assumes 10 TFLOPs/s hardware throughput. It is a FLOP-limited estimate, not measured runtime.

---

## 11. Packaging Results for Upload

Package all quadrant result summaries:

```bash
cd /shared/tshanableh/NeRV/HNeRV-DWT

tar -czf quadrant_6seq_results_q8p10.tar.gz \
  $(find runs -path "*quadrant*quadrant_3m_300ep_q8p10*" \( \
      -name "summary_total_quant.csv" -o \
      -name "part_metrics_quant.csv" -o \
      -path "*/recon_quant/metrics.csv" \
  \)) \
  runs/aggregate_4tile_16tile_6seq_q8p10.csv
```

Package all 16-tile result summaries:

```bash
cd /shared/tshanableh/NeRV/HNeRV-DWT

tar -czf tile16_6seq_results_q8p10.tar.gz \
  $(find runs -path "*tiles16*tile16_3m_300ep_q8p10*" \( \
      -name "summary_total_quant.csv" -o \
      -name "part_metrics_quant.csv" -o \
      -path "*/recon_quant/metrics.csv" \
  \)) \
  runs/aggregate_4tile_16tile_6seq_q8p10.csv
```

Verify:

```bash
ls -lh quadrant_6seq_results_q8p10.tar.gz
ls -lh tile16_6seq_results_q8p10.tar.gz
```

---

## 12. Common Issues

### Wrong Python is used

If logs show paths such as:

```text
/usr/local/bin/python
/home/tshanableh/.local
/opt/hpc-env/gpu311
```

then the wrong Python is being used. Always run with:

```bash
/shared/tshanableh/conda_envs/hnerv310/bin/python
```

### MS-SSIM crash for 16-tile mode

For `160×320` tiles, MS-SSIM may fail because the smaller side is 160. The project should already be patched to use single-scale SSIM fallback for small tiles.

Verify:

```bash
grep -n "smaller_side <= 160" HNeRV-main/hnerv_utils.py
```

### Rerun after failure

Use the same command with:

```bash
--skip_existing
```

Completed sequences will be skipped.

### Avoid collisions

Do not run two jobs with the same method, sequence, and tag at the same time.

Safe parallel use:

```text
GPU 1: tile16 honeybee → jockey → readysetgo
GPU 0: tile16 yachtride → bosphorus
```

Unsafe:

```text
GPU 1: tile16 honeybee
GPU 0: tile16 honeybee
```

---

## 13. Manuscript Reporting

For the manuscript, report:

1. Dense HNeRV baseline from the previous paper.
2. LR-wide HNeRV b3&4 from the previous paper.
3. Proposed 4-quadrant split from current runs.
4. Proposed 16-tile split from current runs.

Do not report BPP in the main table if the focus is decoder-compute efficiency. Report:

```text
Params
GFLOPs/frame
GFLOPs reduction
Conceptual FPS @10TFLOPs
PSNR
SSIM
```

The conceptual FPS is not measured runtime. It is only a FLOP-limited estimate.
