#!/usr/bin/env bash
# Historical rank-22 configuration. For manuscript rank-23 candidate use spatial-custom explicitly.
set -euo pipefail
cd "$(dirname "$0")/.."
python run_framework.py spatial16 \
  --data_root ../data \
  --sequence bunny \
  --epochs 300 \
  --eval_freq 30 \
  --expected_frames 132 \
  --batch_size 2 \
  --gpu 0
