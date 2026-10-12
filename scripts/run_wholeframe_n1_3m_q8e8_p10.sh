#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
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
