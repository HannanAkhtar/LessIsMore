#!/usr/bin/env bash
set -euo pipefail
cd /shared/tshanableh/NeRV/HNeRV-DWT-v3
unset PYTHONPATH
export PYTHONNOUSERSITE=1
/shared/tshanableh/conda_envs/hnerv310/bin/python run_experiment.py debug \
  --config /shared/tshanableh/NeRV/HNeRV-DWT-v3/configs/tile16_equal_3m_lrwide.json \
  --sequence bunny \
  --tag tile16_3m_300ep_q8p10 \
  --epochs 300 \
  --eval_freq 30 \
  --gpu 0 \
  --expected_frames 132 \
  --clean
