# Bunny 3M checkpoints and data

This `checkpoints/` folder contains the final Bunny checkpoints and video-frame data for five model variants. Paths below are relative to this folder.

## Folder layout

- `data/bunny/`: the 132 original Bunny PNG frames at 1280 × 720, shared by the three whole-frame variants. The model pipeline center-crops these frames to 1280 × 640.
- `dense_whole/`: dense HNeRV baseline; final float32 checkpoint at `last.pt`.
- `lowrank_same_width_whole/`: low-rank same-width ablation; final float32 checkpoint at `last.pt`.
- `widened_whole/`: proposed widened whole-frame model; final float32 checkpoint at `last.pt`.
- `widened_4_tiles/`: Q0 top-left, Q1 top-right, Q2 bottom-left, and Q3 bottom-right. Pre-cropped frames are in `data/bunny_Q0/` through `data/bunny_Q3/` (640 × 320 pixels each).
- `widened_16_tiles/`: T00 through T33 in row-major order. Pre-cropped frames are in the matching `data/` subfolders (320 × 160 pixels each).
- `manifest.json`: records file roles, dimensions, architecture run IDs, sizes, and SHA-256 hashes.

Each tile folder contains its full epoch-300 training checkpoint (`epoch300.pth`), quantized frame embeddings (`quant_vid.pth`), and TorchScript decoder (`img_decoder.pth`). The tile playback files use 8-bit decoder weights, 6-bit frame embeddings, and 10% pruning; inference dequantizes them to floating-point kernels.
