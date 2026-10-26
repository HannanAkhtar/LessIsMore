# Manuscript-to-code map

All manuscript figures and numbers below are **reference-only** in this checkout. No training frames, checkpoints, packed archives, or GPU logs are committed. Use `docs/experiment_manifest.json` to record provenance as those assets become available.

| Manuscript experiment | Local training implementation and configuration | Evaluation route | Missing evidence |
| --- | --- | --- | --- |
| Whole-frame dense and LR-wide at 0.75M, 1.5M, 3M | `src/run_lowrank_ablation_grid.py` through `run_framework.py wholeframe`; `--baseline_decoder_m` selects budget, `--lowrank_specs widened:3,4`, rank 88 | `python -m src.evaluate_packed` with matching checkpoint and Q8/E8/P10 | Checkpoints, exact frame files, original packed archives |
| Factorization and widening ablations | `src/run_lowrank_ablation_grid.py --lowrank_ablation_specs` supports same-width/global-widened block sets. The manuscript also has targeted-widening results. | Floating-point and quantized metrics in `src/experiment.py` / `src/quant_eval.py` | Targeted-widening training/evaluation entry point and its checkpoints; separate Table 5/6 provenance |
| Dense versus LR-wide four-tile control | `spatial/configs/quadrant_dense_3m_reference.json` and `quadrant_equal_lrwide.json`, `spatial/run_experiment.py` | `spatial/evaluate_packed.py` with four `quant_vid.pth` files | Paired validated Honeybee seed-2 and other checkpoint sets |
| Four-tile spatial | `spatial/configs/quadrant_equal_lrwide.json`, rank 43, 0.75M target per tile | Full-frame stitch and packed Q8/E6/P10 | Frame/quantized model artifacts |
| Sixteen-tile spatial | Historical rank 22 in `tile16_equal_3m_lrwide.json`; candidate rank 23 in `tile16_rank23_0p198_unverified.json` | `spatial/evaluate_packed.py` with sixteen tile payloads | Tensor-shape inspection of published rank-23 checkpoints and exact config metadata |
| Packed rate–distortion | `src/packed_archive.py` fixed-width ZIP; `src/evaluate_packed.py`, `spatial/evaluate_packed.py` | BPP from all archive bytes; PSNR and windowed SSIM from decoded full frames | Original archive format/files for exact agreement with manuscript rates |
| Hardware latency, memory, energy | Decoder timing exposed in packed evaluation; `src/flops.py` counts decoder MACs | Run on exclusive A10G with one warm-up, five passes, synchronized full-frame transfer, 2 ms memory sampling and NVML board-energy counters | Original measurement harness, A10G, NVML logs; present timing is diagnostic only |
| Native seven-sequence UVG | General spatial padding/cropping is in `spatial/run_experiment.py`; no original native config | Four-tile archive decode can crop padded output to 1080×1920; MAC profiler counts padded regions | Actual 3.2M native config, six 600-frame/one 300-frame input manifests, checkpoints, 28.80 GMAC trace |

## Commands and conventions

Read the top-level README for complete install and CLI examples. To evaluate an existing whole-frame checkpoint without retraining:

```bash
python -m src.evaluate_packed --checkpoint CHECKPOINT.pt --model-type lrwide --frames DATA/SEQUENCE --archive runs/eval/sequence.zip --device cuda
```

The saved `args` in the whole-frame checkpoint drive architecture construction. For older low-rank checkpoints, provide `--blocks` matching the training specification (default `3,4`). The loader checks state-dict keys and shapes strictly. Spatial `quant_vid.pth` files require the exact per-tile config and matching tile IDs; the packed loader checks their keys and frame counts.

The paper's 640×1280 sequence protocol and the native 1080×1920 UVG protocol must not be merged. At 640×1280 use 132 naturally ordered center-cropped frames, whole-frame Q8/E8/P10, and spatial Q8/E6/P10. Native UVG uses 540×960 valid quadrants, padded for decoder strides and cropped after reconstruction. The source-frame start offsets and original native padding/config parameters remain unknown here.

For Honeybee, the manuscript's primary paired four-tile control uses a validated seed-2 checkpoint after an erroneous preliminary run was excluded. The evidence trail is recorded in the manuscript; the checkpoint identifier and error diagnosis are not in this repository. Do not substitute a seed-1 or preliminary Honeybee run into paper aggregates.

**Metric definitions:** average per-frame RGB PSNR in dB; full-frame Gaussian-window SSIM; `packed_bpp = 8 × archive bytes / (T × H × W)`; decoder MACs sum Conv2d/Linear operations over all processed tile pixels; decoder GFLOPs use two operations per MAC. Parameter budget is aggregate decoder weights plus embeddings where the paper says playback parameters. The archives include weight symbols, embeddings, quantization scales, architecture and tile metadata, padding descriptors, and container overhead. Unstructured pruned zeros remain ordinary fixed-width symbols.
