# Spatial experiments

Use the [top-level README](../README.md) for setup, training, packed playback, and manuscript caveats. `run_experiment.py` prepares independent tiles, invokes the modified HNeRV trainer in `HNeRV-main/`, extracts predictions, stitches full frames, and produces legacy summaries. `evaluate_packed.py` serializes saved quantized decoder weights and embeddings, reloads them, and computes measured packed BPP with full-frame PSNR/windowed SSIM.

Configurations:

- `configs/quadrant_equal_lrwide.json`: four low-rank widened tiles, rank 43.
- `configs/quadrant_dense_3m_reference.json`: dense four-tile reference target. No local checkpoint proves exact paper parameter count.
- `configs/tile16_equal_3m_lrwide.json`: historical committed rank 22.
- `configs/tile16_rank23_0p198_unverified.json`: rank 23 candidate based on the old v2 generator; published checkpoint provenance must still be checked.

The default `run_framework.py spatial16` route deliberately keeps the historical rank-22 file. Select the rank-23 candidate explicitly with `spatial-custom --config ...`. The native 1080×1920 manuscript run requires its missing original configuration and checkpoints; generic tile padding/cropping in `run_experiment.py` alone is not numerical reproduction.

`summary_total_quant.csv` retains `sum_bpp` for historical readers and labels it `legacy_sum_of_tile_estimates_not_measured_packed_bpp`. Use `packed_bpp` from `evaluate_packed.py` for measured archive storage. `run_experiment.py score` uses a legacy global SSIM surrogate; use packed playback for publication-protocol windowed SSIM.
