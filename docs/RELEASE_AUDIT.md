# Release audit (local checkout, 2026-10-26)

## Initial findings

| Area | Evidence before edits | Current disposition |
| --- | --- | --- |
| Whole-frame model/training | `src/models.py`, `src/experiment.py`, `src/run_lowrank_ablation_grid.py`, `run_wholeframe_lrwide.py` | Preserved operations and checkpoint tensor names. Added dense-only and rank/budget/seed CLI controls. |
| Spatial model/training | `spatial/HNeRV-main/model_all.py`, `train_nerv_all.py`, `spatial/run_experiment.py` | Preserved reference architecture. Added configuration validation, seed forwarding, natural ordering, padding/cropping, and safeguards against accidental output replacement. |
| Inference | Whole-frame quantized evaluation simulated dequantized playback; spatial extracted saved `GT|prediction` images. | Added serialized archive readers and evaluation CLIs. Model execution remains untested without PyTorch/checkpoints. |
| BPP | `src/metrics.py:compression_bpp` estimates weight/embedding bits; spatial `sum_bpp` sums HNeRV's legacy `bits/pixel`. Neither is a measured complete archive. | Kept historical fields with explicit protocol tags. New `packed_bpp` uses actual ZIP bytes and frame geometry. |
| SSIM | Spatial `run_experiment.py` uses a global RGB surrogate; the old whole-frame helper pads the window. | New packed paths share a full-frame valid-window Gaussian SSIM function; historical scoring paths remain marked legacy. |
| Complexity | `src/flops.py` counts Conv2d/Linear MACs; spatial profiler calls a two-FLOP convention GFLOPs. | New archive reports both GMACs and GFLOPs explicitly, summing every tile's full padded output. |
| Rank 16 | Committed `tile16_equal_3m_lrwide.json` has rank 22; manuscript page 7 says evaluated checkpoint tensors show rank 23. | Historical rank-22 file preserved. Added `tile16_rank23_0p198_unverified.json` from the existing v2 generator values. Actual checkpoint shapes unavailable, so provenance unverified. |
| Native UVG | No native 1080×1920 configuration, training manifest, or checkpoints in repository. | General padding/cropping mechanics added, but numerical native result remains reference-only. |
| Dependencies | Whole-frame requirements pin CUDA PyTorch 2.11.0; spatial pins PyTorch 1.8.1, torchvision 0.9.1, NumPy 1.19.2. | Separate documented environments; CPU-only test set verified locally. Model environments not installed or validated. |
| Paths/secrets | Primary README and old spatial examples contained `/shared/tshanableh/...`; no credential-like token was found in tracked source by pattern scan. | Public README uses configurable paths. Historical notes/scripts retained under `docs/legacy/`; `spatial/run_batch_splits.py` no longer has cluster setup commands. |
| License | No repository or vendored HNeRV `LICENSE` file in this checkout. | Attribution retained. Licensing decision is required before public redistribution. |

## Scientific and implementation caveats

1. The two HNeRV implementations have different decoder definitions and parameter-budget solvers. A whole-frame rank/width cannot be transferred directly to a spatial tile.
2. Whole-frame checkpoints store model tensors and arguments. Spatial `model_latest.pth` holds training state, while `quant_vid.pth` stores quantized decoder tensors and embeddings. Only the latter is sufficient for the spatial packed playback path when paired with its JSON configuration.
3. Packed archives use ZIP_STORED entries and bit-packed 6-bit embeddings. Their container overhead is real and counted. Their byte layout may differ from the archive generator used for the manuscript; matching paper BPP requires the original archive specification and files.
4. `src/metrics.py` retains the legacy `bpp` field for compatibility. It excludes quantization scales and container overhead. Spatial `sum_bpp` has a distinct legacy protocol tag. Neither should be cited as a measured packed rate.
5. The new profiler counts Conv2d/Linear multiplications and additions only. The manuscript's exact native 28.80 GMAC figure cannot be checked without its models and padded-tile configuration.
6. Spatial training writes quantized predictions as PNGs. The legacy `score` path reads those 8-bit saved images and uses global SSIM. The packed path scores decoded tensors using windowed SSIM; small differences from the legacy numbers are expected.
7. Existing rank-23 generator `spatial/make_tile16_3m_exact_v2_config.py` selects a 0.198M per-tile target. No local run logs or checkpoints prove that this precise target generated the published 3.040M aggregate result.

## Assets and decisions needed before public release

- Five-sequence and native seven-sequence source-frame selection manifests, including filenames, hashes, and frame start offsets.
- Validated 3M whole-frame and 4/16-tile checkpoints or `quant_vid.pth` files, especially the paired Honeybee seed-2 set and rank-23 sixteen-tile set.
- For each sixteen-tile checkpoint, run `python scripts/inspect_spatial_rank.py CHECKPOINT.pth` and record the stage-3/stage-4 tensor ranks in the manifest.
- Native UVG training config, actual padded tile dimensions, per-sequence checkpoint IDs, and measurement scripts/logs for the reported 28.80 GMACs and 32.70 dB.
- The original paper archive writer or archived representations to test byte-for-byte BPP correspondence with the new container.
- A GPU environment for model imports, checkpoint loading, archive round-trip playback, and 300-epoch reproducibility checks; an A10G/NVML environment for the latency, memory, and board-energy protocol.
- Author decision on repository license and permission/terms for the modified upstream HNeRV code.

No commits, remote changes, model retraining, downloads, or experiment result overwrites were made during this audit.

## Verification performed

- `py -3.11 -m pytest -q tests/test_release_cpu.py`: **6 passed, 1 skipped**. The skipped case constructs PyTorch models and checks checkpoint/quantization/metric/MAC behavior; PyTorch is not installed.
- `python -m compileall -q run_framework.py run_wholeframe_lrwide.py src spatial scripts tests`: passed.
- `run_framework.py`, `run_wholeframe_lrwide.py`, `spatial/run_experiment.py`, both packed evaluators, the rank inspector, and the frame-preparation tool: `--help` passed.
- Eight checked JSON files parsed successfully; `git diff --check` reported no whitespace errors. A pattern scan of active files found no obvious credential string. This is not a full secret-history review.

## Publication-readiness checklist

- [x] Public entry points and metric conventions documented.
- [x] Historical rank-22 configuration retained; rank-23 candidate labeled unverified.
- [x] Archive byte accounting and CPU round-trip tests added.
- [x] Legacy cluster commands removed from primary instructions; historical material preserved.
- [ ] Install and run the whole-frame and spatial model environments on supported GPU hardware.
- [ ] Verify old and new checkpoints through decoded archive playback and compare against reported tables.
- [ ] Recover exact frame manifests and checkpoint IDs, including Honeybee seed 2 and native UVG.
- [ ] Recover the original packed archives/serializer and A10G/NVML measurement records.
- [ ] Decide and document repository and upstream-code redistribution licenses.
