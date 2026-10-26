# Quantized/Pruned Evaluation Patch for Previous HNeRV Low-Rank Runs

This patch changes the previous low-rank HNeRV code so that reported PSNR/SSIM are computed from actual quantized/pruned playback output, not only from compression-accounting values.

## What changed

The old evaluation path computed:

```text
pred = model(x)
PSNR/SSIM = metric(pred, x)
```

and then used `model_bits`, `embed_bits`, and `prune_ratio` only inside BPP/accounting.

The patched evaluation path computes:

```text
z = encoder(x)
z_q = quantize_dequantize(z, embed_bits)
model_qp = deepcopy(model)
prune non-encoder weights in model_qp
quantize/dequantize non-encoder tensors in model_qp
pred_qp = model_qp.decoder(z_q)
PSNR/SSIM = metric(pred_qp, x)
```

This follows the same concept used in the spatial-split HNeRV implementation: prune first, quantize/dequantize decoder weights and embeddings, then compute metrics from the quantized playback output.

## Important output columns

Patched CSVs include these verification columns:

```text
eval_variant
actual_model_quantized
actual_embed_quantized
actual_pruned
model_bits_used_for_eval
embed_bits_used_for_eval
pruned_weights
prunable_weights
quantized_decoder_tensors
```

For the corrected results, `eval_variant` should be:

```text
quant_pruned_playback
```

and `actual_model_quantized`, `actual_embed_quantized`, and `actual_pruned` should be `True` when `prune_ratio > 0`.

## Regenerate all baseline + LR b3&4 results

From the low-rank project root:

```bash
cd /shared/tshanableh/NeRV/HNeRV-Low-Rank-Conv

unset PYTHONPATH
export PYTHONNOUSERSITE=1

/shared/tshanableh/conda_envs/hnerv310/bin/python run_all_lowrank_quant_pruned.py   --data_root ../data   --runs_root runs_full_originals_qp_eval_b34_3M   --only bunny honeybee jockey readysetgo yachtride   --epochs_baseline 300   --epochs_lowrank 300   --batch_size 2   --model_bits 8   --embed_bits 8   --prune_ratio 0.10   --gpu 0
```

This trains/evaluates:

```text
baseline
lowrank_samew_b34
lowrank_wide_b34
```

## Recompute corrected metrics from existing checkpoints only

If the checkpoints already exist and you only want to recompute the corrected Q/P metrics, use:

```bash
/shared/tshanableh/conda_envs/hnerv310/bin/python run_all_lowrank_quant_pruned.py   --data_root ../data   --runs_root runs_full_originals_qp_eval_b34_3M   --only bunny honeybee jockey readysetgo yachtride   --epochs_baseline 300   --epochs_lowrank 300   --batch_size 2   --model_bits 8   --embed_bits 8   --prune_ratio 0.10   --gpu 0   --skip_existing_models
```

Do not use `--skip_existing` when recomputing corrected metrics, because that would skip reading the checkpoint and skip rewriting `summary.csv`.

## Final results file

The combined corrected results are saved to:

```text
runs_full_originals_qp_eval_b34_3M/all_results.csv
```

Check the verification columns:

```bash
/shared/tshanableh/conda_envs/hnerv310/bin/python - <<'INNERPY'
import pandas as pd
p = 'runs_full_originals_qp_eval_b34_3M/all_results.csv'
df = pd.read_csv(p)
print(df[['sequence','model_type','psnr','ssim','gflops_per_frame_decoder_only','eval_variant','actual_model_quantized','actual_embed_quantized','actual_pruned','model_bits_used_for_eval','embed_bits_used_for_eval','pruned_weights','prunable_weights']])
INNERPY
```
