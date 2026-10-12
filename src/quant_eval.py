from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .flops import count_model_flops
from .metrics import batch_psnr, batch_ssim, compression_bpp
from .models import (
    decoder_num_parameters,
    encoder_num_parameters,
    infer_embedding_shape,
    total_num_parameters,
)
from .utils import move_batch


def quant_tensor_like_hnerv(t: torch.Tensor, bits: int = 8) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Quantize/dequantize a tensor using the same idea as the HNeRV reference code.

    The reference implementation tries whole-tensor quantization and a small set
    of per-axis min/scale candidates, then keeps the candidate with the lowest
    mean absolute reconstruction error. The returned second tensor is the
    dequantized tensor that should be used for quantized evaluation.
    """
    if bits is None or int(bits) < 0:
        return {"quant": t.detach().clone(), "min": torch.zeros((), device=t.device), "scale": torch.ones((), device=t.device)}, t.detach().clone()

    bits = int(bits)
    qmax = float(2 ** bits - 1)
    eps = torch.finfo(t.float().dtype).eps
    t_work = t.detach()

    candidates = []
    t_min, t_max = t_work.min(), t_work.max()
    scale = (t_max - t_min) / qmax
    scale = torch.clamp(scale, min=eps)
    candidates.append((t_min, scale))

    for axis in range(t_work.dim()):
        t_min_axis = t_work.min(axis, keepdim=True)[0]
        t_max_axis = t_work.max(axis, keepdim=True)[0]
        if t_min_axis.nelement() / max(1, t_work.nelement()) < 0.02:
            scale_axis = (t_max_axis - t_min_axis) / qmax
            scale_axis = torch.clamp(scale_axis, min=eps)
            candidates.append((t_min_axis.to(torch.float16), scale_axis.to(torch.float16)))

    best = None
    best_err = None
    for t_min_c, scale_c in candidates:
        t_min_e = t_min_c.to(device=t_work.device).expand_as(t_work)
        scale_e = scale_c.to(device=t_work.device).expand_as(t_work)
        quant = ((t_work - t_min_e) / scale_e).round().clamp(0, qmax)
        dequant = t_min_e + scale_e * quant
        err = (t_work - dequant).abs().mean()
        if best_err is None or err < best_err:
            best_err = err
            best = (quant, dequant, t_min_c, scale_c)

    quant, dequant, best_min, best_scale = best
    quant_dtype = torch.uint8 if bits <= 8 else torch.int16
    quant_pack = {
        "quant": quant.to(quant_dtype),
        "min": best_min.detach().clone(),
        "scale": best_scale.detach().clone(),
    }
    return quant_pack, dequant.to(dtype=t.dtype)


def apply_global_magnitude_pruning(model: nn.Module, prune_ratio: float) -> Tuple[int, int]:
    """Zero the smallest-magnitude non-encoder Conv/Linear-style weights.

    This mirrors the spatial-split HNeRV implementation: parameters with fewer
    than two dimensions are skipped, and any parameter whose name contains
    "encoder" is skipped. This means playback decoder weights are pruned while
    the training-only encoder is left unchanged.
    """
    prune_ratio = float(prune_ratio or 0.0)
    if prune_ratio <= 0.0:
        return 0, 0

    params = []
    for name, p in model.named_parameters():
        lname = name.lower()
        if not p.requires_grad or p.ndim < 2:
            continue
        if "encoder" in lname:
            continue
        params.append((name, p))

    total = sum(p.numel() for _, p in params)
    if total == 0:
        return 0, 0

    k = int(round(total * prune_ratio))
    if k <= 0:
        return 0, total

    with torch.no_grad():
        all_abs = torch.cat([p.detach().abs().flatten() for _, p in params])
        k = min(max(k, 1), all_abs.numel())
        threshold = torch.kthvalue(all_abs, k).values
        pruned = 0
        for _, p in params:
            mask = p.detach().abs() > threshold
            pruned += int((~mask).sum().item())
            p.mul_(mask)

    return int(pruned), int(total)


def quantize_non_encoder_state_dict_(model: nn.Module, model_bits: int) -> int:
    """In-place dequantized simulation of weight quantization for non-encoder tensors."""
    if model_bits is None or int(model_bits) < 0:
        return 0

    state = model.state_dict()
    quantized_tensors = 0
    for key, value in list(state.items()):
        if "encoder" in key.lower():
            continue
        if not torch.is_floating_point(value):
            continue
        _, dequant = quant_tensor_like_hnerv(value, int(model_bits))
        state[key] = dequant.to(dtype=value.dtype, device=value.device)
        quantized_tensors += 1
    model.load_state_dict(state, strict=True)
    return int(quantized_tensors)


@torch.no_grad()
def collect_frame_embeddings(model: nn.Module, dataset, args, device) -> torch.Tensor:
    """Collect encoder embeddings for every frame in dataset order."""
    if not hasattr(model, "encoder"):
        raise ValueError("Quantized playback evaluation requires a model with an encoder and decoder.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    model.eval().to(device)
    embeddings = []
    for batch in tqdm(loader, desc="collect embeddings", leave=False):
        batch = move_batch(batch, device)
        x = batch["image"]
        embeddings.append(model.encoder(x).detach())
    return torch.cat(embeddings, dim=0)


@torch.no_grad()
def evaluate_quantized_pruned_playback_model(
    model: nn.Module,
    dataset,
    args,
    device,
    model_type: str,
) -> Dict[str, Any]:
    """Evaluate HNeRV playback from quantized embeddings and quantized/pruned decoder.

    This is the actual Q/P evaluation path. It does not evaluate model(x).
    Instead it:
      1. extracts all frame embeddings with the training encoder,
      2. quantizes/dequantizes those embeddings using args.embed_bits,
      3. deep-copies the model,
      4. applies global magnitude pruning to non-encoder weights,
      5. quantizes/dequantizes non-encoder state_dict tensors using args.model_bits,
      6. reconstructs frames as quantized_model.decoder(dequantized_embedding).
    """
    source_model = model.eval().to(device)
    z = collect_frame_embeddings(source_model, dataset, args, device)
    _, z_dequant = quant_tensor_like_hnerv(z, int(args.embed_bits))
    z_dequant = z_dequant.to(device=device)

    playback_model = copy.deepcopy(source_model).to(device).eval()
    pruned_weights, prunable_weights = apply_global_magnitude_pruning(playback_model, float(args.prune_ratio))
    quantized_decoder_tensors = quantize_non_encoder_state_dict_(playback_model, int(args.model_bits))
    playback_model.eval()

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    psnrs, ssims = [], []
    offset = 0
    for batch in tqdm(loader, desc=f"eval {model_type} quant/pruned", leave=False):
        batch = move_batch(batch, device)
        x = batch["image"]
        bsz = x.shape[0]
        z_batch = z_dequant[offset:offset + bsz]
        offset += bsz
        pred = playback_model.decoder(z_batch)
        psnrs.append(batch_psnr(pred, x))
        ssims.append(batch_ssim(pred, x))

    T = len(dataset)
    H, W = args.img_size
    emb_shape = infer_embedding_shape((H, W), args.enc_strides, args.emb_dim)
    embedding_values = T * int(np.prod(emb_shape))
    dec_params = decoder_num_parameters(playback_model)
    bpp_stats = compression_bpp(
        dec_params, embedding_values, T, H, W,
        args.model_bits, args.embed_bits, 0, args.prune_ratio,
    )
    flops = count_model_flops(playback_model.decoder, (1, *emb_shape), device=str(device), flop_mult=args.flop_mult)

    return {
        "model_type": model_type,
        "sequence": Path(args.data_path).name,
        "frames": T,
        "height": H,
        "width": W,
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "decoder_params": int(dec_params),
        "encoder_params_not_stored": int(encoder_num_parameters(playback_model)),
        "total_trainable_params": int(total_num_parameters(playback_model)),
        "embedding_values": int(embedding_values),
        "cluster_bits": 0,
        "gflops_per_frame_decoder_only": float(flops / 1e9),
        "eval_variant": "quant_pruned_playback",
        "actual_model_quantized": True,
        "actual_embed_quantized": True,
        "actual_pruned": float(args.prune_ratio) > 0.0,
        "model_bits_used_for_eval": int(args.model_bits),
        "embed_bits_used_for_eval": int(args.embed_bits),
        "pruned_weights": int(pruned_weights),
        "prunable_weights": int(prunable_weights),
        "quantized_decoder_tensors": int(quantized_decoder_tensors),
        **bpp_stats,
    }
