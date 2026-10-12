from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def mse_to_psnr(mse: float, max_val: float = 1.0) -> float:
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def batch_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.detach().clamp(0, 1).float()
    target = target.detach().clamp(0, 1).float()
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    psnr = torch.tensor([mse_to_psnr(float(m.item())) for m in mse], device=pred.device)
    return float(psnr.mean().item())


def _ssim_window(channels: int, window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = (g[:, None] @ g[None, :]).view(1, 1, window_size, window_size)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def batch_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> float:
    """Compute mean single-scale SSIM for a batch using only PyTorch.

    This removes the previous scikit-image dependency, so the project can run
    in minimal Windows/AWS environments. Inputs are expected as NCHW tensors in
    [0, 1].
    """
    pred = pred.detach().clamp(0, 1).float()
    target = target.detach().clamp(0, 1).float()

    if pred.ndim != 4 or target.ndim != 4:
        raise ValueError("batch_ssim expects NCHW tensors")
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shape mismatch: {pred.shape} vs {target.shape}")

    n, c, h, w = pred.shape
    # SSIM needs a window not larger than the image. Keep it odd.
    ws = min(window_size, h, w)
    if ws % 2 == 0:
        ws -= 1
    if ws < 3:
        # Fallback for extremely small images: use a global SSIM-like score.
        mu_x = pred.mean(dim=(2, 3), keepdim=True)
        mu_y = target.mean(dim=(2, 3), keepdim=True)
        sigma_x = ((pred - mu_x) ** 2).mean(dim=(2, 3), keepdim=True)
        sigma_y = ((target - mu_y) ** 2).mean(dim=(2, 3), keepdim=True)
        sigma_xy = ((pred - mu_x) * (target - mu_y)).mean(dim=(2, 3), keepdim=True)
    else:
        window = _ssim_window(c, ws, sigma, pred.device, pred.dtype)
        padding = ws // 2
        mu_x = F.conv2d(pred, window, padding=padding, groups=c)
        mu_y = F.conv2d(target, window, padding=padding, groups=c)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x = F.conv2d(pred * pred, window, padding=padding, groups=c) - mu_x2
        sigma_y = F.conv2d(target * target, window, padding=padding, groups=c) - mu_y2
        sigma_xy = F.conv2d(pred * target, window, padding=padding, groups=c) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / (denominator + 1e-12)
    ssim_per_image = ssim_map.flatten(1).mean(dim=1)
    return float(ssim_per_image.mean().item())


def compression_bpp(
    decoder_params: int,
    embedding_values: int,
    num_frames: int,
    height: int,
    width: int,
    model_bits: int = 8,
    embed_bits: int = 8,
    cluster_bits: int = 0,
    prune_ratio: float = 0.0,
) -> Dict[str, float]:
    if not 0.0 <= prune_ratio < 1.0:
        raise ValueError(f"prune_ratio must be in [0, 1), got {prune_ratio}")
    effective_decoder_params = decoder_params * (1.0 - prune_ratio)
    decoder_bits = effective_decoder_params * model_bits
    embedding_bits = embedding_values * embed_bits
    total_bits = decoder_bits + embedding_bits + cluster_bits
    denom = max(1, num_frames * height * width)
    return {
        "decoder_bits": float(decoder_bits),
        "effective_decoder_params_after_prune": float(effective_decoder_params),
        "prune_ratio": float(prune_ratio),
        "embedding_bits": float(embedding_bits),
        "cluster_bits": float(cluster_bits),
        "total_bits": float(total_bits),
        "bpp": float(total_bits / denom),
    }
