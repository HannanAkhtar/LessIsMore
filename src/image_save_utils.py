"""
Utility for saving reconstructed frames during HNeRV evaluation.

Copy this file into:

    HNeRV-Low-Rank-Conv/src/image_save_utils.py

Then import and call save_reconstruction_batch(...) inside the evaluation loop
where PSNR/SSIM are computed.
"""

from pathlib import Path

import numpy as np
from PIL import Image


def _to_numpy_batch(x):
    """
    Convert torch tensor or numpy array to numpy.
    Accepts:
      B x C x H x W
      B x H x W x C
      C x H x W
      H x W x C
    """
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().float().cpu().numpy()
    except Exception:
        pass

    x = np.asarray(x)

    if x.ndim == 3:
        x = x[None, ...]

    if x.ndim != 4:
        raise ValueError(f"Expected 3D or 4D tensor/array, got shape {x.shape}")

    # Convert B x C x H x W to B x H x W x C.
    if x.shape[1] in (1, 3) and x.shape[-1] not in (1, 3):
        x = np.transpose(x, (0, 2, 3, 1))

    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)

    if x.shape[-1] != 3:
        raise ValueError(f"Expected 3 channels after conversion, got shape {x.shape}")

    return x


def save_reconstruction_batch(pred, out_dir, frame_ids=None, start_idx=0):
    """
    Save a batch of reconstructed frames as PNG.

    Args:
        pred:
            Predicted images as torch tensor or numpy array. Values should be in [0,1].
        out_dir:
            Output directory.
        frame_ids:
            Optional list/tensor of frame indices. If omitted, uses start_idx, start_idx+1, ...
        start_idx:
            Counter used when frame_ids is omitted.

    Returns:
        next_start_idx
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    batch = _to_numpy_batch(pred)
    b = batch.shape[0]

    if frame_ids is not None:
        try:
            import torch
            if isinstance(frame_ids, torch.Tensor):
                frame_ids = frame_ids.detach().cpu().numpy().tolist()
        except Exception:
            pass
        frame_ids = [int(v) for v in frame_ids]
        if len(frame_ids) != b:
            raise ValueError(f"frame_ids length {len(frame_ids)} does not match batch size {b}")
    else:
        frame_ids = list(range(start_idx, start_idx + b))

    for img, idx in zip(batch, frame_ids):
        img_u8 = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
        Image.fromarray(img_u8, mode="RGB").save(out_dir / f"{idx:06d}.png")

    return max(frame_ids) + 1 if frame_ids else start_idx + b
