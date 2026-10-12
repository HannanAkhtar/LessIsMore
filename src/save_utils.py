from pathlib import Path
import numpy as np
from PIL import Image


def _to_numpy_batch(x):
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
        raise ValueError(f"Expected 3D or 4D tensor/array, got {x.shape}")

    # B,C,H,W -> B,H,W,C
    if x.shape[1] in (1, 3) and x.shape[-1] not in (1, 3):
        x = np.transpose(x, (0, 2, 3, 1))

    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)

    if x.shape[-1] != 3:
        raise ValueError(f"Expected RGB output, got {x.shape}")

    return x


def save_reconstruction_batch(pred, out_dir, frame_ids=None, start_idx=0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    batch = _to_numpy_batch(pred)
    b = batch.shape[0]

    if frame_ids is None:
        frame_ids = list(range(start_idx, start_idx + b))
    else:
        try:
            import torch
            if isinstance(frame_ids, torch.Tensor):
                frame_ids = frame_ids.detach().cpu().numpy().tolist()
        except Exception:
            pass
        frame_ids = [int(x) for x in frame_ids]

    for img, idx in zip(batch, frame_ids):
        img_u8 = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
        Image.fromarray(img_u8, mode="RGB").save(out_dir / f"{idx:06d}.png")

    return max(frame_ids) + 1
