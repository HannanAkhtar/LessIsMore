from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def list_frame_paths(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Frame folder not found: {folder}")
    frames = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    frames.sort(key=natural_key)
    if not frames:
        raise RuntimeError(f"No image frames found in {folder}. Expected {sorted(IMAGE_EXTS)}")
    return frames


def load_image_tensor(path: Path, img_size: Tuple[int, int], preprocess: str = "center_crop") -> torch.Tensor:
    """Load RGB image as float tensor in [0,1], shape C,H,W.

    img_size is (H, W). preprocess:
      - center_crop: crop to H,W when source is larger; resize if source is smaller.
      - resize: resize directly to H,W.
      - none: no resize/crop; img_size ignored.
    """
    img = Image.open(path).convert("RGB")
    H, W = img_size
    if preprocess == "resize":
        img = img.resize((W, H), Image.BICUBIC)
    elif preprocess == "center_crop":
        w, h = img.size
        if h < H or w < W:
            img = img.resize((W, H), Image.BICUBIC)
        else:
            left = (w - W) // 2
            top = (h - H) // 2
            img = img.crop((left, top, left + W, top + H))
    elif preprocess == "none":
        pass
    else:
        raise ValueError(f"Unknown preprocess={preprocess}. Use center_crop, resize, or none.")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class FrameFolderDataset(Dataset):
    def __init__(self, data_path: str | Path, img_size: Tuple[int, int], preprocess: str = "center_crop"):
        self.data_path = Path(data_path)
        self.frames = list_frame_paths(self.data_path)
        self.img_size = tuple(img_size)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = load_image_tensor(self.frames[idx], self.img_size, self.preprocess)
        return {"image": x, "index": idx, "path": str(self.frames[idx])}


class EmbeddingFrameDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, targets: torch.Tensor, indices: Optional[torch.Tensor] = None):
        self.embeddings = embeddings.float().cpu()
        self.targets = targets.float().cpu()
        if indices is None:
            self.indices = torch.arange(len(self.embeddings), dtype=torch.long)
        else:
            self.indices = indices.long().cpu()

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"embedding": self.embeddings[idx], "image": self.targets[idx], "index": int(self.indices[idx])}

class EmbeddingFramePathDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, frame_paths: List[str], img_size: Tuple[int, int], preprocess: str = "center_crop", indices: Optional[List[int]] = None):
        self.embeddings = embeddings.float().cpu()
        self.frame_paths = [str(p) for p in frame_paths]
        self.img_size = tuple(img_size)
        self.preprocess = preprocess
        if indices is None:
            self.indices = list(range(len(self.frame_paths)))
        else:
            self.indices = list(map(int, indices))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        global_idx = self.indices[idx]
        x = load_image_tensor(Path(self.frame_paths[global_idx]), self.img_size, self.preprocess)
        return {"embedding": self.embeddings[global_idx], "image": x, "index": global_idx}
