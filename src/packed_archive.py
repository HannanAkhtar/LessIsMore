"""Fixed-width, decodable playback container. The decoder software is external.

All bytes in the ZIP_STORED container count toward measured BPP, including its
directory, descriptors, tensor scales, and bit-packed embedding symbols.
This format is intentionally not an entropy-coded video bitstream.
"""
from __future__ import annotations

import json
import math
from itertools import zip_longest
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import numpy as np


FORMAT = "lessismore-packed-v1"


def packed_bpp(archive_path, frames: int, height: int, width: int) -> float:
    if min(frames, height, width) <= 0:
        raise ValueError("frames, height, and width must be positive")
    return 8 * Path(archive_path).stat().st_size / (frames * height * width)


def pack_symbols(values, bits: int) -> bytes:
    if not 1 <= bits <= 8:
        raise ValueError("fixed-width symbols require 1 to 8 bits")
    values = np.asarray(values, dtype=np.uint8).ravel()
    if values.size and np.any(values > (1 << bits) - 1):
        raise ValueError("symbol exceeds its declared bit depth")
    if bits == 8:
        return values.tobytes()
    planes = ((values[:, None] >> np.arange(bits, dtype=np.uint8)) & 1).astype(np.uint8)
    return np.packbits(planes.ravel(), bitorder="little").tobytes()


def unpack_symbols(data: bytes, count: int, bits: int) -> np.ndarray:
    if not 1 <= bits <= 8 or count < 0:
        raise ValueError("invalid symbol count or bit depth")
    if len(data) != math.ceil(count * bits / 8):
        raise ValueError("symbol payload has the wrong byte length")
    if bits == 8:
        return np.frombuffer(data, dtype=np.uint8).copy()
    planes = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")[:count * bits]
    return (planes.reshape(count, bits).astype(np.uint16) * (1 << np.arange(bits))).sum(axis=1).astype(np.uint8)


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _write_tensor(archive: ZipFile, prefix: str, pack: dict, bits: int) -> dict:
    symbols = _array(pack["quant"]).astype(np.uint8, copy=False)
    minimum = _array(pack["min"])
    scale = _array(pack["scale"])
    if minimum.dtype not in (np.float16, np.float32, np.float64) or scale.dtype not in (np.float16, np.float32, np.float64):
        raise ValueError("quantization scales must be floating point")
    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(scale)) or np.any(scale < 0):
        raise ValueError("invalid quantization min/scale")
    for scale_shape in (minimum.shape, scale.shape):
        for source_dim, scale_dim in zip_longest(reversed(symbols.shape),
                                                 reversed(scale_shape), fillvalue=1):
            if scale_dim not in (1, source_dim):
                raise ValueError("quantization scales do not broadcast to the symbols")
    archive.writestr(prefix + ".symbols", pack_symbols(symbols, bits))
    archive.writestr(prefix + ".min", minimum.tobytes())
    archive.writestr(prefix + ".scale", scale.tobytes())
    return {
        "prefix": prefix, "bits": bits, "shape": list(symbols.shape),
        "min_shape": list(minimum.shape), "scale_shape": list(scale.shape),
        "min_dtype": minimum.dtype.name, "scale_dtype": scale.dtype.name,
    }


def _read_tensor(archive: ZipFile, meta: dict) -> dict:
    shape = tuple(meta["shape"])
    count = math.prod(shape)
    prefix = meta["prefix"]
    symbols = unpack_symbols(archive.read(prefix + ".symbols"), count, meta["bits"]).reshape(shape)
    arrays = {}
    for name in ("min", "scale"):
        dtype = np.dtype(meta[name + "_dtype"])
        size = tuple(meta[name + "_shape"])
        raw = archive.read(prefix + "." + name)
        if len(raw) != math.prod(size) * dtype.itemsize:
            raise ValueError(f"invalid {name} payload")
        arrays[name] = np.frombuffer(raw, dtype=dtype).copy().reshape(size)
    return {"quant": symbols, **arrays}


def dequantize(pack: dict) -> np.ndarray:
    return _array(pack["min"]).astype(np.float32) + _array(pack["scale"]).astype(np.float32) * _array(pack["quant"]).astype(np.float32)


def write_archive(path, *, tiles: dict, frames: int, height: int, width: int,
                  layout: dict, model_bits: int, embedding_bits: int) -> float:
    """Write quantized decoder states and per-frame embeddings, then return measured BPP.

    A tile is {architecture: JSON object, weights: {state_key: quant pack},
    embeddings: quant pack}. The architecture must be sufficient for the caller's
    decoder factory to rebuild the model. Original RGB frames are never stored.
    """
    if not tiles or min(frames, height, width) <= 0:
        raise ValueError("nonempty tiles and positive frame geometry are required")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"format": FORMAT, "frames": frames, "height": height, "width": width,
                "layout": layout, "model_bits": model_bits, "embedding_bits": embedding_bits, "tiles": {}}
    with ZipFile(path, "w", compression=ZIP_STORED) as archive:
        for tile_id, tile in sorted(tiles.items()):
            if not tile_id.isalnum() or not tile["weights"]:
                raise ValueError("tile IDs must be alphanumeric and weights nonempty")
            if _array(tile["embeddings"]["quant"]).shape[0] != frames:
                raise ValueError(f"embedding frame count differs for {tile_id}")
            weights = {}
            for index, (key, pack) in enumerate(sorted(tile["weights"].items())):
                weights[key] = _write_tensor(archive, f"tiles/{tile_id}/w{index}", pack, model_bits)
            manifest["tiles"][tile_id] = {
                "architecture": tile["architecture"], "weights": weights,
                "embeddings": _write_tensor(archive, f"tiles/{tile_id}/embeddings", tile["embeddings"], embedding_bits),
            }
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    return packed_bpp(path, frames, height, width)


def read_archive(path) -> dict:
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("format") != FORMAT:
            raise ValueError("unsupported packed archive format")
        tiles = {}
        for tile_id, tile in manifest["tiles"].items():
            tiles[tile_id] = {
                "architecture": tile["architecture"],
                "weights": {key: _read_tensor(archive, value) for key, value in tile["weights"].items()},
                "embeddings": _read_tensor(archive, tile["embeddings"]),
            }
        return {**manifest, "tiles": tiles}
