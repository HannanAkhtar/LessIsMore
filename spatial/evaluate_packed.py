"""Pack and evaluate spatial HNeRV quant_vid.pth files without retraining.

Run from the repository root. Checkpoint files are PyTorch pickle files: load
only files whose origin you trust. The resulting ZIP archive is data-only.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent / "HNeRV-main"))

from src.packed_archive import dequantize, read_archive, write_archive  # noqa: E402

try:
    import torch
    from model_all import HNeRV, HNeRVDecoder
    from src.data import list_frame_paths
    from src.flops import count_model_flops
    from src.metrics import batch_psnr, batch_publication_ssim
    IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    IMPORT_ERROR = exc


def _tensor_pack(pack):
    return {key: value.detach().cpu() for key, value in pack.items()}


def _model_from_descriptor(desc):
    args = argparse.Namespace(**desc)
    return HNeRVDecoder(HNeRV(args))


def _descriptor(cfg, band, quant_state):
    key = "decoder.0.conv.downconv.weight"
    if key not in quant_state:
        raise ValueError(f"Cannot infer fc_dim: missing {key} in spatial quantized model")
    fc_dim = int(quant_state[key]["quant"].shape[0])
    selected = {int(x) for x in str(cfg.get("lowrank_blocks", "")).split(",") if x.strip()}
    for stage in selected:
        factor_key = f"decoder.{stage + 1}.conv.upconv.0.weight"
        if factor_key not in quant_state:
            raise ValueError(f"Missing factorized stage {stage} in {band}: {factor_key}")
        actual_rank = int(quant_state[factor_key]["quant"].shape[0])
        expected_rank = int(cfg["bands"][band]["rank"])
        if actual_rank != expected_rank:
            raise ValueError(f"Rank mismatch for {band} stage {stage}: checkpoint {actual_rank}, config {expected_rank}")
    return {
        "embed": "", "ks": cfg["ks"], "num_blks": cfg.get("num_blks", "1_1"),
        "enc_strds": cfg["enc_strds"], "enc_dim": cfg["enc_dim"],
        "conv_type": cfg["conv_type"], "fc_dim": fc_dim,
        "fc_hw": cfg.get("fc_hw", "9_16"), "reduce": cfg["reduce"],
        "lower_width": cfg["lower_width"], "dec_strds": cfg["dec_strds"],
        "norm": cfg["norm"], "act": cfg["act"],
        "lowrank_blocks": cfg.get("lowrank_blocks", ""),
        "lowrank_rank": cfg["bands"][band]["rank"],
        "out_bias": cfg.get("out_bias", "tanh"),
    }


def _expected_bands(mode):
    if mode == "quadrant":
        return [f"Q{i}" for i in range(4)]
    if mode in ("tiles16", "tile16"):
        return [f"T{r}{c}" for r in range(4) for c in range(4)]
    raise ValueError("Packed spatial playback supports quadrant and 16-tile modes")


def build_archive(args):
    cfg = json.loads(Path(args.config).read_text())
    bands = _expected_bands(cfg["mode"])
    given = dict(item.split("=", 1) for item in args.tile)
    if set(given) != set(bands):
        raise ValueError(f"Supply one --tile BAND=quant_vid.pth for each of {bands}")
    frame_count = len(list_frame_paths(args.frames))
    if args.expected_frames and frame_count != args.expected_frames:
        raise ValueError(f"Expected {args.expected_frames} frames, found {frame_count}")
    tiles = {}
    for band in bands:
        payload = torch.load(given[band], map_location="cpu")
        if set(payload) != {"embed", "model"}:
            raise ValueError(f"Unexpected quant_vid.pth structure for {band}")
        if payload["embed"]["quant"].shape[0] != frame_count:
            raise ValueError(f"Embedding count differs for {band}")
        tiles[band] = {
            "architecture": _descriptor(cfg, band, payload["model"]),
            "weights": {key: _tensor_pack(pack) for key, pack in payload["model"].items()},
            "embeddings": _tensor_pack(payload["embed"]),
        }
    height, width = args.size
    bpp = write_archive(args.archive, tiles=tiles, frames=frame_count,
                         height=height, width=width,
                         layout={"mode": cfg["mode"], "bands": bands,
                                 "padded_tile": [int(x) for x in cfg["crop_list"].split("_")]},
                         model_bits=cfg["quant_model_bit"],
                         embedding_bits=cfg["quant_embed_bit"])
    written = read_archive(args.archive)
    for band, tile in tiles.items():
        if not np.array_equal(written["tiles"][band]["embeddings"]["quant"],
                              tile["embeddings"]["quant"].numpy()):
            raise RuntimeError(f"Embedding symbols changed during archive round-trip: {band}")
        for key, pack in tile["weights"].items():
            if not np.array_equal(written["tiles"][band]["weights"][key]["quant"],
                                  pack["quant"].numpy()):
                raise RuntimeError(f"Weight symbols changed during archive round-trip: {band}/{key}")
    return bpp


def load_decoders(archive, device):
    result = {}
    for band, tile in archive["tiles"].items():
        decoder = _model_from_descriptor(tile["architecture"]).to(device).eval()
        state = decoder.state_dict()
        if set(state) != set(tile["weights"]):
            raise ValueError(f"Decoder state keys mismatch for {band}")
        for key, original in state.items():
            state[key] = torch.from_numpy(dequantize(tile["weights"][key]).copy()).to(
                device=device, dtype=original.dtype)
        decoder.load_state_dict(state, strict=True)
        embeddings = torch.from_numpy(dequantize(tile["embeddings"]).copy()).to(device)
        result[band] = decoder, embeddings
    return result


def stitch(parts, mode, valid_h, valid_w):
    if mode == "quadrant":
        rows = [["Q0", "Q1"], ["Q2", "Q3"]]
    else:
        rows = [[f"T{r}{c}" for c in range(4)] for r in range(4)]
    return torch.cat([torch.cat([parts[band][..., :valid_h, :valid_w] for band in row], dim=-1)
                      for row in rows], dim=-2)


def evaluate(args):
    if args.tile and not Path(args.config).is_file():
        raise FileNotFoundError(f"Spatial config not found: {args.config}")
    if not args.tile and not Path(args.archive).is_file():
        raise FileNotFoundError(f"Packed archive not found: {args.archive}")
    if not Path(args.frames).is_dir():
        raise FileNotFoundError(f"Frame directory not found: {args.frames}")
    if IMPORT_ERROR is not None:
        raise RuntimeError(f"Spatial model dependency unavailable: {IMPORT_ERROR}") from IMPORT_ERROR
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.tile and Path(args.archive).exists():
        raise FileExistsError(f"Archive output already exists: {args.archive}")
    if args.report and Path(args.report).exists():
        raise FileExistsError(f"Report output already exists: {args.report}")
    if args.recon_dir and Path(args.recon_dir).exists() and any(Path(args.recon_dir).iterdir()):
        raise FileExistsError(f"Reconstruction output is not empty: {args.recon_dir}")
    if args.tile:
        build_archive(args)
    archive = read_archive(args.archive)
    bands = _expected_bands(archive["layout"]["mode"])
    if set(archive["tiles"]) != set(bands):
        raise ValueError("Archive tile set does not match its layout")
    if tuple(args.size) != (archive["height"], archive["width"]):
        raise ValueError("Requested full-frame size differs from archive metadata")
    height, width = args.size
    grid = 2 if len(bands) == 4 else 4
    if height % grid or width % grid:
        raise ValueError("Full-frame dimensions must be divisible by tile grid")
    valid_h, valid_w = height // grid, width // grid
    decoders = load_decoders(archive, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    paths = list_frame_paths(args.frames)
    if len(paths) != archive["frames"]:
        raise ValueError("Source frame count differs from archive")
    if args.expected_frames and len(paths) != args.expected_frames:
        raise ValueError(f"Expected {args.expected_frames} frames, found {len(paths)}")
    if args.recon_dir:
        Path(args.recon_dir).mkdir(parents=True, exist_ok=True)
    psnr, ssim, times = [], [], []
    with torch.no_grad():
        for i, path in enumerate(paths):
            source = Image.open(path).convert("RGB")
            if source.height < height or source.width < width:
                raise ValueError(f"Source frame is smaller than {height}x{width}: {path}")
            left = (source.width - width) // 2
            top = (source.height - height) // 2
            target = np.asarray(source.crop((left, top, left + width, top + height)), dtype=np.float32) / 255
            target_tensor = torch.from_numpy(target).permute(2, 0, 1)[None].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            parts = {band: decoder(z[i:i + 1]) for band, (decoder, z) in decoders.items()}
            if any(p.shape[-2] < valid_h or p.shape[-1] < valid_w for p in parts.values()):
                raise ValueError("Decoder output is smaller than valid tile region")
            full = stitch(parts, archive["layout"]["mode"], valid_h, valid_w)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
            psnr.append(batch_psnr(full, target_tensor))
            ssim.append(batch_publication_ssim(full, target_tensor))
            if args.recon_dir:
                rgb = (full[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + .5).astype(np.uint8)
                Image.fromarray(rgb).save(Path(args.recon_dir) / f"{i:06d}.png")
    macs = sum(count_model_flops(decoder, tuple(z[:1].shape), str(device), 1)
               for decoder, z in decoders.values())
    decoder_params = sum(sum(p.numel() for p in decoder.parameters())
                         for decoder, _ in decoders.values())
    embed_values = sum(z.numel() for _, z in decoders.values())
    report = {
        "protocol": "decoded_fixed_width_spatial_archive", "archive": str(args.archive),
        "frames": len(paths), "height": height, "width": width,
        "archive_bytes": Path(args.archive).stat().st_size,
        "packed_bpp": 8 * Path(args.archive).stat().st_size / (len(paths) * height * width),
        "full_frame_psnr_db": statistics.mean(psnr),
        "full_frame_windowed_ssim": statistics.mean(ssim),
        "decoder_parameters_all_tiles": decoder_params,
        "embedding_values_all_tiles": embed_values,
        "playback_parameters_all_tiles": decoder_params + embed_values,
        "decoder_macs_per_frame": macs, "decoder_gmacs_per_frame": macs / 1e9,
        "decoder_gflops_per_frame": 2 * macs / 1e9,
        "flop_convention": "one MAC = two FLOPs; Conv2d/Linear multiplications and additions on every padded tile pixel",
        "median_decode_and_stitch_ms_per_frame": statistics.median(times),
        "diagnostic_decode_and_stitch_fps": 1000 / statistics.median(times),
    }
    if device.type == "cuda":
        report["diagnostic_peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Spatial JSON configuration, required when packing quant_vid files")
    parser.add_argument("--tile", action="append", default=[], help="BAND=path/to/quant_vid.pth; repeat for every tile")
    parser.add_argument("--frames", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--size", nargs=2, type=int, metavar=("H", "W"), default=[640, 1280])
    parser.add_argument("--expected-frames", type=int, default=132)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--recon-dir")
    parser.add_argument("--report")
    args = parser.parse_args()
    if args.tile and not args.config:
        parser.error("--config is required with --tile")
    evaluate(args)


if __name__ == "__main__":
    main()
