"""Evaluate a trusted whole-frame checkpoint through a decodable packed archive."""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .packed_archive import dequantize, read_archive, write_archive

try:
    import torch
    from .data import FrameFolderDataset
    from .flops import count_model_flops
    from .metrics import batch_psnr, batch_publication_ssim
    from .models import (HNeRVDecoder, LowRankHNeRVDecoder,
                         build_hnerv_with_target_decoder_params, decoder_num_parameters)
    from .quant_eval import (apply_global_magnitude_pruning,
                             quant_tensor_like_hnerv)
    from .run_lowrank_ablation_grid import build_lowrank_ablation_model
    IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    IMPORT_ERROR = exc


def build_checkpoint_model(ckpt: dict, kind: str, blocks: list[int]):
    saved = argparse.Namespace(**ckpt["args"])
    common = dict(target_m=saved.baseline_decoder_m, emb_dim=saved.emb_dim,
                  c1=saved.c1, enc_strides=saved.enc_strides,
                  dec_strides=saved.dec_strides, reduce=saved.reduce,
                  kmin=saved.kmin, kmax=saved.kmax,
                  blocks_per_stage=saved.blocks_per_stage)
    dense = build_hnerv_with_target_decoder_params(**common)
    if kind == "dense":
        model = dense
    else:
        model = build_lowrank_ablation_model(saved, "widened", blocks, dense.decoder.dec_width)
    model.load_state_dict(ckpt["model"], strict=True)
    return model


def decoder_descriptor(decoder):
    desc = {key: getattr(decoder, key) for key in
            ("emb_dim", "dec_width", "dec_strides", "reduce", "kmin", "kmax")}
    if isinstance(decoder, LowRankHNeRVDecoder):
        desc.update(kind="lowrank", lowrank_rank=decoder.lowrank_rank,
                    lowrank_factorize_from=decoder.lowrank_factorize_from,
                    lowrank_blocks=decoder.lowrank_blocks)
    else:
        desc["kind"] = "dense"
    return desc


def decoder_from_descriptor(desc):
    kwargs = {k: v for k, v in desc.items() if k != "kind"}
    if desc["kind"] == "dense":
        return HNeRVDecoder(**kwargs)
    if desc["kind"] == "lowrank":
        return LowRankHNeRVDecoder(**kwargs)
    raise ValueError(f"Unknown decoder architecture: {desc['kind']}")


def playback_from_archive(path, device):
    archive = read_archive(path)
    if archive["layout"] != {"mode": "wholeframe"} or set(archive["tiles"]) != {"F"}:
        raise ValueError("This entry point expects one whole-frame tile")
    tile = archive["tiles"]["F"]
    decoder = decoder_from_descriptor(tile["architecture"]).to(device).eval()
    state = decoder.state_dict()
    if set(state) != set(tile["weights"]):
        raise ValueError("Archive decoder keys do not match its architecture")
    for key, tensor in state.items():
        state[key] = torch.from_numpy(dequantize(tile["weights"][key]).copy()).to(
            device=device, dtype=tensor.dtype)
    decoder.load_state_dict(state, strict=True)
    embeddings = torch.from_numpy(dequantize(tile["embeddings"]).copy()).to(device)
    return decoder, embeddings


def evaluate(args):
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not Path(args.frames).is_dir():
        raise FileNotFoundError(f"Frame directory not found: {args.frames}")
    if not 1 <= args.model_bits <= 8 or not 1 <= args.embedding_bits <= 8:
        raise ValueError("Fixed-width archive supports 1 to 8 bits for weights and embeddings")
    if not 0 <= args.prune_ratio < 1:
        raise ValueError("--prune-ratio must be in [0, 1)")
    if IMPORT_ERROR is not None:
        raise RuntimeError(f"Model evaluation dependency unavailable: {IMPORT_ERROR}") from IMPORT_ERROR
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if Path(args.archive).exists():
        raise FileExistsError(f"Archive output already exists: {args.archive}")
    if args.report and Path(args.report).exists():
        raise FileExistsError(f"Report output already exists: {args.report}")
    if args.recon_dir and Path(args.recon_dir).exists() and any(Path(args.recon_dir).iterdir()):
        raise FileExistsError(f"Reconstruction output is not empty: {args.recon_dir}")
    dataset = FrameFolderDataset(args.frames, tuple(args.size), "center_crop")
    if args.expected_frames and len(dataset) != args.expected_frames:
        raise ValueError(f"Expected {args.expected_frames} frames, found {len(dataset)}")
    # PyTorch checkpoints use pickle. Only load checkpoints from trusted sources.
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_checkpoint_model(ckpt, args.model_type,
                                   [int(x) for x in args.blocks.split(",")]).to(device).eval()
    z = []
    with torch.no_grad():
        for sample in dataset:
            z.append(model.encoder(sample["image"].unsqueeze(0).to(device)).cpu())
    embeddings = torch.cat(z, dim=0)
    embedding_pack, _ = quant_tensor_like_hnerv(embeddings, args.embedding_bits)
    playback = copy.deepcopy(model).to(device).eval()
    apply_global_magnitude_pruning(playback, args.prune_ratio)
    weight_packs = {}
    for key, value in playback.decoder.state_dict().items():
        weight_packs[key], _ = quant_tensor_like_hnerv(value.cpu(), args.model_bits)
    height, width = args.size
    bpp = write_archive(args.archive, tiles={"F": {
        "architecture": decoder_descriptor(playback.decoder),
        "weights": weight_packs, "embeddings": embedding_pack,
    }}, frames=len(dataset), height=height, width=width,
        layout={"mode": "wholeframe"}, model_bits=args.model_bits,
        embedding_bits=args.embedding_bits)
    decoder, stored_z = playback_from_archive(args.archive, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stored = read_archive(args.archive)["tiles"]["F"]
    if not np.array_equal(stored["embeddings"]["quant"], embedding_pack["quant"].numpy()):
        raise RuntimeError("Embedding symbols changed during archive round-trip")
    for key, pack in weight_packs.items():
        if not np.array_equal(stored["weights"][key]["quant"], pack["quant"].numpy()):
            raise RuntimeError(f"Weight symbols changed during archive round-trip: {key}")
    if args.recon_dir:
        Path(args.recon_dir).mkdir(parents=True, exist_ok=True)
    psnr, ssim, latencies = [], [], []
    with torch.no_grad():
        for i, sample in enumerate(dataset):
            target = sample["image"].unsqueeze(0).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = decoder(stored_z[i:i + 1])
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)
            psnr.append(batch_psnr(pred, target))
            ssim.append(batch_publication_ssim(pred, target))
            if args.recon_dir:
                rgb = (pred[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
                Image.fromarray(rgb).save(Path(args.recon_dir) / f"{i:06d}.png")
    macs = count_model_flops(decoder, tuple(stored_z[:1].shape), str(device), flop_mult=1)
    report = {
        "protocol": "decoded_fixed_width_archive", "checkpoint": str(args.checkpoint),
        "archive": str(args.archive), "frames": len(dataset), "height": height, "width": width,
        "model_bits": args.model_bits, "embedding_bits": args.embedding_bits,
        "prune_ratio": args.prune_ratio, "archive_bytes": Path(args.archive).stat().st_size,
        "packed_bpp": bpp, "full_frame_psnr_db": statistics.mean(psnr),
        "full_frame_windowed_ssim": statistics.mean(ssim),
        "decoder_parameters": decoder_num_parameters(decoder),
        "embedding_values": stored_z.numel(),
        "playback_parameters": decoder_num_parameters(decoder) + stored_z.numel(),
        "decoder_macs_per_frame": macs, "decoder_gmacs_per_frame": macs / 1e9,
        "decoder_gflops_per_frame": 2 * macs / 1e9,
        "flop_convention": "one multiply-accumulate = two FLOPs; Conv2d/Linear multiplies and adds, excluding activation, bias, and data movement",
        "median_decoder_ms_per_frame": statistics.median(latencies),
        "diagnostic_decoder_fps": 1000 / statistics.median(latencies),
    }
    if device.type == "cuda":
        report["diagnostic_peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames", required=True, help="Ordered input frame folder")
    parser.add_argument("--archive", required=True, help="Output decodable ZIP archive")
    parser.add_argument("--model-type", choices=["dense", "lrwide"], required=True)
    parser.add_argument("--blocks", default="3,4", help="Factorized stage IDs for LR-wide")
    parser.add_argument("--size", type=int, nargs=2, metavar=("H", "W"), default=[640, 1280])
    parser.add_argument("--expected-frames", type=int, default=132)
    parser.add_argument("--model-bits", type=int, default=8)
    parser.add_argument("--embedding-bits", type=int, default=8)
    parser.add_argument("--prune-ratio", type=float, default=0.10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--recon-dir")
    parser.add_argument("--report")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
