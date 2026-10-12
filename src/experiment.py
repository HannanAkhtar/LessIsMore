from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .data import FrameFolderDataset, EmbeddingFramePathDataset
from .models import (
    build_hnerv_with_target_decoder_params,
    build_decoder_with_target_params,
    build_lowrank_decoder_with_target_params,
    build_multihead_hnerv_with_width,
    build_multihead_hnerv_with_target_decoder_params,
    build_multihead_refine_hnerv_with_width,
    build_multihead_refine_hnerv_with_target_decoder_params,
    decoder_num_parameters,
    encoder_num_parameters,
    total_num_parameters,
    model_config_dict,
    infer_embedding_shape,
)
from .metrics import batch_psnr, batch_ssim, compression_bpp
from .quant_eval import evaluate_quantized_pruned_playback_model
from .flops import count_model_flops
from .utils import ensure_dir, save_json, pick_device, move_batch, set_seed


def simple_kmeans_predict(X: np.ndarray, n_clusters: int, seed: int = 0, n_init: int = 10, max_iter: int = 100) -> np.ndarray:
    """Small NumPy-only K-Means used to avoid a hard scikit-learn dependency.

    Args:
        X: Array of shape [num_samples, num_features].
        n_clusters: Number of clusters/experts.
        seed: Random seed.
        n_init: Number of random initializations.
        max_iter: Maximum iterations per initialization.

    Returns:
        Integer labels of shape [num_samples].
    """
    X = np.asarray(X, dtype=np.float32)
    n_samples = X.shape[0]
    if n_samples == 0:
        return np.zeros((0,), dtype=np.int64)
    n_clusters = int(min(max(1, n_clusters), n_samples))

    rng = np.random.default_rng(seed)
    best_labels = None
    best_inertia = np.inf

    for _ in range(max(1, n_init)):
        init_idx = rng.choice(n_samples, size=n_clusters, replace=False)
        centers = X[init_idx].copy()
        labels = np.zeros(n_samples, dtype=np.int64)

        for _iter in range(max_iter):
            # Squared Euclidean distances: [num_samples, n_clusters]
            dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = np.argmin(dists, axis=1).astype(np.int64)

            if np.array_equal(new_labels, labels) and _iter > 0:
                break
            labels = new_labels

            # Update centers. If a cluster becomes empty, re-seed it from a random sample.
            for k in range(n_clusters):
                members = X[labels == k]
                if len(members) == 0:
                    centers[k] = X[rng.integers(0, n_samples)]
                else:
                    centers[k] = members.mean(axis=0)

        inertia = float(((X - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    # If requested clusters > samples, labels can only cover n_samples clusters. This is fine for debug.
    return best_labels.astype(np.int64)


def train_one_baseline(args) -> Dict[str, Any]:
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = pick_device(args.device)
    dataset = FrameFolderDataset(args.data_path, img_size=(args.img_size[0], args.img_size[1]), preprocess=args.preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    model = build_hnerv_with_target_decoder_params(
        target_m=args.baseline_decoder_m,
        emb_dim=args.emb_dim,
        c1=args.c1,
        enc_strides=args.enc_strides,
        dec_strides=args.dec_strides,
        reduce=args.reduce,
        kmin=args.kmin,
        kmax=args.kmax,
        blocks_per_stage=args.blocks_per_stage,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs_baseline))
    loss_fn = nn.MSELoss()

    best_psnr = -1.0
    best_path = out_dir / "baseline_best.pt"
    last_path = out_dir / "baseline_last.pt"
    t0 = time.time()
    for epoch in range(1, args.epochs_baseline + 1):
        model.train()
        losses = []
        pbar = tqdm(loader, desc=f"baseline epoch {epoch}/{args.epochs_baseline}", leave=False)
        for batch in pbar:
            batch = move_batch(batch, device)
            x = batch["image"]
            pred = model(x)
            loss = loss_fn(pred, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=np.mean(losses))
        sched.step()
        if epoch % args.eval_every == 0 or epoch == args.epochs_baseline:
            metrics = evaluate_baseline_model(model, dataset, args, device)
            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]
                torch.save({
                    "model": model.state_dict(),
                    "config": model_config_dict(model),
                    "args": vars(args),
                    "metrics": metrics,
                }, best_path)
    torch.save({
        "model": model.state_dict(),
        "config": model_config_dict(model),
        "args": vars(args),
    }, last_path)
    metrics = evaluate_baseline_model(model, dataset, args, device)
    metrics.update({
        "train_seconds": time.time() - t0,
        "checkpoint": str(last_path),
        "best_checkpoint": str(best_path),
    })
    save_json(metrics, out_dir / "baseline_metrics.json")
    return metrics


@torch.no_grad()
def evaluate_baseline_model(model, dataset, args, device) -> Dict[str, Any]:
    if getattr(args, "eval_variant", "quant") == "quant":
        return evaluate_quantized_pruned_playback_model(
            model=model,
            dataset=dataset,
            args=args,
            device=device,
            model_type="baseline",
        )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model.eval().to(device)
    psnrs, ssims = [], []
    for batch in tqdm(loader, desc="eval baseline orig", leave=False):
        batch = move_batch(batch, device)
        x = batch["image"]
        pred = model(x)
        psnrs.append(batch_psnr(pred, x))
        ssims.append(batch_ssim(pred, x))
    T = len(dataset)
    H, W = args.img_size
    emb_shape = infer_embedding_shape((H, W), args.enc_strides, args.emb_dim)
    embedding_values = T * int(np.prod(emb_shape))
    dec_params = decoder_num_parameters(model)
    bpp_stats = compression_bpp(dec_params, embedding_values, T, H, W, args.model_bits, args.embed_bits, 0, args.prune_ratio)
    flops = count_model_flops(model.decoder, (1, *emb_shape), device=str(device), flop_mult=args.flop_mult)
    return {
        "model_type": "baseline",
        "sequence": Path(args.data_path).name,
        "frames": T,
        "height": H,
        "width": W,
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "decoder_params": int(dec_params),
        "encoder_params_not_stored": int(encoder_num_parameters(model)),
        "total_trainable_params": int(total_num_parameters(model)),
        "embedding_values": int(embedding_values),
        "cluster_bits": 0,
        "gflops_per_frame_decoder_only": float(flops / 1e9),
        "eval_variant": "orig_model_forward",
        "actual_model_quantized": False,
        "actual_embed_quantized": False,
        "actual_pruned": False,
        **bpp_stats,
    }


@torch.no_grad()
def extract_embeddings(args) -> Dict[str, Any]:
    out_dir = ensure_dir(args.out_dir)
    device = pick_device(args.device)
    dataset = FrameFolderDataset(args.data_path, img_size=tuple(args.img_size), preprocess=args.preprocess)
    # Rebuild model with same architecture and load checkpoint.
    model = build_hnerv_with_target_decoder_params(
        target_m=args.baseline_decoder_m,
        emb_dim=args.emb_dim,
        c1=args.c1,
        enc_strides=args.enc_strides,
        dec_strides=args.dec_strides,
        reduce=args.reduce,
        kmin=args.kmin,
        kmax=args.kmax,
        blocks_per_stage=args.blocks_per_stage,
    ).to(device)
    ckpt = torch.load(out_dir / "baseline_last.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    embeddings = []
    errors = []
    paths = []
    for batch in tqdm(loader, desc="extract embeddings", leave=False):
        batch = move_batch(batch, device)
        x = batch["image"]
        z = model.encoder(x)
        pred = model.decoder(z)
        err = torch.mean((pred - x) ** 2, dim=(1,2,3))
        embeddings.append(z.cpu())
        errors.append(err.cpu())
        paths += batch["path"]
    embeddings = torch.cat(embeddings, dim=0)
    errors = torch.cat(errors, dim=0)
    payload = {
        "embeddings": embeddings,
        "errors": errors,
        "paths": paths,
        "img_size": list(args.img_size),
        "preprocess": args.preprocess,
    }
    path = out_dir / "embeddings.pt"
    torch.save(payload, path)
    return {"embeddings_path": str(path), "num_embeddings": embeddings.shape[0], "embedding_shape": list(embeddings.shape[1:])}


def cluster_embeddings(args) -> Dict[str, Any]:
    out_dir = ensure_dir(args.out_dir)
    payload = torch.load(out_dir / "embeddings.pt", map_location="cpu")
    z = payload["embeddings"]
    errors = payload["errors"]
    Z = z.flatten(1).numpy().astype(np.float32)
    if args.cluster_use_error:
        e = errors.numpy().astype(np.float32)
        e = (e - e.mean()) / (e.std() + 1e-8)
        Z = np.concatenate([Z, args.error_weight * e[:, None]], axis=1)
    if args.cluster_method == "embedding":
        labels = simple_kmeans_predict(Z, n_clusters=args.num_experts, seed=args.seed, n_init=10, max_iter=100)
    elif args.cluster_method == "random":
        rng = np.random.default_rng(args.seed)
        labels = rng.integers(0, args.num_experts, size=Z.shape[0])
    elif args.cluster_method == "temporal":
        labels = np.floor(np.linspace(0, args.num_experts, Z.shape[0], endpoint=False)).astype(int)
    else:
        raise ValueError(f"Unknown cluster_method={args.cluster_method}")
    labels = labels.astype(np.int64)
    counts = {int(k): int((labels == k).sum()) for k in range(args.num_experts)}
    torch.save({"labels": torch.tensor(labels), "counts": counts, "method": args.cluster_method}, out_dir / "clusters.pt")
    save_json({"counts": counts, "method": args.cluster_method, "num_experts": args.num_experts}, out_dir / "clusters.json")
    return {"cluster_counts": counts}


def _build_expert_decoder(args):
    """Build the selected expert decoder type.

    dense:    searches a narrow HNeRV decoder to match --expert_decoder_m.
    lowrank: keeps the dense baseline decoder width and searches low-rank
              bottleneck rank to match --expert_decoder_m.
    """
    expert_type = getattr(args, "expert_decoder_type", "dense")
    if expert_type == "lowrank":
        base_m = getattr(args, "lowrank_base_decoder_m", None) or args.baseline_decoder_m
        base = build_hnerv_with_target_decoder_params(
            target_m=base_m,
            emb_dim=args.emb_dim,
            c1=args.c1,
            enc_strides=args.enc_strides,
            dec_strides=args.dec_strides,
            reduce=args.reduce,
            kmin=args.kmin,
            kmax=args.kmax,
            blocks_per_stage=args.blocks_per_stage,
        )
        factorize_from = getattr(args, "lowrank_factorize_from", "auto")
        return build_lowrank_decoder_with_target_params(
            target_m=args.expert_decoder_m,
            emb_dim=args.emb_dim,
            dec_strides=args.dec_strides,
            reduce=args.reduce,
            kmin=args.kmin,
            kmax=args.kmax,
            base_dec_width=base.decoder.dec_width,
            lowrank_factorize_from=factorize_from,
            rank_search_max=getattr(args, "lowrank_rank_search_max", 2048),
        )
    return build_decoder_with_target_params(
        args.expert_decoder_m,
        args.emb_dim,
        args.dec_strides,
        args.reduce,
        args.kmin,
        args.kmax,
    )


def train_experts(args) -> Dict[str, Any]:
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = pick_device(args.device)
    payload = torch.load(out_dir / "embeddings.pt", map_location="cpu")
    clusters = torch.load(out_dir / "clusters.pt", map_location="cpu")
    embeddings = payload["embeddings"]
    frame_paths = payload["paths"]
    labels = clusters["labels"].long()

    expert_metrics = []
    expert_param_sum = 0
    loss_fn = nn.MSELoss()

    for k in range(args.num_experts):
        indices = torch.where(labels == k)[0].tolist()
        if not indices:
            # Empty clusters can happen for random/small test data. Store None and continue.
            expert_metrics.append({"expert": k, "frames": 0, "decoder_params": 0, "checkpoint": None})
            continue
        dataset = EmbeddingFramePathDataset(embeddings, frame_paths, img_size=tuple(args.img_size), preprocess=args.preprocess, indices=indices)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        decoder = _build_expert_decoder(args).to(device)
        opt = torch.optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs_experts))
        t0 = time.time()
        for epoch in range(1, args.epochs_experts + 1):
            decoder.train()
            losses = []
            pbar = tqdm(loader, desc=f"expert {k} epoch {epoch}/{args.epochs_experts}", leave=False)
            for batch in pbar:
                batch = move_batch(batch, device)
                z = batch["embedding"]
                x = batch["image"]
                pred = decoder(z)
                loss = loss_fn(pred, x)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
                pbar.set_postfix(loss=np.mean(losses))
            sched.step()
        ckpt_path = out_dir / f"expert_{k}.pt"
        torch.save({"decoder": decoder.state_dict(), "config": model_config_dict(decoder), "expert": k, "indices": indices}, ckpt_path)
        params = decoder_num_parameters(decoder)
        expert_param_sum += params
        expert_metrics.append({"expert": k, "frames": len(indices), "decoder_params": int(params), "train_seconds": time.time() - t0, "checkpoint": str(ckpt_path)})
    save_json({"experts": expert_metrics, "expert_decoder_param_sum": int(expert_param_sum)}, out_dir / "expert_train_metrics.json")
    return {"experts": expert_metrics, "expert_decoder_param_sum": int(expert_param_sum)}


@torch.no_grad()
def evaluate_experts(args) -> Dict[str, Any]:
    out_dir = ensure_dir(args.out_dir)
    device = pick_device(args.device)
    payload = torch.load(out_dir / "embeddings.pt", map_location="cpu")
    clusters = torch.load(out_dir / "clusters.pt", map_location="cpu")
    embeddings = payload["embeddings"]
    frame_paths = payload["paths"]
    labels = clusters["labels"].long()

    expert_decoders = {}
    total_dec_params = 0
    for k in range(args.num_experts):
        ckpt_path = out_dir / f"expert_{k}.pt"
        if not ckpt_path.exists():
            continue
        decoder = _build_expert_decoder(args).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        decoder.load_state_dict(ckpt["decoder"])
        decoder.eval()
        expert_decoders[k] = decoder
        total_dec_params += decoder_num_parameters(decoder)

    dataset = EmbeddingFramePathDataset(embeddings, frame_paths, img_size=tuple(args.img_size), preprocess=args.preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    psnrs, ssims = [], []
    for batch in tqdm(loader, desc="eval experts", leave=False):
        # Each batch can have mixed expert IDs, so group inside batch.
        idx = batch["index"]
        x = batch["image"].to(device)
        z = batch["embedding"].to(device)
        pred = torch.empty_like(x)
        for k, dec in expert_decoders.items():
            mask = (labels[idx] == k)
            if mask.any():
                pos = torch.where(mask)[0].to(device)
                pred[pos] = dec(z[pos])
        psnrs.append(batch_psnr(pred, x))
        ssims.append(batch_ssim(pred, x))

    T = len(dataset)
    H, W = args.img_size
    emb_shape = tuple(embeddings.shape[1:])
    embedding_values = T * int(np.prod(emb_shape))
    cluster_bits = T * math.ceil(math.log2(args.num_experts))
    bpp_stats = compression_bpp(total_dec_params, embedding_values, T, H, W, args.model_bits, args.embed_bits, cluster_bits, args.prune_ratio)
    # GFLOPs per decoded frame: weighted average by cluster frequency.
    gflops_weighted = 0.0
    for k, dec in expert_decoders.items():
        n_k = int((labels == k).sum().item())
        fl = count_model_flops(dec, (1, *emb_shape), device=str(device), flop_mult=args.flop_mult)
        gflops_weighted += (n_k / max(1, T)) * (fl / 1e9)

    metrics = {
        "model_type": "lowrank_clustered_experts" if getattr(args, "expert_decoder_type", "dense") == "lowrank" else "clustered_experts",
        "expert_decoder_type": getattr(args, "expert_decoder_type", "dense"),
        "expert_decoder_m": float(args.expert_decoder_m),
        "sequence": Path(args.data_path).name,
        "frames": T,
        "height": H,
        "width": W,
        "num_experts": args.num_experts,
        "cluster_method": args.cluster_method,
        "lowrank_factorize_from": getattr(args, "lowrank_factorize_from", None),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "decoder_params": int(total_dec_params),
        "embedding_values": int(embedding_values),
        "cluster_bits": int(cluster_bits),
        "gflops_per_frame_decoder_only": float(gflops_weighted),
        **bpp_stats,
    }
    save_json(metrics, out_dir / "clustered_expert_metrics.json")
    return metrics


class _ActiveHeadDecoderWrapper(nn.Module):
    """Small wrapper so the FLOP counter measures shared trunk + one selected head."""
    def __init__(self, decoder, head_id: int = 0):
        super().__init__()
        self.decoder = decoder
        self.head_id = int(head_id)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder.forward_single_head(z, self.head_id)


def _build_baseline_arch(args):
    return build_hnerv_with_target_decoder_params(
        target_m=args.baseline_decoder_m,
        emb_dim=args.emb_dim,
        c1=args.c1,
        enc_strides=args.enc_strides,
        dec_strides=args.dec_strides,
        reduce=args.reduce,
        kmin=args.kmin,
        kmax=args.kmax,
        blocks_per_stage=args.blocks_per_stage,
    )


def _build_multihead_arch(args):
    # Default diagnostic mode: use the same dec_width as baseline, then add K heads/refiners.
    # This keeps initialization from the trained baseline simple. Extra params are reported.
    baseline_arch = _build_baseline_arch(args)
    baseline_width = baseline_arch.decoder.dec_width
    head_type = getattr(args, "multihead_head_type", "rgb")
    if getattr(args, "multihead_budget_neutral", False):
        target_m = getattr(args, "multihead_decoder_m", None) or args.baseline_decoder_m
        if head_type == "refine":
            return build_multihead_refine_hnerv_with_target_decoder_params(
                target_m=target_m,
                num_heads=args.num_experts,
                emb_dim=args.emb_dim,
                c1=args.c1,
                enc_strides=args.enc_strides,
                dec_strides=args.dec_strides,
                reduce=args.reduce,
                kmin=args.kmin,
                kmax=args.kmax,
                blocks_per_stage=args.blocks_per_stage,
                head_kernel_size=args.multihead_head_kernel,
                refine_hidden_ch=args.refine_hidden_ch,
                refine_kernel_size=args.refine_kernel_size,
            )
        return build_multihead_hnerv_with_target_decoder_params(
            target_m=target_m,
            num_heads=args.num_experts,
            emb_dim=args.emb_dim,
            c1=args.c1,
            enc_strides=args.enc_strides,
            dec_strides=args.dec_strides,
            reduce=args.reduce,
            kmin=args.kmin,
            kmax=args.kmax,
            blocks_per_stage=args.blocks_per_stage,
            head_kernel_size=args.multihead_head_kernel,
        )
    if head_type == "refine":
        return build_multihead_refine_hnerv_with_width(
            dec_width=baseline_width,
            num_heads=args.num_experts,
            emb_dim=args.emb_dim,
            c1=args.c1,
            enc_strides=args.enc_strides,
            dec_strides=args.dec_strides,
            reduce=args.reduce,
            kmin=args.kmin,
            kmax=args.kmax,
            blocks_per_stage=args.blocks_per_stage,
            head_kernel_size=args.multihead_head_kernel,
            refine_hidden_ch=args.refine_hidden_ch,
            refine_kernel_size=args.refine_kernel_size,
        )
    return build_multihead_hnerv_with_width(
        dec_width=baseline_width,
        num_heads=args.num_experts,
        emb_dim=args.emb_dim,
        c1=args.c1,
        enc_strides=args.enc_strides,
        dec_strides=args.dec_strides,
        reduce=args.reduce,
        kmin=args.kmin,
        kmax=args.kmax,
        blocks_per_stage=args.blocks_per_stage,
        head_kernel_size=args.multihead_head_kernel,
    )

def _try_initialize_multihead_from_baseline(multihead_model, args, device) -> bool:
    """Copy encoder + shared decoder trunk + RGB head weights from baseline checkpoint.

    This is intentionally used for the simple multi-head diagnostic. If the width differs
    because --multihead_budget_neutral changed dec_width, shapes may not match; in that
    case the function returns False and training starts randomly.
    """
    ckpt_path = Path(args.out_dir) / "baseline_last.pt"
    if not ckpt_path.exists():
        return False
    try:
        baseline = _build_baseline_arch(args).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        baseline.load_state_dict(ckpt["model"])
        multihead_model.encoder.load_state_dict(baseline.encoder.state_dict())
        multihead_model.decoder.embed_to_dec.load_state_dict(baseline.decoder.embed_to_dec.state_dict())
        if len(multihead_model.decoder.blocks) != len(baseline.decoder.blocks):
            return False
        for mh_block, base_block in zip(multihead_model.decoder.blocks, baseline.decoder.blocks):
            mh_block.load_state_dict(base_block.state_dict())
        for head in multihead_model.decoder.heads:
            if hasattr(head, "load_rgb_from_baseline"):
                head.load_rgb_from_baseline(baseline.decoder.to_rgb)
            else:
                head.load_state_dict(baseline.decoder.to_rgb.state_dict())
        return True
    except Exception as exc:
        print(f"[multihead] Baseline initialization skipped due to mismatch/error: {exc}")
        return False


def train_multihead(args) -> Dict[str, Any]:
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = pick_device(args.device)
    dataset = FrameFolderDataset(args.data_path, img_size=tuple(args.img_size), preprocess=args.preprocess)
    clusters = torch.load(out_dir / "clusters.pt", map_location="cpu")
    labels = clusters["labels"].long()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    model = _build_multihead_arch(args).to(device)
    initialized = False
    if getattr(args, "multihead_init_from_baseline", True):
        initialized = _try_initialize_multihead_from_baseline(model, args, device)
        print(f"[multihead] initialized_from_baseline={initialized}")

    if getattr(args, "train_heads_only", False):
        for p in model.parameters():
            p.requires_grad = False
        for head in model.decoder.heads:
            for p in head.parameters():
                p.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[multihead] train_heads_only=True, trainable_params={trainable}, total_params={total}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters selected. Check --train_heads_only / freezing options.")
    opt = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    epochs = args.epochs_multihead if args.epochs_multihead is not None else args.epochs_experts
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    loss_fn = nn.MSELoss()

    best_psnr = -1.0
    best_path = out_dir / "multihead_best.pt"
    last_path = out_dir / "multihead_last.pt"
    t0 = time.time()

    # Save the initialized baseline-equivalent model if it is already best.
    metrics = evaluate_multihead_model(model, dataset, labels, args, device)
    best_psnr = metrics["psnr"]
    torch.save({
        "model": model.state_dict(),
        "config": model_config_dict(model),
        "args": vars(args),
        "metrics": metrics,
        "initialized_from_baseline": initialized,
    }, best_path)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(loader, desc=f"multihead epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            batch = move_batch(batch, device)
            x = batch["image"]
            idx = batch["index"].long().cpu()
            head_ids = labels[idx].to(device)
            pred = model(x, head_ids=head_ids)
            loss = loss_fn(pred, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=np.mean(losses))
        sched.step()
        if epoch % args.eval_every == 0 or epoch == epochs:
            metrics = evaluate_multihead_model(model, dataset, labels, args, device)
            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]
                torch.save({
                    "model": model.state_dict(),
                    "config": model_config_dict(model),
                    "args": vars(args),
                    "metrics": metrics,
                    "initialized_from_baseline": initialized,
                }, best_path)

    torch.save({
        "model": model.state_dict(),
        "config": model_config_dict(model),
        "args": vars(args),
        "initialized_from_baseline": initialized,
    }, last_path)
    best_metrics = evaluate_saved_multihead(args)
    best_metrics.update({
        "train_seconds": time.time() - t0,
        "checkpoint": str(last_path),
        "best_checkpoint": str(best_path),
        "initialized_from_baseline": bool(initialized),
    })
    save_json(best_metrics, out_dir / "multihead_metrics.json")
    return best_metrics


@torch.no_grad()
def evaluate_multihead_model(model, dataset, labels: torch.Tensor, args, device) -> Dict[str, Any]:
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model.eval().to(device)
    psnrs, ssims = [], []
    for batch in tqdm(loader, desc="eval multihead", leave=False):
        batch = move_batch(batch, device)
        x = batch["image"]
        idx = batch["index"].long().cpu()
        head_ids = labels[idx].to(device)
        pred = model(x, head_ids=head_ids)
        psnrs.append(batch_psnr(pred, x))
        ssims.append(batch_ssim(pred, x))

    T = len(dataset)
    H, W = args.img_size
    emb_shape = infer_embedding_shape((H, W), args.enc_strides, args.emb_dim)
    embedding_values = T * int(np.prod(emb_shape))
    cluster_bits = T * math.ceil(math.log2(args.num_experts)) if args.num_experts > 1 else 0
    dec_params = decoder_num_parameters(model)
    bpp_stats = compression_bpp(dec_params, embedding_values, T, H, W, args.model_bits, args.embed_bits, cluster_bits, args.prune_ratio)
    active_decoder = _ActiveHeadDecoderWrapper(model.decoder, head_id=0).to(device)
    active_flops = count_model_flops(active_decoder, (1, *emb_shape), device=str(device), flop_mult=args.flop_mult)

    counts = {int(k): int((labels == k).sum().item()) for k in range(args.num_experts)}
    return {
        "model_type": "multihead_refine_hnerv" if getattr(args, "multihead_head_type", "rgb") == "refine" else "multihead_hnerv",
        "sequence": Path(args.data_path).name,
        "frames": T,
        "height": H,
        "width": W,
        "num_heads": args.num_experts,
        "cluster_method": args.cluster_method,
        "lowrank_factorize_from": getattr(args, "lowrank_factorize_from", None),
        "cluster_counts": counts,
        "multihead_budget_neutral": bool(getattr(args, "multihead_budget_neutral", False)),
        "head_kernel_size": int(args.multihead_head_kernel),
        "multihead_head_type": getattr(args, "multihead_head_type", "rgb"),
        "refine_hidden_ch": int(getattr(args, "refine_hidden_ch", 0)) if getattr(args, "multihead_head_type", "rgb") == "refine" else 0,
        "refine_kernel_size": int(getattr(args, "refine_kernel_size", 0)) if getattr(args, "multihead_head_type", "rgb") == "refine" else 0,
        "train_heads_only": bool(getattr(args, "train_heads_only", False)),
        "trainable_params_selected": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "decoder_params": int(dec_params),
        "encoder_params_not_stored": int(encoder_num_parameters(model)),
        "total_trainable_params": int(total_num_parameters(model)),
        "embedding_values": int(embedding_values),
        "cluster_bits": int(cluster_bits),
        "gflops_per_frame_decoder_only": float(active_flops / 1e9),
        **bpp_stats,
    }


@torch.no_grad()
def evaluate_saved_multihead(args) -> Dict[str, Any]:
    device = pick_device(args.device)
    dataset = FrameFolderDataset(args.data_path, img_size=tuple(args.img_size), preprocess=args.preprocess)
    clusters = torch.load(Path(args.out_dir) / "clusters.pt", map_location="cpu")
    labels = clusters["labels"].long()
    model = _build_multihead_arch(args).to(device)
    ckpt_path = Path(args.out_dir) / "multihead_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return evaluate_multihead_model(model, dataset, labels, args, device)


def run_single_sequence_multihead(args) -> Dict[str, Any]:
    ensure_dir(args.out_dir)
    train_one_baseline(args)
    extract_embeddings(args)
    cluster_embeddings(args)
    multihead = train_multihead(args)
    baseline = evaluate_saved_baseline(args)
    rows = [baseline, multihead]
    pd.DataFrame(rows).to_csv(Path(args.out_dir) / "summary.csv", index=False)
    model_key = multihead.get("model_type", "multihead_hnerv")
    save_json({"baseline": baseline, model_key: multihead}, Path(args.out_dir) / "summary.json")
    return {"baseline": baseline, model_key: multihead}


def run_single_sequence(args) -> Dict[str, Any]:
    ensure_dir(args.out_dir)
    train_one_baseline(args)
    extract_embeddings(args)
    cluster_embeddings(args)
    train_experts(args)
    baseline = evaluate_saved_baseline(args)
    experts = evaluate_experts(args)
    rows = [baseline, experts]
    pd.DataFrame(rows).to_csv(Path(args.out_dir) / "summary.csv", index=False)
    save_json({"baseline": baseline, "clustered_experts": experts}, Path(args.out_dir) / "summary.json")
    return {"baseline": baseline, "clustered_experts": experts}


@torch.no_grad()
def evaluate_saved_baseline(args) -> Dict[str, Any]:
    device = pick_device(args.device)
    dataset = FrameFolderDataset(args.data_path, img_size=tuple(args.img_size), preprocess=args.preprocess)
    model = build_hnerv_with_target_decoder_params(
        target_m=args.baseline_decoder_m,
        emb_dim=args.emb_dim,
        c1=args.c1,
        enc_strides=args.enc_strides,
        dec_strides=args.dec_strides,
        reduce=args.reduce,
        kmin=args.kmin,
        kmax=args.kmax,
        blocks_per_stage=args.blocks_per_stage,
    ).to(device)
    ckpt = torch.load(Path(args.out_dir) / "baseline_last.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    return evaluate_baseline_model(model, dataset, args, device)
