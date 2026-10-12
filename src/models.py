from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """LayerNorm over channel dimension for NCHW tensors."""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style residual block used in the HNeRV encoder."""
    def __init__(self, dim: int, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + x


class HNeRVEncoder(nn.Module):
    """Progressive downsampling ConvNeXt encoder.

    For 640x1280 with strides [5,4,4,2,2], c1=64, emb_dim=16,
    output is [B,16,2,4].
    """
    def __init__(self, in_ch: int = 3, c1: int = 64, emb_dim: int = 16, strides: List[int] | Tuple[int, ...] = (5,4,4,2,2), blocks_per_stage: int = 1):
        super().__init__()
        strides = list(strides)
        dims = [c1] * (len(strides) - 1) + [emb_dim]
        stages = []
        prev = in_ch
        for i, (s, dim) in enumerate(zip(strides, dims)):
            # HNeRV-style downsampling. Kernel=stride keeps exact integer dimensions for the recommended sizes.
            stages.append(nn.Sequential(
                nn.Conv2d(prev, dim, kernel_size=s, stride=s, padding=0),
                LayerNorm2d(dim),
                *[ConvNeXtBlock(dim) for _ in range(blocks_per_stage)]
            ))
            prev = dim
        self.stages = nn.ModuleList(stages)
        self.out_channels = emb_dim
        self.strides = strides

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return x


class HNeRVBlock(nn.Module):
    """HNeRV decoder block: convolution -> PixelShuffle -> activation."""
    def __init__(self, cin: int, cout: int, stride: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(cin, cout * stride * stride, kernel_size=kernel_size, stride=1, padding=padding)
        self.shuffle = nn.PixelShuffle(stride)
        self.act = nn.GELU()
        self.cin = cin
        self.cout = cout
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.shuffle(self.conv(x)))


class LowRankHNeRVBlock(nn.Module):
    """Low-rank HNeRV decoder block: Conv(KxK) -> Conv(1x1) -> PixelShuffle -> activation.

    Dense HNeRV block:
        Conv2d(Cin, Cout * stride^2, KxK) -> PixelShuffle(stride)

    Low-rank block:
        Conv2d(Cin, rank, KxK) -> Conv2d(rank, Cout * stride^2, 1x1) -> PixelShuffle(stride)

    The intermediate rank is allowed to be larger than Cin. This is useful for
    HNeRV PixelShuffle blocks because the dense output channel count
    Cout * stride^2 can be very large.
    """
    def __init__(self, cin: int, cout: int, stride: int, kernel_size: int, rank: int):
        super().__init__()
        padding = kernel_size // 2
        rank = int(max(1, rank))
        self.conv_k = nn.Conv2d(cin, rank, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv_1 = nn.Conv2d(rank, cout * stride * stride, kernel_size=1, stride=1, padding=0)
        self.shuffle = nn.PixelShuffle(stride)
        self.act = nn.GELU()
        self.cin = int(cin)
        self.cout = int(cout)
        self.stride = int(stride)
        self.kernel_size = int(kernel_size)
        self.rank = int(rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_k(x)
        x = self.conv_1(x)
        return self.act(self.shuffle(x))


class LowRankHNeRVDecoder(nn.Module):
    """HNeRV decoder with selected PixelShuffle convolutions replaced by low-rank blocks.

    The decoder keeps a HNeRV channel schedule but can replace selected dense
    convolutional PixelShuffle blocks with a two-stage low-rank form.

    By default, blocks with index >= ``lowrank_factorize_from`` are factorized.
    For ablations, pass ``lowrank_blocks`` to factorize an explicit set of
    decoder block indices, e.g. [3, 4] or [2, 3, 4].
    """
    def __init__(self, emb_dim: int = 16, dec_width: int = 97,
                 dec_strides: List[int] | Tuple[int, ...] = (5,4,4,2,2),
                 reduce: float = 1.2, kmin: int = 1, kmax: int = 5,
                 lowrank_rank: int = 88, lowrank_factorize_from: int = 2,
                 lowrank_blocks: Optional[List[int]] = None):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.dec_width = int(dec_width)
        self.dec_strides = list(dec_strides)
        self.reduce = float(reduce)
        self.kmin = int(kmin)
        self.kmax = int(kmax)
        self.lowrank_rank = int(lowrank_rank)
        self.lowrank_factorize_from = int(lowrank_factorize_from)
        self.lowrank_blocks = None if lowrank_blocks is None else [int(i) for i in lowrank_blocks]
        self.embed_to_dec = nn.Conv2d(emb_dim, int(dec_width), kernel_size=1)
        blocks = []
        cin = int(dec_width)
        for i, s in enumerate(self.dec_strides):
            if i == 0:
                k = self.kmin
            elif i == 1:
                k = 3 if self.kmax >= 3 else self.kmax
            else:
                k = self.kmax
            cout = max(4, int(round(dec_width / (self.reduce ** (i + 1)))))
            if self.lowrank_blocks is None:
                use_lowrank = i >= self.lowrank_factorize_from
            else:
                use_lowrank = i in self.lowrank_blocks
            if use_lowrank and k > 1:
                blocks.append(LowRankHNeRVBlock(cin, cout, stride=int(s), kernel_size=int(k), rank=self.lowrank_rank))
            else:
                blocks.append(HNeRVBlock(cin, cout, stride=int(s), kernel_size=int(k)))
            cin = cout
        self.blocks = nn.ModuleList(blocks)
        self.to_rgb = nn.Conv2d(cin, 3, kernel_size=3, padding=1)

    def forward_features(self, z: torch.Tensor) -> torch.Tensor:
        x = self.embed_to_dec(z)
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(z)
        return torch.sigmoid(self.to_rgb(x))


class HNeRVDecoder(nn.Module):
    def __init__(self, emb_dim: int = 16, dec_width: int = 97, dec_strides: List[int] | Tuple[int, ...] = (5,4,4,2,2), reduce: float = 1.2, kmin: int = 1, kmax: int = 5):
        super().__init__()
        self.emb_dim = emb_dim
        self.dec_width = int(dec_width)
        self.dec_strides = list(dec_strides)
        self.reduce = float(reduce)
        self.kmin = int(kmin)
        self.kmax = int(kmax)
        self.embed_to_dec = nn.Conv2d(emb_dim, int(dec_width), kernel_size=1)
        blocks = []
        cin = int(dec_width)
        for i, s in enumerate(self.dec_strides):
            if i == 0:
                k = self.kmin
            elif i == 1:
                k = 3 if self.kmax >= 3 else self.kmax
            else:
                k = self.kmax
            cout = max(4, int(round(dec_width / (self.reduce ** (i + 1)))))
            blocks.append(HNeRVBlock(cin, cout, stride=int(s), kernel_size=int(k)))
            cin = cout
        self.blocks = nn.ModuleList(blocks)
        self.to_rgb = nn.Conv2d(cin, 3, kernel_size=3, padding=1)

    def forward_features(self, z: torch.Tensor) -> torch.Tensor:
        x = self.embed_to_dec(z)
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(z)
        return torch.sigmoid(self.to_rgb(x))


class RefinementRGBHead(nn.Module):
    """Cluster-specific residual refinement block followed by RGB projection.

    The residual branch is initialized to zero, so when the RGB projection is
    copied from the baseline HNeRV head the whole multi-head/refinement model is
    exactly baseline-equivalent before fine-tuning.
    """
    def __init__(self, channels: int, hidden_channels: int = 8, refine_kernel_size: int = 3, rgb_kernel_size: int = 3):
        super().__init__()
        hidden_channels = int(max(1, hidden_channels))
        refine_kernel_size = int(refine_kernel_size)
        rgb_kernel_size = int(rgb_kernel_size)
        self.channels = int(channels)
        self.hidden_channels = hidden_channels
        self.refine_kernel_size = refine_kernel_size
        self.rgb_kernel_size = rgb_kernel_size
        self.refine = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=refine_kernel_size, padding=refine_kernel_size // 2),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, padding=0),
        )
        # Zero-init the final refinement projection so the residual branch starts
        # as an exact identity: y = x + 0.
        nn.init.zeros_(self.refine[-1].weight)
        if self.refine[-1].bias is not None:
            nn.init.zeros_(self.refine[-1].bias)
        self.to_rgb = nn.Conv2d(channels, 3, kernel_size=rgb_kernel_size, padding=rgb_kernel_size // 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.refine(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.to_rgb(x)

    def load_rgb_from_baseline(self, baseline_to_rgb: nn.Conv2d) -> None:
        self.to_rgb.load_state_dict(baseline_to_rgb.state_dict())


class MultiHeadHNeRVDecoder(nn.Module):
    """HNeRV decoder with one shared trunk and K small RGB heads.

    The shared trunk is identical to HNeRVDecoder up to the last feature map.
    During decoding, each frame uses exactly one selected head. This preserves
    shared decoder capacity while allowing content/cluster-specific RGB output
    mappings.
    """
    def __init__(self, emb_dim: int = 16, dec_width: int = 97,
                 dec_strides: List[int] | Tuple[int, ...] = (5,4,4,2,2),
                 reduce: float = 1.2, kmin: int = 1, kmax: int = 5,
                 num_heads: int = 4, head_kernel_size: int = 3):
        super().__init__()
        self.emb_dim = emb_dim
        self.dec_width = int(dec_width)
        self.dec_strides = list(dec_strides)
        self.reduce = float(reduce)
        self.kmin = int(kmin)
        self.kmax = int(kmax)
        self.num_heads = int(num_heads)
        self.head_kernel_size = int(head_kernel_size)

        self.embed_to_dec = nn.Conv2d(emb_dim, int(dec_width), kernel_size=1)
        blocks = []
        cin = int(dec_width)
        for i, s in enumerate(self.dec_strides):
            if i == 0:
                k = self.kmin
            elif i == 1:
                k = 3 if self.kmax >= 3 else self.kmax
            else:
                k = self.kmax
            cout = max(4, int(round(dec_width / (self.reduce ** (i + 1)))))
            blocks.append(HNeRVBlock(cin, cout, stride=int(s), kernel_size=int(k)))
            cin = cout
        self.blocks = nn.ModuleList(blocks)
        pad = self.head_kernel_size // 2
        self.heads = nn.ModuleList([
            nn.Conv2d(cin, 3, kernel_size=self.head_kernel_size, padding=pad)
            for _ in range(self.num_heads)
        ])
        self.final_feature_channels = cin

    def forward_features(self, z: torch.Tensor) -> torch.Tensor:
        x = self.embed_to_dec(z)
        for block in self.blocks:
            x = block(x)
        return x

    def forward_single_head(self, z: torch.Tensor, head_id: int = 0) -> torch.Tensor:
        x = self.forward_features(z)
        head_id = int(head_id)
        return torch.sigmoid(self.heads[head_id](x))

    def forward(self, z: torch.Tensor, head_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.forward_features(z)
        if head_ids is None:
            return torch.sigmoid(self.heads[0](x))
        head_ids = head_ids.to(device=x.device, dtype=torch.long).view(-1)
        if head_ids.numel() != x.shape[0]:
            raise ValueError(f"head_ids must have batch length {x.shape[0]}, got {head_ids.numel()}")
        out = torch.empty((x.shape[0], 3, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
        for k, head in enumerate(self.heads):
            mask = head_ids == k
            if mask.any():
                out[mask] = torch.sigmoid(head(x[mask]))
        return out


class MultiHeadHNeRV(nn.Module):
    def __init__(self, in_ch: int = 3, c1: int = 64, emb_dim: int = 16,
                 enc_strides: List[int] | Tuple[int, ...] = (5,4,4,2,2),
                 dec_strides: List[int] | Tuple[int, ...] = (5,4,4,2,2),
                 dec_width: int = 97, reduce: float = 1.2, kmin: int = 1, kmax: int = 5,
                 blocks_per_stage: int = 1, num_heads: int = 4, head_kernel_size: int = 3):
        super().__init__()
        self.encoder = HNeRVEncoder(in_ch=in_ch, c1=c1, emb_dim=emb_dim, strides=enc_strides, blocks_per_stage=blocks_per_stage)
        self.decoder = MultiHeadHNeRVDecoder(
            emb_dim=emb_dim, dec_width=dec_width, dec_strides=dec_strides,
            reduce=reduce, kmin=kmin, kmax=kmax,
            num_heads=num_heads, head_kernel_size=head_kernel_size,
        )

    def forward(self, x: torch.Tensor, head_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.decoder(self.encoder(x), head_ids=head_ids)


class MultiHeadRefineHNeRVDecoder(nn.Module):
    """Shared HNeRV trunk with cluster-specific final refinement blocks.

    Each frame passes through the same shared decoder trunk and then exactly one
    selected refinement head. A refinement head is a small residual bottleneck
    block operating on the final full-resolution feature map, followed by an RGB
    projection. This is stronger than RGB-only heads while preserving almost all
    of the shared HNeRV decoder.
    """
    def __init__(self, emb_dim: int = 16, dec_width: int = 97,
                 dec_strides: List[int] | Tuple[int, ...] = (5,4,4,2,2),
                 reduce: float = 1.2, kmin: int = 1, kmax: int = 5,
                 num_heads: int = 4, head_kernel_size: int = 3,
                 refine_hidden_ch: int = 8, refine_kernel_size: int = 3):
        super().__init__()
        self.emb_dim = emb_dim
        self.dec_width = int(dec_width)
        self.dec_strides = list(dec_strides)
        self.reduce = float(reduce)
        self.kmin = int(kmin)
        self.kmax = int(kmax)
        self.num_heads = int(num_heads)
        self.head_kernel_size = int(head_kernel_size)
        self.refine_hidden_ch = int(refine_hidden_ch)
        self.refine_kernel_size = int(refine_kernel_size)

        self.embed_to_dec = nn.Conv2d(emb_dim, int(dec_width), kernel_size=1)
        blocks = []
        cin = int(dec_width)
        for i, s in enumerate(self.dec_strides):
            if i == 0:
                k = self.kmin
            elif i == 1:
                k = 3 if self.kmax >= 3 else self.kmax
            else:
                k = self.kmax
            cout = max(4, int(round(dec_width / (self.reduce ** (i + 1)))))
            blocks.append(HNeRVBlock(cin, cout, stride=int(s), kernel_size=int(k)))
            cin = cout
        self.blocks = nn.ModuleList(blocks)
        self.final_feature_channels = cin
        self.heads = nn.ModuleList([
            RefinementRGBHead(
                channels=cin,
                hidden_channels=self.refine_hidden_ch,
                refine_kernel_size=self.refine_kernel_size,
                rgb_kernel_size=self.head_kernel_size,
            )
            for _ in range(self.num_heads)
        ])

    def forward_features(self, z: torch.Tensor) -> torch.Tensor:
        x = self.embed_to_dec(z)
        for block in self.blocks:
            x = block(x)
        return x

    def forward_single_head(self, z: torch.Tensor, head_id: int = 0) -> torch.Tensor:
        x = self.forward_features(z)
        return torch.sigmoid(self.heads[int(head_id)](x))

    def forward(self, z: torch.Tensor, head_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.forward_features(z)
        if head_ids is None:
            return torch.sigmoid(self.heads[0](x))
        head_ids = head_ids.to(device=x.device, dtype=torch.long).view(-1)
        if head_ids.numel() != x.shape[0]:
            raise ValueError(f"head_ids must have batch length {x.shape[0]}, got {head_ids.numel()}")
        out = torch.empty((x.shape[0], 3, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
        for k, head in enumerate(self.heads):
            mask = head_ids == k
            if mask.any():
                out[mask] = torch.sigmoid(head(x[mask]))
        return out


class MultiHeadRefineHNeRV(nn.Module):
    def __init__(self, in_ch: int = 3, c1: int = 64, emb_dim: int = 16,
                 enc_strides: List[int] | Tuple[int, ...] = (5,4,4,2,2),
                 dec_strides: List[int] | Tuple[int, ...] = (5,4,4,2,2),
                 dec_width: int = 97, reduce: float = 1.2, kmin: int = 1, kmax: int = 5,
                 blocks_per_stage: int = 1, num_heads: int = 4, head_kernel_size: int = 3,
                 refine_hidden_ch: int = 8, refine_kernel_size: int = 3):
        super().__init__()
        self.encoder = HNeRVEncoder(in_ch=in_ch, c1=c1, emb_dim=emb_dim, strides=enc_strides, blocks_per_stage=blocks_per_stage)
        self.decoder = MultiHeadRefineHNeRVDecoder(
            emb_dim=emb_dim, dec_width=dec_width, dec_strides=dec_strides,
            reduce=reduce, kmin=kmin, kmax=kmax,
            num_heads=num_heads, head_kernel_size=head_kernel_size,
            refine_hidden_ch=refine_hidden_ch, refine_kernel_size=refine_kernel_size,
        )

    def forward(self, x: torch.Tensor, head_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.decoder(self.encoder(x), head_ids=head_ids)


class HNeRV(nn.Module):
    def __init__(self, in_ch: int = 3, c1: int = 64, emb_dim: int = 16,
                 enc_strides: List[int] | Tuple[int, ...] = (5,4,4,2,2),
                 dec_strides: List[int] | Tuple[int, ...] = (5,4,4,2,2),
                 dec_width: int = 97, reduce: float = 1.2, kmin: int = 1, kmax: int = 5,
                 blocks_per_stage: int = 1):
        super().__init__()
        self.encoder = HNeRVEncoder(in_ch=in_ch, c1=c1, emb_dim=emb_dim, strides=enc_strides, blocks_per_stage=blocks_per_stage)
        self.decoder = HNeRVDecoder(emb_dim=emb_dim, dec_width=dec_width, dec_strides=dec_strides, reduce=reduce, kmin=kmin, kmax=kmax)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class LowRankHNeRV(nn.Module):
    """Full HNeRV model with a low-rank decoder.

    This is used for the proposed ablations:
      1) same-width low-rank decoder, and
      2) widened low-rank decoder under the same parameter budget as dense HNeRV.
    """
    def __init__(self, in_ch: int = 3, c1: int = 64, emb_dim: int = 16,
                 enc_strides: List[int] | Tuple[int, ...] = (5,4,4,2,2),
                 dec_strides: List[int] | Tuple[int, ...] = (5,4,4,2,2),
                 dec_width: int = 97, reduce: float = 1.2, kmin: int = 1, kmax: int = 5,
                 blocks_per_stage: int = 1, lowrank_rank: int = 88,
                 lowrank_factorize_from: int = 2, lowrank_blocks: Optional[List[int]] = None):
        super().__init__()
        self.encoder = HNeRVEncoder(in_ch=in_ch, c1=c1, emb_dim=emb_dim, strides=enc_strides, blocks_per_stage=blocks_per_stage)
        self.decoder = LowRankHNeRVDecoder(
            emb_dim=emb_dim, dec_width=dec_width, dec_strides=dec_strides,
            reduce=reduce, kmin=kmin, kmax=kmax,
            lowrank_rank=lowrank_rank,
            lowrank_factorize_from=lowrank_factorize_from,
            lowrank_blocks=lowrank_blocks,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def decoder_num_parameters(model: nn.Module) -> int:
    if hasattr(model, "decoder"):
        return sum(p.numel() for p in model.decoder.parameters())
    return sum(p.numel() for p in model.parameters())


def encoder_num_parameters(model: nn.Module) -> int:
    if not hasattr(model, "encoder"):
        return 0
    return sum(p.numel() for p in model.encoder.parameters())


def total_num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def make_kernel_schedule(num_blocks: int, kmin: int, kmax: int) -> List[int]:
    ks = []
    for i in range(num_blocks):
        if i == 0:
            ks.append(kmin)
        elif i == 1:
            ks.append(3 if kmax >= 3 else kmax)
        else:
            ks.append(kmax)
    return ks


def build_hnerv_with_target_decoder_params(target_m: float, emb_dim: int, c1: int,
                                           enc_strides: List[int], dec_strides: List[int],
                                           reduce: float = 1.2, kmin: int = 1, kmax: int = 5,
                                           blocks_per_stage: int = 1,
                                           tolerance: float = 0.08) -> HNeRV:
    """Build HNeRV by searching dec_width so decoder params are near target_m million."""
    target = int(target_m * 1_000_000)
    lo, hi = 4, 2048
    best_model = None
    best_err = float("inf")
    for _ in range(20):
        mid = (lo + hi) // 2
        model = HNeRV(c1=c1, emb_dim=emb_dim, enc_strides=enc_strides, dec_strides=dec_strides,
                      dec_width=mid, reduce=reduce, kmin=kmin, kmax=kmax, blocks_per_stage=blocks_per_stage)
        n = decoder_num_parameters(model)
        err = abs(n - target)
        if err < best_err:
            best_err = err
            best_model = model
        if n < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return best_model


def build_decoder_with_target_params(target_m: float, emb_dim: int, dec_strides: List[int],
                                     reduce: float = 1.2, kmin: int = 1, kmax: int = 5) -> HNeRVDecoder:
    target = int(target_m * 1_000_000)
    lo, hi = 4, 2048
    best = None
    best_err = float("inf")
    for _ in range(20):
        mid = (lo + hi) // 2
        dec = HNeRVDecoder(emb_dim=emb_dim, dec_width=mid, dec_strides=dec_strides, reduce=reduce, kmin=kmin, kmax=kmax)
        n = decoder_num_parameters(dec)
        err = abs(n - target)
        if err < best_err:
            best_err = err
            best = dec
        if n < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return best



def build_lowrank_decoder_with_target_params(target_m: float, emb_dim: int, dec_strides: List[int],
                                            reduce: float = 1.2, kmin: int = 1, kmax: int = 5,
                                            base_dec_width: int = 97,
                                            lowrank_factorize_from: int | str = "auto",
                                            rank_search_max: int = 2048,
                                            lowrank_blocks: Optional[List[int]] = None) -> LowRankHNeRVDecoder:
    """Build a low-rank HNeRV decoder close to target_m while preserving base width.

    Unlike build_decoder_with_target_params, this does not shrink dec_width to
    hit the parameter budget. It keeps the dense baseline width and searches the
    intermediate low-rank channel count. This is useful for expert decoders where
    preserving feature width may be more important than using a narrow decoder.

    If lowrank_factorize_from="auto":
      - target >= 1.2M: factorize blocks 2..end, preserving the early 3x3 block.
      - target <  1.2M: factorize blocks 1..end to reach smaller budgets.
    """
    target = int(target_m * 1_000_000)
    if lowrank_factorize_from == "auto":
        factorize_from = 2 if target_m >= 1.2 else 1
    else:
        factorize_from = int(lowrank_factorize_from)

    best = None
    best_err = float("inf")
    # Search integer rank directly. Ranks larger than Cin are allowed because
    # the factorized PixelShuffle output channel count can be much larger.
    for rank in range(1, int(rank_search_max) + 1):
        dec = LowRankHNeRVDecoder(
            emb_dim=emb_dim,
            dec_width=int(base_dec_width),
            dec_strides=dec_strides,
            reduce=reduce,
            kmin=kmin,
            kmax=kmax,
            lowrank_rank=rank,
            lowrank_factorize_from=factorize_from,
            lowrank_blocks=lowrank_blocks,
        )
        n = decoder_num_parameters(dec)
        err = abs(n - target)
        if err < best_err:
            best_err = err
            best = dec
        # After the chosen factorization point, parameter count is monotonic in rank.
        if n > target and rank > 1:
            # Continue a few extra steps is unnecessary for absolute error.
            break
    return best


def build_lowrank_hnerv_with_width(dec_width: int, emb_dim: int, c1: int,
                                   enc_strides: List[int], dec_strides: List[int],
                                   reduce: float = 1.2, kmin: int = 1, kmax: int = 5,
                                   blocks_per_stage: int = 1, lowrank_rank: int = 88,
                                   lowrank_blocks: Optional[List[int]] = None,
                                   lowrank_factorize_from: int = 2) -> LowRankHNeRV:
    return LowRankHNeRV(
        c1=c1, emb_dim=emb_dim, enc_strides=enc_strides, dec_strides=dec_strides,
        dec_width=int(dec_width), reduce=reduce, kmin=kmin, kmax=kmax,
        blocks_per_stage=blocks_per_stage, lowrank_rank=int(lowrank_rank),
        lowrank_factorize_from=int(lowrank_factorize_from), lowrank_blocks=lowrank_blocks,
    )


def build_lowrank_hnerv_with_target_decoder_params(target_m: float, emb_dim: int, c1: int,
                                                   enc_strides: List[int], dec_strides: List[int],
                                                   reduce: float = 1.2, kmin: int = 1, kmax: int = 5,
                                                   blocks_per_stage: int = 1, lowrank_rank: int = 88,
                                                   lowrank_blocks: Optional[List[int]] = None,
                                                   lowrank_factorize_from: int = 2,
                                                   max_width: int = 4096) -> LowRankHNeRV:
    """Search decoder width so a low-rank HNeRV decoder is near target_m params.

    This is for the widened-low-rank ablation: factorize a chosen set of
    decoder blocks and then spend the saved parameters by increasing width until
    total decoder parameters are close to the dense HNeRV baseline budget.
    """
    target = int(target_m * 1_000_000)
    lo, hi = 4, int(max_width)
    best_model = None
    best_err = float("inf")
    for _ in range(24):
        mid = (lo + hi) // 2
        model = build_lowrank_hnerv_with_width(
            dec_width=mid, emb_dim=emb_dim, c1=c1,
            enc_strides=enc_strides, dec_strides=dec_strides,
            reduce=reduce, kmin=kmin, kmax=kmax,
            blocks_per_stage=blocks_per_stage, lowrank_rank=lowrank_rank,
            lowrank_blocks=lowrank_blocks, lowrank_factorize_from=lowrank_factorize_from,
        )
        n = decoder_num_parameters(model)
        err = abs(n - target)
        if err < best_err:
            best_err = err
            best_model = model
        if n < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return best_model


def build_multihead_hnerv_with_width(dec_width: int, num_heads: int, emb_dim: int, c1: int,
                                     enc_strides: List[int], dec_strides: List[int],
                                     reduce: float = 1.2, kmin: int = 1, kmax: int = 5,
                                     blocks_per_stage: int = 1,
                                     head_kernel_size: int = 3) -> MultiHeadHNeRV:
    return MultiHeadHNeRV(
        c1=c1, emb_dim=emb_dim, enc_strides=enc_strides, dec_strides=dec_strides,
        dec_width=int(dec_width), reduce=reduce, kmin=kmin, kmax=kmax,
        blocks_per_stage=blocks_per_stage, num_heads=num_heads, head_kernel_size=head_kernel_size,
    )


def build_multihead_hnerv_with_target_decoder_params(target_m: float, num_heads: int, emb_dim: int, c1: int,
                                                     enc_strides: List[int], dec_strides: List[int],
                                                     reduce: float = 1.2, kmin: int = 1, kmax: int = 5,
                                                     blocks_per_stage: int = 1,
                                                     head_kernel_size: int = 3) -> MultiHeadHNeRV:
    """Search dec_width so shared trunk + all heads are near target_m million params."""
    target = int(target_m * 1_000_000)
    lo, hi = 4, 2048
    best_model = None
    best_err = float("inf")
    for _ in range(20):
        mid = (lo + hi) // 2
        model = build_multihead_hnerv_with_width(
            dec_width=mid, num_heads=num_heads, emb_dim=emb_dim, c1=c1,
            enc_strides=enc_strides, dec_strides=dec_strides,
            reduce=reduce, kmin=kmin, kmax=kmax,
            blocks_per_stage=blocks_per_stage, head_kernel_size=head_kernel_size,
        )
        n = decoder_num_parameters(model)
        err = abs(n - target)
        if err < best_err:
            best_err = err
            best_model = model
        if n < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return best_model


def build_multihead_refine_hnerv_with_width(dec_width: int, num_heads: int, emb_dim: int, c1: int,
                                            enc_strides: List[int], dec_strides: List[int],
                                            reduce: float = 1.2, kmin: int = 1, kmax: int = 5,
                                            blocks_per_stage: int = 1,
                                            head_kernel_size: int = 3,
                                            refine_hidden_ch: int = 8,
                                            refine_kernel_size: int = 3) -> MultiHeadRefineHNeRV:
    return MultiHeadRefineHNeRV(
        c1=c1, emb_dim=emb_dim, enc_strides=enc_strides, dec_strides=dec_strides,
        dec_width=int(dec_width), reduce=reduce, kmin=kmin, kmax=kmax,
        blocks_per_stage=blocks_per_stage, num_heads=num_heads, head_kernel_size=head_kernel_size,
        refine_hidden_ch=refine_hidden_ch, refine_kernel_size=refine_kernel_size,
    )


def build_multihead_refine_hnerv_with_target_decoder_params(target_m: float, num_heads: int, emb_dim: int, c1: int,
                                                            enc_strides: List[int], dec_strides: List[int],
                                                            reduce: float = 1.2, kmin: int = 1, kmax: int = 5,
                                                            blocks_per_stage: int = 1,
                                                            head_kernel_size: int = 3,
                                                            refine_hidden_ch: int = 8,
                                                            refine_kernel_size: int = 3) -> MultiHeadRefineHNeRV:
    """Search dec_width so shared trunk + all refinement heads are near target_m million params."""
    target = int(target_m * 1_000_000)
    lo, hi = 4, 2048
    best_model = None
    best_err = float("inf")
    for _ in range(20):
        mid = (lo + hi) // 2
        model = build_multihead_refine_hnerv_with_width(
            dec_width=mid, num_heads=num_heads, emb_dim=emb_dim, c1=c1,
            enc_strides=enc_strides, dec_strides=dec_strides,
            reduce=reduce, kmin=kmin, kmax=kmax,
            blocks_per_stage=blocks_per_stage, head_kernel_size=head_kernel_size,
            refine_hidden_ch=refine_hidden_ch, refine_kernel_size=refine_kernel_size,
        )
        n = decoder_num_parameters(model)
        err = abs(n - target)
        if err < best_err:
            best_err = err
            best_model = model
        if n < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return best_model


def infer_embedding_shape(img_size: Tuple[int, int], enc_strides: List[int], emb_dim: int) -> Tuple[int, int, int]:
    H, W = img_size
    scale = 1
    for s in enc_strides:
        scale *= int(s)
    if H % scale != 0 or W % scale != 0:
        raise ValueError(
            f"Image size {img_size} is not divisible by total stride {scale}. "
            f"Use --preprocess center_crop with --img_size 640 1280 for 1280x720 source frames, "
            f"or choose strides whose product divides H and W."
        )
    return emb_dim, H // scale, W // scale


def model_config_dict(model: nn.Module) -> Dict[str, Any]:
    if isinstance(model, MultiHeadRefineHNeRV):
        return {
            "model_type": "multihead_refine_hnerv",
            "c1": model.encoder.stages[0][0].out_channels,
            "emb_dim": model.encoder.out_channels,
            "enc_strides": model.encoder.strides,
            "dec_width": model.decoder.dec_width,
            "dec_strides": model.decoder.dec_strides,
            "reduce": model.decoder.reduce,
            "kmin": model.decoder.kmin,
            "kmax": model.decoder.kmax,
            "num_heads": model.decoder.num_heads,
            "head_kernel_size": model.decoder.head_kernel_size,
            "refine_hidden_ch": model.decoder.refine_hidden_ch,
            "refine_kernel_size": model.decoder.refine_kernel_size,
        }
    if isinstance(model, MultiHeadHNeRV):
        return {
            "model_type": "multihead_hnerv",
            "c1": model.encoder.stages[0][0].out_channels,
            "emb_dim": model.encoder.out_channels,
            "enc_strides": model.encoder.strides,
            "dec_width": model.decoder.dec_width,
            "dec_strides": model.decoder.dec_strides,
            "reduce": model.decoder.reduce,
            "kmin": model.decoder.kmin,
            "kmax": model.decoder.kmax,
            "num_heads": model.decoder.num_heads,
            "head_kernel_size": model.decoder.head_kernel_size,
        }
    if isinstance(model, HNeRV):
        return {
            "c1": model.encoder.stages[0][0].out_channels,
            "emb_dim": model.encoder.out_channels,
            "enc_strides": model.encoder.strides,
            "dec_width": model.decoder.dec_width,
            "dec_strides": model.decoder.dec_strides,
            "reduce": model.decoder.reduce,
            "kmin": model.decoder.kmin,
            "kmax": model.decoder.kmax,
        }
    if isinstance(model, MultiHeadRefineHNeRVDecoder):
        return {
            "model_type": "multihead_refine_hnerv_decoder",
            "emb_dim": model.emb_dim,
            "dec_width": model.dec_width,
            "dec_strides": model.dec_strides,
            "reduce": model.reduce,
            "kmin": model.kmin,
            "kmax": model.kmax,
            "num_heads": model.num_heads,
            "head_kernel_size": model.head_kernel_size,
            "refine_hidden_ch": model.refine_hidden_ch,
            "refine_kernel_size": model.refine_kernel_size,
        }
    if isinstance(model, MultiHeadHNeRVDecoder):
        return {
            "model_type": "multihead_hnerv_decoder",
            "emb_dim": model.emb_dim,
            "dec_width": model.dec_width,
            "dec_strides": model.dec_strides,
            "reduce": model.reduce,
            "kmin": model.kmin,
            "kmax": model.kmax,
            "num_heads": model.num_heads,
            "head_kernel_size": model.head_kernel_size,
        }
    if isinstance(model, LowRankHNeRVDecoder):
        return {
            "model_type": "lowrank_hnerv_decoder",
            "emb_dim": model.emb_dim,
            "dec_width": model.dec_width,
            "dec_strides": model.dec_strides,
            "reduce": model.reduce,
            "kmin": model.kmin,
            "kmax": model.kmax,
            "lowrank_rank": model.lowrank_rank,
            "lowrank_factorize_from": model.lowrank_factorize_from,
        }
    if isinstance(model, HNeRVDecoder):
        return {
            "emb_dim": model.emb_dim,
            "dec_width": model.dec_width,
            "dec_strides": model.dec_strides,
            "reduce": model.reduce,
            "kmin": model.kmin,
            "kmax": model.kmax,
        }
    return {}
