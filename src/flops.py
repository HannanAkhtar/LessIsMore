from __future__ import annotations

from typing import Tuple, Dict, List

import torch
import torch.nn as nn


def count_model_flops(model: nn.Module, input_shape: Tuple[int, ...], device: str = "cpu", flop_mult: int = 1) -> int:
    """Count Conv2d and Linear operations with forward hooks.

    By default, one multiply-accumulate is counted as one operation, which is common in vision papers.
    Set flop_mult=2 if you want multiply and add to count separately.
    """
    model = model.to(device)
    model.eval()
    total = {"ops": 0}
    hooks = []

    def conv_hook(m: nn.Conv2d, inp, out):
        x = inp[0]
        batch = x.shape[0]
        out_h, out_w = out.shape[-2], out.shape[-1]
        out_channels = m.out_channels
        kernel_ops = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        ops = batch * out_h * out_w * out_channels * kernel_ops * flop_mult
        total["ops"] += int(ops)

    def linear_hook(m: nn.Linear, inp, out):
        x = inp[0]
        # all dimensions except last are independent positions
        positions = int(x.numel() // x.shape[-1])
        ops = positions * m.in_features * m.out_features * flop_mult
        total["ops"] += int(ops)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    with torch.no_grad():
        x = torch.zeros(input_shape, device=device)
        _ = model(x)

    for h in hooks:
        h.remove()
    return total["ops"]
