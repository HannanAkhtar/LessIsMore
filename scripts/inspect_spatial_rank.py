"""Read factorization ranks from a trusted spatial checkpoint or quant_vid file."""
from __future__ import annotations

import argparse
import json

try:
    import torch
except ModuleNotFoundError:
    torch = None


def inspect(path):
    if torch is None:
        raise RuntimeError("PyTorch is required to inspect spatial checkpoint tensors")
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload.get("state_dict", payload))
    ranks = {}
    for stage in (3, 4):
        suffix = f"decoder.{stage + 1}.conv.upconv.0.weight"
        matches = [(key, value) for key, value in state.items() if key.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"Expected one factorized weight ending in {suffix}, found {len(matches)}")
        value = matches[0][1]
        symbols = value["quant"] if isinstance(value, dict) else value
        ranks[f"stage_{stage}"] = int(symbols.shape[0])
    return ranks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Trusted model_latest.pth or quant_vid.pth")
    args = parser.parse_args()
    print(json.dumps({"checkpoint": args.checkpoint, "ranks": inspect(args.checkpoint)}, indent=2))


if __name__ == "__main__":
    main()
