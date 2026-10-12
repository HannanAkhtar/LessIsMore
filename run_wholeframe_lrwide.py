from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate dense baseline and LR-conv results with actual quantized/pruned playback evaluation.")
    parser.add_argument("--data_root", type=str, default=str(Path("..") / "data"))
    parser.add_argument("--runs_root", type=str, default="runs_full_originals_qp_eval_b34_3M")
    parser.add_argument("--only", nargs="*", default=["bunny", "honeybee", "jockey", "readysetgo", "yachtride"])
    parser.add_argument("--epochs_baseline", type=int, default=300)
    parser.add_argument("--epochs_lowrank", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_bits", type=int, default=8)
    parser.add_argument("--embed_bits", type=int, default=8)
    parser.add_argument("--prune_ratio", type=float, default=0.10)
    parser.add_argument("--gpu", type=str, default=None, help="Optional CUDA_VISIBLE_DEVICES value, e.g. 0 or 1.")
    parser.add_argument("--skip_existing_models", action="store_true", help="Reuse existing checkpoints and only recompute Q/P metrics.")
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "src.run_lowrank_ablation_grid",
        "--data_root", args.data_root,
        "--runs_root", args.runs_root,
        "--img_size", "640", "1280",
        "--preprocess", "center_crop",
        "--epochs_baseline", str(args.epochs_baseline),
        "--epochs_lowrank", str(args.epochs_lowrank),
        "--batch_size", str(args.batch_size),
        "--baseline_decoder_m", "3.0",
        "--lowrank_ablation_rank", "88",
        "--lowrank_ablation_specs", "same_width:3,4", "widened:3,4",
        "--model_bits", str(args.model_bits),
        "--embed_bits", str(args.embed_bits),
        "--prune_ratio", str(args.prune_ratio),
        "--eval_variant", "quant",
        "--flop_mult", "2",
        "--device", args.device,
    ]
    if args.only:
        cmd += ["--only", *args.only]
    if args.skip_existing_models:
        cmd += ["--skip_existing_models"]

    if args.gpu is not None:
        shell_cmd = "unset PYTHONPATH; export PYTHONNOUSERSITE=1; export CUDA_VISIBLE_DEVICES=" + args.gpu + "; " + " ".join(cmd)
        run(["bash", "-lc", shell_cmd])
    else:
        run(cmd)


if __name__ == "__main__":
    main()
