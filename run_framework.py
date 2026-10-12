from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], cwd: Path | None = None, gpu: str | None = None) -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd or ROOT), env=env, check=True)


def default_spatial_tag(mode: str, epochs: int) -> str:
    if mode == "spatial4":
        return f"quadrant_3m_{epochs}ep_q8e6_p10"
    if mode == "spatial16":
        return f"tile16_3m_{epochs}ep_q8e6_p10"
    return f"spatial_custom_{epochs}ep"


def run_wholeframe(args: argparse.Namespace) -> None:
    sequences = args.only or ([args.sequence] if args.sequence else [])
    cmd = [
        sys.executable,
        str(ROOT / "run_wholeframe_lrwide.py"),
        "--data_root", args.data_root,
        "--runs_root", args.runs_root,
        "--epochs_baseline", str(args.epochs),
        "--epochs_lowrank", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--device", args.device,
        "--model_bits", str(args.model_bits),
        "--embed_bits", str(args.embed_bits),
        "--prune_ratio", str(args.prune_ratio),
    ]
    if sequences:
        cmd += ["--only", *sequences]
    if args.gpu is not None:
        cmd += ["--gpu", str(args.gpu)]
    if args.skip_existing_models:
        cmd.append("--skip_existing_models")
    run(cmd, cwd=ROOT, gpu=args.gpu)


def run_spatial(args: argparse.Namespace, config: Path, mode: str) -> None:
    tag = args.tag or default_spatial_tag(mode, args.epochs)
    cmd = [
        sys.executable,
        str(ROOT / "spatial" / "run_experiment.py"),
        "debug",
        "--config", str(config),
        "--sequence", args.sequence,
        "--tag", tag,
        "--epochs", str(args.epochs),
        "--eval_freq", str(args.eval_freq),
        "--gpu", str(args.gpu or "0"),
        "--expected_frames", str(args.expected_frames),
        "--orig_data", args.data_root,
        "--work_root", args.spatial_runs_root,
        "--prediction_variant", args.prediction_variant,
    ]
    if args.clean:
        cmd.append("--clean")
    run(cmd, cwd=ROOT / "spatial", gpu=args.gpu)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified runner for dense/LR-wide HNeRV (N=1) and spatial LR-wide HNeRV (N=4 or N=16)."
    )
    parser.add_argument("mode", choices=["wholeframe", "spatial4", "spatial16", "spatial-custom"],
                        help="wholeframe = N=1 LR-wide experiments; spatial4 = 4 quadrants; spatial16 = 4x4 tiles.")
    parser.add_argument("--data_root", default=str(ROOT.parent / "data"),
                        help="Folder containing sequence subfolders, e.g. data/bunny/*.png")
    parser.add_argument("--runs_root", default=str(ROOT / "runs" / "wholeframe"),
                        help="Output root for whole-frame experiments.")
    parser.add_argument("--spatial_runs_root", default=str(ROOT / "runs" / "spatial"),
                        help="Output root for spatial experiments.")
    parser.add_argument("--sequence", default="bunny", help="Single sequence for spatial runs, or fallback for wholeframe.")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Sequence list for wholeframe runs, e.g. --only bunny honeybee jockey.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_freq", type=int, default=30)
    parser.add_argument("--expected_frames", type=int, default=132)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_bits", type=int, default=8)
    parser.add_argument("--embed_bits", type=int, default=8,
                        help="Used by wholeframe runs. Spatial configs currently use their own quant_embed_bit, usually 6.")
    parser.add_argument("--prune_ratio", type=float, default=0.10)
    parser.add_argument("--prediction_variant", choices=["orig", "quant"], default="quant")
    parser.add_argument("--tag", default=None, help="Optional spatial experiment tag.")
    parser.add_argument("--config", default=None, help="Config path for spatial-custom.")
    parser.add_argument("--clean", action="store_true", help="Delete an existing spatial run folder before running.")
    parser.add_argument("--skip_existing_models", action="store_true",
                        help="For wholeframe mode: reuse existing checkpoints and recompute quantized/pruned metrics.")
    args = parser.parse_args()

    if args.mode == "wholeframe":
        run_wholeframe(args)
    elif args.mode == "spatial4":
        run_spatial(args, ROOT / "spatial" / "configs" / "quadrant_equal_lrwide.json", args.mode)
    elif args.mode == "spatial16":
        run_spatial(args, ROOT / "spatial" / "configs" / "tile16_equal_3m_lrwide.json", args.mode)
    else:
        if not args.config:
            raise SystemExit("spatial-custom requires --config")
        run_spatial(args, Path(args.config), args.mode)


if __name__ == "__main__":
    main()
