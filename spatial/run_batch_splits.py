"""Historical batch runner for cropped spatial experiments.

This aggregates legacy spatial summaries. Its BPP column is an estimate and
must not be compared to measured packed BPP from spatial/evaluate_packed.py.
"""
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)

DEFAULT_SEQUENCES = [
    #"bunny",
    #"honeybee",
    #"jockey",
    #"readysetgo",
    #"yachtride",
    "bosphorus",   # change/remove if your 6th sequence has a different name
]


METHODS = {
    "quadrant": {
        "config": PROJECT_ROOT / "configs/quadrant_equal_lrwide.json",
        "tag": "quadrant_3m_300ep_q8p10",
        "mode_prefix": "quadrant",
    },
    "tile16": {
        "config": PROJECT_ROOT / "configs/tile16_equal_3m_lrwide.json",
        "tag": "tile16_3m_300ep_q8p10",
        "mode_prefix": "tiles16",
    },
}


def run_one(method: str, sequence: str, gpu: int, epochs: int, eval_freq: int,
            expected_frames: int, clean: bool, skip_existing: bool) -> None:
    info = METHODS[method]
    config = info["config"]
    tag = info["tag"]
    mode_prefix = info["mode_prefix"]

    run_dir = PROJECT_ROOT / "runs" / f"{mode_prefix}_{sequence}_{tag}"
    summary = run_dir / "summary_total_quant.csv"

    if skip_existing and summary.exists():
        print(f"[SKIP] {method} {sequence}: existing summary found: {summary}")
        return

    log_dir = PROJECT_ROOT / "runs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{sequence}_{tag}.log"

    cmd = [
        str(PYTHON),
        "run_experiment.py",
        "debug",
        "--config", str(config),
        "--sequence", sequence,
        "--tag", tag,
        "--epochs", str(epochs),
        "--eval_freq", str(eval_freq),
        "--gpu", str(gpu),
        "--expected_frames", str(expected_frames),
        "--prediction_variant", "quant",
    ]

    if clean:
        cmd.append("--clean")

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"

    print("=" * 80)
    print(f"[RUN] method={method} sequence={sequence} gpu={gpu}")
    print(f"[LOG] {log_path}")
    print("=" * 80)

    with log_path.open("w") as log:
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    if proc.returncode != 0:
        print(f"[FAILED] {method} {sequence}. Check log:")
        print(log_path)
        raise SystemExit(proc.returncode)

    print(f"[DONE] {method} {sequence}")


def aggregate() -> None:
    rows = []

    for summary in sorted((PROJECT_ROOT / "runs").glob("*/summary_total_quant.csv")):
        with summary.open() as f:
            kv = {r["summary_key"]: r["summary_value"] for r in csv.DictReader(f)}

        # Keep only these experiment tags.
        tag = kv.get("tag", "")
        if tag not in {
            "quadrant_3m_300ep_q8p10",
            "tile16_3m_300ep_q8p10",
        }:
            continue

        rows.append({
            "run": summary.parent.name,
            "mode": kv.get("mode", ""),
            "sequence": kv.get("sequence", ""),
            "tag": tag,
            "prediction_variant": kv.get("prediction_variant", ""),
            "psnr": kv.get("final_full_psnr", ""),
            "ssim": kv.get("final_full_ssim", ""),
            "params_m": kv.get("sum_playback_total_m", ""),
            "gflops": kv.get("sum_decoder_gflops", ""),
            "legacy_estimated_bpp": kv.get("sum_bpp", ""),
            "quant_model_bit": kv.get("quant_model_bit", ""),
            "quant_embed_bit": kv.get("quant_embed_bit", ""),
            "prune_ratio": kv.get("prune_ratio", ""),
        })

    out = PROJECT_ROOT / "runs" / "aggregate_4tile_16tile_6seq_q8p10.csv"

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run",
            "mode",
            "sequence",
            "tag",
            "prediction_variant",
            "psnr",
            "ssim",
            "params_m",
            "gflops",
            "legacy_estimated_bpp",
            "quant_model_bit",
            "quant_embed_bit",
            "prune_ratio",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[AGGREGATE SAVED] {out}")
    for r in rows:
        print(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["quadrant", "tile16"], required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--eval_freq", type=int, default=30)
    parser.add_argument("--expected_frames", type=int, default=132)
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    parser.add_argument("--clean", action="store_true", help="Explicitly replace an existing run directory")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--aggregate_after", action="store_true")
    args = parser.parse_args()

    for seq in args.sequences:
        run_one(
            method=args.method,
            sequence=seq,
            gpu=args.gpu,
            epochs=args.epochs,
            eval_freq=args.eval_freq,
            expected_frames=args.expected_frames,
            clean=args.clean,
            skip_existing=args.skip_existing,
        )

    if args.aggregate_after:
        aggregate()


if __name__ == "__main__":
    main()
