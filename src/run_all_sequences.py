from __future__ import annotations

import argparse
import copy
from pathlib import Path
import pandas as pd

from .args import add_common_args
from .experiment import run_single_sequence
from .utils import ensure_dir


def find_sequences(data_root: Path, exclude_prefix: str | None = None):
    seqs = []
    for p in sorted(data_root.iterdir()):
        if not p.is_dir():
            continue
        if exclude_prefix and p.name.startswith(exclude_prefix):
            continue
        if any(x.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.webp'} for x in p.iterdir()):
            seqs.append(p)
    return seqs


def main():
    parser = argparse.ArgumentParser(description="Run the clustered expert experiment for every sequence folder in a data root.")
    parser.add_argument("--data_root", type=str, required=True, help="Path containing sequence folders, e.g. ..\\data")
    parser.add_argument("--runs_root", type=str, default="runs_all", help="Root folder for all output runs")
    parser.add_argument("--only", type=str, nargs="*", default=None, help="Optional sequence folder names to run")
    parser.add_argument("--skip_existing", action="store_true", help="Skip sequences with summary.csv already present")
    parser.add_argument("--exclude_prefix", type=str, default=None, help="Exclude sequence folders starting with this prefix, e.g. mixed_")
    # Reuse common args, but data_path/out_dir are not used from CLI here.
    add_common_args(parser)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    runs_root = ensure_dir(args.runs_root)
    seqs = find_sequences(data_root, args.exclude_prefix)
    if args.only:
        wanted = set(args.only)
        seqs = [p for p in seqs if p.name in wanted]
    if not seqs:
        raise RuntimeError(f"No sequence folders found in {data_root}")

    all_rows = []
    for seq in seqs:
        seq_out = runs_root / seq.name
        summary_path = seq_out / "summary.csv"
        if args.skip_existing and summary_path.exists():
            df = pd.read_csv(summary_path)
            all_rows.append(df)
            print(f"Skipping {seq.name}; existing {summary_path}")
            continue
        seq_args = copy.deepcopy(args)
        seq_args.data_path = str(seq)
        seq_args.out_dir = str(seq_out)
        print(f"\n=== Running sequence: {seq.name} ===")
        result = run_single_sequence(seq_args)
        df = pd.DataFrame([result["baseline"], result["clustered_experts"]])
        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    combined_path = runs_root / "all_results.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nAll results saved to: {combined_path}")
    print(combined[["sequence", "model_type", "psnr", "ssim", "bpp", "decoder_params", "gflops_per_frame_decoder_only"]])


if __name__ == "__main__":
    main()
