from __future__ import annotations

import argparse
import copy
from pathlib import Path
import pandas as pd

from .args import add_common_args
from .experiment import run_single_sequence_multihead
from .utils import ensure_dir
from .run_all_sequences import find_sequences


def main():
    parser = argparse.ArgumentParser(description="Run shared-trunk multi-head HNeRV for every sequence folder in a data root.")
    parser.add_argument("--data_root", type=str, required=True, help="Path containing sequence folders, e.g. ../data")
    parser.add_argument("--runs_root", type=str, default="runs_multihead", help="Root folder for all output runs")
    parser.add_argument("--only", type=str, nargs="*", default=None, help="Optional sequence folder names to run")
    parser.add_argument("--skip_existing", action="store_true", help="Skip sequences with summary.csv already present")
    parser.add_argument("--exclude_prefix", type=str, default=None, help="Exclude sequence folders starting with this prefix, e.g. mixed_")
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
        print(f"\n=== Running multi-head sequence: {seq.name} ===")
        result = run_single_sequence_multihead(seq_args)
        model_key = next(k for k in result.keys() if k != "baseline")
        df = pd.DataFrame([result["baseline"], result[model_key]])
        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    combined_path = runs_root / "all_results.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nAll multi-head results saved to: {combined_path}")
    cols = ["sequence", "model_type", "psnr", "ssim", "bpp", "decoder_params", "gflops_per_frame_decoder_only"]
    extra = ["num_heads", "cluster_method", "cluster_bits", "multihead_head_type", "refine_hidden_ch", "train_heads_only", "trainable_params_selected"]
    print(combined[[c for c in cols + extra if c in combined.columns]])


if __name__ == "__main__":
    main()
