from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path

import pandas as pd

from .args import add_common_args
from .experiment import (
    train_one_baseline,
    extract_embeddings,
    cluster_embeddings,
    train_experts,
    evaluate_saved_baseline,
    evaluate_experts,
)
from .run_all_sequences import find_sequences
from .utils import ensure_dir, save_json


def _prepare_k_dir(shared_dir: Path, k_dir: Path) -> None:
    ensure_dir(k_dir)
    for name in ["baseline_last.pt", "baseline_best.pt", "embeddings.pt"]:
        src = shared_dir / name
        if src.exists():
            shutil.copy2(src, k_dir / name)


def _run_one_sequence_grid(args, seq: Path, seq_out: Path):
    ensure_dir(seq_out)
    shared_dir = ensure_dir(seq_out / "baseline_shared")

    shared_args = copy.deepcopy(args)
    shared_args.data_path = str(seq)
    shared_args.out_dir = str(shared_dir)

    if not (shared_dir / "baseline_last.pt").exists():
        train_one_baseline(shared_args)
    if not (shared_dir / "embeddings.pt").exists():
        extract_embeddings(shared_args)

    baseline = evaluate_saved_baseline(shared_args)
    rows = [baseline]

    for k in args.grid_num_experts:
        k = int(k)
        k_dir = ensure_dir(seq_out / f"lowrank_k{k}")
        _prepare_k_dir(shared_dir, k_dir)

        k_args = copy.deepcopy(args)
        k_args.data_path = str(seq)
        k_args.out_dir = str(k_dir)
        k_args.num_experts = k
        k_args.expert_decoder_type = "lowrank"
        k_args.expert_decoder_m = float(args.baseline_decoder_m) / k
        # Keep the low-rank experts' channel width equal to the dense baseline width.
        k_args.lowrank_base_decoder_m = args.baseline_decoder_m

        print(f"\n--- Low-rank expert comparison: {seq.name}, K={k}, target per expert={k_args.expert_decoder_m:.4f}M ---")
        cluster_embeddings(k_args)
        train_experts(k_args)
        metrics = evaluate_experts(k_args)
        metrics["model_type"] = f"lowrank_experts_k{k}"
        metrics["target_total_decoder_m"] = float(args.baseline_decoder_m)
        metrics["target_per_expert_m"] = float(k_args.expert_decoder_m)
        rows.append(metrics)
        pd.DataFrame(rows).to_csv(seq_out / "summary_partial.csv", index=False)

    pd.DataFrame(rows).to_csv(seq_out / "summary.csv", index=False)
    save_json({"rows": rows}, seq_out / "summary.json")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compare dense 3M HNeRV baseline against low-rank expert decoders with total expert params near the baseline budget.")
    parser.add_argument("--data_root", type=str, required=True, help="Path containing sequence folders, e.g. ..\\data")
    parser.add_argument("--runs_root", type=str, default="runs_lowrank_expert_grid", help="Root folder for all output runs")
    parser.add_argument("--only", type=str, nargs="*", default=None, help="Optional sequence folder names to run")
    parser.add_argument("--exclude_prefix", type=str, default=None, help="Exclude sequence folders starting with this prefix, e.g. mixed_")
    parser.add_argument("--grid_num_experts", type=int, nargs="+", default=[2, 3], help="Expert counts to compare. Each run uses baseline_decoder_m / K per expert.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip sequences with summary.csv already present")
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

    all_dfs = []
    for seq in seqs:
        seq_out = runs_root / seq.name
        if args.skip_existing and (seq_out / "summary.csv").exists():
            df = pd.read_csv(seq_out / "summary.csv")
            all_dfs.append(df)
            print(f"Skipping {seq.name}; existing {seq_out / 'summary.csv'}")
            continue
        print(f"\n=== Running low-rank expert grid sequence: {seq.name} ===")
        rows = _run_one_sequence_grid(args, seq, seq_out)
        all_dfs.append(pd.DataFrame(rows))

    combined = pd.concat(all_dfs, ignore_index=True)
    combined_path = runs_root / "all_results.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nAll low-rank expert-grid results saved to: {combined_path}")
    cols = [c for c in ["sequence", "model_type", "psnr", "ssim", "bpp", "decoder_params", "gflops_per_frame_decoder_only", "num_experts", "expert_decoder_type", "expert_decoder_m"] if c in combined.columns]
    print(combined[cols])


if __name__ == "__main__":
    main()
