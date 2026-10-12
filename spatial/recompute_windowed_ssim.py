#!/usr/bin/env python3
"""
Recompute full-frame windowed SSIM for HNeRV / LRConv / spatial-split runs.

This script replaces global pseudo-SSIM / mixed SSIM reporting with a single
standard windowed SSIM implementation from pytorch_msssim.

It does not retrain models and does not overwrite old summaries by default.
It reads saved reconstructed frames, matches them with ground-truth frames, and
writes a new CSV with recomputed SSIM values.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from pytorch_msssim import ssim as windowed_ssim


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

DEFAULT_SEQUENCES = [
    "bunny",
    "honeybee",
    "jockey",
    "readysetgo",
    "yachtride",
    "bosphorus",
    "shakendry",
]

DEFAULT_PRED_DIR_NAMES = [
    "recon_quant",
    "recon",
    "reconstructions",
    "predictions_quant",
    "predictions",
    "pred",
    "visualize_model_quant",
    "visualize_model",
]

DEFAULT_GT_DIR_NAMES = [
    "gt",
    "ground_truth",
    "groundtruth",
    "target",
    "targets",
    "frames",
    "data",
]


def natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def list_images(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=natural_key,
    )


def read_rgb01(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def center_crop_to(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if h < target_h or w < target_w:
        raise ValueError(f"Cannot center-crop shape {arr.shape} to {(target_h, target_w)}")
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return arr[top:top + target_h, left:left + target_w, :]


def crop_gt_to_pred(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    ph, pw = pred.shape[:2]
    gh, gw = gt.shape[:2]
    if (gh, gw) == (ph, pw):
        return gt
    return center_crop_to(gt, ph, pw)


def np_rgb_to_torch(img: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)


@torch.no_grad()
def compute_windowed_ssim(pred: np.ndarray, gt: np.ndarray, device: torch.device) -> float:
    x = np_rgb_to_torch(pred, device)
    y = np_rgb_to_torch(gt, device)
    return float(windowed_ssim(x, y, data_range=1.0, size_average=True).detach().cpu())


def compute_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2))
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def infer_sequence_from_path(path: Path, sequences: Sequence[str]) -> Optional[str]:
    parts = [p.lower() for p in path.parts]
    joined = "_".join(parts)
    for seq in sequences:
        s = seq.lower()
        if s in parts or s in joined:
            return seq
    return None


def find_prediction_dirs(roots: Sequence[Path], pred_dir_names: Sequence[str], max_depth: int) -> List[Path]:
    names = {n.lower() for n in pred_dir_names}
    found: List[Path] = []

    for root in roots:
        if not root.exists():
            print(f"[WARN] runs_root does not exist: {root}")
            continue

        root_depth = len(root.resolve().parts)

        for d in root.rglob("*"):
            if not d.is_dir():
                continue
            try:
                depth = len(d.resolve().parts) - root_depth
            except FileNotFoundError:
                continue
            if depth > max_depth:
                continue
            if d.name.lower() in names and list_images(d):
                found.append(d)

    unique = []
    seen = set()
    for d in sorted(found):
        key = str(d.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def find_gt_dir_inside_run(run_folder: Path, gt_dir_names: Sequence[str]) -> Optional[Path]:
    names = {n.lower() for n in gt_dir_names}
    for d in run_folder.rglob("*"):
        if d.is_dir() and d.name.lower() in names and list_images(d):
            return d
    return None


def get_gt_images(sequence: str, data_root: Optional[Path], run_folder: Path, gt_dir_names: Sequence[str]) -> Tuple[List[Path], str]:
    if data_root is not None:
        candidates = [
            data_root / sequence,
            data_root / sequence.lower(),
            data_root / sequence.upper(),
        ]
        for c in candidates:
            imgs = list_images(c)
            if imgs:
                return imgs, str(c)

    gt_dir = find_gt_dir_inside_run(run_folder, gt_dir_names)
    if gt_dir is not None:
        imgs = list_images(gt_dir)
        if imgs:
            return imgs, str(gt_dir)

    return [], ""


def pair_images(pred_imgs: List[Path], gt_imgs: List[Path]) -> List[Tuple[Path, Path]]:
    gt_by_name: Dict[str, Path] = {p.name: p for p in gt_imgs}
    exact = [(p, gt_by_name[p.name]) for p in pred_imgs if p.name in gt_by_name]
    if len(exact) >= max(1, int(0.8 * len(pred_imgs))):
        return exact
    n = min(len(pred_imgs), len(gt_imgs))
    return list(zip(pred_imgs[:n], gt_imgs[:n]))


def read_optional_summary(run_folder: Path) -> Dict[str, object]:
    out: Dict[str, object] = {}

    for name in ["summary_total_quant.csv", "summary.csv", "summary_partial.csv", "summary.json"]:
        p = run_folder / name
        if not p.exists():
            continue

        try:
            if p.suffix.lower() == ".json":
                data = json.loads(p.read_text())
                if isinstance(data, dict):
                    out.update({f"old_{k}": v for k, v in data.items()})
            else:
                with p.open(newline="") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                if len(rows) == 1:
                    out.update({f"old_{k}": v for k, v in rows[0].items()})
                else:
                    for r in rows:
                        if "summary_key" in r and "summary_value" in r:
                            out[f"old_{r['summary_key']}"] = r["summary_value"]
        except Exception as e:
            print(f"[WARN] could not read metadata from {p}: {e}")

    return out


def safe_float(x: object) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", nargs="+", required=True, help="One or more run roots to scan recursively.")
    parser.add_argument("--data_root", default=None, help="Dataset root containing sequence folders, e.g. ../data")
    parser.add_argument("--out", default="windowed_ssim_recalc.csv", help="Output CSV path.")
    parser.add_argument("--pred_dir_names", default=",".join(DEFAULT_PRED_DIR_NAMES), help="Comma-separated prediction folder names to scan for.")
    parser.add_argument("--gt_dir_names", default=",".join(DEFAULT_GT_DIR_NAMES), help="Comma-separated GT folder names if data_root is not provided.")
    parser.add_argument("--sequences", default=",".join(DEFAULT_SEQUENCES), help="Comma-separated sequence names.")
    parser.add_argument("--max_depth", type=int, default=8, help="Maximum recursive depth below each runs_root.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for SSIM computation.")
    parser.add_argument("--no_crop_gt_to_pred", action="store_true", help="Disable center-cropping GT to prediction size.")
    parser.add_argument("--write_per_frame", action="store_true", help="Also write per-frame CSV next to the aggregate output.")
    args = parser.parse_args()

    roots = [Path(p) for p in args.runs_root]
    data_root = Path(args.data_root) if args.data_root else None
    pred_dir_names = [s.strip() for s in args.pred_dir_names.split(",") if s.strip()]
    gt_dir_names = [s.strip() for s in args.gt_dir_names.split(",") if s.strip()]
    sequences = [s.strip() for s in args.sequences.split(",") if s.strip()]
    crop_enabled = not args.no_crop_gt_to_pred

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    pred_dirs = find_prediction_dirs(roots, pred_dir_names, args.max_depth)
    print(f"[INFO] Found {len(pred_dirs)} prediction folders.")
    for d in pred_dirs:
        print(f"  - {d}")

    aggregate_rows: List[Dict[str, object]] = []
    per_frame_rows: List[Dict[str, object]] = []

    for pred_dir in pred_dirs:
        run_folder = pred_dir.parent
        sequence = infer_sequence_from_path(run_folder, sequences) or infer_sequence_from_path(pred_dir, sequences)

        if sequence is None:
            print(f"[SKIP] Could not infer sequence from path: {pred_dir}")
            continue

        pred_imgs = list_images(pred_dir)
        gt_imgs, gt_source = get_gt_images(sequence, data_root, run_folder, gt_dir_names)

        if not pred_imgs:
            print(f"[SKIP] No prediction images in {pred_dir}")
            continue
        if not gt_imgs:
            print(f"[SKIP] No GT images found for sequence={sequence}, run={run_folder}")
            continue

        pairs = pair_images(pred_imgs, gt_imgs)
        if not pairs:
            print(f"[SKIP] No image pairs for {pred_dir}")
            continue

        ssims = []
        psnrs = []
        used = 0

        for pred_path, gt_path in pairs:
            try:
                pred = read_rgb01(pred_path)
                gt = read_rgb01(gt_path)

                if crop_enabled:
                    gt = crop_gt_to_pred(gt, pred)
                elif pred.shape != gt.shape:
                    raise ValueError(f"shape mismatch: pred {pred.shape}, gt {gt.shape}")

                ssim_val = compute_windowed_ssim(pred, gt, device)
                psnr_val = compute_psnr(pred, gt)

                ssims.append(ssim_val)
                psnrs.append(psnr_val)
                used += 1

                if args.write_per_frame:
                    per_frame_rows.append({
                        "run_folder": str(run_folder),
                        "pred_dir": str(pred_dir),
                        "sequence": sequence,
                        "frame_index": used - 1,
                        "pred_file": pred_path.name,
                        "gt_file": gt_path.name,
                        "windowed_ssim": ssim_val,
                        "psnr_recomputed": psnr_val,
                    })

            except Exception as e:
                print(f"[WARN] failed pair pred={pred_path}, gt={gt_path}: {e}")

        if used == 0:
            print(f"[SKIP] No successfully scored frames for {pred_dir}")
            continue

        meta = read_optional_summary(run_folder)

        avg_ssim = float(np.mean(ssims))
        avg_psnr = float(np.mean(psnrs))

        old_ssim = safe_float(meta.get("old_final_full_ssim", meta.get("old_ssim")))
        old_psnr = safe_float(meta.get("old_final_full_psnr", meta.get("old_psnr")))

        row: Dict[str, object] = {
            "run_folder": str(run_folder),
            "pred_dir": str(pred_dir),
            "sequence": sequence,
            "num_frames_scored": used,
            "gt_source": gt_source,
            "windowed_ssim": avg_ssim,
            "psnr_recomputed": avg_psnr,
            "old_ssim_if_found": old_ssim,
            "old_psnr_if_found": old_psnr,
            "delta_ssim_new_minus_old": (avg_ssim - old_ssim) if old_ssim is not None else None,
            "delta_psnr_new_minus_old": (avg_psnr - old_psnr) if old_psnr is not None else None,
        }

        for key in [
            "old_mode",
            "old_tag",
            "old_prediction_variant",
            "old_sum_playback_total_m",
            "old_sum_decoder_gflops",
            "old_quant_model_bit",
            "old_quant_embed_bit",
            "old_prune_ratio",
            "old_model_type",
            "old_decoder_params",
            "old_gflops_per_frame_decoder_only",
            "old_eval_variant",
        ]:
            if key in meta:
                row[key] = meta[key]

        aggregate_rows.append(row)

        print(
            f"[OK] {run_folder} | seq={sequence} | frames={used} | "
            f"windowed_ssim={avg_ssim:.6f} | psnr={avg_psnr:.3f}"
        )

    if not aggregate_rows:
        raise SystemExit("No runs were scored. Check --runs_root, --data_root, and prediction folder names.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({k for r in aggregate_rows for k in r.keys()})
    preferred = [
        "run_folder", "pred_dir", "sequence", "num_frames_scored", "gt_source",
        "windowed_ssim", "psnr_recomputed", "old_ssim_if_found", "old_psnr_if_found",
        "delta_ssim_new_minus_old", "delta_psnr_new_minus_old",
        "old_mode", "old_tag", "old_prediction_variant",
        "old_sum_playback_total_m", "old_sum_decoder_gflops",
        "old_quant_model_bit", "old_quant_embed_bit", "old_prune_ratio",
        "old_model_type", "old_decoder_params", "old_gflops_per_frame_decoder_only",
        "old_eval_variant",
    ]
    fieldnames = [f for f in preferred if f in fieldnames] + [f for f in fieldnames if f not in preferred]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print(f"\n[WROTE] {out_path}")

    if args.write_per_frame:
        per_frame_path = out_path.with_name(out_path.stem + "_per_frame.csv")
        pf_fields = sorted({k for r in per_frame_rows for k in r.keys()})
        with per_frame_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=pf_fields)
            writer.writeheader()
            writer.writerows(per_frame_rows)
        print(f"[WROTE] {per_frame_path}")


if __name__ == "__main__":
    main()
