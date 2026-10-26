import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

def natural_key(path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


DWT_BANDS = ("LL", "LH", "HL", "HH")
QUAD_BANDS = ("Q0", "Q1", "Q2", "Q3")
TILE16_BANDS = tuple(f"T{r}{c}" for r in range(4) for c in range(4))


def list_images(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if not folder.is_dir():
        raise FileNotFoundError(f"Frame folder not found: {folder}")
    files = [p for p in folder.iterdir() if p.suffix.lower() in exts]
    files.sort(key=natural_key)
    if not files:
        raise FileNotFoundError(f"No image files found in {folder}")
    return files


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def save_rgb(path: Path, arr01: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(arr01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def center_crop(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h and w == target_w:
        return img
    if h < target_h or w < target_w:
        raise ValueError(f"Cannot center-crop {h}x{w} to {target_h}x{target_w}")
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return img[top:top + target_h, left:left + target_w, :]


def dwt_haar_rgb(img):
    h, w, _ = img.shape
    if h % 2 or w % 2:
        raise ValueError(f"Image size must be even, got {h}x{w}")
    a = img[0::2, 0::2, :]
    b = img[0::2, 1::2, :]
    c_ = img[1::2, 0::2, :]
    d = img[1::2, 1::2, :]
    return {
        "LL": (a + b + c_ + d) / 2.0,
        "LH": (a - b + c_ - d) / 2.0,
        "HL": (a + b - c_ - d) / 2.0,
        "HH": (a - b - c_ + d) / 2.0,
    }


def idwt_haar_rgb(coeffs):
    ll, lh, hl, hh = coeffs["LL"], coeffs["LH"], coeffs["HL"], coeffs["HH"]
    a = (ll + lh + hl + hh) / 2.0
    b = (ll - lh + hl - hh) / 2.0
    c = (ll + lh - hl - hh) / 2.0
    d = (ll - lh - hl + hh) / 2.0
    h2, w2, ch = ll.shape
    out = np.empty((2 * h2, 2 * w2, ch), dtype=np.float32)
    out[0::2, 0::2, :] = a
    out[0::2, 1::2, :] = b
    out[1::2, 0::2, :] = c
    out[1::2, 1::2, :] = d
    return out


def split_quadrants(img):
    h, w, _ = img.shape
    return {
        "Q0": img[:h // 2, :w // 2, :],
        "Q1": img[:h // 2, w // 2:, :],
        "Q2": img[h // 2:, :w // 2, :],
        "Q3": img[h // 2:, w // 2:, :],
    }


def merge_quadrants(parts):
    top = np.concatenate([parts["Q0"], parts["Q1"]], axis=1)
    bot = np.concatenate([parts["Q2"], parts["Q3"]], axis=1)
    return np.concatenate([top, bot], axis=0)


def split_tiles(img, rows=4, cols=4):
    h, w, _ = img.shape
    if h % rows or w % cols:
        raise ValueError(f"Image size {h}x{w} not divisible by tile grid {rows}x{cols}")
    th, tw = h // rows, w // cols
    parts = {}
    for r in range(rows):
        for c in range(cols):
            parts[f"T{r}{c}"] = img[r * th:(r + 1) * th, c * tw:(c + 1) * tw, :]
    return parts


def merge_tiles(parts, rows=4, cols=4):
    row_imgs = []
    for r in range(rows):
        row_imgs.append(np.concatenate([parts[f"T{r}{c}"] for c in range(cols)], axis=1))
    return np.concatenate(row_imgs, axis=0)


def compute_minmax(parts_per_frame, bands):
    mins = {b: np.full((3,), np.inf, dtype=np.float64) for b in bands}
    maxs = {b: np.full((3,), -np.inf, dtype=np.float64) for b in bands}
    for parts in parts_per_frame:
        for b in bands:
            flat = parts[b].reshape(-1, 3)
            mins[b] = np.minimum(mins[b], flat.min(axis=0))
            maxs[b] = np.maximum(maxs[b], flat.max(axis=0))
    return {b: {"min": mins[b].tolist(), "max": maxs[b].tolist()} for b in bands}


def normalize(arr, meta):
    mn = np.asarray(meta["min"], dtype=np.float32).reshape(1, 1, 3)
    mx = np.asarray(meta["max"], dtype=np.float32).reshape(1, 1, 3)
    return (arr - mn) / np.maximum(mx - mn, 1e-8)


def denormalize(arr01, meta):
    mn = np.asarray(meta["min"], dtype=np.float32).reshape(1, 1, 3)
    mx = np.asarray(meta["max"], dtype=np.float32).reshape(1, 1, 3)
    return arr01 * (mx - mn) + mn


def load_cfg(args):
    cfg = json.loads(Path(args.config).read_text())
    if "bands" not in cfg:
        raise ValueError(f"Config must include a bands dictionary: {args.config}")
    cfg["bands_list"] = list(cfg["bands"].keys())
    expected = expected_bands_for_mode(cfg["mode"])
    if tuple(cfg["bands_list"]) != expected:
        raise ValueError(f"Expected bands {expected} for {cfg['mode']}")
    if any(float(part["modelsize"]) <= 0 for part in cfg["bands"].values()):
        raise ValueError("All per-tile modelsize targets must be positive")
    if cfg.get("lowrank_blocks") and any(int(part["rank"]) <= 0 for part in cfg["bands"].values()):
        raise ValueError("Low-rank tile ranks must be positive")
    return cfg


def run_root(args, cfg):
    return Path(args.work_root) / f"{cfg['mode']}_{args.sequence}_{args.tag}"


def data_root(args, cfg):
    return run_root(args, cfg) / "data"


def pred_root(args, cfg):
    return run_root(args, cfg) / f"predictions_{args.prediction_variant}"


def recon_root(args, cfg):
    return run_root(args, cfg) / f"recon_{args.prediction_variant}"


def hnerv_output_root(args, cfg):
    return run_root(args, cfg) / "hnerv_output"


def expected_bands_for_mode(mode):
    if mode == "dwt":
        return DWT_BANDS
    if mode == "quadrant":
        return QUAD_BANDS
    if mode in ["tiles16", "tile16"]:
        return TILE16_BANDS
    raise ValueError(f"Unknown mode: {mode}")


def prepare(args):
    cfg = load_cfg(args)
    mode = cfg["mode"]
    grid = 4 if mode in ("tiles16", "tile16") else 2
    if mode != "dwt" and (args.full_height % grid or args.full_width % grid):
        raise ValueError(f"Full-frame dimensions must be divisible by the {grid}x{grid} tile grid")
    bands = cfg["bands_list"]
    expected = expected_bands_for_mode(mode)
    if tuple(bands) != tuple(expected):
        raise ValueError(f"Bands in config do not match mode={mode}. Expected {expected}, got {bands}")

    gt_dir = Path(args.orig_data) / args.sequence
    out_root = data_root(args, cfg)
    out_root.mkdir(parents=True, exist_ok=True)

    frames = list_images(gt_dir)
    if args.expected_frames is not None and len(frames) != args.expected_frames:
        raise ValueError(f"Expected {args.expected_frames} source frames in {gt_dir}, found {len(frames)}")
    decomposed = []
    tile_h, tile_w = map(int, cfg["crop_list"].split("_"))
    stride_product = math.prod(cfg["enc_strds"])
    if mode != "dwt" and (tile_h % stride_product or tile_w % stride_product):
        raise ValueError(f"Configured tile dimensions {tile_h}x{tile_w} must be divisible by encoder stride {stride_product}")
    valid_h = args.full_height // (4 if mode in ["tiles16", "tile16"] else 2)
    valid_w = args.full_width // (4 if mode in ["tiles16", "tile16"] else 2)
    if mode != "dwt" and (valid_h > tile_h or valid_w > tile_w):
        raise ValueError(f"Valid tile {valid_h}x{valid_w} exceeds configured padded tile {tile_h}x{tile_w}")
    for p in frames:
        img = center_crop(read_rgb(p), args.full_height, args.full_width)
        if mode == "dwt":
            decomposed.append(dwt_haar_rgb(img))
        elif mode == "quadrant":
            decomposed.append({k: np.pad(v, ((0, tile_h - v.shape[0]),
                                                  (0, tile_w - v.shape[1]), (0, 0)))
                               for k, v in split_quadrants(img).items()})
        elif mode in ["tiles16", "tile16"]:
            decomposed.append({k: np.pad(v, ((0, tile_h - v.shape[0]),
                                                  (0, tile_w - v.shape[1]), (0, 0)))
                               for k, v in split_tiles(img, 4, 4).items()})
        else:
            raise ValueError(mode)

    if mode == "dwt":
        meta = compute_minmax(decomposed, bands)
    else:
        meta = {b: {"min": [0, 0, 0], "max": [1, 1, 1]} for b in bands}

    for i, parts in enumerate(decomposed):
        name = f"{i:06d}.png"
        for b in bands:
            save_rgb(out_root / f"{args.sequence}_{b}" / name, normalize(parts[b], meta[b]))

    meta_json = {
        "sequence": args.sequence,
        "mode": mode,
        "bands": bands,
        "num_frames": len(frames),
        "source_frames": [p.name for p in frames],
        "orig_data": str(gt_dir),
        "full_height": args.full_height,
        "full_width": args.full_width,
        "valid_tile_height": valid_h if mode != "dwt" else None,
        "valid_tile_width": valid_w if mode != "dwt" else None,
        "padded_tile_height": tile_h,
        "padded_tile_width": tile_w,
        "normalization": "per_band_per_channel_minmax" if mode == "dwt" else "none",
        "meta": meta,
    }
    (out_root / f"{args.sequence}_{mode}_meta.json").write_text(json.dumps(meta_json, indent=2))
    print(f"Prepared {len(frames)} frames in {mode} mode: {out_root}")


def build_train_cmd(args, cfg, band):
    mode = cfg["mode"]
    vid = f"{args.sequence}_{band}"
    # train_nerv_all prepends "output/" to --outf. Compute a relative path
    # so spatial runs can live in any --work_root, not only ./runs.
    hnerv_output_dir = hnerv_output_root(args, cfg) / vid
    hnerv_output_dir.mkdir(parents=True, exist_ok=True)
    hnerv_output_base = Path(args.hnerv_repo).resolve() / "output"
    local_outf = os.path.relpath(hnerv_output_dir.resolve(), hnerv_output_base)
    band_cfg = cfg["bands"][band]

    cmd = [
        sys.executable, "train_nerv_all.py",
        "--outf", local_outf,
        "--data_path", str(data_root(args, cfg) / vid),
        "--vid", vid,
        "--conv_type", cfg["conv_type"][0], cfg["conv_type"][1],
        "--act", cfg["act"],
        "--norm", cfg["norm"],
        "--crop_list", cfg["crop_list"],
        "--resize_list", cfg["resize_list"],
        "--loss", cfg["loss"],
        "--enc_strds", *[str(x) for x in cfg["enc_strds"]],
        "--enc_dim", cfg["enc_dim"],
        "--dec_strds", *[str(x) for x in cfg["dec_strds"]],
        "--ks", cfg["ks"],
        "--reduce", str(cfg["reduce"]),
        "--modelsize", str(band_cfg["modelsize"]),
        "--lowrank_blocks", cfg["lowrank_blocks"],
        "--lowrank_rank", str(band_cfg["rank"]),
        "--manualSeed", str(args.manual_seed),
        "-e", str(args.epochs),
        "--eval_freq", str(args.eval_freq),
        "--lower_width", str(cfg["lower_width"]),
        "-b", str(args.batch_size),
        "--lr", str(cfg["lr"]),
        "--quant_model_bit", str(cfg.get("quant_model_bit", 8)),
        "--quant_embed_bit", str(cfg.get("quant_embed_bit", 6)),
        "--prune_ratio", str(cfg.get("prune_ratio", 0.10)),
        "--dump_images",
    ]
    if cfg.get("lowrank_blocks"):
        cmd.append("--auto_widen_lowrank")
    return cmd


def train(args):
    cfg = load_cfg(args)
    hnerv_repo = Path(args.hnerv_repo)
    if not (hnerv_repo / "train_nerv_all.py").exists():
        raise FileNotFoundError(f"Missing train_nerv_all.py in {hnerv_repo}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONPATH", None)

    for band in cfg["bands_list"]:
        cmd = build_train_cmd(args, cfg, band)
        print("\n" + "=" * 100)
        print(f"Training {cfg['mode']} part {band}")
        print(" ".join(str(x) for x in cmd))
        print("=" * 100 + "\n")
        subprocess.run(cmd, cwd=hnerv_repo, env=env, check=True)


def find_visualize_dir(args, cfg, band):
    base = hnerv_output_root(args, cfg) / f"{args.sequence}_{band}"
    visual_name = "visualize_model_quant" if args.prediction_variant == "quant" else "visualize_model_orig"
    matches = list(base.glob(f"**/{visual_name}"))
    if not matches:
        raise FileNotFoundError(f"No {visual_name} found under {base}. Did --dump_images run?")
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def extract(args):
    cfg = load_cfg(args)
    out_root = pred_root(args, cfg)
    for band in cfg["bands_list"]:
        vis = find_visualize_dir(args, cfg, band)
        out = out_root / f"{args.sequence}_{band}"
        out.mkdir(parents=True, exist_ok=True)

        count = 0
        for f in list_images(vis):
            m = re.search(r"pred_(\d+)", f.name)
            if not m:
                continue
            idx = int(m.group(1))
            arr = read_rgb(f)
            pred = arr[:, arr.shape[1] // 2:, :]  # official HNeRV saves GT|prediction
            save_rgb(out / f"{idx:06d}.png", pred)
            count += 1
        print(f"{band}: extracted {count} {args.prediction_variant} predictions from {vis}")


def check(args):
    cfg = load_cfg(args)
    counts = {}
    for band in cfg["bands_list"]:
        counts[band] = len(list_images(pred_root(args, cfg) / f"{args.sequence}_{band}"))
        print(f"{band}: {counts[band]} frames")
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"Frame-count mismatch: {counts}")
    if args.expected_frames is not None and next(iter(counts.values())) != args.expected_frames:
        raise RuntimeError(f"Expected {args.expected_frames}, got {counts}")


def reconstruct(args):
    cfg = load_cfg(args)
    mode = cfg["mode"]
    bands = cfg["bands_list"]
    rroot = recon_root(args, cfg)
    rroot.mkdir(parents=True, exist_ok=True)

    meta = json.loads((data_root(args, cfg) / f"{args.sequence}_{mode}_meta.json").read_text())
    band_meta = meta["meta"]

    files = {b: list_images(pred_root(args, cfg) / f"{args.sequence}_{b}") for b in bands}
    n = len(files[bands[0]])
    names = [p.name for p in files[bands[0]]]
    if any([p.name for p in files[b]] != names for b in bands):
        raise ValueError("Tile prediction filenames do not align")
    if n != meta["num_frames"]:
        raise ValueError(f"Expected {meta['num_frames']} tile predictions, got {n}")

    for i in range(n):
        parts = {}
        for b in bands:
            part = denormalize(read_rgb(files[b][i]), band_meta[b])
            if mode != "dwt":
                grid = 4 if mode in ("tiles16", "tile16") else 2
                valid_h = meta.get("valid_tile_height") or meta["full_height"] // grid
                valid_w = meta.get("valid_tile_width") or meta["full_width"] // grid
                part = part[:valid_h, :valid_w]
            parts[b] = part

        if mode == "dwt":
            rec = idwt_haar_rgb(parts)
        elif mode == "quadrant":
            rec = merge_quadrants(parts)
        elif mode in ["tiles16", "tile16"]:
            rec = merge_tiles(parts, 4, 4)
        else:
            raise ValueError(mode)
        if rec.shape[:2] != (meta["full_height"], meta["full_width"]):
            raise ValueError(f"Reconstruction has shape {rec.shape[:2]}, expected full frame")
        save_rgb(rroot / names[i], rec)
    print(f"Saved {n} reconstructed frames to {rroot}")


def psnr(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    if mse <= 1e-12:
        return 99.0
    return -10.0 * math.log10(mse)


def ssim_global_rgb(pred, gt):
    vals = []
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    for c in range(3):
        x = pred[:, :, c].astype(np.float64)
        y = gt[:, :, c].astype(np.float64)
        mux, muy = x.mean(), y.mean()
        varx, vary = x.var(), y.var()
        cov = ((x - mux) * (y - muy)).mean()
        vals.append(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux ** 2 + muy ** 2 + c1) * (varx + vary + c2)))
    return float(np.mean(vals))


def score_folder(gt_files, pred_files, crop_to_pred=True):
    if len(gt_files) != len(pred_files):
        raise ValueError(f"Ground-truth/prediction frame-count mismatch: {len(gt_files)} vs {len(pred_files)}")
    rows, psnrs, ssims = [], [], []
    for i, (g, p) in enumerate(zip(gt_files, pred_files)):
        gt, pred = read_rgb(g), read_rgb(p)
        if crop_to_pred:
            gt = center_crop(gt, pred.shape[0], pred.shape[1])
        pval = psnr(pred, gt)
        sval = ssim_global_rgb(pred, gt)
        rows.append({"frame": i, "psnr": pval, "ssim": sval})
        psnrs.append(pval)
        ssims.append(sval)
    return rows, float(np.mean(psnrs)), float(np.mean(ssims))


def score(args):
    cfg = load_cfg(args)
    gt_files = list_images(Path(args.orig_data) / args.sequence)
    pred_files = list_images(recon_root(args, cfg))

    rows, avg_psnr, avg_ssim = score_folder(gt_files, pred_files, crop_to_pred=True)
    out_csv = recon_root(args, cfg) / "metrics.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"frame": "average", "psnr": avg_psnr, "ssim": avg_ssim})
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.6f}")
    print(f"Saved {out_csv}")


def find_latest_csv(base: Path):
    csvs = list(base.glob("**/*.csv"))
    if not csvs:
        return None
    epoch_csvs = [p for p in csvs if re.search(r"epoch\d+\.csv$", p.name)]
    if epoch_csvs:
        epoch_csvs.sort(key=lambda p: int(re.search(r"epoch(\d+)\.csv$", p.name).group(1)))
        return epoch_csvs[-1]
    eval_csvs = [p for p in csvs if p.name == "eval.csv"]
    if eval_csvs:
        eval_csvs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return eval_csvs[0]
    csvs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0]


def score_parts(args):
    cfg = load_cfg(args)
    rows_out = []
    for band in cfg["bands_list"]:
        gt_dir = data_root(args, cfg) / f"{args.sequence}_{band}"
        pred_dir = pred_root(args, cfg) / f"{args.sequence}_{band}"
        gt_files = list_images(gt_dir)
        pred_files = list_images(pred_dir)
        _, avg_psnr, avg_ssim = score_folder(gt_files, pred_files, crop_to_pred=False)
        rows_out.append({"part": band, "part_psnr": avg_psnr, "part_ssim": avg_ssim})
        print(f"{band}: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.6f}")

    out_csv = run_root(args, cfg) / f"part_metrics_{args.prediction_variant}.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["part", "part_psnr", "part_ssim"])
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Saved {out_csv}")
    return rows_out


def summarize(args):
    cfg = load_cfg(args)
    mode = cfg["mode"]
    part_rows = score_parts(args)
    part_map = {r["part"]: r for r in part_rows}

    total_encoder_m = 0.0
    total_decoder_m = 0.0
    total_playback_m = 0.0
    total_gflops = 0.0
    total_bpp = 0.0
    rows = []

    for band in cfg["bands_list"]:
        base = hnerv_output_root(args, cfg) / f"{args.sequence}_{band}"
        csv_path = find_latest_csv(base)
        row = {}
        if csv_path is not None:
            with csv_path.open(newline="") as f:
                csv_rows = list(csv.DictReader(f))
                row = csv_rows[0] if csv_rows else {}
        size_str = row.get("Size (M)", "0_0_0")
        try:
            encoder_m, decoder_m, total_m = [float(x) for x in size_str.split("_")]
        except Exception:
            encoder_m = decoder_m = total_m = 0.0
        gflops = float(row.get("Decoder_GFLOPs", -1) or -1)
        bpp = float(row.get("bits/pixel", 0) or 0)  # legacy estimate, not archive bytes

        total_encoder_m += encoder_m
        total_decoder_m += decoder_m
        total_playback_m += total_m
        if gflops > 0:
            total_gflops += gflops
        total_bpp += bpp

        rows.append({
            "part": band,
            "modelsize_target_m": cfg["bands"][band]["modelsize"],
            "rank": cfg["bands"][band]["rank"],
            "encoder_m": encoder_m,
            "decoder_m": decoder_m,
            "playback_total_m": total_m,
            "decoder_gflops": gflops,
            "bpp": bpp,
            "quant": row.get("Quant", ""),
            "prune_ratio": row.get("Prune_ratio", ""),
            "pruned_weights": row.get("Pruned_weights", ""),
            "prunable_weights": row.get("Prunable_weights", ""),
            "hnerv_pred_seen_psnr": row.get("pred_seen_psnr", ""),
            "hnerv_pred_seen_ssim": row.get("pred_seen_ssim", ""),
            "hnerv_quant_seen_psnr": row.get("quant_seen_psnr", ""),
            "hnerv_quant_seen_ssim": row.get("quant_seen_ssim", ""),
            "part_psnr": part_map[band]["part_psnr"],
            "part_ssim": part_map[band]["part_ssim"],
            "csv_path": str(csv_path) if csv_path else "",
        })

    full_metrics = recon_root(args, cfg) / "metrics.csv"
    final_psnr = ""
    final_ssim = ""
    if full_metrics.exists():
        with full_metrics.open(newline="") as f:
            mrows = list(csv.DictReader(f))
        if mrows:
            final_psnr = mrows[-1].get("psnr", "")
            final_ssim = mrows[-1].get("ssim", "")

    out_csv = run_root(args, cfg) / f"summary_total_{args.prediction_variant}.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["summary_key", "summary_value"])
        writer.writerow(["mode", mode])
        writer.writerow(["sequence", args.sequence])
        writer.writerow(["tag", args.tag])
        writer.writerow(["prediction_variant", args.prediction_variant])
        writer.writerow(["final_full_psnr", final_psnr])
        writer.writerow(["final_full_ssim", final_ssim])
        writer.writerow(["sum_encoder_m", total_encoder_m])
        writer.writerow(["sum_decoder_m", total_decoder_m])
        writer.writerow(["sum_playback_total_m", total_playback_m])
        writer.writerow(["sum_decoder_gflops", total_gflops])
        writer.writerow(["sum_bpp", total_bpp])  # retained for historical CSV readers
        writer.writerow(["bpp_protocol", "legacy_sum_of_tile_estimates_not_measured_packed_bpp"])
        writer.writerow(["quant_model_bit", cfg.get("quant_model_bit", "")])
        writer.writerow(["quant_embed_bit", cfg.get("quant_embed_bit", "")])
        writer.writerow(["prune_ratio", cfg.get("prune_ratio", "")])
        writer.writerow([])
        header = list(rows[0].keys()) if rows else []
        writer.writerow(header)
        for r in rows:
            writer.writerow([r[h] for h in header])

    print(f"Saved summary: {out_csv}")
    print(f"Total playback params: {total_playback_m:.4f}M")
    print(f"Total decoder GFLOPs: {total_gflops:.4f}")
    print(f"Final full-frame PSNR: {final_psnr}")
    print(f"Final full-frame SSIM: {final_ssim}")


def clean(args):
    cfg = load_cfg(args)
    rr = run_root(args, cfg)
    if rr.exists():
        print(f"Removing {rr}")
        shutil.rmtree(rr)


def debug(args):
    if args.clean:
        clean(args)
    prepare(args)
    train(args)
    extract(args)
    check(args)
    reconstruct(args)
    score(args)
    summarize(args)


def parse_args():
    project_root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="HNeRV LR-wide DWT/quadrant/16-tile experiments")
    p.add_argument("command", choices=["prepare", "train", "extract", "check", "reconstruct", "score", "score_parts", "summarize", "debug", "clean"])
    p.add_argument("--hnerv_repo", default=str(project_root / "HNeRV-main"))
    p.add_argument("--orig_data", default=str(project_root.parent.parent / "data"))
    p.add_argument("--work_root", default=str(project_root / "runs"))
    p.add_argument("--sequence", default="bunny")
    p.add_argument("--tag", default="debug_2ep")
    p.add_argument("--config", default=str(project_root / "configs" / "quadrant_equal_lrwide.json"))
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gpu", default="0")
    p.add_argument("--manual_seed", type=int, default=1)
    p.add_argument("--eval_freq", type=int, default=1)
    p.add_argument("--expected_frames", type=int, default=132)
    p.add_argument("--full_height", type=int, default=640)
    p.add_argument("--full_width", type=int, default=1280)
    p.add_argument("--prediction_variant", choices=["orig", "quant"], default="quant",
                   help="Reconstruct from visualize_model_quant or visualize_model_orig. Default is quant for Q8/P10 reporting.")
    p.add_argument("--clean", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.command == "debug" and run_root(args, load_cfg(args)).exists() and not args.clean:
        raise FileExistsError("Run directory already exists; choose a new --tag. Existing experiment outputs were preserved.")
    if args.command == "prepare":
        prepare(args)
    elif args.command == "train":
        train(args)
    elif args.command == "extract":
        extract(args)
    elif args.command == "check":
        check(args)
    elif args.command == "reconstruct":
        reconstruct(args)
    elif args.command == "score":
        score(args)
    elif args.command == "score_parts":
        score_parts(args)
    elif args.command == "summarize":
        summarize(args)
    elif args.command == "debug":
        debug(args)
    elif args.command == "clean":
        clean(args)


if __name__ == "__main__":
    main()
