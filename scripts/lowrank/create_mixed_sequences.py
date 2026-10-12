from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import Iterable

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def natural_key(path: Path):
    import re
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def list_frames(folder: Path):
    frames = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    frames.sort(key=natural_key)
    return frames


def find_source_sequences(data_root: Path, sources: Iterable[str] | None = None, exclude_prefix: str = "mixed_"):
    source_set = set(sources) if sources else None
    seqs = []
    for p in sorted(data_root.iterdir()):
        if not p.is_dir():
            continue
        if exclude_prefix and p.name.startswith(exclude_prefix):
            continue
        if source_set is not None and p.name not in source_set:
            continue
        if list_frames(p):
            seqs.append(p)
    if not seqs:
        raise RuntimeError(f"No source sequences found under {data_root}")
    return seqs


def split_lengths(total_frames: int, n_segments: int, rng: random.Random, balanced_shuffle: bool = True) -> list[int]:
    """Return n segment lengths that sum to total_frames and differ by at most 1."""
    if n_segments <= 0:
        raise ValueError("n_segments must be positive")
    if total_frames < n_segments:
        raise ValueError(
            f"frames_per_output={total_frames} is smaller than the number of source videos={n_segments}. "
            "Increase frames_per_output or reduce --sources."
        )
    base = total_frames // n_segments
    rem = total_frames % n_segments
    lengths = [base + 1] * rem + [base] * (n_segments - rem)
    if balanced_shuffle:
        rng.shuffle(lengths)
    return lengths


def sample_random_frames(rng: random.Random, seqs: list[Path], n: int):
    chosen = []
    frame_cache = {s: list_frames(s) for s in seqs}
    for _ in range(n):
        s = rng.choice(seqs)
        f = rng.choice(frame_cache[s])
        chosen.append({"source_sequence": s, "source_frame": f, "segment_index": -1, "segment_len": 1})
    return chosen


def sample_random_segments(rng: random.Random, seqs: list[Path], n: int, segment_len: int):
    chosen = []
    frame_cache = {s: list_frames(s) for s in seqs}
    prev = None
    seg_idx = 0
    while len(chosen) < n:
        available = [s for s in seqs if s != prev] or seqs
        s = rng.choice(available)
        frames = frame_cache[s]
        seg = min(segment_len, n - len(chosen), len(frames))
        start = rng.randint(0, max(0, len(frames) - seg))
        for f in frames[start:start + seg]:
            chosen.append({"source_sequence": s, "source_frame": f, "segment_index": seg_idx, "segment_len": seg})
            if len(chosen) >= n:
                break
        prev = s
        seg_idx += 1
    return chosen


def sample_one_segment_per_source(rng: random.Random, seqs: list[Path], n: int, shuffle_sources: bool = True):
    """
    Create one mixed sequence where each source video appears exactly once as a single contiguous segment.

    Example for 6 source videos and 132 output frames:
      lengths are six 22-frame contiguous segments.
      each source contributes exactly one segment, and the order of segments is shuffled per output.
    """
    if n < len(seqs):
        raise ValueError(
            f"frames_per_output={n} is smaller than number of source videos={len(seqs)}. "
            "Need at least one frame from each source."
        )
    frame_cache = {s: list_frames(s) for s in seqs}
    segment_order = list(seqs)
    if shuffle_sources:
        rng.shuffle(segment_order)

    lengths = split_lengths(n, len(segment_order), rng, balanced_shuffle=True)
    chosen = []
    for seg_idx, (s, seg_len) in enumerate(zip(segment_order, lengths)):
        frames = frame_cache[s]
        if len(frames) < seg_len:
            raise ValueError(f"Source sequence {s} has only {len(frames)} frames, but segment length {seg_len} was requested.")
        start = rng.randint(0, len(frames) - seg_len)
        segment_frames = frames[start:start + seg_len]
        for f in segment_frames:
            chosen.append({"source_sequence": s, "source_frame": f, "segment_index": seg_idx, "segment_len": seg_len})
    assert len(chosen) == n
    return chosen


def main():
    parser = argparse.ArgumentParser(description="Create mixed-scene sequences from existing frame folders.")
    parser.add_argument("--data_root", type=str, required=True, help="Folder containing source sequences, e.g. ..\\data")
    parser.add_argument("--out_root", type=str, default=None, help="Where mixed sequences are created. Default: data_root")
    parser.add_argument("--sources", type=str, nargs="*", default=None, help="Optional source folder names. Default: all non-mixed frame folders.")
    parser.add_argument("--num_outputs", type=int, default=6)
    parser.add_argument("--frames_per_output", type=int, default=132)
    parser.add_argument(
        "--mode",
        type=str,
        default="one_segment_per_source",
        choices=["one_segment_per_source", "random_segments", "random_frames"],
        help=(
            "one_segment_per_source: each source video contributes exactly one contiguous segment to each mixed sequence. "
            "random_segments: older behavior, repeatedly samples segments from random sources. "
            "random_frames: samples independent frames."
        ),
    )
    parser.add_argument("--segment_len", type=int, default=11, help="Used only for random_segments.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--prefix", type=str, default="mixed_")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_shuffle_sources", action="store_true", help="For one_segment_per_source, keep source-folder order instead of shuffling segments.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data_root = Path(args.data_root)
    out_root = Path(args.out_root) if args.out_root else data_root
    out_root.mkdir(parents=True, exist_ok=True)
    seqs = find_source_sequences(data_root, args.sources, exclude_prefix=args.prefix)
    print("Source sequences:", ", ".join(s.name for s in seqs))

    for i in range(args.num_outputs):
        out_dir = out_root / f"{args.prefix}{i:02d}"
        if out_dir.exists():
            if args.overwrite:
                shutil.rmtree(out_dir)
            else:
                raise FileExistsError(f"{out_dir} exists. Use --overwrite to replace it.")
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.mode == "random_frames":
            chosen = sample_random_frames(rng, seqs, args.frames_per_output)
        elif args.mode == "random_segments":
            chosen = sample_random_segments(rng, seqs, args.frames_per_output, args.segment_len)
        else:
            chosen = sample_one_segment_per_source(
                rng,
                seqs,
                args.frames_per_output,
                shuffle_sources=not args.no_shuffle_sources,
            )

        manifest_rows = []
        for j, item in enumerate(chosen):
            src_seq = item["source_sequence"]
            src_frame = item["source_frame"]
            dst = out_dir / f"frame_{j:04d}{src_frame.suffix.lower()}"
            shutil.copy2(src_frame, dst)
            manifest_rows.append({
                "output_frame": dst.name,
                "source_sequence": src_seq.name,
                "source_frame": src_frame.name,
                "segment_index": item["segment_index"],
                "segment_len": item["segment_len"],
                "source_path": str(src_frame),
            })

        with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"Created {out_dir} with {len(chosen)} frames using mode={args.mode}")


if __name__ == "__main__":
    main()
