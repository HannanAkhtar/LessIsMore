"""Select naturally ordered RGB frames, crop/validate them, and record hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from PIL import Image


def natural_key(path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(source, output, *, count, start_index, height, width, mode):
    source, output = Path(source), Path(output)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    frames = sorted((p for p in source.iterdir() if p.suffix.lower() in
                     {".png", ".jpg", ".jpeg", ".bmp", ".webp"}), key=natural_key)
    selected = frames[start_index:start_index + count]
    if count <= 0 or start_index < 0 or len(selected) != count:
        raise ValueError(f"Requested {count} frames from index {start_index}, found {len(selected)}")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, path in enumerate(selected):
        image = Image.open(path).convert("RGB")
        if mode == "crop":
            if image.height < height or image.width < width:
                raise ValueError(f"Cannot center-crop {path} to {height}x{width}")
            left = (image.width - width) // 2
            top = (image.height - height) // 2
            image = image.crop((left, top, left + width, top + height))
        elif image.size != (width, height):
            raise ValueError(f"Native frame size mismatch for {path}: {image.size}")
        destination = output / f"{index:06d}.png"
        image.save(destination)
        records.append({"source": path.name, "source_sha256": digest(path),
                        "output": destination.name, "output_sha256": digest(destination)})
    manifest = {"source_directory": str(source.resolve()), "output_directory": str(output.resolve()),
                "start_index": start_index, "count": count, "height": height, "width": width,
                "mode": mode, "frames": records}
    (output / "frame_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=132)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--mode", choices=["crop", "native"], default="crop")
    args = parser.parse_args()
    prepare(args.source, args.output, count=args.count, start_index=args.start_index,
            height=args.height, width=args.width, mode=args.mode)


if __name__ == "__main__":
    main()
