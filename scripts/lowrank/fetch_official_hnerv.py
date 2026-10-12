from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone the official HNeRV implementation for reference only.")
    parser.add_argument("--out_dir", type=str, default="official_HNeRV")
    parser.add_argument("--repo", type=str, default="https://github.com/haochen-rye/HNeRV")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        print(f"{out_dir} already exists; skipping clone.")
        return
    subprocess.run(["git", "clone", args.repo, str(out_dir)], check=True)


if __name__ == "__main__":
    main()
