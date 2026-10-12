from __future__ import annotations

import argparse
from .args import add_common_args
from .experiment import run_single_sequence


def main():
    parser = argparse.ArgumentParser(description="Run baseline HNeRV and clustered expert HNeRV on one sequence.")
    add_common_args(parser)
    args = parser.parse_args()
    if args.data_path is None or args.out_dir is None:
        parser.error("--data_path and --out_dir are required for src.run_sequence")
    results = run_single_sequence(args)
    print("\nDone. Summary:")
    for name, r in results.items():
        print(f"{name}: PSNR={r['psnr']:.4f}, SSIM={r['ssim']:.5f}, BPP={r['bpp']:.6f}, params={r['decoder_params']}, GFLOPs={r['gflops_per_frame_decoder_only']:.4f}")


if __name__ == "__main__":
    main()
