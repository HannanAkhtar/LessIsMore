# Code provenance and entry points

`src/` and `run_wholeframe_lrwide.py` came from the whole-frame low-rank HNeRV code. `spatial/HNeRV-main/` adapts the [official HNeRV reference implementation](https://github.com/haochen-rye/HNeRV), retaining its authors' README and attribution. `spatial/run_experiment.py` wraps that model with independent spatial tiles. `run_framework.py` dispatches the two implementations without changing their checkpoint tensor layouts.

The two decoder implementations and historical dependency files are distinct. See [README.md](README.md) for supported commands, [docs/RELEASE_AUDIT.md](docs/RELEASE_AUDIT.md) for the release audit, and [docs/PAPER_REPRODUCTION.md](docs/PAPER_REPRODUCTION.md) for manuscript mapping. Prior lab notes, a pre-patch source copy, and cluster scripts are retained under `docs/legacy/` for traceability.
