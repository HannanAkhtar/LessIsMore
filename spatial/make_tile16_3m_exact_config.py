from pathlib import Path
import json

src = Path("configs/tile16_equal_3m_lrwide.json")
dst = Path("configs/tile16_equal_3m_exact_lrwide.json")

cfg = json.loads(src.read_text())

# Previous actual total playback params: 2.72M = 0.17M per tile.
# Target total playback params: ~3.00M = 0.1875M per tile.
# Raise the requested modelsize slightly because auto-widening underfilled before.
NEW_MODELSIZE = 0.207
NEW_RANK = 24

changed = 0

def patch(obj):
    global changed

    if isinstance(obj, dict):
        # Format: {"T00": {"modelsize": ..., "rank": ...}}
        for k, v in obj.items():
            if isinstance(k, str) and k.startswith("T") and isinstance(v, dict):
                if "modelsize" in v:
                    v["modelsize"] = NEW_MODELSIZE
                if "modelsize_target_m" in v:
                    v["modelsize_target_m"] = NEW_MODELSIZE
                if "rank" in v:
                    v["rank"] = NEW_RANK
                if "lowrank_rank" in v:
                    v["lowrank_rank"] = NEW_RANK
                changed += 1

        # Format: {"part": "T00", "modelsize": ..., "rank": ...}
        part_name = str(obj.get("part", obj.get("band", obj.get("name", ""))))
        if part_name.startswith("T"):
            if "modelsize" in obj:
                obj["modelsize"] = NEW_MODELSIZE
            if "modelsize_target_m" in obj:
                obj["modelsize_target_m"] = NEW_MODELSIZE
            if "rank" in obj:
                obj["rank"] = NEW_RANK
            if "lowrank_rank" in obj:
                obj["lowrank_rank"] = NEW_RANK
            changed += 1

        for v in obj.values():
            patch(v)

    elif isinstance(obj, list):
        for v in obj:
            patch(v)

patch(cfg)

cfg["quant_model_bit"] = 8
cfg["quant_embed_bit"] = 6
cfg["prune_ratio"] = 0.10

dst.write_text(json.dumps(cfg, indent=2))

print(f"Wrote: {dst}")
print(f"Patched tile entries: {changed}")
print(f"New per-tile target modelsize: {NEW_MODELSIZE}M")
print(f"New rank: {NEW_RANK}")
print("Expected total playback params: close to 3.0M. Verify after debug run.")
