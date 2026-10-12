from pathlib import Path
import json

src = Path("configs/tile16_equal_3m_lrwide.json")
dst = Path("configs/tile16_equal_3m_exact_v2_lrwide.json")

cfg = json.loads(src.read_text())

NEW_MODELSIZE = 0.198
NEW_RANK = 23

def patch(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.startswith("T") and isinstance(v, dict):
                for key in ["modelsize", "modelsize_target_m"]:
                    if key in v:
                        v[key] = NEW_MODELSIZE
                for key in ["rank", "lowrank_rank"]:
                    if key in v:
                        v[key] = NEW_RANK

        part_name = str(obj.get("part", obj.get("band", obj.get("name", ""))))
        if part_name.startswith("T"):
            for key in ["modelsize", "modelsize_target_m"]:
                if key in obj:
                    obj[key] = NEW_MODELSIZE
            for key in ["rank", "lowrank_rank"]:
                if key in obj:
                    obj[key] = NEW_RANK

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

print(f"Wrote {dst}")
print(f"Per-tile modelsize target: {NEW_MODELSIZE}M")
print(f"Rank: {NEW_RANK}")
