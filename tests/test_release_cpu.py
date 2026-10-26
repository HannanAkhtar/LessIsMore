import json
from pathlib import Path

import numpy as np
import pytest

from src.packed_archive import (dequantize, pack_symbols, packed_bpp,
                                read_archive, unpack_symbols, write_archive)
from spatial.run_experiment import (load_cfg, merge_quadrants, merge_tiles,
                                    split_quadrants, split_tiles, prepare,
                                    reconstruct, data_root, pred_root, recon_root)


def test_six_bit_symbols_and_measured_archive_roundtrip(tmp_path):
    symbols = np.array([0, 1, 63, 7, 34, 17, 5], dtype=np.uint8)
    assert np.array_equal(unpack_symbols(pack_symbols(symbols, 6), len(symbols), 6), symbols)
    pack = {"quant": symbols.reshape(1, 7, 1, 1),
            "min": np.array(-0.5, dtype=np.float32),
            "scale": np.array(0.25, dtype=np.float32)}
    weight = {"quant": np.array([0, 255], dtype=np.uint8),
              "min": np.array(0, dtype=np.float16),
              "scale": np.array(0.01, dtype=np.float16)}
    path = tmp_path / "playback.zip"
    bpp = write_archive(path, tiles={"Q0": {
        "architecture": {"kind": "synthetic"}, "weights": {"layer.weight": weight},
        "embeddings": pack,
    }}, frames=1, height=2, width=7, layout={"mode": "synthetic"},
        model_bits=8, embedding_bits=6)
    loaded = read_archive(path)
    assert bpp == packed_bpp(path, 1, 2, 7)
    assert loaded["format"] == "lessismore-packed-v1"
    assert np.array_equal(loaded["tiles"]["Q0"]["embeddings"]["quant"], pack["quant"])
    assert np.allclose(dequantize(loaded["tiles"]["Q0"]["embeddings"]),
                       dequantize(pack))
    assert np.array_equal(loaded["tiles"]["Q0"]["weights"]["layer.weight"]["quant"],
                          weight["quant"])


def test_symbol_validation():
    with pytest.raises(ValueError):
        pack_symbols([64], 6)
    with pytest.raises(ValueError):
        unpack_symbols(b"", 2, 6)


def test_spatial_splitting_and_stitching():
    frame = np.arange(8 * 12 * 3, dtype=np.float32).reshape(8, 12, 3)
    assert np.array_equal(merge_quadrants(split_quadrants(frame)), frame)
    assert np.array_equal(merge_tiles(split_tiles(frame)), frame)


def test_release_configs_and_rank_provenance():
    root = Path(__file__).resolve().parents[1]
    historical = json.loads((root / "spatial/configs/tile16_equal_3m_lrwide.json").read_text())
    candidate = json.loads((root / "spatial/configs/tile16_rank23_0p198_unverified.json").read_text())
    dense = json.loads((root / "spatial/configs/quadrant_dense_3m_reference.json").read_text())
    assert {x["rank"] for x in historical["bands"].values()} == {22}
    assert {x["rank"] for x in candidate["bands"].values()} == {23}
    assert dense["lowrank_blocks"] == ""
    assert len(dense["bands"]) == 4
    from types import SimpleNamespace
    assert load_cfg(SimpleNamespace(config=root / "spatial/configs/quadrant_equal_lrwide.json"))["mode"] == "quadrant"


def test_spatial_preparation_padding_order_and_stitch(tmp_path):
    import shutil
    from types import SimpleNamespace
    from PIL import Image

    source = tmp_path / "source" / "demo"
    source.mkdir(parents=True)
    images = {}
    for name, shift in (("frame10.png", 9), ("frame2.png", 3)):
        image = (np.arange(10 * 18 * 3).reshape(10, 18, 3) + shift).astype(np.uint8)
        Image.fromarray(image).save(source / name)
        images[name] = image
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"mode": "quadrant", "crop_list": "6_10",
                                    "enc_strds": [2], "bands": {f"Q{i}": {"modelsize": 1, "rank": 3} for i in range(4)},
                                    "lowrank_blocks": "3,4"}))
    args = SimpleNamespace(config=cfg_path, orig_data=tmp_path / "source",
                           work_root=tmp_path / "runs", sequence="demo", tag="test",
                           full_height=10, full_width=18, expected_frames=2,
                           prediction_variant="quant")
    prepare(args)
    cfg = load_cfg(args)
    metadata = json.loads((data_root(args, cfg) / "demo_quadrant_meta.json").read_text())
    assert metadata["source_frames"] == ["frame2.png", "frame10.png"]
    assert metadata["valid_tile_height"] == 5
    assert metadata["padded_tile_height"] == 6
    for band in cfg["bands"]:
        src = data_root(args, cfg) / f"demo_{band}"
        dst = pred_root(args, cfg) / f"demo_{band}"
        dst.mkdir(parents=True)
        for frame in src.glob("*.png"):
            shutil.copy2(frame, dst / frame.name)
        assert Image.open(src / "000000.png").size == (10, 6)
    reconstruct(args)
    assert np.array_equal(np.asarray(Image.open(recon_root(args, cfg) / "000000.png")), images["frame2.png"])


def test_frame_selection_manifest_and_no_overwrite(tmp_path):
    from PIL import Image
    from scripts.prepare_frames import prepare as select_frames

    source, output = tmp_path / "raw", tmp_path / "selected"
    source.mkdir()
    for number in (10, 2):
        Image.fromarray(np.full((6, 8, 3), number, dtype=np.uint8)).save(source / f"frame{number}.png")
    manifest = select_frames(source, output, count=2, start_index=0,
                             height=4, width=8, mode="crop")
    assert [item["source"] for item in manifest["frames"]] == ["frame2.png", "frame10.png"]
    assert Image.open(output / "000000.png").size == (8, 4)
    assert len(manifest["frames"][0]["output_sha256"]) == 64
    with pytest.raises(FileExistsError):
        select_frames(source, output, count=2, start_index=0,
                      height=4, width=8, mode="crop")


def test_torch_models_and_checkpoint_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from src.models import HNeRV, LowRankHNeRV
    from src.quant_eval import quant_tensor_like_hnerv

    kwargs = dict(c1=8, emb_dim=4, enc_strides=[2, 2], dec_strides=[2, 2],
                  dec_width=8, reduce=1.2, kmin=1, kmax=3)
    x = torch.rand(1, 3, 16, 16)
    for cls, extra in ((HNeRV, {}), (LowRankHNeRV, {"lowrank_rank": 3, "lowrank_blocks": [1]})):
        model = cls(**kwargs, **extra).eval()
        with torch.no_grad():
            expected = model(x)
        path = tmp_path / (cls.__name__ + ".pt")
        torch.save(model.state_dict(), path)
        loaded = cls(**kwargs, **extra).eval()
        loaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        with torch.no_grad():
            assert torch.allclose(loaded(x), expected)
        assert expected.shape == x.shape
        from src.flops import count_model_flops
        macs = count_model_flops(model.decoder, tuple(model.encoder(x).shape), flop_mult=1)
        flops = count_model_flops(model.decoder, tuple(model.encoder(x).shape), flop_mult=2)
        assert macs > 0 and flops == 2 * macs
        from src.metrics import batch_psnr, batch_ssim, batch_publication_ssim
        assert batch_psnr(expected, expected) == 99.0
        assert batch_ssim(expected, expected) > 0.99
        assert batch_publication_ssim(expected, expected) > 0.99
    pack, restored = quant_tensor_like_hnerv(torch.linspace(-1, 1, 31), 6)
    assert pack["quant"].dtype == torch.uint8
    assert torch.max(torch.abs(restored - torch.linspace(-1, 1, 31))) < 0.04
