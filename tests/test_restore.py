"""Portable restore contract; generated files/models are engineering fixtures."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import stat
import zipfile

import numpy as np
import pytest
import torch

from umud_attention.restore import restore_model_bundle


def payload(prefix="saved_run", tasks=("apo", "fasc"), last=False):
    files = {}
    for name in ("train_manifest.csv", "test_manifest.csv", "data_audit.json"):
        files[f"{prefix}/{name}"] = b"{}" if name.endswith("json") else b"image_path\n/content/original/data.tif\n"
    for task in tasks:
        folder = f"{prefix}/checkpoints/{task}_fold0"
        files[f"{folder}/best.pt"] = b"opaque fixture: restore must not deserialize this"
        for name in ("config.json", "report.json"):
            files[f"{folder}/{name}"] = json.dumps({"task": task, "fold": 0}).encode()
        if last:
            files[f"{folder}/last.pt"] = b"optional resume checkpoint fixture"
    return files


def make_zip(path, files):
    with zipfile.ZipFile(path, "w") as zipped:
        for name, data in files.items():
            zipped.writestr(name, data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_restore_preserves_metadata_paths_and_hashes_without_loading_weights(tmp_path):
    files = payload(last=True)
    archive = tmp_path / "model_bundle.zip"
    digest = make_zip(archive, files)
    restored = restore_model_bundle(archive, tmp_path / "restored", expected_sha256=digest)
    assert len(restored["checkpoints"]) == len(restored["last_checkpoints"]) == 2
    assert restored["all_last_checkpoints_present"] is True
    assert Path(restored["train_manifest"]).read_bytes() == files["saved_run/train_manifest.csv"]
    receipt = json.loads(Path(restored["receipt"]).read_text())
    assert not receipt["data_paths_rewritten"] and not receipt["training_performed"]
    assert not receipt["checkpoint_deserialization_performed"]
    for name, data in files.items():
        assert receipt["files"][name]["sha256"] == hashlib.sha256(data).hexdigest()
    with pytest.raises(FileExistsError, match="already exists"):
        restore_model_bundle(archive, tmp_path / "restored", expected_sha256=digest)


def test_checksum_mismatch_extracts_nothing(tmp_path):
    archive = tmp_path / "bundle.zip"
    make_zip(archive, payload())
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        restore_model_bundle(archive, tmp_path / "out", expected_sha256="0" * 64)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "folder/../../escape", "C:/escape", "folder\\escape"])
def test_unsafe_archive_paths_leave_no_partial_restore(tmp_path, name):
    archive = tmp_path / "bundle.zip"
    digest = make_zip(archive, {**payload(), name: b"forbidden"})
    with pytest.raises(ValueError, match="Unsafe"):
        restore_model_bundle(archive, tmp_path / "out", expected_sha256=digest)
    assert not (tmp_path / "out").exists() and not list(tmp_path.glob(".umud_restore_*"))


def test_symlink_and_duplicate_entries_are_rejected(tmp_path):
    for kind in ("symlink", "duplicate"):
        archive = tmp_path / f"{kind}.zip"
        make_zip(archive, payload())
        with zipfile.ZipFile(archive, "a") as zipped:
            if kind == "symlink":
                info = zipfile.ZipInfo("link")
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                zipped.writestr(info, "../escape")
            else:
                with pytest.warns(UserWarning, match="Duplicate"):
                    zipped.writestr("saved_run/data_audit.json", "{}")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        with pytest.raises(ValueError, match="Unsafe|Duplicate"):
            restore_model_bundle(archive, tmp_path / kind, expected_sha256=digest)
        assert not (tmp_path / kind).exists()


def test_unpaired_checkpoints_and_size_limit_fail_before_publish(tmp_path):
    archive = tmp_path / "bundle.zip"
    digest = make_zip(archive, payload(tasks=("apo",)))
    with pytest.raises(ValueError, match="matching apo/fasc"):
        restore_model_bundle(archive, tmp_path / "out", expected_sha256=digest)
    with pytest.raises(ValueError, match="size/member limit"):
        restore_model_bundle(archive, tmp_path / "out", expected_sha256=digest, max_uncompressed_bytes=1)
    assert not (tmp_path / "out").exists()


def test_restored_checkpoint_runs_inference_without_training_or_pretrained_download(tmp_path, monkeypatch):
    from umud_attention.infer import Predictor
    import umud_attention.infer as inference
    from umud_attention.train import TrainConfig, model_from_config
    import umud_attention.train as training
    torch.set_num_threads(2)
    files = payload()
    # Real serialized model structure; random weights, not a trained-data claim.
    for task in ("apo", "fasc"):
        config = asdict(TrainConfig(task=task, size=64, encoder="resnet18", decoder_width=8,
                                   pretrained=False, device="cpu", workers=0))
        model = model_from_config(config)
        path = tmp_path / f"{task}.pt"
        torch.save({"config": config, "model": model.state_dict(), "threshold": .5,
                    "manifest_sha256": "engineering-fixture", "best_dice": None}, path)
        files[f"saved_run/checkpoints/{task}_fold0/best.pt"] = path.read_bytes()
        files[f"saved_run/checkpoints/{task}_fold0/config.json"] = json.dumps(config).encode()
    archive = tmp_path / "bundle.zip"
    digest = make_zip(archive, files)
    restored = restore_model_bundle(archive, tmp_path / "restored", expected_sha256=digest)
    def forbidden(*args, **kwargs):
        raise AssertionError("Restore/inference must not train or fetch pretrained weights")
    monkeypatch.setattr(training, "train_fold", forbidden)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden)
    original_factory = inference.model_from_config
    def saved_weights_only(config, pretrained=False):
        assert pretrained is False
        return original_factory(config, pretrained=False)
    monkeypatch.setattr(inference, "model_from_config", saved_weights_only)
    predictor = Predictor(restored["checkpoints"], device="cpu")
    image = np.random.default_rng(3).integers(0, 255, (41, 77, 3), dtype=np.uint8)
    masks = predictor.predict(image, horizontal_tta=False)
    assert set(masks) == {"apo", "fasc"}
    assert all(m.shape == (41, 77) and np.isfinite(m).all() for m in masks.values())
