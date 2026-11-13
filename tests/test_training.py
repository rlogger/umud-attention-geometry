"""Tiny generated fixtures verify integration/resume, never real-data accuracy."""

from dataclasses import replace
import hashlib
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from umud_attention.data import assign_folds, discover_dataset
from umud_attention.train import TrainConfig, train_fold


def _save(path, pixels):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


@pytest.fixture(scope="module")
def engineering_run(tmp_path_factory):
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("generated_training_contract")
    for task_index, task in enumerate(("apo", "fasc")):
        for i in range(6):
            rng = np.random.default_rng(100 * task_index + i)
            pixels = rng.integers(0, 90, (48, 80), dtype=np.uint8)
            mask = np.zeros((24, 40), dtype=np.uint8)
            mask[7:9, 4:36] = 255
            mask[16:18, 4:36] = 255
            pixels[14:18, 8:72] = 200
            pixels[32:36, 8:72] = 200
            if task == "apo":
                mask = 255 - mask
            _save(root / f"{task}_imgs_v1" / f"fixture_{i}.png", pixels)
            _save(root / f"{task}_masks_v1" / f"fixture_{i}.png", mask)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        train, _ = discover_dataset(root)
        manifest = assign_folds(train, n_folds=3, seed=12)
    path = root / "manifest.csv"
    manifest.to_csv(path, index=False)
    config = TrainConfig(task="apo", fold=0, epochs=1, size=64, batch_size=2,
                         accumulation=2, encoder="resnet18", decoder_width=8,
                         pretrained=False, device="cpu", workers=0, amp=False,
                         seed=12, mask_polarity="dark")
    output = root / "run"
    report = train_fold(path, output, config)
    return {"root": root, "manifest": path, "output": output,
            "config": config, "report": report}


def test_real_dataset_dict_trains_and_checkpoint_roundtrips(engineering_run):
    case = engineering_run
    report = case["report"]
    assert report["train_images"] == 4 and report["validation_images"] == 2
    assert np.isfinite(report["best_validation_dice"])
    checkpoint = Path(report["checkpoint"])
    assert checkpoint.name == "best.pt" and checkpoint.is_file()
    best = torch.load(checkpoint, map_location="cpu", weights_only=True)
    last = torch.load(checkpoint.with_name("last.pt"), map_location="cpu", weights_only=True)
    assert best["epoch"] == last["epoch"] == 0
    assert len(last["history"]) == 1
    assert np.isfinite(last["history"][0]["train_loss"])
    assert np.isfinite(last["history"][0]["val_loss"])
    assert "optimizer" in last and "optimizer" not in best
    assert "NOT UMUD" in report["metric_scope"]


def test_completed_resume_does_not_train_or_rewrite_checkpoint(engineering_run):
    case = engineering_run
    checkpoint = Path(case["report"]["checkpoint"]).with_name("last.pt")
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = train_fold(case["manifest"], case["output"], case["config"], resume=True)
    assert report["best_validation_dice"] == case["report"]["best_validation_dice"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert len(state["history"]) == 1


def test_changed_manifest_cannot_resume(engineering_run):
    case = engineering_run
    changed = pd.read_csv(case["manifest"])
    changed["review_note"] = "changed manifest"
    with pytest.raises(ValueError, match="Manifest changed"):
        train_fold(changed, case["output"], case["config"], resume=True)


def test_changed_mask_pixels_cannot_resume_same_manifest(engineering_run):
    case = engineering_run
    rows = pd.read_csv(case["manifest"])
    path = Path(rows[rows.task == "apo"].iloc[0].mask_path)
    original_bytes = path.read_bytes()
    try:
        pixels = np.array(Image.open(path))
        pixels[0, 0] = 255 - pixels[0, 0]
        Image.fromarray(pixels).save(path)
        with pytest.raises(ValueError, match="(?i)(mask|preprocess|content).*(changed|differ)|(?i:changed).*(mask|preprocess|content)"):
            train_fold(case["manifest"], case["output"], case["config"], resume=True)
    finally:
        path.write_bytes(original_bytes)


def test_existing_run_does_not_silently_restart(engineering_run):
    case = engineering_run
    with pytest.raises(ValueError, match="(?i)(existing|checkpoint|resume|new output)"):
        train_fold(case["manifest"], case["output"], case["config"], resume=False)


def test_second_task_plain_control_uses_only_its_labels(engineering_run):
    case = engineering_run
    config = replace(case["config"], task="fasc", attention=False, mask_polarity="light")
    report = train_fold(case["manifest"], case["root"] / "plain_control", config)
    assert report["task"] == "fasc"
    assert report["train_images"] == 4 and report["validation_images"] == 2
    state = torch.load(report["checkpoint"], map_location="cpu", weights_only=True)
    assert state["config"]["attention"] is False
    assert np.isfinite(state["history"][0]["train_loss"])


def test_bounded_pilot_resumes_same_planned_configuration(engineering_run):
    case = engineering_run
    config = replace(case["config"], epochs=2, patience=10)
    output = case["root"] / "bounded_pilot"
    first = train_fold(case["manifest"], output, config, max_epochs_this_call=1)
    last_path = Path(first["checkpoint"]).with_name("last.pt")
    state = torch.load(last_path, map_location="cpu", weights_only=True)
    assert state["epoch"] == 0 and len(state["history"]) == 1
    assert state["config"]["epochs"] == 2
    train_fold(case["manifest"], output, config, max_epochs_this_call=1)
    resumed = torch.load(last_path, map_location="cpu", weights_only=True)
    assert resumed["epoch"] == 1 and len(resumed["history"]) == 2
    assert resumed["history"][0] == state["history"][0]
    assert resumed["config"] == state["config"]
    before = hashlib.sha256(last_path.read_bytes()).hexdigest()
    train_fold(case["manifest"], output, config, max_epochs_this_call=1)
    assert hashlib.sha256(last_path.read_bytes()).hexdigest() == before
