from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from umud_attention.data import (IMAGE_MEAN, IMAGE_STD, SegmentationDataset,
                                 assign_folds, discover_dataset, letterbox,
                                 load_image)


def save(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    return path


def sample(root, task, name, pixels=None, dark=False, mask_shape=None):
    if pixels is None:
        pixels = np.random.default_rng(int(name)).integers(0, 255, (24, 40), dtype=np.uint8)
    mask = np.zeros(mask_shape or pixels.shape[:2], dtype=np.uint8)
    mask[mask.shape[0]//4:mask.shape[0]//3+1, 3:-3] = 255
    if dark:
        mask = 255 - mask
    image = save(root / f"{task}_imgs_v1" / "nested" / f"{name}.tif", pixels)
    target = save(root / f"{task}_masks_v1" / "nested" / f"{name}.png", mask)
    return image, target


def frame_for(image, mask, task="apo"):
    return pd.DataFrame([{"task": task, "image_id": image.name,
                          "image_path": str(image), "mask_path": str(mask)}])


def test_nested_discovery_pairs_tasks_without_inventing_other_labels(tmp_path):
    root = tmp_path / "download" / "competition"
    sample(root, "apo", "1", dark=True)
    sample(root, "fasc", "2")
    save(root / "test_images_v2" / "test_set_v2" / "first.tif", np.zeros((20, 30), np.uint8))
    save(root / "test_images_v2" / "test_set_v2" / "second.png", np.ones((20, 30), np.uint8))
    train, test = discover_dataset(tmp_path)
    assert train[["task", "image_id"]].values.tolist() == [["apo", "1.tif"], ["fasc", "2.tif"]]
    assert train.mask_path.notna().all() and len(train) == 2
    assert test.image_id.tolist() == ["first.tif", "second.png"]
    assert {"group", "fold", "mask_hash", "mask_light_fraction"} <= set(train)


def test_unpaired_images_fail_instead_of_becoming_background(tmp_path):
    sample(tmp_path, "apo", "1")
    save(tmp_path / "fasc_imgs_v1" / "unlabeled.tif", np.zeros((20, 30), np.uint8))
    with pytest.raises(ValueError, match="Unpaired fasc"):
        discover_dataset(tmp_path)


def test_ambiguous_stems_and_test_basenames_fail(tmp_path):
    sample(tmp_path, "apo", "1")
    duplicate = save(tmp_path / "apo_imgs_v1" / "other" / "1.png", np.zeros((20, 30), np.uint8))
    with pytest.raises(ValueError, match="Ambiguous duplicate stem"):
        discover_dataset(tmp_path)
    duplicate.unlink()
    save(tmp_path / "test_images_v2" / "a" / "same.png", np.zeros((20, 30), np.uint8))
    save(tmp_path / "test_images_v2" / "b" / "same.png", np.zeros((20, 30), np.uint8))
    with pytest.raises(ValueError, match="duplicate basename"):
        discover_dataset(tmp_path)


def test_mask_canvas_mismatch_is_audited_and_explicitly_supported(tmp_path):
    image, mask = sample(tmp_path, "apo", "1", mask_shape=(12, 20), dark=True)
    with pytest.warns(UserWarning, match="different canvas"):
        train, _ = discover_dataset(tmp_path)
    assert train.attrs["mask_canvas_mismatch_count"] == 1
    dataset = SegmentationDataset(train, size=80, mask_polarity="dark", mask_alignment="full_frame")
    output = dataset[0]
    native = (np.asarray(Image.open(mask)) < 128).astype(np.float32)
    native = np.asarray(Image.fromarray(native).resize((40, 24), Image.Resampling.NEAREST))
    _, expected, valid, _ = letterbox(load_image(image), 80, native)
    np.testing.assert_array_equal(output["mask"][0].numpy(), expected)
    np.testing.assert_array_equal(output["valid"][0].numpy(), valid)
    assert output["mask_original_hw"].tolist() == [12, 20]
    strict = SegmentationDataset(train, mask_polarity="dark", mask_alignment="require_native")
    with pytest.raises(ValueError, match="canvas mismatch"):
        strict[0]


def test_decoded_duplicates_cross_tasks_and_encoding_stay_together(tmp_path):
    shared = np.random.default_rng(900).integers(0, 255, (24, 40), dtype=np.uint8)
    sample(tmp_path, "apo", "1", shared)
    sample(tmp_path, "fasc", "9", np.repeat(shared[..., None], 3, axis=2))
    for task in ("apo", "fasc"):
        for i in range(2, 8):
            sample(tmp_path, task, str(i))
    train, _ = discover_dataset(tmp_path)
    with pytest.warns(UserWarning, match="subject independence"):
        folds = assign_folds(train, n_folds=3, seed=22)
    paired = folds[(folds.image_id == "1.tif") | (folds.image_id == "9.tif")]
    assert paired.image_hash.nunique() == paired.group.nunique() == paired.fold.nunique() == 1
    assert (folds.groupby("image_hash").fold.nunique() == 1).all()
    assert set(folds.fold) == {0, 1, 2}
    assert len(folds) == len(train)
    with pytest.warns(UserWarning):
        reordered = assign_folds(train.sample(frac=1, random_state=5), n_folds=3, seed=22)
    pd.testing.assert_frame_equal(folds, reordered)


def test_explicit_groups_merge_transitively_with_duplicates(tmp_path):
    for task in ("apo", "fasc"):
        for i in range(1, 9):
            sample(tmp_path, task, str(i))
    train, _ = discover_dataset(tmp_path)
    groups = pd.DataFrame([{"task": "apo", "image_id": "1.tif", "group": "session_a"},
                           {"task": "apo", "image_id": "2.tif", "group": "session_a"}])
    with pytest.warns(UserWarning):
        folds = assign_folds(train, n_folds=3, groups_csv=groups)
    connected = folds[folds.image_id.isin(["1.tif", "2.tif"])]
    assert len(connected) == 4 and connected.group.nunique() == connected.fold.nunique() == 1
    bad = groups.copy()
    bad.loc[0, "image_id"] = "unknown.tif"
    with pytest.raises(ValueError, match="Unknown explicit group"):
        assign_folds(train, groups_csv=bad)
    with pytest.raises(ValueError, match="Duplicate group assignment"):
        assign_folds(train, groups_csv=pd.concat([groups, groups.iloc[[0]]]))


def test_horizontal_flip_families_preserve_identity_and_merge_reviewed_groups(tmp_path):
    shared = np.random.default_rng(908).integers(0, 255, (24, 40), dtype=np.uint8)
    sample(tmp_path, "apo", "1", shared)
    sample(tmp_path, "fasc", "9", np.repeat(shared[:, ::-1, None], 3, axis=2))
    for task in ("apo", "fasc"):
        for i in range(2, 8):
            sample(tmp_path, task, str(i))
    train, _ = discover_dataset(tmp_path)
    pair = train[train.image_id.isin(["1.tif", "9.tif"])]
    assert pair.image_hash.nunique() == 2
    assert pair.image_family_hash.nunique() == 1
    explicit = pd.DataFrame([{"task": "fasc", "image_id": "9.tif", "group": "reviewed_session"},
                             {"task": "apo", "image_id": "2.tif", "group": "reviewed_session"}])
    with pytest.warns(UserWarning, match="horizontal flips"):
        folded = assign_folds(train, groups_csv=explicit)
    connected = folded[folded.image_id.isin(["1.tif", "9.tif", "2.tif"])]
    assert len(connected) == 4
    assert connected.group.nunique() == connected.fold.nunique() == 1
    assert folded.groupby("image_family_hash").fold.nunique().max() == 1
    # An old manifest must be upgraded, not allowed to retain exact-only grouping.
    with pytest.warns(UserWarning):
        upgraded = assign_folds(train.drop(columns="image_family_hash"), groups_csv=explicit)
    pd.testing.assert_frame_equal(folded, upgraded[folded.columns])


def test_too_few_independent_groups_fails(tmp_path):
    for task in ("apo", "fasc"):
        sample(tmp_path, task, "1")
    train, _ = discover_dataset(tmp_path)
    with pytest.raises(ValueError, match="Too few independent groups"):
        assign_folds(train, n_folds=3)


def test_grayscale_rgb_replication_and_letterbox_restoration_metadata(tmp_path):
    source = np.arange(15*40, dtype=np.uint16).reshape(15, 40).astype(np.uint8)
    path = save(tmp_path / "gray.png", source)
    image = load_image(path)
    np.testing.assert_array_equal(image[..., 0], source)
    np.testing.assert_array_equal(image[..., 0], image[..., 2])
    mask = (source > 128).astype(np.float32)
    boxed, target, valid, meta = letterbox(image, 80, mask)
    assert boxed.shape == (80, 80, 3)
    assert meta == {"original_hw": [15, 40], "resized_hw": [30, 80],
                    "pad_tlbr": [25, 0, 25, 0], "scale_xy": [2, 2]}
    assert valid.sum() == 30 * 80
    assert target[valid == 0].sum() == 0
    restored = np.asarray(Image.fromarray(target[25:55]).resize((40, 15), Image.Resampling.NEAREST))
    np.testing.assert_array_equal(restored, mask)


def test_dark_light_policies_normalization_and_collation(tmp_path):
    image_a, mask_a = sample(tmp_path, "apo", "1", dark=True)
    image_f, mask_f = sample(tmp_path, "fasc", "2")
    frame = pd.concat([frame_for(image_a, mask_a), frame_for(image_f, mask_f, "fasc")])
    ds = SegmentationDataset(frame, size=64, mask_polarity={"apo": "dark", "fasc": "light"})
    batch = next(iter(DataLoader(ds, batch_size=2)))
    assert batch["image"].shape == (2, 3, 64, 64)
    assert batch["mask"].shape == batch["valid"].shape == (2, 1, 64, 64)
    assert batch["mask_polarity"] == ["dark", "light"]
    assert set(torch.unique(batch["mask"]).tolist()) == {0, 1}
    assert torch.all(batch["mask"][batch["valid"] == 0] == 0)
    item = ds[0]
    boxed, _, _, _ = letterbox(load_image(image_a), 64)
    reconstructed = item["image"].numpy().transpose(1, 2, 0) * np.array(IMAGE_STD) + np.array(IMAGE_MEAN)
    np.testing.assert_allclose(reconstructed, boxed / 255, atol=1e-7)
    assert len(ds.frame.mask_hash.iloc[0]) == 64
    reordered = SegmentationDataset(frame.iloc[::-1], size=64, mask_polarity={"apo": "dark", "fasc": "light"})
    assert ds.mask_manifest_hash == reordered.mask_manifest_hash
    changed_policy = SegmentationDataset(frame, size=64, mask_polarity="light")
    assert changed_policy.mask_manifest_hash != ds.mask_manifest_hash


def test_auto_polarity_sparse_side_and_ambiguous_cases(tmp_path):
    image, mask = sample(tmp_path, "apo", "1", dark=True)
    with pytest.warns(UserWarning, match="Auto mask polarity"):
        ds = SegmentationDataset(frame_for(image, mask), mask_polarity="auto")
    assert ds.frame.mask_polarity.iloc[0] == "dark"
    values = np.zeros((24, 40), np.uint8)
    values[:12] = 255
    save(mask, values)
    with pytest.raises(ValueError, match="ambiguous/uniform"):
        SegmentationDataset(frame_for(image, mask), mask_polarity="auto")
    save(mask, np.zeros((24, 40), np.uint8))
    with pytest.raises(ValueError, match="ambiguous/uniform"):
        SegmentationDataset(frame_for(image, mask), mask_polarity="auto")
    explicit = SegmentationDataset(frame_for(image, mask), mask_polarity="light")
    assert explicit[0]["mask"].sum() == 0


def test_augmentation_is_paired_binary_and_never_vertical_flip(tmp_path):
    pixels = np.zeros((64, 64), np.uint8)
    pixels[6:24, 12:48] = 255
    image = save(tmp_path / "source.png", pixels)
    mask = save(tmp_path / "mask.png", pixels)
    ds = SegmentationDataset(frame_for(image, mask), size=64, augment=True, mask_polarity="light")
    for seed in [1, 7, 19]:
        torch.manual_seed(seed)
        item = ds[0]
        restored = item["image"][0].numpy() * IMAGE_STD[0] + IMAGE_MEAN[0]
        predicted = restored >= .5
        target = item["mask"][0].numpy().astype(bool)
        assert (predicted & target).sum() / (predicted | target).sum() > .95
        assert np.where(target)[0].mean() < 32
        assert set(torch.unique(item["mask"]).tolist()) == {0, 1}
        assert item["augmented"].item() is True


def test_multiframe_high_bitdepth_and_colored_masks_fail_explicitly(tmp_path):
    frames = [Image.fromarray(np.zeros((10, 10), np.uint8)), Image.fromarray(np.ones((10, 10), np.uint8))]
    multi = tmp_path / "multi.tif"
    frames[0].save(multi, save_all=True, append_images=frames[1:])
    with pytest.raises(ValueError, match="Multi-frame"):
        load_image(multi)
    high = save(tmp_path / "high.tif", np.full((10, 10), 1024, np.uint16))
    with pytest.raises(ValueError, match="intensity scaling"):
        load_image(high)
    image, mask = sample(tmp_path, "apo", "1")
    colored = np.zeros((24, 40, 3), np.uint8)
    colored[..., 0] = 255
    save(mask, colored)
    with pytest.raises(ValueError, match="identical RGB"):
        SegmentationDataset(frame_for(image, mask), mask_polarity="dark")
