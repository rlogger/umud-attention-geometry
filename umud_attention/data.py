"""UMUD task-specific pairing, duplicate-grouped folds, and paired image transforms.

No labels are synthesized for the other task. Exact decoded-image duplicates
and horizontal flips share folds across both tasks. This is NOT proof of subject-independent splits:
provide reviewed groups.csv (task,image_id,group) for related acquisitions.

Different mask canvas sizes are recorded. `mask_alignment='full_frame'` means
an explicit full-frame nearest-neighbour resize, NOT a recovered crop transform.
Inspect paired overlays before training to confirm that assumption on real data.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
FOLDERS = {"apo_imgs_v1": "apo_image", "apo_masks_v1": "apo_mask",
           "fasc_imgs_v1": "fasc_image", "fasc_masks_v1": "fasc_mask",
           "test_images_v2": "test"}


def _open_single(path):
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"Multi-frame input requires an explicit frame policy: {path}")
        image.load()
        return image.copy()


def load_image(path):
    """Load one 8-bit image as RGB uint8 HWC; grayscale is replicated to RGB."""
    image = _open_single(path)
    if image.mode not in {"1", "L", "LA", "RGB", "RGBA", "P"}:
        raise ValueError(f"Unsupported image mode {image.mode}; specify intensity scaling: {path}")
    if image.mode == "P":
        image = image.convert("RGBA" if "transparency" in image.info else "RGB")
    if image.mode in {"LA", "RGBA"} and np.any(np.asarray(image)[..., -1] != 255):
        raise ValueError(f"Nonopaque image alpha requires an explicit compositing policy: {path}")
    return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _fingerprint(path):
    # Lossless canonical representation for supported inputs: repeated RGB
    # grayscale channels do not conceal duplicates encoded as L versus RGB.
    pixels = load_image(path)
    shape = pixels.shape
    if np.array_equal(pixels[..., 0], pixels[..., 1]) and np.array_equal(pixels[..., 0], pixels[..., 2]):
        pixels = pixels[..., 0]
    header = json.dumps({"shape": pixels.shape, "dtype": pixels.dtype.str}, sort_keys=True)
    digest = hashlib.sha256(header.encode() + b"\0" + np.ascontiguousarray(pixels).tobytes()).hexdigest()
    flip_digest = hashlib.sha256(header.encode() + b"\0" + np.ascontiguousarray(pixels[:, ::-1]).tobytes()).hexdigest()
    return {"image_hash": digest, "image_family_hash": min(digest, flip_digest),
            "height": shape[0], "width": shape[1]}


def _mask_values(path):
    image = _open_single(path)
    if image.mode == "P":
        image = image.convert("RGB")
    values = np.asarray(image)
    if values.ndim == 3:
        if values.shape[2] == 4 and np.all(values[..., 3] == 255):
            values = values[..., :3]
        if values.shape[2] != 3 or not (np.array_equal(values[..., 0], values[..., 1])
                                       and np.array_equal(values[..., 0], values[..., 2])):
            raise ValueError(f"Mask must be scalar or identical RGB channels: {path}")
        values = values[..., 0]
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"Mask must be a finite single-channel image: {path}")
    if values.min() < 0 or values.max() > 255:
        raise ValueError(f"Unsupported mask range; need 0/1 or 0..255: {path}")
    scale = 1.0 if values.max() <= 1 else 255.0
    return values.astype(np.float32) / scale


def _mask_metadata(path):
    values = _mask_values(path)
    return {"mask_height": values.shape[0], "mask_width": values.shape[1],
            "mask_light_fraction": float((values >= .5).mean()),
            "mask_hash": hashlib.sha256(np.ascontiguousarray(values).tobytes()
                                        + str(values.shape).encode()).hexdigest()}


def _unique(paths, role, stem=True):
    indexed = {}
    for path in paths:
        key = path.stem if stem else path.name
        if key in indexed:
            raise ValueError(f"Ambiguous duplicate {'stem' if stem else 'basename'} in {role}: {key}")
        indexed[key] = path
    return indexed


def discover_dataset(root):
    """Return train/test DataFrames from exact UMUD v1/v2 folder components.

    Training pairs use exact filename stems within each task (extensions may
    differ). Missing partners and duplicate stems fail; tasks never cross-pair.
    Train has task,image_id,image_path,mask_path,group,fold plus audit columns.
    No sample-submission rows are used to infer the complete test cohort.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    buckets = {role: [] for role in FOLDERS.values()}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if any(part.startswith(".") or part == "__MACOSX" for part in relative.parts):
            continue
        roles = {FOLDERS[part] for part in relative.parts[:-1] if part in FOLDERS}
        if len(roles) > 1:
            raise ValueError(f"Ambiguous dataset directory roles: {path}")
        if roles:
            buckets[roles.pop()].append(path)
    records = []
    for task in ("apo", "fasc"):
        images = _unique(buckets[f"{task}_image"], f"{task} images")
        masks = _unique(buckets[f"{task}_mask"], f"{task} masks")
        if set(images) != set(masks):
            raise ValueError(f"Unpaired {task} files: images without masks={sorted(set(images)-set(masks))[:5]}, "
                             f"masks without images={sorted(set(masks)-set(images))[:5]}")
        for stem in sorted(images):
            image_info = _fingerprint(images[stem])
            mask_info = _mask_metadata(masks[stem])
            same_size = (image_info["height"], image_info["width"]) == (mask_info["mask_height"], mask_info["mask_width"])
            records.append({"task": task, "image_id": images[stem].name,
                            "image_path": str(images[stem]), "mask_path": str(masks[stem]),
                            "group": image_info["image_hash"], "fold": -1,
                            **image_info, **mask_info, "mask_size_matches_image": same_size})
    if not records:
        raise ValueError("No paired apo/fasc training data found in known UMUD directories")
    train = pd.DataFrame(records)
    mismatch = int((~train.mask_size_matches_image).sum())
    if mismatch:
        warnings.warn(f"{mismatch} masks use a different canvas. Inspect paired overlays before accepting a full-frame resize.", UserWarning)
    tests = _unique(buckets["test"], "test images", stem=False)
    test = pd.DataFrame([{"image_id": name, "image_path": str(path), **_fingerprint(path)}
                         for name, path in sorted(tests.items())],
                        columns=["image_id", "image_path", "image_hash", "image_family_hash", "height", "width"])
    train.attrs["mask_canvas_mismatch_count"] = mismatch
    return train, test


def assign_folds(train, n_folds=3, seed=42, groups_csv=None):
    """Group exact decoded images, horizontal flips and reviewed acquisitions.

    Explicit groups are merged transitively with duplicate groups across tasks.
    Rows are retained; no group crosses folds. groups.csv may cover a subset,
    but must not reference unknown/duplicate (task,image_id) keys.
    """
    if not isinstance(n_folds, (int, np.integer)) or n_folds < 2:
        raise ValueError("n_folds must be an integer >=2")
    required = {"task", "image_id", "image_path"}
    if not required <= set(train.columns) or train.empty:
        raise ValueError(f"Nonempty train manifest requires {sorted(required)}")
    frame = train.sort_values(["task", "image_id"]).reset_index(drop=True).copy()
    if frame.duplicated(["task", "image_id"]).any():
        raise ValueError("Duplicate (task,image_id) training keys")
    if not set(frame.task) <= {"apo", "fasc"}:
        raise ValueError("Tasks must be apo or fasc")
    if "image_hash" not in frame or "image_family_hash" not in frame:
        # Upgrade older manifests rather than silently keeping their weaker split.
        fingerprints = {p: _fingerprint(p) for p in frame.image_path.unique()}
        hashes = frame.image_path.map(lambda p: fingerprints[p]["image_hash"])
        if "image_hash" in frame and not frame.image_hash.equals(hashes):
            raise ValueError("Decoded-image hash changed while upgrading family groups")
        frame["image_hash"] = hashes
        frame["image_family_hash"] = frame.image_path.map(lambda p: fingerprints[p]["image_family_hash"])
    if frame.image_hash.isna().any() or frame.image_hash.eq("").any():
        raise ValueError("Missing decoded-image hashes")
    if frame.image_family_hash.isna().any() or frame.image_family_hash.eq("").any():
        raise ValueError("Missing horizontal-flip family hashes")
    parent = list(range(len(frame)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(indices):
        indices = list(indices)
        for i in indices[1:]:
            parent[find(i)] = find(indices[0])
    for indices in frame.groupby("image_hash", sort=True).indices.values():
        union(indices)
    for indices in frame.groupby("image_family_hash", sort=True).indices.values():
        union(indices)
    if groups_csv is not None:
        explicit = pd.read_csv(groups_csv, dtype=str, keep_default_na=False) if isinstance(groups_csv, (str, Path)) else groups_csv.copy()
        if not {"task", "image_id", "group"} <= set(explicit.columns):
            raise ValueError("groups.csv requires task,image_id,group")
        if explicit.duplicated(["task", "image_id"]).any():
            raise ValueError("Duplicate group assignment for (task,image_id)")
        if explicit[["task", "image_id", "group"]].isna().any().any() or explicit["group"].astype(str).str.strip().eq("").any():
            raise ValueError("Explicit groups must be nonempty")
        lookup = {(r.task, r.image_id): i for i, r in enumerate(frame.itertuples())}
        keys = list(zip(explicit.task, explicit.image_id))
        unknown = set(keys) - set(lookup)
        if unknown:
            raise ValueError(f"Unknown explicit group keys: {sorted(unknown)[:5]}")
        for indices in explicit.groupby("group", sort=True).indices.values():
            union(lookup[keys[i]] for i in indices)
    roots = [find(i) for i in range(len(frame))]
    members = {}
    for i, root in enumerate(roots):
        members.setdefault(root, []).append(i)
    if len(members) < n_folds:
        raise ValueError("Too few independent groups for the requested folds")
    tasks = sorted(frame.task.unique())
    vectors, names = {}, {}
    for root, indices in members.items():
        rows = frame.iloc[indices]
        vectors[root] = np.array([(rows.task == task).sum() for task in tasks], dtype=float)
        text = json.dumps(sorted(zip(rows.task, rows.image_id, rows.image_hash)))
        names[root] = "g_" + hashlib.sha256(text.encode()).hexdigest()[:20]
    for j, task in enumerate(tasks):
        if sum(vector[j] > 0 for vector in vectors.values()) < n_folds:
            raise ValueError(f"Too few independent {task} groups for {n_folds} folds")
    totals = sum(vectors.values())
    rng = np.random.default_rng(seed)
    order = list(sorted(members, key=lambda r: names[r]))
    rng.shuffle(order)
    order.sort(key=lambda r: -float(np.max(vectors[r] / totals)))
    counts = np.zeros((n_folds, len(tasks)))
    assignments = {}
    for root in order:
        added = vectors[root]
        costs = (((counts + added) / totals) ** 2 - (counts / totals) ** 2).sum(axis=1)
        tied = np.flatnonzero(np.isclose(costs, costs.min(), rtol=0, atol=1e-14))
        fold = int(rng.choice(tied))
        assignments[root] = fold
        counts[fold] += added
    if np.any(counts == 0):
        raise ValueError("Grouping produced an empty task/fold; reduce n_folds or review groups")
    frame["group"] = [names[r] for r in roots]
    frame["fold"] = [assignments[r] for r in roots]
    warnings.warn("Exact duplicates and horizontal flips are grouped; other transformed variants, near duplicates and subject independence remain unverified. Add reviewed groups.csv; filenames do not establish subjects.", UserWarning)
    return frame


def _resolve_polarity(light_fraction, requested):
    if requested not in {"dark", "light", "auto"}:
        raise ValueError("mask_polarity must be dark, light, auto, or a per-task mapping")
    if requested != "auto":
        return requested
    if 0 < light_fraction <= .35:
        return "light"
    if .65 <= light_fraction < 1:
        return "dark"
    raise ValueError("Mask polarity is ambiguous/uniform; set an explicit dark/light task policy after inspecting masks")


def letterbox(image, size=512, mask=None):
    """Return (RGB canvas, mask or None, valid mask, restoration metadata).

    The mask must already describe the same full image grid. `valid` excludes
    padding. Undo pad_tlbr and resize to original_hw to restore probabilities.
    """
    if not isinstance(size, (int, np.integer)) or size < 2:
        raise ValueError("size must be an integer >=2")
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("letterbox expects RGB uint8 HWC")
    h, w = image.shape[:2]
    if mask is not None and np.asarray(mask).shape != (h, w):
        raise ValueError("Align mask to the original image grid before letterboxing")
    scale = min(size / h, size / w)
    rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
    top, left = (size - rh) // 2, (size - rw) // 2
    bottom, right = size - rh - top, size - rw - left
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[top:top+rh, left:left+rw] = np.asarray(Image.fromarray(image).resize((rw, rh), Image.Resampling.BILINEAR))
    valid = np.zeros((size, size), dtype=np.float32)
    valid[top:top+rh, left:left+rw] = 1
    target = None
    if mask is not None:
        target = np.zeros((size, size), dtype=np.float32)
        target[top:top+rh, left:left+rw] = np.asarray(Image.fromarray(np.asarray(mask, dtype=np.float32)).resize((rw, rh), Image.Resampling.NEAREST))
    meta = {"original_hw": [h, w], "resized_hw": [rh, rw],
            "pad_tlbr": [top, left, bottom, right], "scale_xy": [rw/w, rh/h]}
    return canvas, target, valid, meta


def _augment(image, mask, valid):
    # All random draws use torch's worker-seeded RNG. No vertical flips.
    flip = bool(torch.rand(()) < .5)
    angle = float(torch.empty(()).uniform_(-7, 7))
    scale = float(torch.empty(()).uniform_(.95, 1.05))
    h, w = mask.shape
    tx, ty = [float(torch.empty(()).uniform_(-.03, .03)) * v for v in (w, h)]
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) * scale
    inverse = np.linalg.inv(rotation)
    centre = np.array([(w-1)/2, (h-1)/2])
    offset = centre - inverse @ (centre + np.array([tx, ty]))
    affine = tuple(np.column_stack([inverse, offset]).ravel())
    outputs = []
    for array, interpolation in [(image, Image.Resampling.BILINEAR), (mask, Image.Resampling.NEAREST), (valid, Image.Resampling.NEAREST)]:
        pil = Image.fromarray(array)
        if flip:
            pil = ImageOps.mirror(pil)
        outputs.append(np.asarray(pil.transform((w, h), Image.Transform.AFFINE, affine,
                                               resample=interpolation, fillcolor=0)).copy())
    return outputs


class SegmentationDataset(Dataset):
    """Single-task labeled samples; ImageNet-normalized RGB tensors.

    mask_polarity may be dark/light/auto or {'apo':..., 'fasc':...}. Auto is a
    minority-occupancy heuristic, NOT verified anatomy. Resolved policies and
    mask hashes are retained in `frame` for the run manifest/checkpoint guard.
    `full_frame` explicitly resizes annotation canvases before letterboxing;
    `require_native` forbids size differences. Neither can recover unknown crops.
    """
    def __init__(self, frame, size=512, augment=False, mask_polarity="auto", mask_alignment="full_frame"):
        if not {"task", "image_id", "image_path", "mask_path"} <= set(frame.columns) or frame.empty:
            raise ValueError("Nonempty labeled manifest requires task,image_id,image_path,mask_path")
        if not isinstance(size, (int, np.integer)) or size < 2:
            raise ValueError("size must be an integer >=2")
        if mask_alignment not in {"full_frame", "require_native"}:
            raise ValueError("mask_alignment must be full_frame or require_native")
        self.frame = frame.reset_index(drop=True).copy()
        if self.frame.duplicated(["task", "image_id"]).any():
            raise ValueError("Duplicate labeled (task,image_id) keys")
        self.size, self.augment, self.mask_alignment = int(size), bool(augment), mask_alignment
        self.mask_polarity = mask_polarity
        for i, row in self.frame.iterrows():
            if row.task not in {"apo", "fasc"}:
                raise ValueError(f"Unknown labeled task: {row.task}")
            requested = mask_polarity.get(row.task) if isinstance(mask_polarity, dict) else mask_polarity
            metadata = _mask_metadata(row.mask_path)
            for key, value in metadata.items():
                self.frame.loc[i, key] = value
            self.frame.loc[i, "mask_polarity"] = _resolve_polarity(metadata["mask_light_fraction"], requested)
        self.polarity_counts = dict(Counter(zip(self.frame.task, self.frame.mask_polarity)))
        audited = self.frame.sort_values(["task", "image_id"])[
            ["task", "image_id", "mask_hash", "mask_polarity", "mask_height", "mask_width"]]
        self.mask_manifest_hash = hashlib.sha256(audited.to_json(orient="records").encode()).hexdigest()
        self.preprocessing = {"size": self.size, "normalization": "ImageNet", "mean": IMAGE_MEAN,
                              "std": IMAGE_STD, "mask_polarity": mask_polarity,
                              "mask_alignment": mask_alignment, "vertical_flip": False,
                              "mask_manifest_hash": self.mask_manifest_hash}
        if (isinstance(mask_polarity, dict) and "auto" in mask_polarity.values()) or mask_polarity == "auto":
            warnings.warn("Auto mask polarity uses minority occupancy; inspect paired previews and retain resolved mask_polarity/mask_hash in the manifest.", UserWarning)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = load_image(row.image_path)
        values = _mask_values(row.mask_path)
        target = (values < .5 if row.mask_polarity == "dark" else values >= .5).astype(np.float32)
        mask_original_hw = list(target.shape)
        if target.shape != image.shape[:2]:
            if self.mask_alignment == "require_native":
                raise ValueError(f"Mask/image canvas mismatch: {row.task}/{row.image_id}")
            target = np.asarray(Image.fromarray(target).resize((image.shape[1], image.shape[0]), Image.Resampling.NEAREST))
        image, target, valid, meta = letterbox(image, self.size, target)
        if self.augment:
            image, target, valid = _augment(image, target, valid)
        pixels = image.astype(np.float32) / 255.0
        pixels = (pixels - np.asarray(IMAGE_MEAN, dtype=np.float32)) / np.asarray(IMAGE_STD, dtype=np.float32)
        return {"image": torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1))),
                "mask": torch.from_numpy(np.ascontiguousarray(target[None], dtype=np.float32)),
                "valid": torch.from_numpy(np.ascontiguousarray(valid[None], dtype=np.float32)),
                "image_id": str(row.image_id), "task": str(row.task),
                "mask_polarity": str(row.mask_polarity), "mask_hash": str(row.mask_hash),
                "original_hw": torch.tensor(meta["original_hw"]),
                "resized_hw": torch.tensor(meta["resized_hw"]),
                "pad_tlbr": torch.tensor(meta["pad_tlbr"]),
                "scale_xy": torch.tensor(meta["scale_xy"], dtype=torch.float32),
                "mask_original_hw": torch.tensor(mask_original_hw),
                "augmented": torch.tensor(self.augment)}
