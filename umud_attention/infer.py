"""Fold/flip ensemble -> restored masks -> calibrated geometry -> exact Kaggle CSV.

No constant or guessed physical measurement is substituted on a failed image.
An optional, explicitly reported blend can retain an existing team's predictions
on unsupported cases. That hybrid depends on the reference CSV and is not a
standalone method for future images. The reference is never a training label.
"""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn import functional as F

from .calibration import infer_known_calibration
from .data import IMAGE_MEAN, IMAGE_STD, letterbox, load_image
from .geometry import GeometryConfig, measure_geometry
from .submission import export_submission, validate_submission, load_expected_ids, TARGETS
from .train import get_device, model_from_config


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def restore_probability(prob, meta):
    """Undo letterboxing before fitting geometry; output is original image H,W."""
    top, left, bottom, right = meta["pad_tlbr"]
    rh, rw = meta["resized_hw"]
    if any(int(v) != v or v < 0 for v in (top, left, bottom, right)) or min(rh, rw) < 1:
        raise ValueError("Malformed letterbox metadata")
    if np.asarray(prob).ndim != 2 or np.asarray(prob).shape != (top + rh + bottom, left + rw + right):
        raise ValueError("Probability canvas does not match letterbox metadata")
    if not np.isfinite(prob).all():
        raise ValueError("Nonfinite probability map")
    cropped = np.asarray(prob)[top:top + rh, left:left + rw]
    h, w = meta["original_hw"]
    return np.asarray(Image.fromarray(cropped.astype(np.float32)).resize((w, h), Image.Resampling.BILINEAR))


class Predictor:
    def __init__(self, checkpoint_paths, device="auto"):
        self.device = get_device(device)
        self.models = {"apo": [], "fasc": []}
        self.provenance = []
        seen = set()
        for path in checkpoint_paths:
            state = torch.load(path, map_location="cpu", weights_only=True)
            c = state["config"]
            key = (c["task"], c["fold"])
            if key in seen:
                raise ValueError(f"Duplicate task/fold checkpoint: {key}")
            if c["task"] not in self.models:
                raise ValueError(f"Unknown task {c['task']}")
            seen.add(key)
            model = model_from_config(c, pretrained=False)
            model.load_state_dict(state["model"])
            model.eval().to(self.device)
            self.models[c["task"]].append((model, c["size"], state.get("threshold", 0.5)))
            self.provenance.append({"checkpoint": str(path), "sha256": file_hash(path),
                                    "task": c["task"], "fold": c["fold"], "config": c,
                                    "segmentation_val_dice": state.get("best_dice"),
                                    "manifest_sha256": state.get("manifest_sha256")})
        if any(not models for models in self.models.values()):
            raise ValueError("Need at least one separately trained checkpoint for EACH task")
        if {c["fold"] for c in self.provenance if c["task"] == "apo"} != {c["fold"] for c in self.provenance if c["task"] == "fasc"}:
            raise ValueError("Apo and fascicle checkpoints must cover the same fold IDs")

    def thresholds(self):
        return {task: float(np.median([threshold for _, _, threshold in models]))
                for task, models in self.models.items()}

    @torch.inference_mode()
    def predict(self, image, *, horizontal_tta=True, sizes=None):
        result = {}
        for task, models in self.models.items():
            total, count = np.zeros(image.shape[:2], np.float32), 0
            for model, trained_size, _ in models:
                for size in (sizes or [trained_size]):
                    canvas, _, _, meta = letterbox(image, size=size)
                    x = canvas.astype(np.float32) / 255
                    x = (x - np.array(IMAGE_MEAN, np.float32)) / np.array(IMAGE_STD, np.float32)
                    x = torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(self.device)
                    with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
                        prob = model(x).float().sigmoid()
                        if horizontal_tta:
                            flipped = model(torch.flip(x, [-1])).float().sigmoid()
                            prob = (prob + torch.flip(flipped, [-1])) / 2
                    total += restore_probability(prob[0, 0].cpu().numpy(), meta)
                    count += 1
            result[task] = np.clip(total / count, 0, 1)
        return result


def read_calibration(path, expected_ids):
    if not path:
        return {}
    frame = pd.read_csv(path)
    required = {"image_id", "sx_mm_per_px", "sy_mm_per_px", "source"}
    if not required <= set(frame) or frame.image_id.isna().any() or frame.image_id.duplicated().any():
        raise ValueError(f"Calibration needs unique IDs and {sorted(required)}")
    if not set(frame.image_id) <= set(expected_ids):
        raise ValueError("Calibration contains unknown IDs")
    values = frame[["sx_mm_per_px", "sy_mm_per_px"]].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Physical spacing must be finite and positive")
    if frame.source.astype("string").fillna("").str.strip().eq("").any():
        raise ValueError("Each calibration requires a source")
    return frame.set_index("image_id").to_dict("index")


def geometry_consensus(apo, fasc, spacing, cfg):
    """Sensitivity to nearby thresholds, not a statistical confidence interval."""
    variants = []
    for delta in (-0.08, 0.0, 0.08):
        trial = replace(cfg, apo_threshold=float(np.clip(cfg.apo_threshold + delta, .05, .95)),
                        fasc_threshold=float(np.clip(cfg.fasc_threshold + delta, .05, .95)))
        variants.append(measure_geometry(apo, fasc, spacing, config=trial))
    valid = [r for r in variants if r["status"] == "ok"]
    central = variants[1]
    if not valid:
        return central
    # Keep a real diagnostic overlay from the successful setting closest to centre.
    selected = central if central["status"] == "ok" else valid[0]
    values = np.array([[r[k] for k in TARGETS] for r in valid])
    selected.update(dict(zip(TARGETS, np.median(values, axis=0).tolist())))
    spread = np.ptp(values, axis=0)
    selected["diagnostics"]["threshold_success_count"] = len(valid)
    selected["diagnostics"]["threshold_measurement_range"] = dict(zip(TARGETS, spread.tolist()))
    selected["diagnostics"]["stable_thresholds"] = bool(len(valid) >= 2 and np.all(spread <= [3.0, 12.0, 3.0]))
    selected["diagnostics"]["aggregation"] = "median of supported threshold variants; plotted overlay is one variant"
    return selected


def geometry_config_for_roi(cfg, original_shape_hw, roi):
    """Normalize horizontal support to the ROI while keeping original coordinates.

    Only minimum aponeurosis span/overlap fractions are translated; unrelated
    display margins must not increase required ultrasound support. Pixel-count,
    residual, fragment-length and extrapolation guards are unchanged.
    """
    shape, bounds = np.asarray(original_shape_hw, dtype=float), np.asarray(roi, dtype=float)
    if (shape.shape != (2,) or bounds.shape != (4,) or not np.isfinite(shape).all() or
            not np.isfinite(bounds).all() or np.any(shape != np.floor(shape)) or
            np.any(bounds != np.floor(bounds))):
        raise ValueError("Original image shape and ROI must contain finite integer coordinates")
    height, width = map(int, shape)
    x0, y0, x1, y1 = map(int, bounds)
    if not (height >= 2 and width >= 2 and 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
            and x1 - x0 >= 2 and y1 - y0 >= 2):
        raise ValueError("ROI must be inside the original image and at least 2x2 pixels")
    ratio = (x1 - x0 - 1) / (width - 1)
    effective = replace(cfg, min_apo_span_fraction=cfg.min_apo_span_fraction * ratio,
                        min_overlap_fraction=cfg.min_overlap_fraction * ratio)
    evidence = {"roi_xyxy": [x0, y0, x1, y1], "original_shape_hw": [height, width],
                "roi_width_pixels": x1 - x0, "original_width_pixels": width,
                "span_normalization_ratio": ratio,
                "minimum_apo_span_original_px": cfg.min_apo_span_fraction * (x1 - x0 - 1),
                "minimum_pair_overlap_original_px": cfg.min_overlap_fraction * (x1 - x0 - 1),
                "coordinate_grid": "original image; ROI right/bottom exclusive; no geometry coordinate resize",
                "changed_fields": ["min_apo_span_fraction", "min_overlap_fraction"],
                "other_geometry_bounds_changed": False}
    return effective, evidence


def predict_dataset(test_manifest, checkpoint_paths, output_dir, *, calibration_csv=None,
                    use_known_rulers=False, reference_csv=None, blend_weight=0.2,
                    config=None, device="auto", save_masks=True, expected_count=309):
    test = pd.read_csv(test_manifest) if isinstance(test_manifest, (str, Path)) else test_manifest.copy()
    if not {"image_id", "image_path"} <= set(test) or len(test) != expected_count or test.image_id.duplicated().any():
        raise ValueError(f"Need a unique {expected_count}-image test manifest")
    if not 0 < blend_weight <= 1:
        raise ValueError("Blend weight must be in (0,1]")
    ids = load_expected_ids(manifest=test, expected_count=expected_count)
    cal = read_calibration(calibration_csv, ids)
    predictor = Predictor(checkpoint_paths, device=device)
    thresholds = predictor.thresholds()
    cfg = config or GeometryConfig(apo_threshold=thresholds["apo"], fasc_threshold=thresholds["fasc"])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if save_masks:
        (out / "masks").mkdir(exist_ok=True)
    rows, qc, details = [], [], []
    for i, row in enumerate(test.to_dict("records")):
        try:
            image = load_image(row["image_path"])
        except (ValueError, OSError) as exc:
            rows.append({"image_id": row["image_id"], **{k: np.nan for k in TARGETS}})
            qc.append({"image_id": row["image_id"], "status": "failed", "reason": str(exc),
                       "accepted_fascicles": 0, "stable_thresholds": False, "eligible_for_blend": False,
                       "calibration_source": "not_evaluated", "square_pixel_assumption": False, "ruler_error": None})
            details.append({"image_id": row["image_id"], "error": str(exc)})
            continue
        calibration, ruler_error = cal.get(row["image_id"]), None
        ruler = None
        if use_known_rulers:
            try:
                ruler = infer_known_calibration(row["image_path"])
                if calibration is None:
                    calibration = ruler
            except ValueError as exc:
                ruler_error = str(exc)
        spacing = None if calibration is None else (calibration["sx_mm_per_px"], calibration["sy_mm_per_px"])
        roi = ruler["roi"] if ruler is not None else (0, 0, image.shape[1], image.shape[0])
        x0, y0, x1, y1 = roi
        per_image_cfg, roi_normalization = geometry_config_for_roi(cfg, image.shape[:2], roi)
        try:
            cropped = predictor.predict(image[y0:y1, x0:x1])
        except (ValueError, RuntimeError) as exc:
            # CUDA OOM/model execution errors should be fixed, not hidden in a hybrid.
            raise RuntimeError(f"Model inference failed for {row['image_id']}: {exc}") from exc
        masks = {k: np.zeros(image.shape[:2], np.float32) for k in cropped}
        for k in masks:
            masks[k][y0:y1, x0:x1] = cropped[k]
        measured = geometry_consensus(masks["apo"], masks["fasc"], spacing, per_image_cfg)
        diag = measured["diagnostics"]
        diag["roi_support_normalization"] = roi_normalization
        diag["base_per_image_geometry_config"] = asdict(per_image_cfg)
        stable = bool(diag.get("stable_thresholds", False))
        eligible = measured["status"] == "ok" and stable and diag.get("n_accepted_fascicles", 0) >= 2
        rows.append({"image_id": row["image_id"], **{k: measured[k] for k in TARGETS}})
        qc.append({"image_id": row["image_id"], "status": measured["status"], "reason": measured["reason"],
                   "accepted_fascicles": diag.get("n_accepted_fascicles", 0), "stable_thresholds": stable,
                   "eligible_for_blend": eligible, "calibration_source": (calibration or {}).get("source", "missing"),
                   "square_pixel_assumption": (calibration or {}).get("square_pixel_assumption", False),
                   "geometry_support_width_pixels": roi_normalization["roi_width_pixels"],
                   "minimum_apo_span_original_px": roi_normalization["minimum_apo_span_original_px"],
                   "minimum_pair_overlap_original_px": roi_normalization["minimum_pair_overlap_original_px"],
                   "ruler_error": ruler_error})
        details.append({"image_id": row["image_id"], "geometry": measured, "calibration": calibration, "roi": roi,
                        "effective_geometry_config": asdict(per_image_cfg), "roi_support_normalization": roi_normalization})
        if save_masks:
            np.savez_compressed(out / "masks" / f"{row['image_id']}.npz", apo=masks["apo"].astype(np.float16),
                                fasc=masks["fasc"].astype(np.float16))
        if i % 10 == 0 or i + 1 == len(test):
            print(f"Inference {i+1}/{len(test)}: {row['image_id']} {measured['status']}", flush=True)
    predictions, quality = pd.DataFrame(rows), pd.DataFrame(qc)
    predictions.to_csv(out / "measurements_partial.csv", index=False)
    quality.to_csv(out / "quality.csv", index=False)
    (out / "geometry_details.json").write_text(json.dumps(_json(details), indent=2, allow_nan=False))
    report = {"n_images": len(test), "complete_measurements": int(np.isfinite(predictions[list(TARGETS)].to_numpy()).all(axis=1).sum()),
              "eligible_for_blend": int(quality.eligible_for_blend.sum()), "checkpoints": predictor.provenance,
              "geometry_config": asdict(cfg), "use_known_rulers": use_known_rulers,
              "leaderboard_score": None, "standalone_submission": None, "hybrid_submission": None}
    if report["complete_measurements"] == len(test):
        destination = export_submission(predictions, ids, out / "submission_attention.csv", expected_count=expected_count)
        report["standalone_submission"] = str(destination)
    if reference_csv is not None:
        reference = validate_submission(reference_csv, ids, expected_count=expected_count)
        eligible = quality.eligible_for_blend.to_numpy(bool)
        hybrid = reference.copy()
        hybrid.loc[eligible, list(TARGETS)] = ((1-blend_weight) * reference.loc[eligible, list(TARGETS)].to_numpy()
                                             + blend_weight * predictions.loc[eligible, list(TARGETS)].to_numpy())
        changed = int((hybrid[list(TARGETS)] != reference[list(TARGETS)]).any(axis=1).sum())
        if changed:
            destination = export_submission(hybrid, ids, out / "submission_hybrid.csv", expected_count=expected_count)
            report["hybrid_submission"] = str(destination)
        report.update(reference_sha256=file_hash(reference_csv), blend_weight=blend_weight,
                      images_changed=changed, images_retaining_reference=len(test)-changed,
                      hybrid_limitation="Depends on existing team CSV; cannot reproduce its reference component on new images")
    (out / "inference_report.json").write_text(json.dumps(_json(report), indent=2, allow_nan=False))
    return report
