"""Held-out mask-derived PIXEL geometry consistency, not physical accuracy.

This evaluates the same geometry function on predicted masks and paired label
masks for the exact same source image. The reference is algorithm-derived from
segmentation labels; it is NOT an independent expert PA/FL/MT measurement and
cannot establish a UMUD score, millimetre accuracy, or superiority to a baseline.
Unknown physical calibration is never invented: all comparisons are in pixels
and pixel-grid degrees. A declared full-frame nearest-neighbour mask alignment
matches the training data adapter; it cannot recover unknown crop transforms.
"""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .data import _fingerprint, _mask_metadata, _mask_values, _resolve_polarity, load_image
from .geometry import GeometryConfig, _configuration, measure_geometry, plot_geometry
from .infer import Predictor, _json


PIXEL_TARGETS = ("pa_pixel_deg", "fl_px", "mt_px")
SCOPE = ("Mask-derived pixel geometry consistency on same-image paired segmentation labels. "
         "Reference and prediction use the SAME geometry algorithm; not independent physical "
         "ground truth, not millimetre accuracy, and not a UMUD leaderboard score.")


def _manifest_hash(frame):
    return hashlib.sha256(frame.sort_values(["task", "image_id"]).to_csv(index=False).encode()).hexdigest()


def _checked_manifest(manifest, fold):
    frame = pd.read_csv(manifest) if isinstance(manifest, (str, Path)) else manifest.copy()
    required = {"task", "image_id", "image_path", "mask_path", "image_hash", "group", "fold"}
    if not required <= set(frame) or frame.empty:
        raise ValueError(f"Nonempty training manifest requires {sorted(required)}")
    if not isinstance(fold, (int, np.integer)) or fold < 0:
        raise ValueError("fold must be a nonnegative integer")
    if not set(frame.task) <= {"apo", "fasc"} or frame.duplicated(["task", "image_id"]).any():
        raise ValueError("Manifest requires unique apo/fasc task,image_id rows")
    for key in ("image_hash", "group"):
        if frame[key].isna().any() or frame[key].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing {key} would invalidate held-out grouping")
    folds = pd.to_numeric(frame.fold, errors="coerce").to_numpy(float)
    if not np.isfinite(folds).all() or np.any(folds < 0) or np.any(folds != np.floor(folds)):
        raise ValueError("Manifest needs assigned nonnegative integer folds")
    if frame.groupby("group").fold.nunique().max() != 1:
        raise ValueError("Acquisition/duplicate group crosses held-out folds")
    if frame.groupby("image_hash").fold.nunique().max() != 1:
        raise ValueError("Same image hash crosses held-out folds")
    return frame


def _checked_predictor(checkpoints, frame, fold, device):
    predictor = Predictor(checkpoints, device=device)
    provenance = predictor.provenance
    if len(provenance) != 2 or {item.get("task") for item in provenance} != {"apo", "fasc"}:
        raise ValueError("Evaluation requires exactly one checkpoint for each task in the held-out fold")
    expected_hash = _manifest_hash(frame)
    for item in provenance:
        config = item.get("config", {})
        if item.get("fold") != fold or config.get("fold") != fold or config.get("task") != item.get("task"):
            raise ValueError("Checkpoint task/fold does not match this held-out evaluation")
        if item.get("manifest_sha256") != expected_hash:
            raise ValueError("Checkpoint training-manifest hash does not match evaluation manifest")
    return predictor


def _reference_branch(rows, task, predictor, image_hash, shape):
    config = next(item["config"] for item in predictor.provenance if item["task"] == task)
    annotations = []
    for row in rows.sort_values(["image_id", "mask_path"]).to_dict("records"):
        current_image = _fingerprint(row["image_path"])
        if current_image["image_hash"] != image_hash:
            raise ValueError(f"Source image changed or image_hash is incorrect: {task}/{row['image_id']}")
        metadata = _mask_metadata(row["mask_path"])
        recorded_hash = row.get("mask_hash")
        if isinstance(recorded_hash, str) and recorded_hash and recorded_hash != metadata["mask_hash"]:
            raise ValueError(f"Annotation mask changed: {task}/{row['image_id']}")
        requested = row.get("mask_polarity")
        if not isinstance(requested, str) or requested not in {"dark", "light", "auto"}:
            requested = config.get("mask_polarity")
            if isinstance(requested, dict):
                requested = requested.get(task)
        if requested not in {"dark", "light", "auto"}:
            raise ValueError(f"Missing explicit/checkpoint mask-polarity policy for {task}")
        polarity = _resolve_polarity(metadata["mask_light_fraction"], requested)
        values = _mask_values(row["mask_path"])
        binary = (values < .5 if polarity == "dark" else values >= .5).astype(np.float32)
        original_shape = list(binary.shape)
        if binary.shape != shape:
            binary = np.asarray(Image.fromarray(binary).resize((shape[1], shape[0]), Image.Resampling.NEAREST))
        annotations.append({"row": row, "mask": binary, "mask_hash": metadata["mask_hash"],
                            "polarity": polarity, "requested_polarity": requested,
                            "mask_original_hw": original_shape, "aligned_hw": list(shape),
                            "alignment": "native" if original_shape == list(shape) else "full_frame_nearest_explicit_assumption"})
    identities = {(item["mask_hash"], item["polarity"]) for item in annotations}
    if len(identities) != 1:
        raise ValueError(f"Ambiguous {task} annotations: multiple distinct mask hashes/polarities for the same image")
    selected = annotations[0]
    source = {key: value for key, value in selected.items() if key not in {"row", "mask"}}
    source.update(image_id=selected["row"]["image_id"], image_path=str(selected["row"]["image_path"]),
                  mask_path=str(selected["row"]["mask_path"]), annotation_rows=len(annotations),
                  all_annotation_paths=[str(item["row"]["mask_path"]) for item in annotations])
    return selected["mask"], source


def _pixel_values(result):
    values = result.get("diagnostics", {}).get("pixel_measurements", {})
    return {key: float(values.get(key, np.nan)) for key in PIXEL_TARGETS}


def evaluate_paired_masks(manifest, checkpoints, outdir, fold=0, device="auto", limit=None, *, max_overlays=2,
                          config=None):
    """Save a bounded, auditable pixel-geometry consistency report and return it.

    Pair only decoded image_hash values present in BOTH tasks in the held-out
    fold. Repeated identical annotations collapse to one source image; differing
    masks/polarities are explicitly excluded. Every selected paired-image group
    remains in the coverage denominator, including exclusions and failures.
    limit selects the first N hash-sorted pairs and reports the full available
    count separately. No missing measurement is replaced with zero.

    Checkpoints must be precisely one apo and one fasc checkpoint trained with
    this manifest and this held-out fold. Do not ensemble other folds here:
    those models may have trained on this cohort. No GPU training is performed.
    config optionally supplies the geometric policy; segmentation thresholds
    always come from the task checkpoints, so evaluation uses learned settings.
    """
    frame = _checked_manifest(manifest, fold)
    if limit is not None and (not isinstance(limit, (int, np.integer)) or limit <= 0):
        raise ValueError("limit must be a positive integer or None")
    if not isinstance(max_overlays, (int, np.integer)) or max_overlays < 0:
        raise ValueError("max_overlays must be a nonnegative integer")
    held = frame[frame.fold == fold]
    paired = [(key, rows) for key, rows in held.groupby("image_hash", sort=True)
              if set(rows.task) == {"apo", "fasc"}]
    available = len(paired)
    selected = paired[:limit] if limit is not None else paired
    predictor = _checked_predictor(checkpoints, frame, fold, device)
    thresholds = predictor.thresholds()
    cfg = replace(_configuration(config), apo_threshold=thresholds["apo"], fasc_threshold=thresholds["fasc"])
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "details").mkdir(exist_ok=True)
    if max_overlays:
        (out / "overlays").mkdir(exist_ok=True)
    records, detail_files, overlays = [], [], []
    for image_hash, rows in selected:
        artifact_id = hashlib.sha256(str(image_hash).encode()).hexdigest()
        record = {"image_hash": image_hash, "fold": int(fold), "status": "excluded", "reason": "",
                  "reference_geometry_status": "not_evaluated", "predicted_geometry_status": "not_evaluated",
                  "reference_complete": False, "prediction_complete": False, "both_complete": False}
        for prefix in ("reference", "predicted", "absolute_difference"):
            record.update({f"{prefix}_{key}": float("nan") for key in PIXEL_TARGETS})
        detail = {"image_hash": image_hash, "scope": SCOPE,
                  "input_rows": rows[["task", "image_id", "image_path", "mask_path"]].to_dict("records")}
        try:
            apo_rows, fasc_rows = rows[rows.task == "apo"], rows[rows.task == "fasc"]
            image = load_image(apo_rows.iloc[0].image_path)
            gt_apo, apo_source = _reference_branch(apo_rows, "apo", predictor, image_hash, image.shape[:2])
            gt_fasc, fasc_source = _reference_branch(fasc_rows, "fasc", predictor, image_hash, image.shape[:2])
            detail["reference_sources"] = {"apo": apo_source, "fasc": fasc_source}
            for task, source in (("apo", apo_source), ("fasc", fasc_source)):
                record.update({f"{task}_image_id": source["image_id"], f"{task}_image_path": source["image_path"],
                               f"{task}_mask_path": source["mask_path"], f"{task}_mask_hash": source["mask_hash"],
                               f"{task}_mask_polarity": source["polarity"], f"{task}_alignment": source["alignment"],
                               f"{task}_annotation_rows": source["annotation_rows"]})
            reference = measure_geometry(gt_apo, gt_fasc, spacing_xy=None, config=cfg)
            ref = _pixel_values(reference)
            record.update({f"reference_{key}": value for key, value in ref.items()})
            record.update(reference_complete=all(np.isfinite(list(ref.values()))),
                          reference_geometry_status=reference["status"])
            detail["reference_geometry"] = reference
            predicted_masks = predictor.predict(image)
            if any(np.asarray(predicted_masks[task]).shape != image.shape[:2] for task in ("apo", "fasc")):
                raise ValueError("Predicted masks must share the original image grid")
            prediction = measure_geometry(predicted_masks["apo"], predicted_masks["fasc"], spacing_xy=None, config=cfg)
            pred = _pixel_values(prediction)
            for key in PIXEL_TARGETS:
                record[f"reference_{key}"] = ref[key]
                record[f"predicted_{key}"] = pred[key]
                if np.isfinite(ref[key]) and np.isfinite(pred[key]):
                    record[f"absolute_difference_{key}"] = abs(pred[key] - ref[key])
            ref_complete, pred_complete = all(np.isfinite(list(ref.values()))), all(np.isfinite(list(pred.values())))
            record.update(reference_complete=ref_complete, prediction_complete=pred_complete,
                          both_complete=ref_complete and pred_complete,
                          reference_geometry_status=reference["status"], predicted_geometry_status=prediction["status"],
                          status="compared" if ref_complete and pred_complete else "partial",
                          reason="pixel_geometry_compared" if ref_complete and pred_complete else "some_pixel_geometry_unavailable")
            detail.update(reference_geometry=reference, predicted_geometry=prediction)
            if len(overlays) < max_overlays:
                try:
                    import matplotlib.pyplot as plt
                    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
                    plot_geometry(image, reference, ax=axes[0])
                    plot_geometry(image, prediction, ax=axes[1])
                    axes[0].set_title("Label-mask geometry · uncalibrated pixels")
                    axes[1].set_title("Predicted-mask geometry · uncalibrated pixels")
                    path = out / "overlays" / f"{artifact_id}.png"
                    figure.savefig(path, dpi=140)
                    plt.close(figure)
                    overlays.append(str(path))
                    record["overlay_path"] = str(path)
                except (ImportError, OSError, ValueError) as exc:
                    detail["overlay_error"] = f"{type(exc).__name__}: {exc}"
        except (ValueError, OSError, KeyError, RuntimeError) as exc:
            record.update(status="excluded", reason=f"{type(exc).__name__}: {exc}")
        detail["comparison"] = record.copy()
        path = out / "details" / f"{artifact_id}.json"
        path.write_text(json.dumps(_json(detail), indent=2, allow_nan=False))
        detail_files.append(str(path))
        records.append(record)
    base_columns = ["image_hash", "fold", "status", "reason", "reference_geometry_status", "predicted_geometry_status",
                    "reference_complete", "prediction_complete", "both_complete"]
    values_columns = [f"{prefix}_{key}" for prefix in ("reference", "predicted", "absolute_difference") for key in PIXEL_TARGETS]
    table = pd.DataFrame(records) if records else pd.DataFrame(columns=base_columns + values_columns)
    comparisons_path = out / "paired_geometry_comparisons.csv"
    table.to_csv(comparisons_path, index=False)
    count = len(table)
    metrics = []
    for key in PIXEL_TARGETS:
        errors = pd.to_numeric(table[f"absolute_difference_{key}"], errors="coerce").to_numpy(float)
        covered = np.isfinite(errors)
        references = pd.to_numeric(table[f"reference_{key}"], errors="coerce").to_numpy(float)
        predictions = pd.to_numeric(table[f"predicted_{key}"], errors="coerce").to_numpy(float)
        metrics.append({"measurement": key, "unit": "pixel-grid degrees" if key == "pa_pixel_deg" else "pixels",
                        "reference_type": "same_geometry_algorithm_on_paired_label_masks",
                        "independent_physical_ground_truth": False,
                        "mae_on_available_pairs": float(errors[covered].mean()) if covered.any() else float("nan"),
                        "median_absolute_difference": float(np.median(errors[covered])) if covered.any() else float("nan"),
                        "n_both_usable": int(covered.sum()), "n_reference_usable": int(np.isfinite(references).sum()),
                        "n_prediction_usable": int(np.isfinite(predictions).sum()), "n_evaluated_pairs": count,
                        "pair_coverage": float(covered.mean()) if count else 0.0})
    metrics_path = out / "pixel_geometry_metrics.csv"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    report = {"scope": SCOPE, "independent_physical_ground_truth": False, "physical_export_allowed": False,
              "umud_score": None, "fold": int(fold), "limit": limit,
              "training_manifest_sha256": _manifest_hash(frame), "checkpoint_provenance": predictor.provenance,
              "heldout_unique_images": int(held.image_hash.nunique()), "paired_images_available": available,
              "paired_images_evaluated": count, "unpaired_heldout_images": int(held.image_hash.nunique())-available,
              "reference_complete_pairs": int(table.reference_complete.sum()),
              "prediction_complete_pairs": int(table.prediction_complete.sum()),
              "both_complete_pairs": int(table.both_complete.sum()),
              "both_complete_coverage": float(table.both_complete.mean()) if count else 0.0,
              "excluded_pairs": int((table.status == "excluded").sum()),
              "coverage_denominator": "all selected paired image hashes, including exclusions/failures; limit subset explicitly reported",
              "geometry_config": asdict(cfg), "metrics": metrics, "comparisons_path": str(comparisons_path),
              "metrics_path": str(metrics_path), "detail_files": detail_files, "overlays": overlays,
              "limitations": ["Shared geometry errors can cancel because both sides use the same algorithm.",
                              "Pixel-grid angles are not physical angles unless aspect calibration is established.",
                              "Mask canvas differences use a full-frame resize assumption, not verified crop recovery.",
                              "Auto polarity, when requested by checkpoint/manifest, is a minority-occupancy heuristic.",
                              "Conditional errors must be read alongside coverage; missing measurements are not zero."]}
    report_path = out / "pixel_geometry_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(_json(report), indent=2, allow_nan=False))
    return _json(report)
