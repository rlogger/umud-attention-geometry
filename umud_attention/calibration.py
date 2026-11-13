"""EXPERIMENTAL opt-in ruler candidates for two known public UMUD layouts.

Adapted from the verified Quick and Dirty version-1 template rules:
https://www.kaggle.com/code/ambrosm/umud-quick-and-dirty

These rules encode known display ruler units; they are not a general scale
detector. Vertical spacing is derived from ruler marks; equal horizontal
spacing is an explicit square-pixel assumption. Review candidates on actual
competition images before trusting metric outputs. Unknown layouts fail.
No physiological output or typical muscle dimension is used as a fallback.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

def read_table(path):
    return pd.read_csv(path)


def _nonempty_ids(frame, label):
    if "image_id" not in frame or frame.image_id.isna().any() or frame.image_id.duplicated().any():
        raise ValueError(f"{label}: missing or duplicate image IDs")
    if frame.image_id.astype(str).str.strip().eq("").any():
        raise ValueError(f"{label}: empty image IDs")
    return frame.image_id.tolist()


REFERENCE = "https://www.kaggle.com/code/ambrosm/umud-quick-and-dirty (version 1)"
TIFF_ROI = {7: (142, 1058), 8: (162, 1038), 9: (210, 990),
            10: (249, 951), 11: (281, 919), 12: (307, 893),
            13: (330, 870), 14: (349, 851), 15: (142, 1058)}


def _first(rows, label):
    if len(rows) == 0:
        raise ValueError(f"Known-layout signature missing {label}")
    return int(rows[0])


def _last(rows, label):
    if len(rows) == 0:
        raise ValueError(f"Known-layout signature missing {label}")
    return int(rows[-1])


def _check_roi(roi, width, height):
    left, top, right, bottom = roi
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f"Known-layout ROI is out of bounds or empty: {roi}")


def infer_known_calibration(image_path):
    """Return an experimental calibration candidate or raise ValueError.

    Both layouts must be native uint8 RGB, H=800 W=1200. PNG additionally
    requires two ruler-column detections and a matching 2.8..2.9 cm ROI width.
    TIFF additionally requires the exact gray-175 ruler signature, recognized
    tick count, and matching 5.7..5.8 cm ROI width. Format is read from file
    contents, not guessed from its suffix. ROI is (left,top,right,bottom) with
    exclusive right/bottom; spacing always refers to the ORIGINAL input grid.
    """
    with Image.open(image_path) as image:
        fmt, mode = image.format, image.mode
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Unknown ruler layout: multiframe image")
        pixels = np.asarray(image)
    if mode != "RGB" or pixels.dtype != np.uint8 or pixels.shape != (800, 1200, 3):
        raise ValueError(f"Unknown ruler layout: expected native uint8 RGB H800 W1200, got {mode} {pixels.shape}")
    gray = pixels.mean(axis=2)
    height, width = gray.shape
    if fmt == "PNG":
        rows6 = np.flatnonzero(gray[:, 6] > 50)
        first = _first(rows6, "PNG first tick at x=6")
        minor = _first(rows6[rows6 >= 150], "PNG minor tick after row 150")
        rows9 = np.flatnonzero(gray[:, 9] > 50)
        major = _first(rows9[rows9 >= 150], "PNG major tick after row 150")
        last = _last(rows6, "PNG last tick at x=6")
        if not (0 <= first < minor <= major <= last < height):
            raise ValueError("PNG ruler detections are not ordered positive intervals")
        if major - first < 3 * (minor - first):
            px_per_cm = float(major - first)
            ruler_units = "major=10mm, minor=5mm"
        else:
            px_per_cm = float(minor - first)
            ruler_units = "major=50mm, minor=10mm"
        half_width = int(np.argmin(gray[:, 600:].sum(axis=0)))
        roi = (600 - half_width, first, 600 + half_width, last)
        _check_roi(roi, width, height)
        width_cm = (roi[2] - roi[0]) / px_per_cm
        if not 2.8 < width_cm < 2.9:
            raise ValueError(f"PNG recognized-device width check failed: {width_cm:.6g} cm")
        template = "quick_dirty_v1_png_ruler_1200x800"
        evidence = {"first_tick_row": first, "minor_tick_row": minor, "major_tick_row": major,
                    "last_tick_row": last, "ruler_units": ruler_units, "roi_width_cm": width_cm}
    elif fmt == "TIFF":
        if not np.all(pixels[87, 1147:1157, :] == 175):
            raise ValueError("Unknown TIFF layout: exact gray-175 signature at row87 x1147:1157 absent")
        tick_rows = np.flatnonzero(gray[:, 1150] > 50)
        first = _first(tick_rows, "TIFF first tick")
        second = _first(tick_rows[tick_rows >= first + 20], "TIFF second tick at least 20 rows later")
        last = _last(tick_rows, "TIFF last tick")
        if not (0 <= first < second <= last < height):
            raise ValueError("TIFF ruler detections are not ordered positive intervals")
        intervals = int(round((last - first) / (second - first)))
        if intervals not in TIFF_ROI:
            raise ValueError(f"Unknown TIFF ruler interval count: {intervals}")
        if intervals == 15:
            px_per_cm = (last - first) / 3.0
            ruler_units = "15 subdivisions span 30mm"
        else:
            px_per_cm = (last - first) / intervals * 2.0
            ruler_units = "minor=5mm"
        left, right = TIFF_ROI[intervals]
        roi = (left, 91, right, last)
        _check_roi(roi, width, height)
        width_cm = (right - left) / px_per_cm
        if not 5.7 < width_cm < 5.8:
            raise ValueError(f"TIFF recognized-device width check failed: {width_cm:.6g} cm")
        template = "quick_dirty_v1_tiff_gray175_1200x800"
        evidence = {"first_tick_row": first, "second_tick_row": second, "last_tick_row": last,
                    "ruler_intervals": intervals, "ruler_units": ruler_units, "roi_width_cm": width_cm}
    else:
        raise ValueError(f"Unknown ruler layout: unsupported file format {fmt}")
    if not np.isfinite(px_per_cm) or px_per_cm <= 0:
        raise ValueError("Ruler pixels-per-centimetre must be finite and positive")
    mm_per_px = 10.0 / px_per_cm
    return {"sx_mm_per_px": float(mm_per_px), "sy_mm_per_px": float(mm_per_px),
            "source": f"EXPERIMENTAL known-display ruler template {template}; {REFERENCE}; "
                      "vertical ruler calibration; horizontal spacing ASSUMES square pixels; review required",
            "template": template, "roi": tuple(map(int, roi)),
            "px_per_cm": float(px_per_cm), "experimental": True, "review_required": True,
            "square_pixel_assumption": True, "calibration_method": "known_public_test_layout_ruler_template",
            "ruler_evidence": evidence}


def calibrate_manifest(manifest):
    """Return (candidate_dataframe, failures_dataframe) without hiding failures.

    This only proposes explicit template candidates; it does not authorize a
    physical export or save a verified-calibration file. Duplicate IDs reject
    the manifest; each unreadable/unknown image receives its own failure row.
    """
    frame = read_table(manifest) if isinstance(manifest, (str, Path)) else pd.DataFrame(manifest).copy()
    _nonempty_ids(frame, "calibration manifest")
    if "image_path" not in frame:
        raise ValueError("Calibration manifest needs image_path")
    candidates, failures = [], []
    for row in frame.to_dict(orient="records"):
        try:
            candidates.append({"image_id": row["image_id"], "image_path": str(row["image_path"]),
                               **infer_known_calibration(row["image_path"])})
        except Exception as exc:
            failures.append({"image_id": row["image_id"], "image_path": str(row["image_path"]),
                             "reason": f"{type(exc).__name__}: {exc}"})
    candidate_columns = ["image_id", "image_path", "sx_mm_per_px", "sy_mm_per_px", "source", "template",
                         "roi", "px_per_cm", "experimental", "review_required", "square_pixel_assumption",
                         "calibration_method", "ruler_evidence"]
    return (pd.DataFrame(candidates, columns=candidate_columns),
            pd.DataFrame(failures, columns=["image_id", "image_path", "reason"]))
