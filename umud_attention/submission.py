"""Strict UMUD CSV helpers for Colab/Kaggle; no downloads or submissions.

Provide observed test IDs or a directory of actual test images. The two-row
official sample is a format example, not a complete ID manifest. Numeric checks
cannot establish physical calibration, prediction accuracy, or model provenance.

    ids = load_expected_ids(manifest="observed_test_manifest.csv")
    checked = validate_submission(predictions, ids)
    deltas = compare_predictions(checked, "reference.csv", ids)
    export_submission(checked, ids, "submission.csv")

Dependencies: numpy and pandas. PA is degrees; FL and MT are millimetres.
"""

import csv
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

SCHEMA = ("image_id", "pa_deg", "fl_mm", "mt_mm")
TARGETS = SCHEMA[1:]
UNITS = {"pa_deg": "degrees", "fl_mm": "mm", "mt_mm": "mm"}
TEST_COUNT = 309
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def _ids(values, label, expected_count):
    ids = list(values)
    if any(not isinstance(v, str) or not v or v != v.strip()
           or "/" in v or "\\" in v or v in {".", ".."} for v in ids):
        raise ValueError(f"{label}: IDs must be exact nonempty image basenames")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label}: duplicate image IDs")
    if len(ids) != expected_count:
        raise ValueError(f"{label}: expected {expected_count} rows, found {len(ids)}")
    return ids


def load_expected_ids(*, manifest=None, test_dir=None, expected_count=TEST_COUNT):
    """Read image_id from a manifest, or enumerate actual image files recursively.

    Supply exactly one source. A manifest can be a CSV path, DataFrame, or an
    explicit sequence of observed IDs. Duplicate basenames are always rejected.
    """
    if (manifest is None) == (test_dir is None):
        raise ValueError("Supply exactly one of manifest or test_dir")
    if test_dir is not None:
        root = Path(test_dir)
        if not root.is_dir():
            raise ValueError(f"Test directory does not exist: {root}")
        values = sorted(p.name for p in root.rglob("*") if p.is_file()
                        and p.suffix.lower() in IMAGE_SUFFIXES
                        and not any(s.startswith(".") or s == "__MACOSX"
                                    for s in p.relative_to(root).parts))
    else:
        if isinstance(manifest, (str, Path)):
            manifest = pd.read_csv(manifest, dtype={"image_id": str},
                                   keep_default_na=False)
        if isinstance(manifest, pd.DataFrame):
            if list(manifest.columns).count("image_id") != 1:
                raise ValueError("Manifest must have one image_id column")
            values = manifest["image_id"].tolist()
        else:
            values = list(manifest)
    return _ids(values, "Expected test manifest", expected_count)


def _table(data):
    if isinstance(data, (str, Path)):
        # Inspect the raw header before pandas can rename duplicate columns.
        with Path(data).open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.reader(stream)
            header = next(reader, [])
            if any(row and len(row) != len(SCHEMA) for row in reader):
                raise ValueError("Every nonempty CSV row must contain exactly four fields")
        if len(header) != len(SCHEMA) or set(header) != set(SCHEMA):
            raise ValueError(f"Expected exactly {SCHEMA}; got {header}")
        data = pd.read_csv(data, dtype={"image_id": str}, keep_default_na=False)
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Predictions must be a CSV path or pandas DataFrame")
    if len(data.columns) != len(SCHEMA) or set(data.columns) != set(SCHEMA):
        raise ValueError(f"Expected exactly {SCHEMA}; got {list(data.columns)}")
    return data.loc[:, list(SCHEMA)].copy()


def validate_submission(predictions, expected_ids, *, expected_count=TEST_COUNT):
    """Return a validated copy in canonical column and supplied manifest order.

    Reject missing/extra/duplicate IDs, unexpected columns, and nonfinite values.
    Does not fill, clip, round, infer calibration, or assess scientific accuracy.
    """
    expected = _ids(expected_ids, "Expected test manifest", expected_count)
    frame = _table(predictions)
    actual = _ids(frame["image_id"], "Predictions", expected_count)
    if set(actual) != set(expected):
        raise ValueError(f"Test ID mismatch: missing={sorted(set(expected)-set(actual))[:5]}, "
                         f"extra={sorted(set(actual)-set(expected))[:5]}")
    for target in TARGETS:
        if frame[target].map(lambda v: isinstance(v, (bool, np.bool_))).any():
            raise ValueError(f"Boolean values are not measurements: {target}")
        frame[target] = pd.to_numeric(frame[target], errors="coerce")
        if not np.isfinite(frame[target].to_numpy(dtype=float)).all():
            raise ValueError(f"Missing, nonnumeric, or nonfinite values: {target}")
    return frame.set_index("image_id").loc[expected].reset_index()


def export_submission(predictions, expected_ids, output_path, *, expected_count=TEST_COUNT):
    """Validate, atomically write comma CSV without an index, and check read-back."""
    expected_ids = list(expected_ids)
    checked = validate_submission(predictions, expected_ids, expected_count=expected_count)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".csv", delete=False) as f:
        temporary = Path(f.name)
    try:
        checked.to_csv(temporary, index=False)
        restored = validate_submission(temporary, expected_ids, expected_count=expected_count)
        np.testing.assert_allclose(restored[list(TARGETS)].to_numpy(float),
                                   checked[list(TARGETS)].to_numpy(float), rtol=1e-12, atol=0)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def compare_predictions(candidate, reference, expected_ids, *, expected_count=TEST_COUNT):
    """Per-target absolute prediction differences, NOT ground-truth errors/score."""
    expected_ids = list(expected_ids)
    a = validate_submission(candidate, expected_ids, expected_count=expected_count)
    b = validate_submission(reference, expected_ids, expected_count=expected_count)
    with np.errstate(over="ignore", invalid="ignore"):
        delta = np.abs(a[list(TARGETS)].to_numpy(float) - b[list(TARGETS)].to_numpy(float))
    if not np.isfinite(delta).all():
        raise ValueError("Prediction differences overflowed")
    rows = []
    for i, target in enumerate(TARGETS):
        values = delta[:, i]
        rows.append({"target": target, "unit": UNITS[target],
                     "mean_abs_delta": float(np.sum(values / len(values))),
                     "median_abs_delta": float(np.median(values)),
                     "p95_abs_delta": float(np.quantile(values, .95)),
                     "max_abs_delta": float(values.max()),
                     "n_changed": int(np.count_nonzero(values))})
    result = pd.DataFrame(rows)
    result.attrs["interpretation"] = "Prediction differences only; reference is not ground truth. No accuracy score."
    return result
