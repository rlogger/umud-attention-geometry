"""Scalar scoring must reject absent independent evaluation evidence."""

import pandas as pd
import pytest

from umud_attention.workflow import score_measurements


@pytest.mark.parametrize("as_csv", [False, True])
def test_empty_scalar_ground_truth_cannot_produce_a_score(tmp_path, as_csv):
    empty = pd.DataFrame(columns=["image_id", "pa_deg", "fl_mm", "mt_mm"])
    ground_truth = empty
    if as_csv:
        ground_truth = tmp_path / "empty_manual_measurements.csv"
        empty.to_csv(ground_truth, index=False)
    with pytest.raises(ValueError, match="at least one image"):
        score_measurements(ground_truth, empty, provenance="Independent manual measurement cohort")


def test_nonempty_scalar_ground_truth_remains_scoreable():
    measured = pd.DataFrame({"image_id": ["example.png"], "pa_deg": [20.],
                             "fl_mm": [60.], "mt_mm": [20.]})
    result = score_measurements(measured, measured.copy(), provenance="Independent manual measurement")
    assert result["score"] == 0.0
    assert result["n"] == 1
