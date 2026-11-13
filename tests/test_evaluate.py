"""TESTS: analytic segmentation fixtures; no physical or leaderboard claims."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from umud_attention import evaluate
from umud_attention.data import _fingerprint, _mask_metadata


def masks():
    apo = np.zeros((130, 240), np.uint8)
    apo[17:21, 10:231] = 255
    apo[80:84, 10:231] = 255
    fasc = np.zeros_like(apo)
    for intercept in (20, 80, 140):
        for y in range(30, 71):
            fasc[y, y+intercept] = 255
    return {'apo': apo, 'fasc': fasc}


@pytest.fixture
def paired_manifest(tmp_path):
    rows = []
    for index, tasks in enumerate([('apo', 'fasc'), ('apo', 'fasc'), ('apo',)]):
        image_path = tmp_path/f'image_{index}.png'
        Image.fromarray(np.full((130, 240, 3), 10*(index+1), np.uint8)).save(image_path)
        fingerprint = _fingerprint(image_path)
        for task in tasks:
            mask_path = tmp_path/f'{task}_{index}.png'
            Image.fromarray(masks()[task]).save(mask_path)
            rows.append({'task': task, 'image_id': f'{task}_{index}.png',
                         'image_path': str(image_path), 'mask_path': str(mask_path),
                         **fingerprint, **_mask_metadata(mask_path),
                         'group': f'group_{index}', 'fold': 0, 'mask_polarity': 'light'})
    return pd.DataFrame(rows)


def use_oracle(monkeypatch, frame, *, missing_marker=None, checkpoint_fold=0, manifest_hash=None):
    class Predictor:
        def __init__(self, checkpoints, device='auto'):
            self.provenance = [{'task': task, 'fold': checkpoint_fold,
                                'config': {'task': task, 'fold': checkpoint_fold, 'mask_polarity': 'light'},
                                'manifest_sha256': manifest_hash or evaluate._manifest_hash(frame),
                                'checkpoint': f'TEST-only-{task}-oracle'} for task in ('apo', 'fasc')]
        def thresholds(self):
            return {'apo': .5, 'fasc': .5}
        def predict(self, image):
            output = {key: value.astype(np.float32)/255 for key, value in masks().items()}
            if int(image[0, 0, 0]) == missing_marker:
                output['fasc'][:] = 0
            return output
    monkeypatch.setattr(evaluate, 'Predictor', Predictor)


def test_same_hash_pairing_known_geometry_and_no_physical_outputs(paired_manifest, tmp_path, monkeypatch):
    use_oracle(monkeypatch, paired_manifest)
    report = evaluate.evaluate_paired_masks(paired_manifest, [], tmp_path/'out', max_overlays=0)
    assert report['paired_images_available'] == 2
    assert report['paired_images_evaluated'] == 2
    assert report['unpaired_heldout_images'] == 1
    assert report['both_complete_pairs'] == 2
    assert report['both_complete_coverage'] == 1
    assert report['independent_physical_ground_truth'] is False
    assert report['physical_export_allowed'] is False
    assert report['umud_score'] is None
    table = pd.read_csv(report['comparisons_path'])
    np.testing.assert_allclose(table.reference_pa_pixel_deg, 45, atol=1e-7)
    np.testing.assert_allclose(table.reference_fl_px, 60*np.sqrt(2), atol=1e-7)
    np.testing.assert_allclose(table.reference_mt_px, 60, atol=1e-7)
    assert all(metric['mae_on_available_pairs'] == 0 for metric in report['metrics'])
    assert not {'pa_deg', 'fl_mm', 'mt_mm'} & set(table.columns)
    saved = json.loads(Path(report['report_path']).read_text())
    assert saved['scope'].startswith('Mask-derived pixel geometry consistency')


def test_missing_geometry_reduces_coverage_never_counts_as_zero_error(paired_manifest, tmp_path, monkeypatch):
    use_oracle(monkeypatch, paired_manifest, missing_marker=20)
    report = evaluate.evaluate_paired_masks(paired_manifest, [], tmp_path/'out', max_overlays=0)
    metrics = {row['measurement']: row for row in report['metrics']}
    assert report['both_complete_coverage'] == .5
    assert metrics['fl_px']['n_both_usable'] == 1
    assert metrics['fl_px']['pair_coverage'] == .5
    assert metrics['fl_px']['n_reference_usable'] == 2
    assert metrics['mt_px']['n_both_usable'] == 2
    table = pd.read_csv(report['comparisons_path'])
    assert table.absolute_difference_fl_px.isna().sum() == 1


def test_zero_coverage_mae_is_null_not_zero(paired_manifest, tmp_path, monkeypatch):
    frame = paired_manifest[paired_manifest.image_path.str.endswith('image_1.png')].copy()
    use_oracle(monkeypatch, frame, missing_marker=20)
    report = evaluate.evaluate_paired_masks(frame, [], tmp_path/'out', max_overlays=0)
    metrics = {row['measurement']: row for row in report['metrics']}
    assert metrics['fl_px']['n_both_usable'] == 0
    assert metrics['fl_px']['mae_on_available_pairs'] is None
    assert metrics['fl_px']['pair_coverage'] == 0


def test_different_annotation_hash_is_excluded_with_denominator_preserved(paired_manifest, tmp_path, monkeypatch):
    extra = paired_manifest.iloc[0].copy()
    extra['image_id'] = 'conflicting_annotation.png'
    new_path = tmp_path/'conflicting_mask.png'
    pixels = masks()['apo'].copy()
    pixels[4, 4] = 255
    Image.fromarray(pixels).save(new_path)
    extra['mask_path'] = str(new_path)
    for key, value in _mask_metadata(new_path).items():
        extra[key] = value
    frame = pd.concat([paired_manifest, pd.DataFrame([extra])], ignore_index=True)
    use_oracle(monkeypatch, frame)
    report = evaluate.evaluate_paired_masks(frame, [], tmp_path/'out', max_overlays=0)
    assert report['paired_images_evaluated'] == 2
    assert report['excluded_pairs'] == 1
    assert report['both_complete_coverage'] == .5
    table = pd.read_csv(report['comparisons_path'])
    assert table.reason.str.contains('Ambiguous apo annotations').any()


def test_identical_duplicate_annotation_is_collapsed_not_double_counted(paired_manifest, tmp_path, monkeypatch):
    extra = paired_manifest.iloc[0].copy()
    extra['image_id'] = 'duplicate_annotation.png'
    frame = pd.concat([paired_manifest, pd.DataFrame([extra])], ignore_index=True)
    use_oracle(monkeypatch, frame)
    report = evaluate.evaluate_paired_masks(frame, [], tmp_path/'out', max_overlays=0)
    assert report['paired_images_evaluated'] == 2
    assert report['excluded_pairs'] == 0
    table = pd.read_csv(report['comparisons_path'])
    assert table.apo_annotation_rows.max() == 2


@pytest.mark.parametrize('bad', ['fold', 'manifest'])
def test_checkpoint_provenance_must_match_heldout_cohort(paired_manifest, tmp_path, monkeypatch, bad):
    use_oracle(monkeypatch, paired_manifest, checkpoint_fold=1 if bad == 'fold' else 0,
               manifest_hash='wrong' if bad == 'manifest' else None)
    with pytest.raises(ValueError, match='(?i)(fold|manifest)'):
        evaluate.evaluate_paired_masks(paired_manifest, [], tmp_path/'out', max_overlays=0)


def test_same_image_cannot_appear_in_training_and_evaluation_fold(paired_manifest, tmp_path, monkeypatch):
    frame = paired_manifest.copy()
    frame.loc[1, 'fold'] = 1
    frame.loc[1, 'group'] = 'misgrouped'
    with pytest.raises(ValueError, match='Same image hash crosses'):
        evaluate.evaluate_paired_masks(frame, [], tmp_path/'out', max_overlays=0)


def test_acquisition_group_cannot_cross_folds(paired_manifest, tmp_path):
    frame = paired_manifest.copy()
    frame.loc[1, 'fold'] = 1
    with pytest.raises(ValueError, match='group crosses'):
        evaluate.evaluate_paired_masks(frame, [], tmp_path/'out', max_overlays=0)


def test_limit_reports_full_available_and_actual_denominator(paired_manifest, tmp_path, monkeypatch):
    use_oracle(monkeypatch, paired_manifest)
    report = evaluate.evaluate_paired_masks(paired_manifest, [], tmp_path/'out', limit=1, max_overlays=0)
    assert report['paired_images_available'] == 2
    assert report['paired_images_evaluated'] == 1
    assert report['both_complete_coverage'] == 1
    assert all(row['n_evaluated_pairs'] == 1 for row in report['metrics'])


def test_dark_annotation_policy_and_explicit_fullframe_alignment(paired_manifest, tmp_path, monkeypatch):
    frame = paired_manifest.copy()
    index = frame.index[frame.task == 'apo'][0]
    path = Path(frame.loc[index, 'mask_path'])
    pixels = 255-masks()['apo']
    # Deliberately different annotation canvas. Full-frame mapping is recorded.
    Image.fromarray(pixels).resize((480, 260), Image.Resampling.NEAREST).save(path)
    for key, value in _mask_metadata(path).items():
        frame.loc[index, key] = value
    frame.loc[index, 'mask_polarity'] = 'dark'
    use_oracle(monkeypatch, frame)
    report = evaluate.evaluate_paired_masks(frame, [], tmp_path/'out', max_overlays=0)
    assert report['both_complete_pairs'] == 2
    table = pd.read_csv(report['comparisons_path'])
    assert 'full_frame_nearest_explicit_assumption' in set(table.apo_alignment)
    assert 'dark' in set(table.apo_mask_polarity)
    assert all(row['mae_on_available_pairs'] == 0 for row in report['metrics'])


def test_changed_image_content_is_excluded(paired_manifest, tmp_path, monkeypatch):
    use_oracle(monkeypatch, paired_manifest)
    path = Path(paired_manifest.iloc[0].image_path)
    Image.fromarray(np.full((130, 240, 3), 99, np.uint8)).save(path)
    report = evaluate.evaluate_paired_masks(paired_manifest, [], tmp_path/'out', max_overlays=0)
    assert report['excluded_pairs'] == 1
    table = pd.read_csv(report['comparisons_path'])
    assert table.reason.str.contains('Source image changed').any()


def test_empty_paired_cohort_has_explicit_zero_coverage(paired_manifest, tmp_path, monkeypatch):
    frame = paired_manifest[paired_manifest.task == 'apo'].copy()
    use_oracle(monkeypatch, frame)
    report = evaluate.evaluate_paired_masks(frame, [], tmp_path/'out', max_overlays=0)
    assert report['paired_images_available'] == report['paired_images_evaluated'] == 0
    assert report['both_complete_coverage'] == 0
    assert all(metric['mae_on_available_pairs'] is None for metric in report['metrics'])


def test_prediction_failure_preserves_available_label_geometry(paired_manifest, tmp_path, monkeypatch):
    use_oracle(monkeypatch, paired_manifest)
    previous = evaluate.Predictor
    class FailingPredictor(previous):
        def predict(self, image):
            raise RuntimeError('TEST-only model failure')
    monkeypatch.setattr(evaluate, 'Predictor', FailingPredictor)
    report = evaluate.evaluate_paired_masks(paired_manifest, [], tmp_path/'out', max_overlays=0)
    assert report['reference_complete_pairs'] == 2
    assert report['prediction_complete_pairs'] == 0
    assert report['excluded_pairs'] == 2
    assert all(metric['n_reference_usable'] == 2 for metric in report['metrics'])
    assert all(metric['mae_on_available_pairs'] is None for metric in report['metrics'])


def test_explicit_geometry_policy_uses_checkpoint_thresholds(paired_manifest, tmp_path, monkeypatch):
    use_oracle(monkeypatch, paired_manifest)
    report = evaluate.evaluate_paired_masks(paired_manifest, [], tmp_path/'policy', max_overlays=0,
        config={'pennation_mode': 'local_orientation', 'max_fascicle_extrapolation_ratio': 0,
                'apo_threshold': .2, 'fasc_threshold': .3})
    config = report['geometry_config']
    assert config['apo_threshold'] == config['fasc_threshold'] == .5
    assert config['pennation_mode'] == 'local_orientation'
    metrics = {row['measurement']: row for row in report['metrics']}
    assert metrics['pa_pixel_deg']['n_both_usable'] == 2
    assert metrics['pa_pixel_deg']['mae_on_available_pairs'] == 0
    assert metrics['fl_px']['n_both_usable'] == 0
