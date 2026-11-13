"""TESTS: generated checkpoint/geometry contracts, never benchmark accuracy."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from umud_attention import infer
from umud_attention.calibration import infer_known_calibration
from umud_attention.data import letterbox
from umud_attention.geometry import GeometryConfig, measure_geometry


class TinyLogitModel(torch.nn.Module):
    """TEST-only persisted model; no pretrained weights or downloaded code."""
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Conv2d(3, 1, 1)

    def forward(self, image):
        return self.head(image)


@pytest.fixture
def tiny_checkpoints(tmp_path, monkeypatch):
    torch.set_num_threads(2)

    def make_model(config, pretrained=False):
        assert pretrained is False
        return TinyLogitModel()

    monkeypatch.setattr(infer, 'model_from_config', make_model)
    paths = []
    expected = {'apo': [.5, .75], 'fasc': [.25, .5]}
    for task, values in expected.items():
        for fold, probability in enumerate(values):
            model = TinyLogitModel()
            with torch.no_grad():
                model.head.weight.zero_()
                model.head.bias.fill_(np.log(probability/(1-probability)))
            path = tmp_path/f'{task}_fold{fold}.pt'
            torch.save({'config': {'task': task, 'fold': fold, 'size': 32},
                        'model': model.state_dict(), 'threshold': .3+.2*fold,
                        'manifest_sha256': 'TEST-only generated fixture'}, path)
            paths.append(path)
    return paths


def test_checkpoint_fold_flip_ensemble_restores_native_shape(tiny_checkpoints):
    predictor = infer.Predictor(tiny_checkpoints, device='cpu')
    image = np.zeros((47, 113, 3), dtype=np.uint8)
    output = predictor.predict(image, horizontal_tta=True, sizes=[32, 48])
    assert set(output) == {'apo', 'fasc'}
    assert all(array.shape == (47, 113) for array in output.values())
    np.testing.assert_allclose(output['apo'], .625, atol=1e-6)
    np.testing.assert_allclose(output['fasc'], .375, atol=1e-6)
    assert predictor.thresholds() == pytest.approx({'apo': .4, 'fasc': .4})
    assert len(predictor.provenance) == 4
    assert all(len(item['sha256']) == 64 for item in predictor.provenance)


def test_predictor_rejects_duplicate_or_unpaired_folds(tiny_checkpoints):
    with pytest.raises(ValueError, match='Duplicate'):
        infer.Predictor(tiny_checkpoints + [tiny_checkpoints[0]], device='cpu')
    with pytest.raises(ValueError, match='same fold IDs'):
        infer.Predictor(tiny_checkpoints[:-1], device='cpu')
    with pytest.raises(ValueError, match='EACH task'):
        infer.Predictor(tiny_checkpoints[:2], device='cpu')


def test_horizontal_tta_unflips_spatial_predictions(tiny_checkpoints):
    for path in tiny_checkpoints:
        state = torch.load(path, weights_only=True)
        state['model']['head.weight'][0, 0, 0, 0] = 1
        torch.save(state, path)
    predictor = infer.Predictor(tiny_checkpoints, device='cpu')
    # Portrait canvas gives asymmetric horizontal padding; red increases left
    # to right, so forgetting to unflip the TTA result would flatten this signal.
    image = np.zeros((113, 47, 3), dtype=np.uint8)
    image[:, :, 0] = np.linspace(0, 255, 47, dtype=np.uint8)
    plain = predictor.predict(image, horizontal_tta=False)
    augmented = predictor.predict(image, horizontal_tta=True)
    for task in ('apo', 'fasc'):
        assert plain[task][50, -5] > plain[task][50, 5] + .2
        np.testing.assert_allclose(plain[task], augmented[task], atol=1e-6)


def test_letterbox_restoration_removes_odd_padding():
    image = np.zeros((47, 113, 3), dtype=np.uint8)
    _, _, valid, meta = letterbox(image, size=32)
    restored = infer.restore_probability(valid, meta)
    assert restored.shape == (47, 113)
    np.testing.assert_allclose(restored, 1, atol=1e-7)


def test_restoration_rejects_wrong_canvas_shape():
    image = np.zeros((47, 113, 3), dtype=np.uint8)
    _, _, _, meta = letterbox(image, size=32)
    with pytest.raises(ValueError, match='(?i)(shape|canvas|grid)'):
        infer.restore_probability(np.ones((31, 32), dtype=np.float32), meta)


def _masks():
    apo = np.zeros((130, 240), dtype=np.float32)
    apo[17:21, 10:231] = 1
    apo[80:84, 10:231] = 1
    fasc = np.zeros_like(apo)
    for intercept in (20, 80, 140):
        for y in range(30, 71):
            fasc[y, y+intercept] = 1
    return apo, fasc


@pytest.fixture
def image_case(tmp_path, monkeypatch):
    apo, fasc = _masks()

    class OraclePredictor:
        """TEST-only segmentation oracle isolates geometry and file plumbing."""
        provenance = [{'checkpoint': 'TEST-only segmentation oracle'}]

        def __init__(self, checkpoint_paths, device='auto'):
            pass

        def thresholds(self):
            return {'apo': .5, 'fasc': .5}

        def predict(self, image):
            assert image.shape[:2] == apo.shape
            return {'apo': apo.copy(), 'fasc': fasc.copy()}

    monkeypatch.setattr(infer, 'Predictor', OraclePredictor)
    rows = []
    for image_id in ['b.png', 'a.png']:
        path = tmp_path/image_id
        Image.fromarray(np.zeros((*apo.shape, 3), np.uint8)).save(path)
        rows.append({'image_id': image_id, 'image_path': str(path)})
    return pd.DataFrame(rows)


def _calibration(tmp_path, ids):
    path = tmp_path/'calibration.csv'
    pd.DataFrame([{'image_id': image_id, 'sx_mm_per_px': .2, 'sy_mm_per_px': .2,
                   'source': 'TEST-only exact synthetic grid'} for image_id in ids]).to_csv(path, index=False)
    return path


def test_consensus_records_sensitivity_not_confidence_interval():
    apo, fasc = _masks()
    result = infer.geometry_consensus(apo, fasc, (.2, .2), GeometryConfig())
    assert result['status'] == 'ok'
    assert result['diagnostics']['threshold_success_count'] == 3
    assert result['diagnostics']['stable_thresholds'] is True
    assert result['diagnostics']['threshold_measurement_range'] == pytest.approx({key: 0 for key in infer.TARGETS})
    assert result['diagnostics']['confidence_is_calibrated'] is False


def test_single_threshold_recovery_is_explicitly_unstable():
    apo, fasc = _masks()
    result = infer.geometry_consensus(apo*.45, fasc, (.2, .2), GeometryConfig())
    assert result['status'] == 'ok'
    assert result['diagnostics']['threshold_success_count'] == 1
    assert result['diagnostics']['stable_thresholds'] is False
    assert result['fl_mm'] == pytest.approx(12*np.sqrt(2))


def test_explicit_roi_masks_are_reembedded_on_original_grid(image_case, tmp_path, monkeypatch):
    apo, fasc = _masks()
    roi = (10, 10, 230, 120)

    class CropPredictor:
        provenance = [{'checkpoint': 'TEST-only cropped segmentation oracle'}]
        def __init__(self, *args, **kwargs):
            pass
        def thresholds(self):
            return {'apo': .5, 'fasc': .5}
        def predict(self, image):
            assert image.shape[:2] == (110, 220)
            return {'apo': apo[10:120, 10:230], 'fasc': fasc[10:120, 10:230]}

    monkeypatch.setattr(infer, 'Predictor', CropPredictor)
    monkeypatch.setattr(infer, 'infer_known_calibration', lambda _: {
        'roi': roi, 'sx_mm_per_px': .2, 'sy_mm_per_px': .2,
        'source': 'TEST-only synthetic ROI and physical grid', 'square_pixel_assumption': False})
    report = infer.predict_dataset(image_case, [], tmp_path/'run', use_known_rulers=True, expected_count=2)
    restored = np.load(tmp_path/'run'/'masks'/'b.png.npz')
    assert restored['apo'].shape == (130, 240)
    np.testing.assert_allclose(restored['apo'][10:120, 10:230], apo[10:120, 10:230])
    assert not restored['apo'][:10].any()
    assert report['complete_measurements'] == 2
    result = pd.read_csv(report['standalone_submission'])
    np.testing.assert_allclose(result.mt_mm, 12, atol=1e-6)


def test_missing_scale_never_becomes_physical_predictions(image_case, tmp_path):
    report = infer.predict_dataset(image_case, [], tmp_path/'run', expected_count=2)
    assert report['complete_measurements'] == 0
    assert report['standalone_submission'] is None
    saved = pd.read_csv(tmp_path/'run'/'measurements_partial.csv')
    assert saved.image_id.tolist() == ['b.png', 'a.png']
    assert saved[list(infer.TARGETS)].isna().all().all()
    detail = json.loads((tmp_path/'run'/'geometry_details.json').read_text())
    assert detail[0]['geometry']['fl_mm'] is None
    assert not detail[0]['geometry']['diagnostics']['physical_export_allowed']
    assert not (tmp_path/'run'/'submission_attention.csv').exists()


def test_narrow_ultrasound_roi_uses_its_support_width_on_original_grid(tmp_path, monkeypatch):
    # TEST: observed viewport dimensions, but exact generated masks. y200/y400
    # boundaries and y=x-250 fascicle imply200px x/y spans on original grid.
    roi = (457, 117, 743, 722)
    apo = np.zeros((800, 1200), np.float32)
    apo[197:201, 457:743] = 1
    apo[400:404, 457:743] = 1
    fasc = np.zeros_like(apo)
    for y in range(250, 351): fasc[y, y+250] = 1
    before = measure_geometry(apo, fasc, (.4, .2))
    assert before['status'] == 'failed'
    assert before['reason'] == 'no_supported_aponeurosis_pair'
    class NarrowOracle:
        provenance = [{'checkpoint': 'TEST-only narrow-ROI oracle'}]
        def __init__(self, *args, **kwargs): pass
        def thresholds(self): return {'apo': .5, 'fasc': .5}
        def predict(self, image):
            assert image.shape[:2] == (605, 286)
            return {'apo': apo[117:722, 457:743], 'fasc': fasc[117:722, 457:743]}
    monkeypatch.setattr(infer, 'Predictor', NarrowOracle)
    monkeypatch.setattr(infer, 'infer_known_calibration', lambda _: {
        'roi': roi, 'sx_mm_per_px': .4, 'sy_mm_per_px': .2,
        'source': 'TEST-only known anisotropic pixel grid', 'square_pixel_assumption': False})
    image_path = tmp_path/'narrow.png'
    Image.fromarray(np.zeros((800, 1200, 3), np.uint8)).save(image_path)
    manifest = pd.DataFrame([{'image_id': image_path.name, 'image_path': str(image_path)}])
    report = infer.predict_dataset(manifest, [], tmp_path/'out', use_known_rulers=True, expected_count=1)
    assert report['complete_measurements'] == 1
    result = pd.read_csv(report['standalone_submission']).iloc[0]
    assert result.pa_deg == pytest.approx(np.degrees(np.arctan(.5)), abs=1e-6)
    assert result.fl_mm == pytest.approx(np.hypot(80, 40), abs=1e-6)
    assert result.mt_mm == pytest.approx(40, abs=1e-6)
    detail = json.loads((tmp_path/'out/geometry_details.json').read_text())[0]
    audit = detail['roi_support_normalization']
    assert audit['minimum_apo_span_original_px'] == pytest.approx(71.25)
    assert audit['minimum_pair_overlap_original_px'] == pytest.approx(57)
    assert audit['other_geometry_bounds_changed'] is False
    assert detail['effective_geometry_config']['max_fascicle_extrapolation_ratio'] == 1.5
    np.testing.assert_allclose(detail['geometry']['diagnostics']['segments'][0]['intersections_px'],
                               [[450, 200], [650, 400]], atol=1e-6)


def test_full_frame_roi_keeps_geometry_configuration_unchanged():
    config = GeometryConfig()
    effective, evidence = infer.geometry_config_for_roi(config, (800, 1200), (0, 0, 1200, 800))
    assert effective == config
    assert evidence['span_normalization_ratio'] == 1


@pytest.mark.parametrize('roi', [(0, 0, 1201, 800), (20, 10, 21, 100), (0.5, 0, 100, 100)])
def test_malformed_roi_cannot_change_geometry_support(roi):
    with pytest.raises(ValueError, match='ROI'):
        infer.geometry_config_for_roi(GeometryConfig(), (800, 1200), roi)


def test_standalone_submission_retains_manifest_order(image_case, tmp_path):
    calibration = _calibration(tmp_path, ['a.png', 'b.png'])
    report = infer.predict_dataset(image_case, [], tmp_path/'run', calibration_csv=calibration,
                                  expected_count=2)
    assert report['complete_measurements'] == 2
    frame = pd.read_csv(report['standalone_submission'])
    assert frame.image_id.tolist() == ['b.png', 'a.png']
    assert list(frame) == ['image_id', 'pa_deg', 'fl_mm', 'mt_mm']
    np.testing.assert_allclose(frame.pa_deg, 45, atol=1e-6)
    np.testing.assert_allclose(frame.fl_mm, 12*np.sqrt(2), atol=1e-6)
    np.testing.assert_allclose(frame.mt_mm, 12, atol=1e-6)
    mask = np.load(tmp_path/'run'/'masks'/'b.png.npz')
    assert mask['apo'].shape == (130, 240)


def test_reference_hybrid_aligns_ids_and_only_changes_supported_rows(image_case, tmp_path):
    calibration = _calibration(tmp_path, ['b.png'])
    # Deliberately reverse the reference file order relative to manifest.
    reference = pd.DataFrame([{'image_id': 'a.png', 'pa_deg': 11., 'fl_mm': 61., 'mt_mm': 21.},
                              {'image_id': 'b.png', 'pa_deg': 22., 'fl_mm': 72., 'mt_mm': 32.}])
    reference_path = tmp_path/'reference.csv'
    reference.to_csv(reference_path, index=False)
    report = infer.predict_dataset(image_case, [], tmp_path/'run', calibration_csv=calibration,
                                  reference_csv=reference_path, blend_weight=.2, expected_count=2)
    assert report['standalone_submission'] is None
    assert report['images_changed'] == 1
    assert report['images_retaining_reference'] == 1
    hybrid = pd.read_csv(report['hybrid_submission'])
    assert hybrid.image_id.tolist() == ['b.png', 'a.png']
    np.testing.assert_allclose(hybrid.loc[0, list(infer.TARGETS)].to_numpy(float),
                               .8*np.array([22, 72, 32]) + .2*np.array([45, 12*np.sqrt(2), 12]))
    np.testing.assert_allclose(hybrid.loc[1, list(infer.TARGETS)].to_numpy(float), [11, 61, 21])
    assert report['reference_sha256'] == infer.file_hash(reference_path)
    assert 'reference' in report['hybrid_limitation']


@pytest.mark.parametrize('bad_id', ['../escape.png', '', None])
def test_invalid_image_ids_rejected_before_any_mask_write(image_case, tmp_path, bad_id):
    image_case.loc[0, 'image_id'] = bad_id
    with pytest.raises(ValueError, match='(?i)(ID|basename|manifest)'):
        infer.predict_dataset(image_case, [], tmp_path/'run', expected_count=2)
    assert not (tmp_path/'escape.png.npz').exists()


def test_failed_image_retains_a_failed_row_and_other_results(image_case, tmp_path):
    missing = pd.DataFrame([{'image_id': 'missing.png', 'image_path': str(tmp_path/'absent.png')}])
    manifest = pd.concat([image_case, missing], ignore_index=True)
    calibration = _calibration(tmp_path, manifest.image_id.tolist())
    report = infer.predict_dataset(manifest, [], tmp_path/'run', calibration_csv=calibration,
                                  expected_count=3, save_masks=False)
    assert report['n_images'] == 3
    assert report['complete_measurements'] == 2
    quality = pd.read_csv(tmp_path/'run'/'quality.csv')
    assert quality.image_id.tolist() == ['b.png', 'a.png', 'missing.png']
    assert quality.iloc[-1].status == 'failed'
    saved = pd.read_csv(tmp_path/'run'/'measurements_partial.csv')
    assert saved.iloc[-1][list(infer.TARGETS)].isna().all()


def test_unknown_ruler_layout_is_not_assigned_a_default_scale(tmp_path):
    path = tmp_path/'unknown.png'
    Image.fromarray(np.zeros((130, 240, 3), np.uint8)).save(path)
    with pytest.raises(ValueError, match='Unknown ruler layout'):
        infer_known_calibration(path)


@pytest.mark.parametrize('column,value', [('sx_mm_per_px', 0), ('sy_mm_per_px', np.nan), ('source', '')])
def test_calibration_requires_positive_scale_and_source(tmp_path, column, value):
    path = _calibration(tmp_path, ['a.png'])
    frame = pd.read_csv(path)
    frame.loc[0, column] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        infer.read_calibration(path, ['a.png'])
