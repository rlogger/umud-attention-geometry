"""TESTS: synthetic mathematical fixtures, NOT competition accuracy evidence."""

import numpy as np
import pytest

try:
    from .geometry import GeometryConfig, measure_geometry
except ImportError:
    from geometry import GeometryConfig, measure_geometry


def exact_fixture():
    """TEST fixture: boundaries y20/y80, diagonal fascicles slope1."""
    apo = np.zeros((130, 240), dtype=bool)
    apo[17:21, 10:231] = True
    apo[80:84, 10:231] = True
    fasc = np.zeros_like(apo)
    for intercept in (20, 80, 140):
        for y in range(30, 71):
            fasc[y, y + intercept] = True
    return apo, fasc


def test_exact_original_grid_geometry():
    apo, fasc = exact_fixture()
    result = measure_geometry(apo, fasc, (.2, .2))
    assert result['status'] == 'ok', result
    assert result['pa_deg'] == pytest.approx(45, abs=1e-8)
    assert result['fl_mm'] == pytest.approx(12 * np.sqrt(2), abs=1e-8)
    assert result['mt_mm'] == pytest.approx(12, abs=1e-8)
    assert result['diagnostics']['n_accepted_fascicles'] == 3
    np.testing.assert_allclose(result['diagnostics']['segments'][0]['intersections_px'],
                               [[40, 20], [100, 80]], atol=1e-8)


def test_anisotropic_spacing_changes_physical_angle():
    apo, fasc = exact_fixture()
    result = measure_geometry(apo, fasc, (.4, .2))
    assert result['status'] == 'ok'
    assert result['pa_deg'] == pytest.approx(np.degrees(np.arctan(.5)), abs=1e-8)
    assert result['fl_mm'] == pytest.approx(np.sqrt(24**2 + 12**2), abs=1e-8)
    assert result['mt_mm'] == pytest.approx(12, abs=1e-8)
    assert result['diagnostics']['pixel_segment_lengths'][0] == pytest.approx(60*np.sqrt(2))


def test_uniform_scale_changes_lengths_only():
    apo, fasc = exact_fixture()
    a = measure_geometry(apo, fasc, (.2, .2))
    b = measure_geometry(apo, fasc, (2, 2))
    assert b['pa_deg'] == pytest.approx(a['pa_deg'])
    for name in ('fl_mm', 'mt_mm'):
        assert b[name] == pytest.approx(10 * a[name])


def test_horizontal_reflection_preserves_measurements():
    apo, fasc = exact_fixture()
    a = measure_geometry(apo, fasc, (.2, .2))
    b = measure_geometry(np.fliplr(apo), np.fliplr(fasc), (.2, .2))
    for name in ('pa_deg', 'fl_mm', 'mt_mm'):
        assert b[name] == pytest.approx(a[name], abs=1e-8)


def test_pair_selected_using_fascicle_support_not_outermost_surfaces():
    apo, fasc = exact_fixture()
    apo[105:109, 10:231] = True
    result = measure_geometry(apo, fasc, (.2, .2))
    assert result['status'] == 'ok'
    assert result['mt_mm'] == pytest.approx(12)
    assert result['diagnostics']['selected_component_ids'] == [1, 2]


def test_similarly_supported_muscle_layers_are_ambiguous():
    apo = np.zeros((190, 240), dtype=bool)
    apo[17:21, 10:231] = True
    apo[80:84, 10:231] = True
    apo[143:147, 10:231] = True
    fasc = np.zeros_like(apo)
    for y in range(30, 71):
        fasc[y, y+20] = True
        fasc[y+63, y+20] = True
    result = measure_geometry(apo, fasc, (.2, .2))
    assert result['status'] == 'failed'
    assert result['reason'] == 'ambiguous_aponeurosis_pair'


def test_missing_spacing_keeps_pixel_diagnostics_without_metric_outputs():
    apo, fasc = exact_fixture()
    result = measure_geometry(apo, fasc, None)
    assert result['status'] == 'partial'
    assert result['reason'] == 'missing_physical_spacing'
    assert all(np.isnan(result[name]) for name in ('pa_deg', 'fl_mm', 'mt_mm'))
    pixels = result['diagnostics']['pixel_measurements']
    assert pixels['fl_px'] == pytest.approx(60*np.sqrt(2))
    assert pixels['mt_px'] == pytest.approx(60)
    assert not result['diagnostics']['physical_export_allowed']


@pytest.mark.parametrize('spacing', [(.1, 0), (-1, .1), (np.nan, .1), (.1, np.inf), .1, (.1,)])
def test_bad_scale_is_not_replaced(spacing):
    apo, fasc = exact_fixture()
    result = measure_geometry(apo, fasc, spacing)
    assert result['status'] == 'failed'
    assert 'invalid_spacing_xy' in result['reason']
    assert np.isnan(result['fl_mm'])


def test_empty_masks_do_not_fabricate_predictions():
    zero = np.zeros((130, 240))
    result = measure_geometry(zero, zero, (.2, .2))
    assert result['status'] == 'failed'
    assert np.isnan(result['pa_deg']) and np.isnan(result['mt_mm'])


def test_thickness_survives_missing_fascicles():
    apo, fasc = exact_fixture()
    result = measure_geometry(apo, np.zeros_like(fasc), (.2, .2))
    assert result['status'] == 'partial'
    assert result['mt_mm'] == pytest.approx(12)
    assert np.isnan(result['pa_deg']) and np.isnan(result['fl_mm'])


def test_near_parallel_fragments_rejected():
    apo, fasc = exact_fixture()
    fasc[:] = False
    for x in range(40, 200):
        fasc[50 + round((x-40)/80), x] = True
    result = measure_geometry(apo, fasc, (.2, .2))
    assert result['status'] == 'partial'
    assert np.isnan(result['fl_mm'])


def test_tiny_components_rejected():
    apo, fasc = exact_fixture()
    fasc[:] = False
    fasc[40, 40] = True
    result = measure_geometry(apo, fasc, (.2, .2))
    assert result['status'] == 'partial'
    assert result['diagnostics']['rejection_counts'] == {'too_few_pixels': 1}


def test_extrapolation_budget_is_effective():
    apo, fasc = exact_fixture()
    result = measure_geometry(apo, fasc, (.2, .2), {'max_fascicle_extrapolation_ratio': 0})
    assert result['status'] == 'partial'
    assert np.isnan(result['fl_mm'])


def test_curve_intersections_and_local_tangent_match_analytic_fixture():
    # TEST: yU=20+.0025(x-120)^2, yD=80+.0025(x-120)^2;
    # fascicle y=x-40. Deep intersection is (120,80), tangent slope0.
    # Upper intersection has x=120+(1-sqrt(1.6))/.005.
    yy, xx = np.indices((170, 240))
    upper = 20 + .0025*(xx-120)**2
    deep = upper + 60
    apo = (((yy <= upper) & (yy >= upper-3)) | ((yy >= deep) & (yy <= deep+3)))
    apo[:, :20] = False
    apo[:, 221:] = False
    fasc = np.zeros_like(apo)
    for x in range(80, 111):
        fasc[x-40, x] = True
    result = measure_geometry(apo, fasc, (1, 1), {'curve_degree': 2})
    assert result['status'] == 'ok', result
    upper_x = 120 + (1-np.sqrt(1.6))/.005
    expected_length = (120-upper_x)*np.sqrt(2)
    assert result['pa_deg'] == pytest.approx(45, abs=1)
    assert result['fl_mm'] == pytest.approx(expected_length, abs=2)
    # Central normal thickness is close to60px, with <=1px raster-edge bias.
    assert result['mt_mm'] == pytest.approx(60, abs=1.5)


def test_sloped_parallel_boundaries_use_normal_not_vertical_thickness():
    # TEST: yU=.5x+10, yD=.5x+60. True normal gap=50/sqrt(1.25).
    yy, xx = np.indices((180, 220))
    upper, deep = .5*xx+10, .5*xx+60
    apo = (((yy <= upper)&(yy >= upper-3)) | ((yy >= deep)&(yy <= deep+3)))
    apo[:, :10] = False
    apo[:, 201:] = False
    result = measure_geometry(apo, np.zeros_like(apo), (1, 1))
    assert result['status'] == 'partial'
    assert result['mt_mm'] == pytest.approx(50/np.sqrt(1.25), abs=.7)
    assert result['mt_mm'] < 48  # It must not report the vertical50px gap.


@pytest.mark.parametrize('value', [np.nan, np.inf, -1, 2])
def test_invalid_probabilities_fail_visibly(value):
    apo, fasc = exact_fixture()
    apo = apo.astype(float)
    apo[0, 0] = value
    result = measure_geometry(apo, fasc, (.2, .2))
    assert result['status'] == 'failed'
    assert np.isnan(result['mt_mm'])


def test_grid_mismatch_fails_without_resizing():
    apo, fasc = exact_fixture()
    result = measure_geometry(apo, fasc.T, (.2, .2))
    assert result['status'] == 'failed'
    assert 'original_grid' in result['reason']


def test_uint8_binary_masks_and_probabilities_agree():
    apo, fasc = exact_fixture()
    result = measure_geometry(apo.astype(np.uint8)*255, fasc.astype(np.uint8)*255, (.2, .2))
    probabilities = measure_geometry(apo*.8+.1, fasc*.8+.1, (.2, .2))
    assert result['status'] == probabilities['status'] == 'ok'
    assert result['fl_mm'] == pytest.approx(probabilities['fl_mm'])


def test_confidence_is_labeled_heuristic_not_accuracy():
    apo, fasc = exact_fixture()
    result = measure_geometry(apo, fasc, (.2, .2))
    assert 0 <= result['diagnostics']['confidence_score'] <= 1
    assert result['diagnostics']['confidence_is_calibrated'] is False


def test_crossing_connected_aponeuroses_fail_closed():
    yy, xx = np.indices((150, 240))
    apo = (np.abs(yy-(20+.4*xx))<=1) | (np.abs(yy-(120-.4*xx))<=1)
    result = measure_geometry(apo, np.zeros_like(apo), (.2, .2))
    assert result['status'] == 'failed'
    assert np.isnan(result['mt_mm'])


def test_small_boundary_gap_can_be_repaired_without_coordinate_change():
    apo, fasc = exact_fixture()
    apo[:, 100:102] = False
    result = measure_geometry(apo, fasc, (.2, .2))
    assert result['status'] == 'ok'
    assert result['mt_mm'] == pytest.approx(12, abs=1e-8)


def sparse_orientation_fixture(offset=40):
    """TEST: short y=x-offset fragment inside y20/y170 boundaries.

    Complete fascicle would span150sqrt(2)px, observed annotation25sqrt(2)px.
    """
    apo = np.zeros((220, 300), bool)
    apo[17:21, 10:291] = True
    apo[170:174, 10:291] = True
    fasc = np.zeros_like(apo)
    for y in range(70, 96):
        fasc[y, y + offset] = True
    return apo, fasc


def test_local_orientation_does_not_require_supported_full_length():
    apo, fasc = sparse_orientation_fixture()
    legacy = measure_geometry(apo, fasc, (.4, .2))
    local = measure_geometry(apo, fasc, (.4, .2), {'pennation_mode': 'local_orientation'})
    assert np.isnan(legacy['pa_deg']) and np.isnan(legacy['fl_mm'])
    assert local['pa_deg'] == pytest.approx(np.degrees(np.arctan(.5)), abs=1e-8)
    assert np.isnan(local['fl_mm'])
    assert local['status'] == 'partial'
    assert local['diagnostics']['n_accepted_orientations'] == 1
    assert local['diagnostics']['n_accepted_fascicles'] == 0


def test_image_extent_policy_reports_short_fragment_extrapolation():
    apo, fasc = sparse_orientation_fixture()
    cfg = {'pennation_mode': 'local_orientation', 'fascicle_support_policy': 'image_extent'}
    result = measure_geometry(apo, fasc, (.4, .2), cfg)
    assert result['status'] == 'ok', result
    assert result['pa_deg'] == pytest.approx(np.degrees(np.arctan(.5)), abs=1e-8)
    assert result['fl_mm'] == pytest.approx(np.hypot(60, 30), abs=1e-8)
    segment = result['diagnostics']['segments'][0]
    assert segment['fascicle_extrapolation_ratio'] == pytest.approx(3)
    assert segment['extrapolated_length_fraction'] == pytest.approx(5/6)
    assert 'image_extent_extrapolation_policy' in result['diagnostics']['quality_flags']
    assert 'accepted_segments_use_extrapolation' in result['diagnostics']['quality_flags']


def test_image_extent_policy_still_rejects_outside_image_limit():
    apo, fasc = sparse_orientation_fixture(offset=-50)
    cfg = {'pennation_mode': 'local_orientation', 'fascicle_support_policy': 'image_extent',
           'max_outside_image_fraction': .05}
    result = measure_geometry(apo, fasc, (1, 1), cfg)
    assert result['pa_deg'] == pytest.approx(45)
    assert np.isnan(result['fl_mm'])
    # Wider declared bounds permit the actual off-screen intersection(-30,20).
    cfg['max_outside_image_fraction'] = .15
    accepted = measure_geometry(apo, fasc, (1, 1), cfg)
    assert accepted['fl_mm'] == pytest.approx(150*np.sqrt(2))


def test_image_extent_policy_still_rejects_boundary_support_limit():
    apo, fasc = sparse_orientation_fixture(offset=-50)
    cfg = {'fascicle_support_policy': 'image_extent', 'max_apo_extrapolation_ratio': 0}
    result = measure_geometry(apo, fasc, (1, 1), cfg)
    assert np.isnan(result['fl_mm'])


def test_local_policy_remains_uncalibrated_and_reflection_invariant():
    apo, fasc = sparse_orientation_fixture()
    cfg = {'pennation_mode': 'local_orientation', 'fascicle_support_policy': 'image_extent'}
    a = measure_geometry(apo, fasc, None, cfg)
    b = measure_geometry(np.fliplr(apo), np.fliplr(fasc), None, cfg)
    assert all(np.isnan(a[k]) for k in ('pa_deg', 'fl_mm', 'mt_mm'))
    assert not a['diagnostics']['physical_export_allowed']
    for name in ('pa_pixel_deg', 'fl_px', 'mt_px'):
        assert a['diagnostics']['pixel_measurements'][name] == pytest.approx(
            b['diagnostics']['pixel_measurements'][name])


@pytest.mark.parametrize('config', [{'pennation_mode': 'guess'}, {'fascicle_support_policy': 'unlimited'}])
def test_unknown_orientation_or_support_policy_rejected(config):
    apo, fasc = exact_fixture()
    result = measure_geometry(apo, fasc, (1, 1), config)
    assert result['status'] == 'failed'
    assert 'Invalid config value' in result['reason']
