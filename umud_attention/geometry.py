"""Original-grid mask geometry for the attention U-Net candidate.

The two masks MUST come from the same image in the same original pixel grid.
Coordinates are (x,y), origin top-left, pixel centres at integer indices. Known
spacing_xy=(sx,sy) transforms points to millimetres BEFORE fitting geometry.
This module does not identify subjects, restore unknown crops, or infer scale.

Robust aponeurosis fits are paired using shared support and the fascicle
evidence between them. The default is linear: it avoids quadratic curvature
amplifying uncertainty when extrapolating beyond observed support. Quadratic
curves remain an explicit curve_degree=2 ablation requiring held-out evaluation.
Each usable fascicle fragment is a straight line. By default PA uses the local deep-curve
tangent at its intersection; the explicit local_orientation option instead
compares the fragment with the deep tangent nearest its centre, independently
of full-length intersection acceptance. FL is the
distance between intersections. MT uses central superficial points' normal
distances to the deep curve. These choices are a research candidate, not a
claim of better leaderboard accuracy. Curved fascicle arclength is not modeled.

Unknown scale produces explicitly UNCALIBRATED pixel diagnostics, with all
physical targets NaN. No typical-angle, mean-length, or anatomy-based fallback.
Dependencies: NumPy and SciPy; Matplotlib is optional for plot_geometry().
"""

from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
from numpy.polynomial import Polynomial
from scipy import ndimage


@dataclass(frozen=True)
class GeometryConfig:
    """Starting settings only: tune on held-out labels, then freeze for test."""

    apo_threshold: float = 0.5
    fasc_threshold: float = 0.5
    apo_closing_radius_px: int = 2
    min_apo_pixels: int = 30
    min_apo_span_fraction: float = 0.25
    min_overlap_fraction: float = 0.20
    max_apo_candidates: int = 8
    curve_degree: int = 1
    robust_iterations: int = 8
    huber_k: float = 1.5
    max_curve_residual_fraction: float = 0.035
    max_apo_tilt_deg: float = 65.0
    min_gap_px: float = 3.0
    pair_ambiguity_margin: float = 0.04
    intervening_boundary_fraction: float = 0.80
    min_fascicle_pixels: int = 20
    min_fragment_length_px: float = 25.0
    max_fascicle_residual_fraction: float = 0.06
    min_pa_deg: float = 3.0
    max_pa_deg: float = 85.0
    max_fascicle_extrapolation_ratio: float = 1.5
    # The legacy default treats observed connected fragments as fascicle support.
    # Explicit image_extent mode is for short orientation annotations: it keeps
    # the image/boundary extension guards but reports rather than gates the
    # extrapolation relative to fragment length. Neither mode proves accuracy.
    fascicle_support_policy: str = "fragment_relative"
    pennation_mode: str = "supported_intersections"
    max_apo_extrapolation_ratio: float = 0.25
    max_outside_image_fraction: float = 0.25
    intersection_tolerance_px: float = 2.0
    thickness_central_fraction: float = 0.20
    thickness_samples: int = 9


def _configuration(config):
    cfg = GeometryConfig(**config) if isinstance(config, dict) else (config or GeometryConfig())
    if not isinstance(cfg, GeometryConfig):
        raise ValueError("config must be a GeometryConfig, dict, or None")
    zero_allowed = {"apo_closing_radius_px", "pair_ambiguity_margin", "max_apo_extrapolation_ratio",
                    "max_fascicle_extrapolation_ratio", "max_outside_image_fraction"}
    choices = {"fascicle_support_policy": {"fragment_relative", "image_extent"},
               "pennation_mode": {"supported_intersections", "local_orientation"}}
    for key, value in asdict(cfg).items():
        if key in choices:
            if value not in choices[key]:
                raise ValueError(f"Invalid config value: {key}")
            continue
        if not np.isfinite(value) or value < 0 or (value == 0 and key not in zero_allowed):
            raise ValueError(f"Invalid config value: {key}")
    for key in ("apo_closing_radius_px", "min_apo_pixels", "max_apo_candidates", "curve_degree",
                "robust_iterations", "min_fascicle_pixels", "thickness_samples"):
        if not isinstance(getattr(cfg, key), (int, np.integer)):
            raise ValueError(f"{key} must be an integer")
    for key in ("apo_threshold", "fasc_threshold", "min_apo_span_fraction", "min_overlap_fraction",
                "pair_ambiguity_margin", "intervening_boundary_fraction", "thickness_central_fraction"):
        if getattr(cfg, key) > 1:
            raise ValueError(f"{key} must be <=1")
    if cfg.curve_degree not in (1, 2) or not (0 < cfg.min_pa_deg < cfg.max_pa_deg <= 90):
        raise ValueError("Curve degree must be 1/2 and angle bounds ordered within (0,90]")
    if cfg.max_apo_tilt_deg >= 90:
        raise ValueError("max_apo_tilt_deg must be <90")
    return cfg


def _mask(mask, name):
    arr = np.asarray(mask)
    if arr.ndim != 2 or min(arr.shape) < 2 or np.iscomplexobj(arr):
        raise ValueError(f"{name} must be a 2D real original-grid mask")
    try:
        arr = arr.astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must contain numeric probabilities") from exc
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains nonfinite pixels")
    if np.all((arr == 0) | (arr == 255)):
        arr = arr / 255.0
    if np.any((arr < 0) | (arr > 1)):
        raise ValueError(f"{name} must be probabilities [0,1] or binary {{0,1}}/{{0,255}}")
    return arr


def _components(binary):
    labels, _ = ndimage.label(binary, structure=np.ones((3, 3), bool))
    for component_id, bounds in enumerate(ndimage.find_objects(labels), start=1):
        if bounds is None:
            continue
        ys, xs = np.nonzero(labels[bounds] == component_id)
        yield component_id, np.column_stack((xs + bounds[1].start, ys + bounds[0].start)).astype(float)


def _edge(points, upper):
    xs = np.unique(points[:, 0])
    reducer = np.max if upper else np.min
    return np.array([(x, reducer(points[points[:, 0] == x, 1])) for x in xs])


def _curve(points, scale, cfg):
    physical = points * scale
    x, y = physical.T
    midpoint, halfspan = (x.min() + x.max()) / 2, np.ptp(x) / 2
    if halfspan <= 0 or len(x) <= cfg.curve_degree:
        raise ValueError("Aponeurosis has insufficient horizontal support")
    design = np.polynomial.polynomial.polyvander((x - midpoint) / halfspan, cfg.curve_degree)
    weights = np.ones(len(y))
    for _ in range(cfg.robust_iterations):
        coefficient = np.linalg.lstsq(design * np.sqrt(weights[:, None]), y * np.sqrt(weights), rcond=None)[0]
        residual = y - design @ coefficient
        mad = np.median(np.abs(residual - np.median(residual)))
        sigma = max(mad / 0.6744897501960817, np.finfo(float).eps * max(np.ptp(y), 1))
        weights = np.minimum(1, cfg.huber_k * sigma / np.maximum(np.abs(residual), sigma * 1e-12))
    polynomial = Polynomial(coefficient)(Polynomial([-midpoint / halfspan, 1 / halfspan]))
    return {"poly": polynomial, "x0": float(x.min()), "x1": float(x.max()),
            "residual_fraction": float(np.percentile(np.abs(residual), 90) / np.ptp(x))}


def _line(points, cfg):
    weights = np.ones(len(points))
    for _ in range(cfg.robust_iterations):
        centre = np.average(points, axis=0, weights=weights)
        delta = points - centre
        covariance = (delta * weights[:, None]).T @ delta / weights.sum()
        _, vectors = np.linalg.eigh(covariance)
        direction = vectors[:, -1]
        if direction[0] < 0 or (direction[0] == 0 and direction[1] < 0):
            direction = -direction
        residual = np.abs(delta @ np.array([-direction[1], direction[0]]))
        sigma = max(np.median(residual) / 0.6744897501960817, np.finfo(float).eps)
        weights = np.minimum(1, cfg.huber_k * sigma / np.maximum(residual, sigma * 1e-12))
    projected = (points - centre) @ direction
    bounds = np.array([projected.min(), projected.max()])
    span = float(np.ptp(bounds))
    return {"point": centre, "direction": direction, "bounds": bounds, "span": span,
            "residual_fraction": float(np.percentile(residual, 90) / max(span, np.finfo(float).eps))}


def _real_roots(polynomial):
    coefficients = np.asarray(polynomial.coef)
    # Drop only numerically negligible high-order terms. This avoids spurious
    # enormous roots when an exactly straight mask is fit with degree two.
    while len(coefficients) > 1 and abs(coefficients[-1]) < 1e-12:
        coefficients = coefficients[:-1]
    if len(coefficients) < 2:
        return []
    roots = Polynomial(coefficients).roots()
    return [float(z.real) for z in roots if abs(z.imag) <= 1e-7 * max(1, abs(z.real))]


def _curve_extension(curve, x):
    return max(0.0, curve["x0"] - x, x - curve["x1"]) / (curve["x1"] - curve["x0"])


def _public_curve(curve, scale):
    xs = np.linspace(curve["x0"], curve["x1"], 80)
    return {"points_px": (np.column_stack((xs, curve["poly"](xs))) / scale).tolist(),
            "coefficients": curve["poly"].coef.tolist(),
            "residual_fraction": curve["residual_fraction"]}


def _choose_pair(apo, fasc, scale, cfg, diagnostics):
    binary = apo >= cfg.apo_threshold
    if cfg.apo_closing_radius_px:
        radius = cfg.apo_closing_radius_px
        binary = ndimage.binary_closing(binary, structure=np.ones((3, 2 * radius + 1), bool))
    width = apo.shape[1]
    candidates = [(idx, pts) for idx, pts in _components(binary)
                  if len(pts) >= cfg.min_apo_pixels and np.ptp(pts[:, 0]) >= cfg.min_apo_span_fraction * (width - 1)]
    candidates.sort(key=lambda item: np.ptp(item[1][:, 0]), reverse=True)
    if len(candidates) > cfg.max_apo_candidates:
        diagnostics["quality_flags"].append("aponeurosis_candidate_limit_reached")
    candidates = candidates[:cfg.max_apo_candidates]
    diagnostics["n_aponeurosis_candidates"] = len(candidates)
    fy, fx = np.nonzero(fasc >= cfg.fasc_threshold)
    pairs = []
    for first, second in combinations(candidates, 2):
        upper_item, lower_item = sorted((first, second), key=lambda item: np.median(item[1][:, 1]))
        upper = _curve(_edge(upper_item[1], True), scale, cfg)
        deep = _curve(_edge(lower_item[1], False), scale, cfg)
        lo, hi = max(upper["x0"], deep["x0"]), min(upper["x1"], deep["x1"])
        if hi - lo < cfg.min_overlap_fraction * (width - 1) * scale[0]:
            continue
        if max(upper["residual_fraction"], deep["residual_fraction"]) > cfg.max_curve_residual_fraction:
            continue
        xs = np.linspace(lo, hi, 101)
        gap_poly = deep["poly"] - upper["poly"]
        critical = [x for x in _real_roots(gap_poly.deriv()) if lo <= x <= hi]
        gap_values = gap_poly(np.r_[xs, critical])
        if np.min(gap_values) < cfg.min_gap_px * scale[1]:
            continue
        # A pair must not jump across another comparably supported boundary.
        # Otherwise two different muscle layers could look attractive simply
        # because their union contains more fascicle pixels.
        crosses_boundary = False
        for other_id, other_points in candidates:
            if other_id in (upper_item[0], lower_item[0]):
                continue
            within_x = other_points[(other_points[:, 0] * scale[0] >= lo) &
                                    (other_points[:, 0] * scale[0] <= hi)] * scale
            if len(within_x) and np.ptp(within_x[:, 0]) >= cfg.intervening_boundary_fraction * (hi - lo):
                between = ((within_x[:, 1] > upper["poly"](within_x[:, 0])) &
                           (within_x[:, 1] < deep["poly"](within_x[:, 0])))
                if between.mean() >= cfg.intervening_boundary_fraction:
                    crosses_boundary = True
                    break
        if crosses_boundary:
            continue
        if any(np.max(np.abs(np.degrees(np.arctan(curve["poly"].deriv()(xs))))) > cfg.max_apo_tilt_deg
               for curve in (upper, deep)):
            continue
        inside = ((fx * scale[0] >= lo) & (fx * scale[0] <= hi) &
                  (fy * scale[1] >= upper["poly"](fx * scale[0])) &
                  (fy * scale[1] <= deep["poly"](fx * scale[0])))
        support = int(inside.sum())
        area_px = (hi - lo) * float(np.mean(gap_values)) / np.prod(scale)
        # Squared supported fraction rewards keeping fascicles together; area
        # penalizes selecting an unrelated distant outer boundary. This score
        # is pair-selection evidence, not a trained/calibrated probability.
        score = support**2 / max(area_px, 1.0) if len(fx) else (hi - lo) / scale[0]
        pairs.append({"upper": upper, "deep": deep, "lo": lo, "hi": hi,
                      "score": float(score), "fascicle_support_pixels": support,
                      "component_ids": [upper_item[0], lower_item[0]]})
    pairs.sort(key=lambda pair: pair["score"], reverse=True)
    diagnostics["pair_candidates"] = [{k: p[k] for k in ("score", "fascicle_support_pixels", "component_ids")}
                                      for p in pairs]
    if not pairs:
        raise ValueError("no_supported_aponeurosis_pair")
    if len(pairs) > 1:
        margin = (pairs[0]["score"] - pairs[1]["score"]) / max(pairs[0]["score"], 1e-12)
        if len(fx) == 0 or margin < cfg.pair_ambiguity_margin:
            raise ValueError("ambiguous_aponeurosis_pair")
    else:
        margin = 1.0
    diagnostics["pair_score_margin"] = float(margin)
    return pairs[0]


def _thickness(pair, scale, cfg):
    centre = (pair["lo"] + pair["hi"]) / 2
    half = (pair["hi"] - pair["lo"]) * cfg.thickness_central_fraction / 2
    deep = pair["deep"]
    segments = []
    for x in np.linspace(centre - half, centre + half, cfg.thickness_samples):
        point = np.array([x, pair["upper"]["poly"](x)])
        # Stationary squared distance => the connecting vector is normal to
        # the local deep-curve tangent. Its polynomial degree is at most three.
        derivative = Polynomial([-point[0], 1]) + (deep["poly"] - point[1]) * deep["poly"].deriv()
        roots = [root for root in _real_roots(derivative)
                 if _curve_extension(deep, root) <= cfg.max_apo_extrapolation_ratio]
        if not roots:
            continue
        feet = [np.array([root, deep["poly"](root)]) for root in roots]
        foot = min(feet, key=lambda candidate: np.linalg.norm(point - candidate))
        segments.append(np.stack((point, foot)))
    if not segments:
        return float("nan"), []
    return float(np.median([np.linalg.norm(p[1] - p[0]) for p in segments])), segments


def _local_orientation(line, pair, cfg):
    """Angle to the deep tangent at the closest supported point to fragment centre.

    The point and tangent use fitted-coordinate units, hence physical axes when
    calibration exists. This is an orientation estimate, not proof that the
    observed fragment extends to either aponeurosis.
    """
    point, deep = line["point"], pair["deep"]
    equation = Polynomial([-point[0], 1]) + (deep["poly"] - point[1]) * deep["poly"].deriv()
    roots = [x for x in _real_roots(equation)
             if _curve_extension(deep, x) <= cfg.max_apo_extrapolation_ratio]
    if not roots:
        return None
    feet = [np.array([x, deep["poly"](x)]) for x in roots]
    foot = min(feet, key=lambda q: np.linalg.norm(q - point))
    tangent = np.array([1.0, deep["poly"].deriv()(foot[0])])
    tangent /= np.linalg.norm(tangent)
    angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(line["direction"], tangent)), 0, 1))))
    if not cfg.min_pa_deg <= angle <= cfg.max_pa_deg:
        return None
    return {"angle": angle, "tangent_point": foot}


def _intersections(line, pair, shape, scale, cfg):
    point, direction = line["point"], line["direction"]
    parameters = []
    for curve in (pair["upper"], pair["deep"]):
        equation = curve["poly"](Polynomial([point[0], direction[0]])) - Polynomial([point[1], direction[1]])
        parameters.append([t for t in _real_roots(equation)
                           if _curve_extension(curve, point[0] + t * direction[0]) <= cfg.max_apo_extrapolation_ratio])
    possibilities = []
    lo, hi = line["bounds"]
    tolerance = cfg.intersection_tolerance_px * max(scale)
    image_bound = np.array([shape[1] - 1, shape[0] - 1])
    for upper_t in parameters[0]:
        for deep_t in parameters[1]:
            start, end = sorted((upper_t, deep_t))
            if start > lo + tolerance or end < hi - tolerance or end - start <= tolerance:
                continue
            extension = max(0.0, lo - start, end - hi) / line["span"]
            if (cfg.fascicle_support_policy == "fragment_relative" and
                    extension > cfg.max_fascicle_extrapolation_ratio):
                continue
            hits = point + np.array([upper_t, deep_t])[:, None] * direction
            hits_px = hits / scale
            if (np.any(hits_px < -image_bound * cfg.max_outside_image_fraction) or
                    np.any(hits_px > image_bound * (1 + cfg.max_outside_image_fraction))):
                continue
            sample = point + np.linspace(start, end, 21)[:, None] * direction
            if (np.any(sample[:, 1] < pair["upper"]["poly"](sample[:, 0]) - tolerance) or
                    np.any(sample[:, 1] > pair["deep"]["poly"](sample[:, 0]) + tolerance)):
                continue
            slope = pair["deep"]["poly"].deriv()(hits[1, 0])
            tangent = np.array([1.0, slope])
            tangent /= np.linalg.norm(tangent)
            angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(direction, tangent)), 0, 1))))
            if not cfg.min_pa_deg <= angle <= cfg.max_pa_deg:
                continue
            apo_extension = max(_curve_extension(pair["upper"], hits[0, 0]),
                                _curve_extension(pair["deep"], hits[1, 0]))
            possibilities.append({"hits": hits, "angle": angle, "length": end - start,
                                  "fascicle_extension": float(extension), "apo_extension": float(apo_extension),
                                  "extrapolated_length_fraction": float(np.clip(1 - line["span"] / (end - start), 0, 1)),
                                  "deep_tangent": tangent})
    return min(possibilities, key=lambda item: (item["fascicle_extension"], item["length"])) if possibilities else None


def measure_geometry(apo_prob, fasc_prob, spacing_xy, config=None):
    """Return PA/FL/MT with explicit quality evidence; never manufacture scale.

    spacing_xy must be two finite positive mm-per-original-pixel values, or
    None for uncalibrated diagnostics. Malformed inputs fail visibly in result
    status/reason. Targets remain NaN on unknown scale even when pixel geometry
    is available. status='ok' means geometric checks passed, NOT proven accuracy.
    Confidence is an uncalibrated support heuristic, not a probability/error bar.
    """
    result = {"pa_deg": float("nan"), "fl_mm": float("nan"), "mt_mm": float("nan"),
              "status": "failed", "reason": "", "diagnostics": {
                  "quality_flags": [], "segments": [], "confidence_score": 0.0,
                  "confidence_is_calibrated": False, "physical_export_allowed": False}}
    diag = result["diagnostics"]
    try:
        cfg = _configuration(config)
        diag["config"] = asdict(cfg)
        apo, fasc = _mask(apo_prob, "apo_prob"), _mask(fasc_prob, "fasc_prob")
        if apo.shape != fasc.shape:
            raise ValueError("masks_must_share_the_same_original_grid")
        calibrated = spacing_xy is not None
        scale = np.asarray(spacing_xy if calibrated else [1, 1], dtype=float)
        if scale.shape != (2,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("invalid_spacing_xy: require two finite positive mm/pixel values or None")
        diag.update(shape_hw=list(apo.shape), spacing_xy=scale.tolist() if calibrated else None,
                    coordinate_units="mm" if calibrated else "uncalibrated_pixels",
                    coordinate_convention="original pixel centres; x right, y down; origin top-left")
        pair = _choose_pair(apo, fasc, scale, cfg, diag)
        diag["selected_component_ids"] = pair["component_ids"]
        diag["aponeuroses"] = {"superficial": _public_curve(pair["upper"], scale),
                               "deep": _public_curve(pair["deep"], scale)}
        mt, thickness_segments = _thickness(pair, scale, cfg)
        diag["thickness_segments_px"] = [(segment / scale).tolist() for segment in thickness_segments]
        yy, xx = np.indices(apo.shape)
        inside = ((yy * scale[1] >= pair["upper"]["poly"](xx * scale[0])) &
                  (yy * scale[1] <= pair["deep"]["poly"](xx * scale[0])))
        active = fasc >= cfg.fasc_threshold
        if np.any(active & ~inside):
            diag["quality_flags"].append("outside_muscle_fascicle_pixels_removed")
        accepted, orientations, rejected = [], [], Counter()
        if cfg.fascicle_support_policy == "image_extent":
            diag["quality_flags"].append("image_extent_extrapolation_policy")
        if cfg.pennation_mode == "local_orientation":
            diag["quality_flags"].append("local_orientation_pa_independent_of_fl")
        for component_id, points_px in _components(active & inside):
            segment = {"component_id": component_id, "n_pixels": len(points_px), "accepted": False}
            diag["segments"].append(segment)
            reason = None
            if len(points_px) < cfg.min_fascicle_pixels:
                reason = "too_few_pixels"
            else:
                fitted = _line(points_px * scale, cfg)
                endpoints = fitted["point"] + fitted["bounds"][:, None] * fitted["direction"]
                segment["observed_endpoints_px"] = (endpoints / scale).tolist()
                segment["fit_residual_fraction"] = fitted["residual_fraction"]
                if np.linalg.norm((endpoints[1] - endpoints[0]) / scale) < cfg.min_fragment_length_px:
                    reason = "fragment_too_short"
                elif fitted["residual_fraction"] > cfg.max_fascicle_residual_fraction:
                    reason = "fragment_not_straight_enough"
                else:
                    local = _local_orientation(fitted, pair, cfg)
                    if local is not None:
                        orientations.append(local)
                        segment.update(orientation_accepted=True, local_pa_deg=local["angle"],
                                       local_tangent_point_px=(local["tangent_point"] / scale).tolist())
                    else:
                        segment["orientation_accepted"] = False
                    candidate = _intersections(fitted, pair, apo.shape, scale, cfg)
                    if candidate is None:
                        reason = "no_supported_intersections_or_angle"
                    else:
                        segment.update(accepted=True, reason="ok", intersections_px=(candidate["hits"] / scale).tolist(),
                                       angle_deg=candidate["angle"], length_coordinate_units=candidate["length"],
                                       fascicle_extrapolation_ratio=candidate["fascicle_extension"],
                                       extrapolated_length_fraction=candidate["extrapolated_length_fraction"],
                                       apo_extrapolation_ratio=candidate["apo_extension"])
                        accepted.append(candidate)
            if reason:
                segment["reason"] = reason
                rejected[reason] += 1
        diag["n_accepted_fascicles"] = len(accepted)
        diag["n_accepted_orientations"] = len(orientations)
        diag["rejection_counts"] = dict(rejected)
        angle_source = orientations if cfg.pennation_mode == "local_orientation" else accepted
        pa = float(np.median([item["angle"] for item in angle_source])) if angle_source else float("nan")
        fl = float(np.median([item["length"] for item in accepted])) if accepted else float("nan")
        pixel_lengths = [float(np.linalg.norm(np.diff(item["hits"] / scale, axis=0))) for item in accepted]
        diag["pixel_segment_lengths"] = pixel_lengths
        diag["pixel_thickness_segment_lengths"] = [float(np.linalg.norm((segment[1] - segment[0]) / scale))
                                                    for segment in thickness_segments]
        if accepted:
            extension = float(np.mean([item["fascicle_extension"] + item["apo_extension"] for item in accepted]))
            diag["confidence_score"] = float(min(1, len(accepted) / 3) * np.exp(-extension) *
                                               (0.5 + 0.5 * diag["pair_score_margin"]))
            if len(accepted) == 1:
                diag["quality_flags"].append("single_fascicle_estimate")
            if extension > 1e-10:
                diag["quality_flags"].append("accepted_segments_use_extrapolation")
            diag["median_extrapolated_length_fraction"] = float(np.median(
                [item["extrapolated_length_fraction"] for item in accepted]))
        if not calibrated:
            diag["pixel_measurements"] = {"pa_pixel_deg": pa, "fl_px": fl, "mt_px": mt}
            diag["quality_flags"].append("uncalibrated_no_physical_outputs")
            result.update(status="partial", reason="missing_physical_spacing")
        else:
            result.update(pa_deg=pa, fl_mm=fl, mt_mm=mt)
            complete = all(np.isfinite(value) for value in (pa, fl, mt))
            result.update(status="ok" if complete else "partial",
                          reason="ok" if complete else "insufficient_supported_geometry")
            diag["physical_export_allowed"] = complete
        return result
    except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
        result["reason"] = str(exc)
        diag["quality_flags"].append("geometry_failed")
        return result


def plot_geometry(image, result, ax=None):
    """Optional diagnostic overlay; all plotted coordinates are ORIGINAL pixels."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image, cmap="gray")
    diag = result["diagnostics"]
    for name, color in (("superficial", "#22b8cf"), ("deep", "#f6ad55")):
        if name in diag.get("aponeuroses", {}):
            points = np.array(diag["aponeuroses"][name]["points_px"])
            ax.plot(points[:, 0], points[:, 1], color=color, lw=2, label=name)
    labeled_projection, labeled_support = False, False
    for segment in diag.get("segments", []):
        if segment["accepted"]:
            points = np.array(segment["intersections_px"])
            ax.plot(points[:, 0], points[:, 1], color="#ffe066", alpha=.8, linestyle="--",
                    label=None if labeled_projection else "fascicle line estimate")
            labeled_projection = True
        if segment.get("orientation_accepted") and "observed_endpoints_px" in segment:
            points = np.array(segment["observed_endpoints_px"])
            ax.plot(points[:, 0], points[:, 1], color="#69db7c", lw=2,
                    label=None if labeled_support else "observed orientation support")
            labeled_support = True
    for segment in diag.get("thickness_segments_px", []):
        points = np.array(segment)
        ax.plot(points[:, 0], points[:, 1], color="#f783ac", alpha=.7)
    if diag.get("aponeuroses"):
        ax.legend()
    ax.set(xlabel="Original x (pixels)", ylabel="Original y (pixels)",
           title=f"{result['status']}: {result['reason']} ({diag.get('coordinate_units', 'no geometry')})")
    return ax
