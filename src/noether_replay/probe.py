"""Exposure-aware fixed-frequency regression with conditional noise covariance."""

from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from fractions import Fraction

import numpy as np

from .conditioning import MAX_DIMENSION, _matrix, validate_covariance
from .model import MAX_JSON_BYTES, Recording, Stream, _integer, array_digest, canonical_json
from .replay import calibration_assignment
from .timing import _sequence_violation, _time

SCHEMA = "noether-replay-probe-response/1"
_REFERENCE_FIELDS = {
    "clock_id", "origin_ns", "origin_uncertainty_ns", "frequency_numerator_hz",
    "frequency_denominator", "frequency_uncertainty_hz", "phase_at_origin_rad",
    "phase_uncertainty_rad", "provenance",
}


def _number(value, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be finite real numeric data")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} exceeds the float64 range") from exc
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return result


def _reference(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != _REFERENCE_FIELDS:
        raise ValueError("Reference fields must match the fixed external oscillator contract")
    result = deepcopy(value)
    clock = result["clock_id"]
    if clock is not None and (not isinstance(clock, str) or not clock):
        raise ValueError("Reference clock must be nonempty text or None")
    for name in ("origin_ns", "origin_uncertainty_ns"):
        item = result[name]
        if item is not None:
            if not _integer(item) or (name.endswith("uncertainty_ns") and item < 0):
                raise ValueError(f"{name} must be integer nanoseconds or None")
            result[name] = int(item)
    for name in ("frequency_numerator_hz", "frequency_denominator"):
        if not _integer(result[name]) or result[name] <= 0:
            raise ValueError("The externally fixed frequency must be a positive rational")
        result[name] = int(result[name])
    for name in ("phase_at_origin_rad", "phase_uncertainty_rad", "frequency_uncertainty_hz"):
        if result[name] is not None:
            result[name] = _number(result[name], name, nonnegative=name != "phase_at_origin_rad")
    if not isinstance(result["provenance"], dict) or not result["provenance"]:
        raise ValueError("Reference requires explicit provenance")
    if len(canonical_json(result)) > MAX_JSON_BYTES:
        raise ValueError("Reference metadata exceeds 16 MiB")
    return result


def _selection(indices, size: int, name: str) -> tuple[int, ...]:
    if not isinstance(indices, (list, tuple, range, np.ndarray)):
        raise ValueError(f"{name} indices must be an explicit bounded sequence")
    if isinstance(indices, np.ndarray) and indices.ndim != 1:
        raise ValueError(f"{name} indices must be one dimensional")
    if not 1 <= len(indices) <= MAX_DIMENSION:
        raise ValueError(f"{name} selection must contain 1 to {MAX_DIMENSION} entries")
    if any(not _integer(i) or not 0 <= i < size for i in indices):
        raise ValueError(f"{name} indices must be in-range integers, not booleans")
    result = tuple(int(i) for i in indices)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} indices must be unique")
    return result


def _basis(start: int, end: int, reference: dict) -> list[float]:
    """Integrate the oscillator; reduce integer cycles before float conversion."""
    numerator = reference["frequency_numerator_hz"]
    denominator = reference["frequency_denominator"] * 1_000_000_000
    duration = Fraction(numerator * (end - start), denominator)
    # Whole periods integrate to exactly zero, including periods beyond float range.
    if duration.denominator == 1:
        attenuation = 0.0
    else:
        remainder = duration % 2
        float_duration = float(duration)
        attenuation = (float(np.sinc(float_duration)) if float_duration < 1
                       else math.sin(math.pi * float(remainder)) / float_duration / math.pi)
    midpoint_cycles = Fraction(
        numerator * (start + end - 2 * reference["origin_ns"]), 2 * denominator,
    ) % 1
    phase = math.tau * float(midpoint_cycles) + math.remainder(
        reference["phase_at_origin_rad"], math.tau,
    )
    return [1.0, attenuation * math.cos(phase), attenuation * math.sin(phase)]


def _phase_bound(start: int, end: int, uncertainty: int, reference: dict) -> float:
    relative_error = Fraction(uncertainty + reference["origin_uncertainty_ns"], 10**9)
    age = Fraction(max(abs(start - reference["origin_ns"]),
                       abs(end - reference["origin_ns"])), 10**9)
    frequency = Fraction(reference["frequency_numerator_hz"], reference["frequency_denominator"])
    frequency_error = Fraction(reference["frequency_uncertainty_hz"])
    cycles = frequency * relative_error + frequency_error * (age + relative_error)
    return reference["phase_uncertainty_rad"] + math.tau * float(cycles)


def _fit(design: np.ndarray, values: np.ndarray, covariance: np.ndarray, rank_rtol: float):
    """Whiten by Cholesky, solve by SVD; no inverse, jitter or residual rescaling."""
    scale = float(np.max(np.abs(covariance)))
    if scale == 0:
        return None, "singular_noise_covariance"
    scaled_covariance = covariance / scale
    eigenvalues = np.linalg.eigvalsh(scaled_covariance)
    covariance_rank = int(np.count_nonzero(eigenvalues > rank_rtol * eigenvalues[-1]))
    noise_diagnostics = {
        "noise_covariance_rank": covariance_rank,
        "noise_covariance_dimension": len(covariance),
        "scaled_noise_covariance_eigenvalues": eigenvalues.tolist(),
    }
    if covariance_rank < len(covariance):
        return noise_diagnostics, "singular_noise_covariance"
    try:
        chol = np.linalg.cholesky(scaled_covariance)
    except np.linalg.LinAlgError:
        return None, "singular_noise_covariance"
    joint = np.kron(design, np.eye(values.shape[1]))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        whitened = np.linalg.solve(chol, joint)
        if not np.isfinite(whitened).all():
            return None, "numerical_range_exceeded"
        left, singular, right = np.linalg.svd(whitened, full_matrices=False)
        rank = int(np.count_nonzero(singular > rank_rtol * singular[0]))
        diagnostics = {**noise_diagnostics, "rank": rank, "parameter_count": joint.shape[1],
                       "whitened_design_singular_values": singular.tolist()}
        if rank < joint.shape[1]:
            return diagnostics, "rank_deficient_design"
        value_scale = float(np.max(np.abs(values))) or 1.0
        whitened_values = np.linalg.solve(chol, values.ravel() / value_scale)
        weighted_right = right.T / singular
        coefficients = (weighted_right @ (left.T @ whitened_values)) * value_scale
        coefficient_covariance = (weighted_right @ weighted_right.T) * scale
        residuals = values - (joint @ coefficients).reshape(values.shape)
        if not all(np.isfinite(x).all() for x in (coefficients, coefficient_covariance, residuals)):
            return diagnostics, "numerical_range_exceeded"
        if np.any(np.diag(coefficient_covariance) <= 0):
            return diagnostics, "coefficient_covariance_underflow"
        coefficients = coefficients.reshape(3, values.shape[1])
        amplitudes = np.hypot(coefficients[1], coefficients[2])
        if not np.isfinite(amplitudes).all():
            return diagnostics, "numerical_range_exceeded"
    result = {
        **diagnostics,
        "coefficients": dict(zip(("offset", "cosine", "sine"), coefficients.tolist())),
        "coefficient_order": "term-major: offset channels, cosine channels, sine channels",
        "coefficient_covariance": coefficient_covariance.tolist(),
        "amplitude": amplitudes.tolist(),
        "phase_lag_rad": [None if a == 0 else math.atan2(s, c)
                          for a, c, s in zip(amplitudes, coefficients[1], coefficients[2])],
        "residuals": residuals.tolist(),
    }
    return result, None


def qualify_probe_response(
    recording: Recording,
    *,
    stream_id: str,
    sample_indices,
    channel_indices,
    reference: dict,
    covariance,
    covariance_provenance: dict,
    observation_model: str,
    max_phase_error_rad: float,
    rank_rtol: float = 1e-12,
) -> dict:
    """Fit offset and two quadratures of a fixed external oscillator.

    The supported observation model is ``uniform_exposure_average`` on a raw
    WFS measurement stream. Full sample-major covariance covers exactly the
    selected coordinates (at most 512). All selected exposures must occupy one
    unambiguous calibration epoch, including gaps and timestamp uncertainty.
    Missing or contradictory evidence withholds the entire window, retaining
    its identity. Invalid arguments/covariance raise. No rows are silently removed.

    Coefficient covariance is conditional on the nominal exposure design and
    supplied measurement-noise covariance. Deterministic clock/phase/frequency
    error bounds are retained separately, never interpreted as random noise.
    ``rank_rtol`` applies to both the covariance eigenvalues and the whitened
    design singular values; numerically unresolved covariance modes withhold.
    No frequency search, amplitude normalization by a requested command, physical
    Jacobian inference, control action or controller-performance claim is made.
    """
    source_digest = recording.digest()
    if not isinstance(stream_id, str) or not stream_id:
        raise ValueError("Select an existing measurement stream")
    source = next((s for s in recording.streams if s.id == stream_id), None)
    if source is None:
        raise ValueError("Select an existing measurement stream")
    uncertainty = None if source.uncertainty_ns is None else int(source.uncertainty_ns)
    samples = _selection(sample_indices, len(source.values), "Sample")
    channels = _selection(channel_indices, math.prod(source.values.shape[1:]), "Channel")
    if len(samples) * len(channels) > MAX_DIMENSION:
        raise ValueError("Joint sample/channel window exceeds 512")
    reference = _reference(reference)
    phase_budget = _number(max_phase_error_rad, "Phase-error budget", nonnegative=True)
    rank_rtol = _number(rank_rtol, "Rank tolerance")
    if not 0 < rank_rtol < 1:
        raise ValueError("Rank tolerance must lie strictly between zero and one")
    if not isinstance(observation_model, str) or not observation_model:
        raise ValueError("Declare the observation model explicitly")
    if not isinstance(covariance_provenance, dict) or not covariance_provenance:
        raise ValueError("Covariance requires explicit provenance")
    raw_covariance = np.asarray(covariance)
    supplied_covariance = _matrix(raw_covariance, "Covariance")
    if supplied_covariance.shape != (len(samples) * len(channels),) * 2:
        raise ValueError("Covariance must cover the complete sample-major selected window")
    validated_covariance = validate_covariance(supplied_covariance)
    configuration = {
        "stream_id": stream_id, "sample_indices": list(samples), "channel_indices": list(channels),
        "reference": reference, "observation_model": observation_model,
        "max_phase_error_rad": phase_budget, "rank_rtol": rank_rtol,
        "covariance": supplied_covariance.tolist(),
        "supplied_covariance_dtype": raw_covariance.dtype.str,
        "supplied_covariance_sha256": array_digest(raw_covariance),
        "covariance_sha256": array_digest(supplied_covariance),
        "validated_covariance_sha256": array_digest(validated_covariance),
        "covariance_provenance": deepcopy(covariance_provenance),
    }
    configuration_bytes = canonical_json(configuration)
    if len(configuration_bytes) > MAX_JSON_BYTES:
        raise ValueError("Probe configuration exceeds 16 MiB")
    reasons = []
    if source.kind != "wfs_measurements" or "conditioning" in source.metadata:
        reasons.append("unsupported_observation_stream")
    if observation_model != "uniform_exposure_average":
        reasons.append("unsupported_observation_model")
    if source.units is None:
        reasons.append("unknown_measurement_units")
    if source.clock_id is None or reference["clock_id"] is None:
        reasons.append("unknown_clock")
    elif source.clock_id != reference["clock_id"]:
        reasons.append("cross_clock_reference")
    if uncertainty is None:
        reasons.append("unknown_exposure_uncertainty")
    for key in ("origin_ns", "origin_uncertainty_ns", "frequency_uncertainty_hz",
                "phase_at_origin_rad", "phase_uncertainty_rad"):
        if reference[key] is None:
            reasons.append(f"unknown_reference_{key}")
    exposures = []
    for index in samples:
        start, end = _time(source, "exposure_start", index), _time(source, "exposure_end", index)
        epoch = calibration_assignment(recording, source, index)
        exposures.append({"sample_index": index,
                          "frame_id": None if source.frame_ids is None else int(source.frame_ids[index]),
                          "start_ns": start, "end_ns": end,
                          "available_ns": _time(source, "available", index),
                          "calibration_epoch_id": epoch})
        if start is None or end is None:
            reasons.append("missing_exposure_endpoint")
        elif end <= start or (uncertainty is not None and end - start <= 2 * uncertainty):
            reasons.append("nonpositive_or_ambiguous_exposure_duration")
        if _sequence_violation(source, index):
            reasons.append("event_sequence_violation")
    epoch = None
    if all(item["start_ns"] is not None and item["end_ns"] is not None for item in exposures):
        hull = Stream(source.id, source.kind, np.zeros((1, 1)), clock_id=source.clock_id,
                      uncertainty_ns=uncertainty,
                      times_ns={"exposure_start": (min(e["start_ns"] for e in exposures),),
                                "exposure_end": (max(e["end_ns"] for e in exposures),)})
        epoch = calibration_assignment(recording, hull, 0)
    if epoch is None or any(item["calibration_epoch_id"] != epoch for item in exposures):
        reasons.append("calibration_continuity_not_established")
    if source.values.ndim == 1:
        selected = source.values[np.asarray(samples)][:, None]
    else:
        coordinates = np.unravel_index(channels, source.values.shape[1:], order="C")
        selected = source.values[(np.asarray(samples)[:, None], *(a[None, :] for a in coordinates))]
    with np.errstate(over="ignore", invalid="ignore"):
        selected = np.asarray(selected, dtype=np.float64)
    if not np.isfinite(selected).all():
        reasons.append("nonfinite_selected_observations")
    design, bounds, fit = None, None, None
    if not reasons:
        try:
            bounds = [_phase_bound(e["start_ns"], e["end_ns"], uncertainty, reference)
                      for e in exposures]
            if not all(math.isfinite(bound) for bound in bounds):
                bounds = None
                reasons.append("phase_bound_exceeds_numerical_range")
            elif max(bounds) > phase_budget:
                reasons.append("phase_error_budget_exceeded")
            else:
                design = np.asarray([_basis(e["start_ns"], e["end_ns"], reference)
                                     for e in exposures])
                fit, reason = _fit(design, selected, validated_covariance, rank_rtol)
                if reason is not None:
                    reasons.append(reason)
        except (OverflowError, np.linalg.LinAlgError):
            reasons.append("numerical_range_exceeded")
    report = {
        "schema": SCHEMA, "recording_sha256": source_digest,
        "configuration": configuration,
        "configuration_sha256": hashlib.sha256(configuration_bytes).hexdigest(),
        "status": "withheld" if reasons else "estimated", "reasons": sorted(set(reasons)),
        "units": source.units, "calibration_epoch_id": epoch, "exposures": exposures,
        "phase_error_bounds_rad": bounds,
        "quadrature_design_error_bounds": None if bounds is None else [min(2.0, b) for b in bounds],
        "design": None if design is None else design.tolist(), "fit": fit,
        "limitations": [
            "Noise covariance is conditional on nominal timing and the supplied covariance model.",
            "Timing bounds are deterministic, conservative, and need not be independent.",
            "A single frequency cannot distinguish clock phase error from physical response lag.",
            "A single probe direction does not identify a full interaction matrix.",
            "A measured closed-loop response is not a physical Jacobian or a safe-control claim.",
            "Hashes bind supplied evidence; they do not authenticate acquisition or calibration.",
        ],
    }
    if recording.digest() != source_digest:
        raise ValueError("Recording changed during probe qualification")
    if len(canonical_json(report)) > MAX_JSON_BYTES:
        raise ValueError("Probe report exceeds 16 MiB")
    return report
