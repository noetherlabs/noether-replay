"""Analytical exposure integrals, independent GLS and fail-closed metadata gates."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from noether_replay.model import Recording, Stream, canonical_json
from noether_replay.probe import qualify_probe_response


def fixture(*, duration=1_000_000, channels=2, offset=0):
    starts = np.array([0, 2, 5, 7, 10, 13, 15, 19, 22, 25, 29, 31]) * 1_000_000
    ends = starts + duration
    omega = 2 * np.pi * 125
    # Independent primitive integration, not the implementation's midpoint/sinc formula.
    t, u = starts * 1e-9, ends * 1e-9
    design = np.column_stack((np.ones(len(t)), (np.sin(omega * u) - np.sin(omega * t))
                             / (omega * (u - t)), (np.cos(omega * t) - np.cos(omega * u))
                             / (omega * (u - t))))
    beta = np.array([[2.0, -1.0], [3.0, 0.0], [0.0, 2.0]])[:, :channels]
    stream = Stream(
        "wfs", "wfs_measurements", design @ beta, "rad",
        frame_ids=tuple(range(100, 100 + len(t))), clock_id="oscillator",
        times_ns={"exposure_start": tuple(int(x) + offset for x in starts),
                  "exposure_end": tuple(int(x) + offset for x in ends)}, uncertainty_ns=0,
    )
    recording = Recording("analytic-probe", (stream,), ({
        "id": "calibration", "clock_id": "oscillator", "start_ns": offset - 1000,
        "end_ns": offset + int(max(ends)) + 1000, "stream_ids": ["wfs"],
        "products": {"response": "b" * 64},
    },), provenance={"source": "analytic exposure integrals"})
    reference = {
        "clock_id": "oscillator", "origin_ns": offset, "origin_uncertainty_ns": 0,
        "frequency_numerator_hz": 125, "frequency_denominator": 1,
        "frequency_uncertainty_hz": 0.0, "phase_at_origin_rad": 0.0,
        "phase_uncertainty_rad": 0.0, "provenance": {"source": "fixed synthetic oscillator"},
    }
    return recording, reference, beta, design


def qualify(recording, reference, **kwargs):
    defaults = {
        "stream_id": "wfs", "sample_indices": list(range(len(recording.streams[0].values))),
        "channel_indices": list(range(recording.streams[0].values.shape[1])),
        "reference": reference,
        "covariance": np.eye(recording.streams[0].values.size) * 0.04,
        "covariance_provenance": {"source": "declared analytic noise"},
        "observation_model": "uniform_exposure_average", "max_phase_error_rad": 0.0,
    }
    defaults.update(kwargs)
    return qualify_probe_response(recording, **defaults)


def coefficients(report):
    return np.array([report["fit"]["coefficients"][key] for key in ("offset", "cosine", "sine")])


def test_irregular_exposure_integrals_recover_both_quadratures_and_offset():
    recording, reference, beta, design = fixture()
    before = recording.digest()
    report = qualify(recording, reference)
    assert report["status"] == "estimated"
    assert report["reasons"] == []
    np.testing.assert_allclose(report["design"], design, atol=2e-14)
    np.testing.assert_allclose(coefficients(report), beta, atol=2e-14)
    np.testing.assert_allclose(report["fit"]["amplitude"], [3, 2], atol=2e-14)
    np.testing.assert_allclose(report["fit"]["phase_lag_rad"], [0, np.pi / 2], atol=2e-14)
    assert all(row["available_ns"] is None for row in report["exposures"])
    assert recording.digest() == before
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_long_exposure_midpoint_comparator_has_analytic_attenuation_bias():
    recording, reference, beta, _ = fixture(duration=3_000_000)
    report = qualify(recording, reference)
    np.testing.assert_allclose(coefficients(report), beta, atol=2e-14)
    times = recording.streams[0].times_ns
    midpoint = (np.array(times["exposure_start"]) + np.array(times["exposure_end"])) * 0.5e-9
    point_design = np.column_stack((np.ones(len(midpoint)), np.cos(2*np.pi*125*midpoint),
                                   np.sin(2*np.pi*125*midpoint)))
    point_beta = np.linalg.lstsq(point_design, recording.streams[0].values, rcond=None)[0]
    np.testing.assert_allclose(point_beta[1:], beta[1:] * np.sinc(125 * 0.003), atol=2e-14)
    assert not np.allclose(point_beta[1:], beta[1:])


def test_compacted_index_comparator_does_not_recover_irregular_phase():
    recording, reference, beta, _ = fixture()
    report = qualify(recording, reference)
    np.testing.assert_allclose(coefficients(report), beta, atol=2e-14)
    assumed_midpoint = (np.arange(12) * 2 + 0.5) * 0.001
    compacted = np.column_stack((np.ones(12), np.sinc(0.125) * np.cos(2*np.pi*125*assumed_midpoint),
                                np.sinc(0.125) * np.sin(2*np.pi*125*assumed_midpoint)))
    wrong = np.linalg.lstsq(compacted, recording.streams[0].values, rcond=None)[0]
    assert not np.allclose(wrong, beta)


def test_full_temporal_cross_channel_covariance_matches_independent_gls():
    recording, reference, _, design = fixture()
    rng = np.random.default_rng(71)
    factor = rng.normal(size=(24, 24))
    covariance = factor @ factor.T + np.eye(24)
    recording.streams[0].values += rng.normal(size=(12, 2)) * 0.1
    report = qualify(recording, reference, covariance=covariance)
    joint = np.kron(design, np.eye(2))
    precision = np.linalg.inv(covariance)
    expected_covariance = np.linalg.inv(joint.T @ precision @ joint)
    estimator = expected_covariance @ joint.T @ precision
    expected = estimator @ recording.streams[0].values.ravel()
    np.testing.assert_allclose(coefficients(report).ravel(), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(report["fit"]["coefficient_covariance"],
                               estimator @ covariance @ estimator.T, rtol=1e-12, atol=1e-12)
    diagonal = qualify(recording, reference, covariance=np.diag(np.diag(covariance)))
    assert not np.allclose(report["fit"]["coefficient_covariance"],
                           diagonal["fit"]["coefficient_covariance"])


def test_order_and_noncontiguous_selected_channels_preserved():
    recording, reference, beta, _ = fixture()
    expanded = np.empty((12, 2, 2))
    expanded[:, :, 0] = recording.streams[0].values
    expanded[:, :, 1] = -999
    recording.streams[0].values = expanded[:, :, ::-1]
    assert not recording.streams[0].values.flags.c_contiguous
    indices = [11, 2, 5, 7, 0, 9]
    report = qualify(recording, reference, sample_indices=indices, channel_indices=[3, 1],
                     covariance=np.eye(12))
    np.testing.assert_allclose(coefficients(report), beta[:, ::-1], atol=2e-14)
    assert [row["sample_index"] for row in report["exposures"]] == indices


def test_large_common_epoch_shift_preserves_design_and_fit_exactly():
    original, reference, _, _ = fixture()
    shifted, shifted_reference, _, _ = fixture(offset=2**90)
    first, second = qualify(original, reference), qualify(shifted, shifted_reference)
    assert first["design"] == second["design"]
    assert first["fit"] == second["fit"]
    assert first["recording_sha256"] != second["recording_sha256"]


@pytest.mark.parametrize("scale", [1e-100, 1e100])
def test_covariance_units_scale_without_jitter_or_residual_rescaling(scale):
    recording, reference, beta, _ = fixture()
    first = qualify(recording, reference)
    second = qualify(recording, reference, covariance=np.eye(24) * 0.04 * scale)
    assert second["status"] == "estimated"
    np.testing.assert_allclose(coefficients(second), beta, atol=2e-14)
    np.testing.assert_allclose(np.asarray(second["fit"]["coefficient_covariance"]) / scale,
                               first["fit"]["coefficient_covariance"], rtol=1e-13, atol=1e-16)


@pytest.mark.parametrize("mutation,reason", [
    ("missing_start", "missing_exposure_endpoint"),
    ("unknown_uncertainty", "unknown_exposure_uncertainty"),
    ("cross_clock", "cross_clock_reference"),
    ("unknown_clock", "unknown_clock"),
    ("ambiguous_duration", "nonpositive_or_ambiguous_exposure_duration"),
    ("available_before_end", "event_sequence_violation"),
    ("missing_calibration", "calibration_continuity_not_established"),
    ("overlapping_epoch", "calibration_continuity_not_established"),
    ("different_epochs", "calibration_continuity_not_established"),
    ("nonfinite", "nonfinite_selected_observations"),
    ("units", "unknown_measurement_units"),
])
def test_metadata_failure_withholds_complete_window(mutation, reason):
    recording, reference, _, _ = fixture()
    stream = recording.streams[0]
    if mutation == "missing_start":
        stream.times_ns["exposure_start"] = (None, *stream.times_ns["exposure_start"][1:])
    elif mutation == "unknown_uncertainty":
        stream.uncertainty_ns = None
    elif mutation == "cross_clock":
        reference["clock_id"] = "unrelated"
    elif mutation == "unknown_clock":
        stream.times_ns = {}
        stream.clock_id = None
    elif mutation == "ambiguous_duration":
        stream.uncertainty_ns = 500_000
    elif mutation == "available_before_end":
        stream.times_ns["available"] = (0,) * 12
    elif mutation == "missing_calibration":
        recording.calibrations = ()
    elif mutation == "overlapping_epoch":
        extra = dict(recording.calibrations[0], id="other", start_ns=3_000_000, end_ns=4_000_000)
        recording.calibrations += (extra,)  # In the gap; no individual exposure crosses it.
    elif mutation == "different_epochs":
        old = recording.calibrations[0]
        recording.calibrations = (dict(old, end_ns=11_000_001),
                                  dict(old, id="next", start_ns=12_000_000))
    elif mutation == "nonfinite":
        stream.values[0, 0] = np.nan
    elif mutation == "units":
        stream.units = None
    report = qualify(recording, reference)
    assert report["status"] == "withheld"
    assert reason in report["reasons"]
    assert report["fit"] is None
    assert len(report["exposures"]) == 12
    assert report["configuration"]["sample_indices"] == list(range(12))


@pytest.mark.parametrize("field", ["origin_ns", "origin_uncertainty_ns", "phase_at_origin_rad",
                                  "phase_uncertainty_rad", "frequency_uncertainty_hz"])
def test_unknown_reference_metadata_withheld(field):
    recording, reference, _, _ = fixture()
    reference[field] = None
    report = qualify(recording, reference)
    assert f"unknown_reference_{field}" in report["reasons"]
    assert report["fit"] is None


def test_clock_frequency_phase_error_budget_is_separate_from_noise_covariance():
    recording, reference, _, _ = fixture()
    stream = recording.streams[0]
    stream.uncertainty_ns = 2
    reference.update(origin_uncertainty_ns=3, frequency_uncertainty_hz=0.5,
                     phase_uncertainty_rad=0.01)
    report = qualify(recording, reference, max_phase_error_rad=0.2)
    age = np.array(stream.times_ns["exposure_end"]) * 1e-9
    expected = 0.01 + 2*np.pi*(125*5e-9 + 0.5*(age + 5e-9))
    np.testing.assert_allclose(report["phase_error_bounds_rad"], expected, rtol=1e-15)
    assert report["status"] == "estimated"
    nominal, nominal_reference, _, _ = fixture()
    assert report["fit"] == qualify(nominal, nominal_reference)["fit"]
    withheld = qualify(recording, reference, max_phase_error_rad=0.0)
    assert withheld["reasons"] == ["phase_error_budget_exceeded"]
    assert withheld["fit"] is None


@pytest.mark.parametrize("case", ["whole_period", "repeated_phase", "too_few"])
def test_rank_failure_is_retained_without_ridge(case):
    recording, reference, _, _ = fixture(duration=8_000_000 if case == "whole_period" else 1_000_000)
    if case == "repeated_phase":
        stream = recording.streams[0]
        stream.times_ns = {"exposure_start": (0,) * 12, "exposure_end": (1_000_000,) * 12}
    kwargs = {"sample_indices": [0, 1], "covariance": np.eye(4)} if case == "too_few" else {}
    report = qualify(recording, reference, **kwargs)
    assert report["status"] == "withheld"
    assert report["reasons"] == ["rank_deficient_design"]
    assert report["fit"]["rank"] < report["fit"]["parameter_count"]
    assert "coefficients" not in report["fit"]


def test_singular_covariance_withheld_invalid_covariance_rejected():
    recording, reference, _, _ = fixture()
    for covariance in (np.zeros((24, 24)), np.ones((24, 24))):
        assert qualify(recording, reference, covariance=covariance)["reasons"] == [
            "singular_noise_covariance"]
    for covariance in (-np.eye(24), np.eye(24) + np.triu(np.ones((24, 24)), 1),
                       np.full((24, 24), np.nan), np.eye(23)):
        with pytest.raises(ValueError):
            qualify(recording, reference, covariance=covariance)


@pytest.mark.parametrize("kwargs", [
    {"sample_indices": [0, 0]}, {"sample_indices": [True]}, {"sample_indices": [-1]},
    {"sample_indices": range(513)}, {"sample_indices": iter([0, 1])},
    {"channel_indices": [0, 0]}, {"channel_indices": [2]}, {"channel_indices": []},
    {"rank_rtol": 0}, {"rank_rtol": 1}, {"rank_rtol": True},
    {"max_phase_error_rad": -1}, {"max_phase_error_rad": float("nan")},
    {"covariance_provenance": {}},
])
def test_malformed_bounded_inputs_raise(kwargs):
    recording, reference, _, _ = fixture()
    with pytest.raises(ValueError):
        qualify(recording, reference, **kwargs)


def test_zero_signal_phase_undefined_and_covariance_still_supplied():
    recording, reference, _, _ = fixture()
    recording.streams[0].values[:] = 0
    report = qualify(recording, reference)
    assert report["fit"]["phase_lag_rad"] == [None, None]
    assert report["fit"]["amplitude"] == [0, 0]
    assert np.trace(report["fit"]["coefficient_covariance"]) > 0


def test_report_metadata_is_detached_and_digests_bind_mutations():
    recording, reference, _, _ = fixture()
    covariance = np.eye(24)
    provenance = {"source": {"name": "independent"}}
    report = qualify(recording, reference, covariance=covariance, covariance_provenance=provenance)
    frozen = copy.deepcopy(report)
    reference["provenance"]["source"] = "edited"
    provenance["source"]["name"] = "edited"
    covariance[0, 0] = 2
    recording.streams[0].values[0, 0] += 1
    assert report == frozen
    next_report = qualify(recording, reference, covariance=covariance, covariance_provenance=provenance)
    assert report["recording_sha256"] != next_report["recording_sha256"]
    assert report["configuration_sha256"] != next_report["configuration_sha256"]
    assert report["configuration_sha256"] == hashlib.sha256(canonical_json(report["configuration"])).hexdigest()


def test_source_mutation_during_fit_refused(monkeypatch):
    from noether_replay import probe

    recording, reference, _, _ = fixture()
    original = probe._fit

    def mutated(*args):
        result = original(*args)
        recording.streams[0].values[0, 0] += 1
        return result

    monkeypatch.setattr(probe, "_fit", mutated)
    with pytest.raises(ValueError, match="changed"):
        qualify(recording, reference)


@pytest.mark.parametrize("uncertainty", [2**62, np.int64(2**62), 2**63, np.uint64(2**63)])
def test_numpy_clock_bounds_do_not_overflow_duration_admission(uncertainty):
    recording, reference, _, _ = fixture()
    recording.streams[0].uncertainty_ns = uncertainty
    recording.calibrations[0].update(start_ns=-(2**66), end_ns=2**66)
    report = qualify(recording, reference, max_phase_error_rad=1e20)
    assert report["status"] == "withheld"
    assert "nonpositive_or_ambiguous_exposure_duration" in report["reasons"]
    assert report["fit"] is None


@pytest.mark.parametrize("field", ["phase_at_origin_rad", "phase_uncertainty_rad",
                                  "frequency_uncertainty_hz"])
def test_complex_reference_scalar_rejected(field):
    recording, reference, _, _ = fixture()
    reference[field] = np.complex128(1 + 2j)
    with pytest.raises(ValueError):
        qualify(recording, reference)


def test_tiny_rational_frequency_does_not_crash_exposure_integral():
    recording, reference, _, _ = fixture()
    reference["frequency_denominator"] = 10**350
    report = qualify(recording, reference)
    assert report["status"] == "withheld"
    assert report["reasons"] == ["rank_deficient_design"]


def test_covariance_digest_preserves_supplied_dtype_identity():
    recording, reference, _, _ = fixture()
    first = qualify(recording, reference, covariance=np.eye(24, dtype=np.float32))
    second = qualify(recording, reference, covariance=np.eye(24, dtype=np.float64))
    assert first["configuration"]["covariance_sha256"] == second["configuration"]["covariance_sha256"]
    assert first["configuration"]["supplied_covariance_sha256"] != second["configuration"]["supplied_covariance_sha256"]
    assert first["fit"] == second["fit"]


@pytest.mark.parametrize("seed,columns", [(0, 20), (2, 23)])
def test_projected_low_rank_covariance_not_admitted_by_roundoff_cholesky(seed, columns):
    recording, reference, _, _ = fixture()
    factor = np.random.default_rng(seed).normal(size=(24, columns))
    report = qualify(recording, reference, covariance=factor @ factor.T)
    assert report["status"] == "withheld"
    assert report["reasons"] == ["singular_noise_covariance"]
    assert report["fit"]["noise_covariance_rank"] == columns
    assert "coefficients" not in report["fit"]


def test_unresolved_positive_covariance_mode_withheld_by_declared_rank_tolerance():
    recording, reference, _, _ = fixture()
    covariance = np.eye(24)
    covariance[0, 0] = 1e-14
    report = qualify(recording, reference, covariance=covariance)
    assert report["reasons"] == ["singular_noise_covariance"]
    assert report["fit"]["noise_covariance_rank"] == 23


def test_positive_covariance_below_output_numeric_range_withheld():
    recording, reference, _, _ = fixture()
    report = qualify(recording, reference, covariance=np.eye(24) * np.nextafter(0.0, 1.0))
    assert report["reasons"] == ["coefficient_covariance_underflow"]
    assert "coefficients" not in report["fit"]
