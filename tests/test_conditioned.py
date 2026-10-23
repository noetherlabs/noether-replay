"""Scientific regressions for conditioned telemetry; no acquired research data."""

from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest

from noether_replay.conditioned import condition_observations
from noether_replay.model import SCHEMA, Recording, Stream, load_recording
from noether_replay.replay import calibration_assignment, run_shadow
from noether_replay.report import build_report


@pytest.fixture
def recording():
    stream = Stream(
        "wfs",
        "wfs_measurements",
        np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0], [13.0, 17.0]]),
        units="rad",
        frame_ids=(10, 11, 12, 13),
        clock_id="camera",
        times_ns={
            "exposure_start": (0, 100, 200, 300),
            "exposure_end": (10, 110, 210, 310),
            "available": (20, 120, 220, 320),
        },
        uncertainty_ns=2,
    )
    epoch = {
        "id": "cal",
        "clock_id": "camera",
        "start_ns": -10,
        "end_ns": 500,
        "stream_ids": ["wfs"],
        "products": {"response": "a" * 64},
    }
    return Recording("fixture", (stream,), (epoch,), provenance={"kind": "synthetic"})


def conditioned(recording, operator=None, covariance=None, **kwargs):
    if operator is None:
        operator = np.eye(4)
    if covariance is None:
        covariance = np.eye(8)
    return condition_observations(
        recording,
        "wfs",
        operator,
        covariance,
        output_id="conditioned",
        covariance_provenance={"kind": "declared_fixture", "version": 1},
        **kwargs,
    )


def assert_shadow_refused_before_callback(recording):
    calls = []

    def policy(values):
        calls.append(values.copy())
        return -values

    with pytest.raises(ValueError):
        run_shadow(recording, "conditioned", policy, policy_id="test", policy_version="1")
    assert calls == []


def test_complete_space_time_covariance_and_repeated_output_are_preserved(recording):
    before = recording.digest()
    source_values = recording.streams[0].values.copy()
    operator = np.array([[0.5, 0.5, 0.0, 0.0], [0.0, -1.0, 2.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
    factor = np.arange(1.0, 65.0).reshape(8, 8) / 64 + np.eye(8)
    covariance = factor @ factor.T
    result = conditioned(recording, operator, covariance, available_ns=(124, 224, 324))
    output = result.streams[-1]
    descriptor = output.metadata["conditioning"]
    expected_values = np.stack(
        (
            (source_values[0] + source_values[1]) / 2,
            2 * source_values[2] - source_values[1],
            (source_values[0] + source_values[1]) / 2,
        )
    )
    # Independent scalar summation checks sample-major layout, channel coupling,
    # temporal covariance and covariance between identical derived observations.
    expected_covariance = np.empty((6, 6))
    for a in range(3):
        for b in range(3):
            for c in range(2):
                for d in range(2):
                    expected_covariance[2 * a + c, 2 * b + d] = sum(
                        operator[a, i] * covariance[2 * i + c, 2 * j + d] * operator[b, j]
                        for i in range(4)
                        for j in range(4)
                    )
    np.testing.assert_array_equal(output.values, expected_values)
    # Duplicate outputs make this covariance singular; harmless eigensolver
    # roundoff projection is permitted, without an absolute physical noise floor.
    np.testing.assert_allclose(
        descriptor["output_covariance"], expected_covariance, rtol=2e-13, atol=0
    )
    assert descriptor["support_indices"] == [[0, 1], [1, 2], [0, 1]]
    assert descriptor["ready_not_before_ns"] == [122, 222, 122]
    assert descriptor["calibration_epoch_ids"] == ["cal"] * 3
    assert output.frame_ids is None
    assert set(output.times_ns) == {"available"}
    assert output.units == "rad"
    assert recording.digest() == before
    np.testing.assert_array_equal(recording.streams[0].values, source_values)
    actions = run_shadow(result, output.id, lambda x: -x, policy_id="test", policy_version="1")
    assert actions["outcome_assessed"] is False
    assert len(actions["samples"]) == 3


def test_noncontiguous_pixels_preserve_explicit_sample_and_flat_channel_order(recording):
    source = recording.streams[0]
    source.values = np.arange(32.0, dtype=float).reshape(4, 2, 4)[:, :, ::2]
    source.kind = "wfs_pixels"
    assert not source.values.flags.c_contiguous
    operator = np.array([[1.0, 0.0, 0.0], [0.0, 0.25, 0.75]])
    result = conditioned(
        recording,
        operator,
        np.eye(6),
        sample_indices=[2, 0, 3],
        channel_indices=[3, 0],
        available_ns=(224, 324),
    )
    output = result.streams[-1]
    selected = np.array([[source.values[i, 1, 1], source.values[i, 0, 0]] for i in [2, 0, 3]])
    np.testing.assert_array_equal(
        output.values, np.stack((selected[0], 0.25 * selected[1] + 0.75 * selected[2]))
    )
    assert output.metadata["conditioning"]["support_indices"] == [[2], [0, 3]]
    assert output.metadata["conditioning"]["sample_indices"] == [2, 0, 3]
    assert output.metadata["conditioning"]["channel_indices"] == [3, 0]


def test_tiny_exact_nonzero_coefficient_still_requires_the_latest_input(recording):
    operator = np.array([[1.0, 0.0, 0.0, np.nextafter(0.0, 1.0)]])
    with pytest.raises(ValueError):
        conditioned(recording, operator, available_ns=(124,))
    result = conditioned(recording, operator, available_ns=(324,))
    descriptor = result.streams[-1].metadata["conditioning"]
    assert descriptor["support_indices"] == [[0, 3]]
    assert descriptor["ready_not_before_ns"] == [322]
    assert descriptor["contributing_exposures"][0][-1]["sample_index"] == 3


def test_availability_accounts_for_both_input_and_output_clock_uncertainty(recording):
    operator = np.array([[0.0, 1.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        conditioned(recording, operator, available_ns=(123,))
    result = conditioned(recording, operator, available_ns=(124,))
    assert result.streams[-1].times_ns["available"] == (124,)


def test_readiness_arithmetic_preserves_python_integer_nanoseconds(recording):
    source = recording.streams[0]
    offset = 2**80
    source.times_ns = {
        key: tuple(offset + t for t in times) for key, times in source.times_ns.items()
    }
    recording.calibrations[0].update(start_ns=offset - 10, end_ns=offset + 500)
    result = conditioned(recording, np.array([[1.0, 0.0, 0.0, 0.0]]), available_ns=(offset + 24,))
    assert result.streams[-1].metadata["conditioning"]["ready_not_before_ns"] == [offset + 22]


@pytest.mark.parametrize("missing", ["availability", "one_availability", "uncertainty", "clock"])
def test_unknown_timing_never_becomes_processing_completion(recording, missing):
    source = recording.streams[0]
    if missing == "availability":
        source.times_ns.pop("available")
    elif missing == "one_availability":
        source.times_ns["available"] = (20, None, 220, 320)
    elif missing == "uncertainty":
        source.uncertainty_ns = None
    else:
        source.clock_id, source.times_ns = None, {}
    result = conditioned(recording, np.array([[0.5, 0.5, 0.0, 0.0]]))
    output = result.streams[-1]
    assert output.metadata["conditioning"]["ready_not_before_ns"] == [None]
    assert output.times_ns.get("available", (None,)) == (None,)
    assert_shadow_refused_before_callback(result)
    with pytest.raises(ValueError):
        conditioned(recording, np.array([[0.5, 0.5, 0.0, 0.0]]), available_ns=(500,))


def test_known_inputs_do_not_manufacture_output_completion(recording):
    result = conditioned(recording)
    assert result.streams[-1].metadata["conditioning"]["ready_not_before_ns"] == [22, 122, 222, 322]
    assert result.streams[-1].times_ns["available"] == (None,) * 4
    assert_shadow_refused_before_callback(result)


def test_unknown_unused_input_does_not_change_exact_support_readiness(recording):
    recording.streams[0].times_ns["available"] = (20, 120, 220, None)
    result = conditioned(recording, np.array([[0.5, 0.5, 0.0, 0.0]]), available_ns=(124,))
    descriptor = result.streams[-1].metadata["conditioning"]
    assert descriptor["support_indices"] == [[0, 1]]
    assert descriptor["ready_not_before_ns"] == [122]


@pytest.mark.parametrize(
    "failure", ["crossing", "missing", "overlap", "exposure_unknown", "hull_overlap"]
)
def test_unqualified_calibration_cannot_enter_shadow(recording, failure):
    operator = np.array([[0.0, 0.5, 0.5, 0.0]])
    if failure == "crossing":
        second = deepcopy(recording.calibrations[0])
        recording.calibrations[0]["end_ns"] = 160
        second.update(id="later", start_ns=160)
        recording.calibrations += (second,)
    elif failure == "missing":
        recording.calibrations = ()
    elif failure in {"overlap", "hull_overlap"}:
        second = deepcopy(recording.calibrations[0])
        second.update(
            id="ambiguous",
            start_ns=103 if failure == "overlap" else 150,
            end_ns=106 if failure == "overlap" else 160,
        )
        recording.calibrations += (second,)
    else:
        recording.streams[0].times_ns["exposure_end"] = (10, None, 210, 310)
    result = conditioned(recording, operator, available_ns=(224,))
    output = result.streams[-1]
    assert output.metadata["conditioning"]["calibration_epoch_ids"] == [None]
    assert calibration_assignment(result, output, 0) is None
    assert_shadow_refused_before_callback(result)
    assert any(
        f["code"] == "unqualified_calibration_interval" for f in build_report(result)["findings"]
    )


@pytest.mark.parametrize(
    "mutation", ["values", "covariance", "operator", "support", "units", "frames"]
)
def test_derived_mutation_is_detected_before_policy_execution(recording, mutation):
    result = conditioned(recording, available_ns=(24, 124, 224, 324))
    output = result.streams[-1]
    descriptor = output.metadata["conditioning"]
    if mutation == "values":
        output.values[0, 0] += 0.1
    elif mutation == "covariance":
        descriptor["output_covariance"][0][0] += 0.1
    elif mutation == "operator":
        descriptor["operator"][0][0] += 0.1
    elif mutation == "support":
        descriptor["support_indices"][0] = [1]
    elif mutation == "units":
        output.units = "nm"
    else:
        output.frame_ids = (10, 11, 12, 13)
    with pytest.raises(ValueError):
        result.validate()
    assert_shadow_refused_before_callback(result)


def test_source_mutation_invalidates_all_derived_bindings(recording):
    result = conditioned(recording, available_ns=(24, 124, 224, 324))
    recording.streams[0].values[0, 0] += 0.1
    with pytest.raises(ValueError):
        build_report(result)
    assert_shadow_refused_before_callback(result)


@pytest.mark.parametrize(
    "part", ["observations", "covariance", "null_observation", "null_covariance"]
)
def test_large_channel_does_not_hide_corruption_of_a_small_channel(recording, part):
    recording.streams[0].values[0] = [1e9, 1e-9]
    if part == "null_observation":
        recording.streams[0].values[0, 1] = 0
    covariance = np.diag([1e9, 1e-9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    result = conditioned(recording, covariance=covariance, available_ns=(24, 124, 224, 324))
    output = result.streams[-1]
    if part == "observations":
        output.values[0, 1] = 1e-3
    elif part == "covariance":
        output.metadata["conditioning"]["output_covariance"][1][1] = 1e-3
    elif part == "null_observation":
        output.values[0, 1] = 1e-300
    else:
        output.metadata["conditioning"]["output_covariance"][0][1] = 1e-300
    with pytest.raises(ValueError):
        result.validate()


def test_external_operator_covariance_and_provenance_are_snapshotted(recording):
    operator, covariance = np.eye(4), np.eye(8)
    provenance = {"kind": "synthetic", "parameters": {"variance": 1.0}}
    result = condition_observations(
        recording,
        "wfs",
        operator,
        covariance,
        output_id="conditioned",
        covariance_provenance=provenance,
    )
    before = result.digest()
    operator[0, 0] = 2
    covariance[0, 0] = 3
    provenance["parameters"]["variance"] = 4
    assert result.digest() == before


def test_report_conditioning_metadata_is_detached_from_mutable_stream(recording):
    result = conditioned(recording, available_ns=(24, 124, 224, 324))
    report = build_report(result)
    snapshot = deepcopy(report)
    summary = next(row for row in report["streams"] if row["id"] == "conditioned")
    assert summary["conditioning"]["source_recording_sha256"] == recording.digest()
    result.streams[-1].metadata["conditioning"]["covariance_provenance"]["version"] = 2
    assert report == snapshot
    assert report["recording_sha256"] != result.digest()


def test_independent_derivations_share_raw_parent_but_chaining_is_refused(recording):
    first = conditioned(recording)
    second = condition_observations(
        first,
        "wfs",
        np.array([[0.5, 0.5, 0.0, 0.0]]),
        np.eye(8),
        output_id="second",
        covariance_provenance={"kind": "fixture"},
    )
    assert (
        second.streams[-1].metadata["conditioning"]["source_recording_sha256"] == recording.digest()
    )
    assert (
        second.streams[-2].metadata["conditioning"]["source_recording_sha256"] == recording.digest()
    )
    second.validate()
    with pytest.raises(ValueError):
        condition_observations(
            second,
            "conditioned",
            np.eye(4),
            np.eye(8),
            output_id="chain",
            covariance_provenance={"kind": "fixture"},
        )


def test_conditioned_json_round_trip_reconstructs_and_binds_all_products(recording, tmp_path):
    result = conditioned(recording, available_ns=(24, 124, 224, 324))
    payload = {
        "schema": SCHEMA,
        "id": result.id,
        "calibrations": result.calibrations,
        "metadata": result.metadata,
        "provenance": result.provenance,
        "streams": [],
    }
    for stream in result.streams:
        payload["streams"].append(
            {
                "id": stream.id,
                "kind": stream.kind,
                "values": stream.values.tolist(),
                "dtype": stream.values.dtype.str,
                "units": stream.units,
                "frame_ids": stream.frame_ids,
                "clock_id": stream.clock_id,
                "times_ns": stream.times_ns,
                "uncertainty_ns": stream.uncertainty_ns,
                "metadata": stream.metadata,
            }
        )
    path = tmp_path / "conditioned.json"
    path.write_text(json.dumps(payload))
    restored = load_recording(path)
    assert restored.digest() == result.digest()
    np.testing.assert_array_equal(restored.streams[-1].values, result.streams[-1].values)


@pytest.mark.parametrize("selection", [[], [0, 0], [-1], [4], [True], [1.0], 1])
def test_malformed_sample_selection_is_rejected(recording, selection):
    with pytest.raises(ValueError):
        conditioned(recording, sample_indices=selection)


@pytest.mark.parametrize("selection", [[], [0, 0], [-1], [2], [False], [1.0], 1])
def test_malformed_channel_selection_is_rejected(recording, selection):
    with pytest.raises(ValueError):
        conditioned(recording, channel_indices=selection)


@pytest.mark.parametrize(
    "covariance",
    [np.eye(7), np.diag([-1.0, *([1.0] * 7)]), np.full((8, 8), np.nan), np.eye(8, dtype=complex)],
)
def test_invalid_joint_covariance_is_rejected(recording, covariance):
    with pytest.raises(ValueError):
        conditioned(recording, covariance=covariance)


def test_zero_support_and_excess_dimensions_cannot_create_observations(recording):
    with pytest.raises(ValueError):
        conditioned(recording, np.zeros((1, 4)))
    with pytest.raises(ValueError):
        conditioned(recording, np.ones((257, 4)))
    with pytest.raises(ValueError):
        conditioned(recording, np.ones((513, 4)))


def test_metadata_budget_is_cumulative_across_independent_windows(recording, monkeypatch):
    from noether_replay import conditioned as module
    from noether_replay.model import canonical_json

    first = conditioned(recording)
    size = len(canonical_json(first.streams[-1].metadata["conditioning"]))
    monkeypatch.setattr(module, "MAX_JSON_BYTES", size + size // 2)
    first.validate()
    with pytest.raises(ValueError, match="metadata exceeds"):
        condition_observations(
            first,
            "wfs",
            np.eye(4),
            np.eye(8),
            output_id="second",
            covariance_provenance={"kind": "declared_fixture", "version": 1},
        )
    assert len(first.streams) == 2


def test_indefinite_retained_covariance_is_rejected_before_reconstruction_tolerance(recording):
    covariance = np.diag([1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    result = conditioned(recording, covariance=covariance)
    result.streams[-1].metadata["conditioning"]["output_covariance"][1][1] = -1e-12
    with pytest.raises(ValueError, match="positive semidefinite"):
        result.validate()


def test_recording_containers_are_independent_and_source_streams_are_shared(recording):
    result = conditioned(recording)
    assert result.streams[0] is recording.streams[0]
    result.calibrations[0]["products"]["response"] = "b" * 64
    result.provenance["kind"] = "changed"
    result.metadata["new"] = "value"
    assert recording.calibrations[0]["products"]["response"] == "a" * 64
    assert recording.provenance == {"kind": "synthetic"}
    assert recording.metadata == {}
    with pytest.raises(ValueError, match="source recording changed"):
        result.validate()


def test_derived_calibration_assignment_requires_the_validated_owning_stream(recording):
    result = conditioned(recording)
    with pytest.raises(ValueError, match="owning recording"):
        calibration_assignment(result, deepcopy(result.streams[-1]), 0)
    result.streams[-1].metadata["conditioning"]["source_stream_id"] = ["wfs"]
    with pytest.raises(ValueError, match="source recording changed"):
        result.validate()


def test_one_dimensional_source_selection_and_nonfinite_values(recording):
    recording.streams[0].values = np.array([1.0, 2.0, 3.0, 4.0])
    result = conditioned(recording, np.array([[0.5, 0.5]]), np.eye(2), sample_indices=[1, 3])
    np.testing.assert_array_equal(result.streams[-1].values, [[3.0]])
    recording.streams[0].values[1] = np.inf
    with pytest.raises(ValueError, match="nonfinite source"):
        conditioned(recording, np.array([[0.5, 0.5]]), np.eye(2), sample_indices=[1, 3])


def test_report_does_not_repeat_dense_reconstruction_for_every_output(recording, monkeypatch):
    from noether_replay import conditioned as module

    result = conditioned(recording, np.ones((100, 4)))
    original = module.validate_conditioned
    calls = []

    def counted(candidate):
        if candidate is result:
            calls.append(True)
        return original(candidate)

    monkeypatch.setattr(module, "validate_conditioned", counted)
    report = build_report(result)
    assert len(calls) <= 2  # Manifest and timeline, independent of output count.
    summary = next(item for item in report["streams"] if item["id"] == "conditioned")
    assert summary["samples"] == summary["calibration_covered_samples"] == 100


def test_signed_integer_covariance_cannot_hide_off_diagonal_corruption(recording):
    result = conditioned(recording, np.array([[1.0, 0.0, 0.0, 0.0]]), np.eye(8) * 2.0**63)
    result.streams[-1].metadata["conditioning"]["output_covariance"] = [
        [2**63 - 1, -(2**63)],
        [-(2**63), 2**63 - 1],
    ]
    with pytest.raises(ValueError, match="covariance does not reproduce"):
        result.validate()


def test_memory_layout_does_not_change_conditioned_reconstruction(recording):
    recording.streams[0].values = np.array([[1e16, 1.0], [1.0, 2.0], [-1e16, 3.0], [1.0, 4.0]])
    operator = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, -1.0], [1.0, 0.0, 1.0, 1.0]])
    factor = np.arange(64.0, dtype=float).reshape(8, 8) / 64 + np.eye(8)
    covariance = factor @ factor.T
    c_order = conditioned(recording, operator, covariance)
    f_order = conditioned(recording, np.asfortranarray(operator), np.asfortranarray(covariance))
    assert c_order.digest() == f_order.digest()
