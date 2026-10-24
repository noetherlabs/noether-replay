"""Current-event policy inputs without attributing alternate closed-loop outcomes."""

from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from noether_replay.conditioned import condition_observations
from noether_replay.model import Recording, Stream, canonical_json
from noether_replay.replay import ShadowObservation, run_shadow


@pytest.fixture
def recording():
    stream = Stream(
        "wfs", "wfs_measurements", np.array([[1.0], [2.0], [3.0]]), "rad",
        (10, 11, 14), "instrument",
        {
            "exposure_start": (0, 100, 400),
            "exposure_end": (10, 110, 410),
            "available": (20, 120, 420),
        },
        0,
        metadata={"future_values": [999.0], "evaluation_truth": [123.0]},
    )
    epochs = tuple(
        {
            "id": name, "clock_id": "instrument", "start_ns": start, "end_ns": end,
            "stream_ids": ["wfs"], "products": {"response": digest * 64},
        }
        for name, start, end, digest in (
            ("early", -1, 300, "a"), ("late", 300, 600, "b"), ("future", 600, 900, "c")
        )
    )
    return Recording("events", (stream,), epochs, metadata={"future_values": [456.0]})


def shadow(recording, policy, **kwargs):
    return run_shadow(
        recording, kwargs.pop("stream_id", "wfs"), policy,
        policy_id="test", policy_version="1", policy_input="event", **kwargs,
    )


def test_event_context_preserves_current_identity_and_calibration_transition(recording):
    seen = []

    def policy(event):
        assert isinstance(event, ShadowObservation)
        seen.append(event)
        return -event.values

    result = shadow(recording, policy)
    assert [event.sample_index for event in seen] == [0, 1, 2]
    assert [event.frame_id for event in seen] == [10, 11, 14]
    assert [event.available_ns for event in seen] == [20, 120, 420]
    assert [event.exposure_start_ns for event in seen] == [0, 100, 400]
    assert [event.exposure_end_ns for event in seen] == [10, 110, 410]
    assert [event.calibration_epoch_id for event in seen] == ["early", "early", "late"]
    assert [dict(event.calibration_products) for event in seen] == [
        {"response": "a" * 64}, {"response": "a" * 64}, {"response": "b" * 64},
    ]
    assert all(event.units == "rad" and event.clock_id == "instrument" for event in seen)
    assert all(event.uncertainty_ns == 0 and event.conditioning is None for event in seen)
    assert result["policy_input"] == "event"
    assert result["mode"] == "shadow" and result["outcome_assessed"] is False
    assert set(result) == {
        "mode", "policy", "recording_sha256", "stream_id", "clock_id", "samples",
        "outcome_assessed", "limitation", "policy_input",
    }


def test_event_input_excludes_other_samples_and_arbitrary_metadata(recording):
    def policy(event):
        assert {field.name for field in fields(event)} == {
            "values", "stream_id", "sample_index", "frame_id", "units", "clock_id",
            "available_ns", "uncertainty_ns", "exposure_start_ns", "exposure_end_ns",
            "calibration_epoch_id", "calibration_products", "conditioning",
        }
        np.testing.assert_array_equal(event.values, [event.sample_index + 1.0])
        assert not np.shares_memory(event.values, recording.streams[0].values)
        assert not hasattr(event, "__dict__")
        return -event.values

    shadow(recording, policy)


def test_metadata_is_frozen_and_values_are_an_independent_writable_copy(recording):
    before = recording.digest()
    seen = []

    def policy(event):
        with pytest.raises(FrozenInstanceError):
            event.sample_index = 99
        with pytest.raises(TypeError):
            event.calibration_products["response"] = "f" * 64
        event.values[:] = -7
        seen.append(event)
        return event.values

    result = shadow(recording, policy)
    assert [row["hypothetical_command"] for row in result["samples"]] == [[-7.0]] * 3
    assert recording.digest() == before
    recording.calibrations[0]["products"]["response"] = "d" * 64
    assert seen[0].calibration_products["response"] == "a" * 64
    assert not np.shares_memory(seen[0].values, seen[1].values)


def test_sorting_keeps_original_sample_identity(recording):
    source = recording.streams[0]
    order = [2, 0, 1]
    source.values = source.values[order]
    source.frame_ids = tuple(source.frame_ids[i] for i in order)
    source.times_ns = {key: tuple(times[i] for i in order) for key, times in source.times_ns.items()}
    seen = []
    result = shadow(recording, lambda event: seen.append(event) or -event.values)
    assert [event.sample_index for event in seen] == [1, 2, 0]
    assert [event.frame_id for event in seen] == [10, 11, 14]
    assert [row["sample_index"] for row in result["samples"]] == [1, 2, 0]
    assert [row["hypothetical_command"] for row in result["samples"]] == [[-1.0], [-2.0], [-3.0]]


def test_irregular_time_policy_uses_exact_integer_differences(recording):
    start = 2**70
    source = recording.streams[0]
    source.values[:] = 1
    source.times_ns = {"available": (start, start + 10, start + 40)}
    recording.calibrations = ()
    previous = None
    integral = 0

    def policy(event):
        nonlocal previous, integral
        assert type(event.available_ns) is int
        if previous is not None:
            integral += (event.available_ns - previous) * event.values[0]
        previous = event.available_ns
        return np.array([-integral])

    result = shadow(recording, policy)
    assert [row["hypothetical_command"] for row in result["samples"]] == [[0.0], [-10.0], [-40.0]]


def test_missing_acquisition_identity_units_and_calibration_stay_unknown(recording):
    source = recording.streams[0]
    source.units = None
    source.frame_ids = None
    source.times_ns = {"available": (20, 120, 420), "exposure_end": (None, 110, None)}
    seen = []
    shadow(recording, lambda event: seen.append(event) or -event.values)
    assert all(event.units is None and event.frame_id is None for event in seen)
    assert all(event.exposure_start_ns is None for event in seen)
    assert [event.exposure_end_ns for event in seen] == [None, 110, None]
    assert all(event.calibration_epoch_id is None and not event.calibration_products for event in seen)


@pytest.mark.parametrize("policy_input", [None, "automatic", "Event", True, []])
def test_invalid_policy_input_is_rejected_before_callbacks(recording, policy_input):
    seen = []
    with pytest.raises(ValueError, match="policy input"):
        run_shadow(
            recording, "wfs", lambda value: seen.append(value) or value,
            policy_id="test", policy_version="1", policy_input=policy_input,
        )
    assert seen == []


@pytest.mark.parametrize("problem", ["unknown", "overlap", "uncertainty", "nonfinite", "sequence"])
def test_all_raw_admission_checks_precede_event_callbacks(recording, problem):
    source = recording.streams[0]
    if problem == "unknown":
        source.times_ns["available"] = (20, None, 420)
    elif problem == "overlap":
        source.times_ns = {"available": (20, 20, 420)}
    elif problem == "uncertainty":
        source.uncertainty_ns = None
    elif problem == "nonfinite":
        source.values[-1, 0] = np.nan
    else:
        source.times_ns["exposure_end"] = (21, 110, 410)
    seen = []
    with pytest.raises(ValueError):
        shadow(recording, lambda event: seen.append(event) or event.values)
    assert seen == []


def make_conditioned(recording, *, available_ns=(424, 524)):
    recording.calibrations = (recording.calibrations[0],)
    recording.calibrations[0]["end_ns"] = 600
    return condition_observations(
        recording, "wfs", np.array([[0.25, 0.75, 0], [0, -1, 2]]), np.eye(3),
        output_id="filtered", covariance_provenance={"kind": "synthetic model"},
        sample_indices=(2, 0, 1), available_ns=available_ns,
    )


def test_conditioned_event_has_only_current_support_with_immutable_lineage(recording):
    candidate = make_conditioned(recording)
    before = candidate.digest()
    seen = []

    def policy(event):
        seen.append(event)
        assert event.exposure_start_ns is None and event.exposure_end_ns is None
        assert event.frame_id is None
        assert set(event.conditioning) == {
            "source_recording_sha256", "source_stream_id", "support_indices",
            "channel_indices", "weights", "contributing_exposures", "ready_not_before_ns",
        }
        with pytest.raises(TypeError):
            event.conditioning["weights"] = (1.0,)
        with pytest.raises(TypeError):
            event.conditioning["contributing_exposures"][0]["sample_index"] = 99
        return -event.values

    shadow(candidate, policy, stream_id="filtered")
    first, second = seen
    assert first.conditioning["support_indices"] == (2, 0)
    assert first.conditioning["weights"] == (0.25, 0.75)
    assert first.conditioning["channel_indices"] == (0,)
    assert second.conditioning["support_indices"] == (0, 1)
    assert second.conditioning["weights"] == (-1, 2)
    assert [item["sample_index"] for item in first.conditioning["contributing_exposures"]] == [2, 0]
    assert [event.calibration_epoch_id for event in seen] == ["early", "early"]
    assert [event.conditioning["ready_not_before_ns"] for event in seen] == [420, 120]
    assert candidate.digest() == before


@pytest.mark.parametrize("problem", ["unknown_availability", "epoch_gap", "corrupt_support"])
def test_conditioned_admission_withholds_every_event_callback(recording, problem):
    if problem == "unknown_availability":
        candidate = make_conditioned(recording, available_ns=None)
    elif problem == "epoch_gap":
        recording.calibrations = ()
        candidate = condition_observations(
            recording, "wfs", np.eye(3), np.eye(3), output_id="filtered",
            covariance_provenance={"kind": "synthetic model"}, available_ns=(20, 120, 420),
        )
    else:
        candidate = make_conditioned(recording)
        candidate.streams[-1].metadata["conditioning"]["support_indices"][0] = [0]
    seen = []
    with pytest.raises(ValueError):
        shadow(candidate, lambda event: seen.append(event) or event.values, stream_id="filtered")
    assert seen == []


def test_external_recording_mutation_still_prevents_a_result(recording):
    def policy(event):
        recording.streams[0].values[event.sample_index, 0] += 1
        return event.values

    with pytest.raises(ValueError, match="Recording changed"):
        shadow(recording, policy)


@pytest.mark.parametrize("bad_action", [np.array([np.nan]), np.zeros((1, 1)), np.array([])])
def test_event_mode_keeps_command_validation(recording, bad_action):
    with pytest.raises(ValueError, match="finite one-dimensional command"):
        shadow(recording, lambda event: bad_action)


def test_array_default_and_explicit_array_result_are_byte_identical(recording):
    options = dict(policy_id="test", policy_version="1")
    default = run_shadow(recording, "wfs", lambda values: -values, **options)
    explicit = run_shadow(recording, "wfs", lambda values: -values, policy_input="array", **options)
    assert canonical_json(default) == canonical_json(explicit)
    assert "policy_input" not in default
