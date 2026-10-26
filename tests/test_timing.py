"""Exact event bounds, explicit lineage and withheld timing evidence."""

from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

from noether_replay.conditioned import condition_observations
from noether_replay.model import Recording, Stream, canonical_json
from noether_replay.timing import qualify_command_timing


@pytest.fixture
def recording():
    first = Stream(
        "wfs", "wfs_measurements", np.array([[1.0], [2.0], [3.0]]),
        "rad", (102, 100, 101), "clock", {"available": (300, 100, 200)}, 10,
    )
    second = Stream(
        "camera", "wfs_pixels", np.zeros((1, 2, 2)),
        clock_id="clock", times_ns={"available": (210,)}, uncertainty_ns=30,
    )
    commands = Stream(
        "dm", "command_applied", np.zeros((3, 1)), frame_ids=(200, 202, 201),
        clock_id="clock", times_ns={"effective": (260, 500, 110)}, uncertainty_ns=5,
    )
    return Recording("timing-fixture", (first, second, commands))


def association(applied_index=0, *references):
    return {
        "applied_index": applied_index,
        "observations": [
            {"stream_id": stream, "sample_index": index}
            for stream, index in (references or (("wfs", 2),))
        ],
    }


def qualify(recording, associations=None, **kwargs):
    return qualify_command_timing(
        recording, applied_stream_id="dm",
        associations=[association()] if associations is None else associations, **kwargs,
    )


def test_multi_observation_interval_uses_maximum_of_both_bounds(recording):
    report = qualify(
        recording, [association(0, ("wfs", 2), ("camera", 0))], latency_budget_ns=60,
    )
    row = report["commands"][0]
    # WFS=[190,210], camera=[180,240]. The latest lower bound belongs
    # to the WFS despite the camera's later nominal availability.
    assert row["latest_availability_interval_ns"] == [190, 240]
    assert row["effective_interval_ns"] == [255, 265]
    assert row["latency_interval_ns"] == [15, 75]
    assert row["causal_order"] == "consistent"
    assert row["budget"] == "ambiguous"
    assert row["reasons"] == []


def test_every_applied_command_is_retained_without_positional_or_frame_pairing(recording):
    report = qualify(recording, [association(2, ("wfs", 1))], latency_budget_ns=50)
    assert [row["applied_index"] for row in report["commands"]] == [0, 1, 2]
    assert [row["frame_id"] for row in report["commands"]] == [200, 202, 201]
    assert report["summary"]["command_count"] == 3
    assert report["summary"]["associated_count"] == 1
    assert report["summary"]["causal_order_counts"]["not_established"] == 2
    for row in report["commands"][:2]:
        assert row["observations"] == []
        assert row["latency_interval_ns"] is None
        assert row["budget"] == "not_established"
        assert row["reasons"] == ["missing_association"]
    assert report["commands"][2]["observations"][0]["sample_index"] == 1
    assert "passed" not in report and "pass" not in report["summary"]


@pytest.mark.parametrize(
    "effective,uncertainty,budget,causal,verdict,interval",
    [
        (220, 0, 20, "consistent", "met", [20, 20]),
        (220, 0, 19, "consistent", "exceeded", [20, 20]),
        (220, 1, 20, "consistent", "ambiguous", [19, 21]),
        (200, 0, 0, "consistent", "met", [0, 0]),
        (200, 1, 1000, "ambiguous", "not_established", [-1, 1]),
        (199, 0, 1000, "violated", "not_established", [-1, -1]),
        (199, 1, 1000, "ambiguous", "not_established", [-2, 0]),
        (220, 0, None, "consistent", "not_requested", [20, 20]),
        (199, 0, None, "violated", "not_requested", [-1, -1]),
    ],
)
def test_exact_causal_and_budget_boundaries(
    recording, effective, uncertainty, budget, causal, verdict, interval,
):
    recording.streams[0].uncertainty_ns = 0
    recording.streams[2].uncertainty_ns = uncertainty
    recording.streams[2].times_ns["effective"] = (effective, 500, 110)
    row = qualify(recording, latency_budget_ns=budget)["commands"][0]
    assert row["latency_interval_ns"] == interval
    assert row["causal_order"] == causal
    assert row["budget"] == verdict


def test_nanoseconds_above_float_and_int64_ranges_remain_exact(recording):
    origin = 2**90
    for stream in recording.streams:
        stream.times_ns = {
            key: tuple(origin + int(time) for time in times)
            for key, times in stream.times_ns.items()
        }
    row = qualify(recording)["commands"][0]
    assert row["effective_ns"] == origin + 260
    assert row["latency_interval_ns"] == [45, 75]
    assert all(type(value) is int for value in row["latency_interval_ns"])
    assert row["latest_availability_interval_ns"] == [origin + 190, origin + 210]


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("source_clock", {"missing_observation_clock", "missing_availability"}),
        ("target_clock", {"missing_command_clock", "missing_effective_time"}),
        ("source_uncertainty", {"unknown_observation_uncertainty"}),
        ("target_uncertainty", {"unknown_command_uncertainty"}),
        ("source_time", {"missing_availability"}),
        ("target_time", {"missing_effective_time"}),
        ("cross_clock", {"cross_clock_reference"}),
    ],
)
def test_missing_evidence_and_cross_clock_links_never_produce_a_latency(recording, mutation, expected):
    source, _, target = recording.streams
    if mutation == "source_clock":
        source.clock_id, source.times_ns = None, {}
    elif mutation == "target_clock":
        target.clock_id, target.times_ns = None, {}
    elif mutation == "source_uncertainty":
        source.uncertainty_ns = None
    elif mutation == "target_uncertainty":
        target.uncertainty_ns = None
    elif mutation == "source_time":
        source.times_ns["available"] = (300, 100, None)
    elif mutation == "target_time":
        target.times_ns["effective"] = (None, 500, 110)
    else:
        source.clock_id = "other-clock"
    row = qualify(recording, latency_budget_ns=100)["commands"][0]
    assert set(row["reasons"]) == expected
    assert row["latest_availability_interval_ns"] is None
    assert row["latency_interval_ns"] is None
    assert row["causal_order"] == row["budget"] == "not_established"


@pytest.mark.parametrize(
    "associations",
    [
        None, {}, (), [None], [{}],
        [{"applied_index": 0, "observations": [], "extra": 1}],
        [{"applied_index": 0, "observations": []}],
        [{"applied_index": 0, "observations": ()}],
        [{"applied_index": True, "observations": [{"stream_id": "wfs", "sample_index": 0}]}],
        [{"applied_index": 0.0, "observations": [{"stream_id": "wfs", "sample_index": 0}]}],
        [association(-1)], [association(3)], [association(), association()],
        [association(0, ("wfs", 0), ("wfs", 0))],
        [association(0, ("missing", 0))], [association(0, ("dm", 0))],
        [association(0, ("wfs", True))], [association(0, ("wfs", 0.0))],
        [association(0, ("wfs", -1))], [association(0, ("wfs", 3))],
        [{"applied_index": 0, "observations": [None]}],
        [{"applied_index": 0, "observations": [{"stream_id": [], "sample_index": 0}]}],
        [{"applied_index": 0, "observations": [{"stream_id": "wfs", "sample_index": 0, "x": 1}]}],
    ],
)
def test_malformed_lineage_is_rejected(recording, associations):
    with pytest.raises(ValueError):
        qualify_command_timing(recording, applied_stream_id="dm", associations=associations)


@pytest.mark.parametrize("budget", [True, np.bool_(False), -1, 1.5, "20", float("nan")])
def test_invalid_budget_is_rejected(recording, budget):
    with pytest.raises(ValueError, match="Latency budget"):
        qualify(recording, latency_budget_ns=budget)


@pytest.mark.parametrize("stream_id", [None, [], "", "missing", "wfs"])
def test_invalid_target_is_rejected(recording, stream_id):
    with pytest.raises(ValueError):
        qualify_command_timing(recording, applied_stream_id=stream_id, associations=[])


def test_recorded_commands_and_empty_applied_streams_are_not_promoted(recording):
    target = recording.streams[2]
    target.kind = "command_recorded"
    with pytest.raises(ValueError, match="command_applied"):
        qualify(recording)
    target.kind = "command_applied"
    target.values = np.zeros((0, 1))
    target.frame_ids, target.times_ns = (), {"effective": ()}
    with pytest.raises(ValueError, match="at least one"):
        qualify(recording, [])


def test_all_missing_associations_remain_counted(recording):
    result = qualify(recording, [], latency_budget_ns=1)
    assert result["summary"]["associated_count"] == 0
    assert result["summary"]["causal_order_counts"]["not_established"] == 3
    assert result["summary"]["budget_counts"]["not_established"] == 3


@pytest.mark.parametrize("stream,stages", [
    (0, {"exposure_start": (None, None, 150), "exposure_end": (None, None, 220)}),
    (2, {"issued": (270, None, None)}),
    (2, {"completed": (270, None, None)}),
])
def test_contradictory_declared_stages_withhold_an_otherwise_met_budget(recording, stream, stages):
    recording.streams[stream].times_ns.update(stages)
    row = qualify(recording, latency_budget_ns=1000)["commands"][0]
    assert row["reasons"] == ["event_sequence_violation"]
    assert row["causal_order"] == row["budget"] == "not_established"
    assert row["latency_interval_ns"] is None


def test_logging_time_and_missing_optional_stages_do_not_change_causal_assessment(recording):
    expected = qualify(recording)["commands"][0]
    recording.streams[0].times_ns["recorded"] = (0, 0, 0)
    recording.streams[2].times_ns["recorded"] = (0, 0, 0)
    recording.streams[2].times_ns["issued"] = (None, None, None)
    assert qualify(recording)["commands"][0] == expected


@pytest.mark.parametrize("stage", ["completed", "issued"])
@pytest.mark.parametrize(
    "time,uncertainty,reason",
    [
        (190, 0, "input_unavailable_at_"),
        (200, 5, "ambiguous_input_availability_at_"),
        (200, 0, None),
        (None, 0, None),
    ],
)
def test_known_completion_and_issue_must_follow_associated_availability(
    recording, stage, time, uncertainty, reason,
):
    recording.streams[0].uncertainty_ns = 0
    target = recording.streams[2]
    target.uncertainty_ns = uncertainty
    target.times_ns[stage] = (time, None, None)
    row = qualify(recording, latency_budget_ns=100)["commands"][0]
    assert row[f"{stage}_ns"] == time
    assert row[f"{stage}_interval_ns"] == (
        None if time is None else [time - uncertainty, time + uncertainty]
    )
    # Actuation still follows availability, but incompatible or ambiguous
    # command generation prevents that interval from earning qualification.
    assert row["latency_interval_ns"] == [60 - uncertainty, 60 + uncertainty]
    if reason is not None:
        assert row["reasons"] == [reason + stage]
        assert row["causal_order"] == row["budget"] == "not_established"
    else:
        assert row["reasons"] == []
        assert row["causal_order"] == "consistent"
        assert row["budget"] == "met"


def test_normalization_detaches_input_and_binds_source_lineage_and_budget(recording):
    references = [association(2, ("wfs", 1)), association(0, ("wfs", 2), ("camera", 0))]
    original = copy.deepcopy(references)
    first = qualify(recording, references, latency_budget_ns=np.int64(70))
    second = qualify(recording, list(reversed(references)), latency_budget_ns=70)
    assert canonical_json(first) == canonical_json(second)
    assert references == original
    assert first["configuration_sha256"] == hashlib.sha256(canonical_json({
        "recording_sha256": first["recording_sha256"], "configuration": first["configuration"],
    })).hexdigest()
    preserved = copy.deepcopy(first)
    references[0]["observations"][0]["sample_index"] = 2
    assert first == preserved
    assert qualify(recording, references, latency_budget_ns=70)["configuration_sha256"] != first["configuration_sha256"]
    assert qualify(recording, original, latency_budget_ns=71)["configuration_sha256"] != first["configuration_sha256"]
    recording.streams[0].values[0, 0] += 1
    changed = qualify(recording, original, latency_budget_ns=70)
    assert changed["recording_sha256"] != first["recording_sha256"]
    assert changed["configuration_sha256"] != first["configuration_sha256"]


def test_nonfinite_values_and_unknown_calibration_do_not_change_timing(recording):
    before = qualify(recording)
    recording.streams[0].values[:] = np.nan
    recording.streams[2].values[:] = np.inf
    after = qualify(recording)
    assert before["commands"] == after["commands"]
    assert before["recording_sha256"] != after["recording_sha256"]
    assert recording.calibrations == ()


def test_validated_conditioned_observation_uses_supplied_completion(recording):
    conditioned = condition_observations(
        recording, "wfs", np.array([[0.5, 0.5]]), np.eye(2), output_id="mean",
        covariance_provenance={"kind": "test"}, sample_indices=[1, 2],
        available_ns=[240],
    )
    row = qualify(conditioned, [association(0, ("mean", 0))])["commands"][0]
    assert row["latency_interval_ns"] == [5, 35]
    assert row["causal_order"] == "consistent"
    # Readiness is 210, but its existence cannot replace missing output availability.
    unknown = condition_observations(
        recording, "wfs", np.array([[0.5, 0.5]]), np.eye(2), output_id="mean",
        covariance_provenance={"kind": "test"}, sample_indices=[1, 2],
    )
    withheld = qualify(unknown, [association(0, ("mean", 0))])["commands"][0]
    assert withheld["causal_order"] == "not_established"
    assert withheld["reasons"] == ["missing_availability"]


def test_conditioned_source_tampering_is_rejected(recording):
    conditioned = condition_observations(
        recording, "wfs", np.array([[1.0]]), np.eye(1), output_id="copy",
        covariance_provenance={"kind": "test"}, sample_indices=[2], available_ns=[225],
    )
    conditioned.streams[-1].values[0, 0] += 1
    with pytest.raises(ValueError, match="reproduce"):
        qualify(conditioned, [association(0, ("copy", 0))])


def test_changed_source_digest_prevents_report(recording, monkeypatch):
    original = recording.digest
    calls = 0

    def digest():
        nonlocal calls
        calls += 1
        if calls == 2:
            recording.streams[0].values[0, 0] += 1
        return original()

    monkeypatch.setattr(recording, "digest", digest)
    with pytest.raises(ValueError, match="Recording changed"):
        qualify(recording)


def test_association_metadata_bound(recording, monkeypatch):
    monkeypatch.setattr("noether_replay.timing.MAX_JSON_BYTES", 1)
    with pytest.raises(ValueError, match="metadata bound"):
        qualify(recording)


@pytest.mark.parametrize("limit_stage", ["configuration", "first_row", "final_report"])
def test_report_size_refusal_preserves_source_and_never_returns_partial_coverage(
    recording, monkeypatch, limit_stage,
):
    report = qualify(recording, [])
    source_digest = recording.digest()
    configuration_size = len(canonical_json({
        "recording_sha256": source_digest, "configuration": report["configuration"],
    }))
    limit = {
        "configuration": configuration_size - 1,
        "first_row": configuration_size + len(canonical_json(report["commands"][0])) - 1,
        "final_report": len(canonical_json(report)) - 1,
    }[limit_stage]
    monkeypatch.setattr("noether_replay.timing.MAX_JSON_BYTES", limit)
    with pytest.raises(ValueError, match="Timing report exceeds"):
        qualify(recording, [])
    assert recording.digest() == source_digest
