"""Exposure-age evidence is distinct from latest-input processing latency."""

from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

import noether_replay.freshness as freshness
from noether_replay.conditioned import condition_observations
from noether_replay.model import Recording, Stream, canonical_json
from noether_replay.timing import qualify_command_timing


@pytest.fixture
def recording():
    return Recording(
        "synthetic-input-age",
        (
            Stream(
                "wfs", "wfs_measurements", np.array([[1.0], [2.0]]), "rad",
                frame_ids=(99, 10), clock_id="clock", uncertainty_ns=0,
                times_ns={"exposure_start": (0, 90), "exposure_end": (10, 95),
                          "available": (11, 96)},
            ),
            Stream(
                "dm", "command_applied", np.zeros((3, 1)), frame_ids=(8, 3, 5),
                clock_id="clock", uncertainty_ns=0,
                times_ns={"effective": (100, 101, 102)},
            ),
        ),
    )


def association(applied_index=0, *references):
    return {
        "applied_index": applied_index,
        "observations": [
            {"stream_id": stream, "sample_index": index}
            for stream, index in (references or (("wfs", 0), ("wfs", 1)))
        ],
    }


def qualify(recording, associations=None, **kwargs):
    return freshness.qualify_input_freshness(
        recording, applied_stream_id="dm",
        associations=[association()] if associations is None else associations,
        max_support_age_ns=kwargs.pop("max_support_age_ns", 20), **kwargs,
    )


def first_contributor(row):
    return row["observations"][0]["contributors"][0]


def conditioned(recording, operator):
    return condition_observations(
        recording, "wfs", [operator], np.eye(2), output_id="filtered",
        covariance_provenance={"source": "analytical unit noise"}, available_ns=(97,),
    )


def test_fresh_latest_input_does_not_conceal_stale_required_history(recording):
    report = qualify(recording, latency_budget_ns=5)
    assert report["timing"] == qualify_command_timing(
        recording, applied_stream_id="dm", associations=[association()], latency_budget_ns=5,
    )
    timing, row = report["timing"]["commands"][0], report["commands"][0]
    assert timing["latency_interval_ns"] == [4, 4]
    assert timing["budget"] == "met"
    assert row["oldest_support_age_interval_ns"] == [100, 100]
    assert row["budget"] == "exceeded"
    assert first_contributor(row)["support_age_hull_ns"] == [90, 100]
    assert row["observations"][1]["contributors"][0]["support_age_hull_ns"] == [5, 10]
    assert report["summary"]["budget_counts"] == {
        "met": 0, "exceeded": 1, "ambiguous": 0, "not_established": 2,
    }


def test_long_exposure_is_not_replaced_by_its_end_or_midpoint(recording):
    recording.streams[0].times_ns["exposure_start"] = (0, 0)
    row = qualify(recording, [association(0, ("wfs", 1))])["commands"][0]
    contributor = first_contributor(row)
    assert contributor["support_age_hull_ns"] == [5, 100]
    assert contributor["oldest_support_age_interval_ns"] == [100, 100]
    assert row["budget"] == "exceeded"


def test_latency_budget_and_age_budget_remain_separate(recording):
    report = qualify(recording, latency_budget_ns=1, max_support_age_ns=200)
    assert report["timing"]["commands"][0]["budget"] == "exceeded"
    assert report["commands"][0]["budget"] == "met"


@pytest.mark.parametrize("operator,indices,age,verdict", [
    ([0.5, 0.5], [0, 1], 100, "exceeded"),
    ([-0.5, 0.5], [0, 1], 100, "exceeded"),
    ([1e-100, 1.0], [0, 1], 100, "exceeded"),
    ([0.0, 1.0], [1], 10, "met"),
])
def test_conditioned_output_expands_every_nonzero_raw_support(
    recording, operator, indices, age, verdict,
):
    derived = conditioned(recording, operator)
    assert "exposure_start" not in derived.streams[-1].times_ns
    report = qualify(derived, [association(0, ("filtered", 0))], latency_budget_ns=5)
    row = report["commands"][0]
    assert report["timing"]["commands"][0]["latency_interval_ns"] == [3, 3]
    contributors = row["observations"][0]["contributors"]
    assert [item["stream_id"] for item in contributors] == ["wfs"] * len(indices)
    assert [item["sample_index"] for item in contributors] == indices
    assert row["oldest_support_age_interval_ns"] == [age, age]
    assert row["budget"] == verdict
    assert row["coverage"]["qualified_contributor_references"] == len(indices)


@pytest.mark.parametrize("effect_uncertainty,bound,interval,verdict", [
    (0, 100, [100, 100], "met"),
    (0, 99, [100, 100], "exceeded"),
    (1, 100, [99, 101], "ambiguous"),
    (1, 101, [99, 101], "met"),
    (1, 98, [99, 101], "exceeded"),
])
def test_exact_age_boundaries(recording, effect_uncertainty, bound, interval, verdict):
    recording.streams[1].uncertainty_ns = effect_uncertainty
    row = qualify(recording, max_support_age_ns=bound)["commands"][0]
    assert row["oldest_support_age_interval_ns"] == interval
    assert row["budget"] == verdict


def test_exposure_and_effect_uncertainties_use_conservative_integer_bounds(recording):
    recording.streams[0].uncertainty_ns = np.int64(1)
    recording.streams[1].uncertainty_ns = np.int64(2)
    row = qualify(recording, max_support_age_ns=np.int64(100))["commands"][0]
    assert row["oldest_support_age_interval_ns"] == [97, 103]
    assert first_contributor(row)["support_age_hull_ns"] == [87, 103]
    assert row["budget"] == "ambiguous"


def test_integer_epochs_above_float_and_int64_ranges(recording):
    origin = 2**90
    for stream in recording.streams:
        stream.times_ns = {
            stage: tuple(origin + int(time) for time in times)
            for stage, times in stream.times_ns.items()
        }
    row = qualify(recording)["commands"][0]
    assert row["oldest_support_age_interval_ns"] == [100, 100]
    assert first_contributor(row)["exposure_start_ns"] == origin
    assert all(type(value) is int for value in row["oldest_support_age_interval_ns"])


def test_every_command_and_unknown_dependency_retained_with_known_stale_witness(recording):
    recording.streams[0].times_ns["exposure_start"] = (0, None)
    report = qualify(recording)
    assert [row["applied_index"] for row in report["commands"]] == [0, 1, 2]
    assert [row["frame_id"] for row in report["commands"]] == [8, 3, 5]
    row = report["commands"][0]
    assert row["coverage"] == {
        "declared_observations": 2, "contributor_references": 2,
        "qualified_contributor_references": 1,
    }
    assert first_contributor(row)["budget"] == "exceeded"
    assert row["observations"][1]["contributors"][0]["reasons"] == ["missing_exposure_start"]
    assert row["oldest_support_age_interval_ns"] is None
    assert row["budget"] == "not_established"
    assert report["summary"]["contributor_reference_budget_counts"]["exceeded"] == 1
    assert report["summary"]["contributor_reference_budget_counts"]["not_established"] == 1
    for absent in report["commands"][1:]:
        assert absent["observations"] == []
        assert absent["budget"] == "not_established"
        assert "missing_association" in absent["reasons"]


@pytest.mark.parametrize("mutation,reason", [
    ("start", "missing_exposure_start"), ("end", "missing_exposure_end"),
    ("source_uncertainty", "unknown_exposure_uncertainty"),
    ("target_uncertainty", "missing_effective_time_or_uncertainty"),
    ("effect", "missing_effective_time_or_uncertainty"),
    ("source_clock", "missing_exposure_clock"), ("target_clock", "missing_command_clock"),
    ("cross_clock", "cross_clock_reference"),
    ("reversed_exposure", "exposure_event_sequence_violation"),
    ("reversed_command", "command_event_sequence_violation"),
])
def test_missing_or_contradictory_evidence_never_meets_age_bound(recording, mutation, reason):
    source, target = recording.streams
    if mutation in {"start", "end"}:
        source.times_ns[f"exposure_{mutation}"] = (None, 95)
    elif mutation == "source_uncertainty":
        source.uncertainty_ns = None
    elif mutation == "target_uncertainty":
        target.uncertainty_ns = None
    elif mutation == "effect":
        target.times_ns["effective"] = (None, 101, 102)
    elif mutation == "source_clock":
        source.clock_id, source.times_ns = None, {}
    elif mutation == "target_clock":
        target.clock_id, target.times_ns = None, {}
    elif mutation == "cross_clock":
        source.clock_id = "another-clock"
    elif mutation == "reversed_exposure":
        source.times_ns["exposure_start"] = (12, 90)
    else:
        target.times_ns["issued"] = (105, 100, 100)
    row = qualify(recording, max_support_age_ns=1000)["commands"][0]
    assert row["budget"] == "not_established"
    assert first_contributor(row)["budget"] == "not_established"
    assert reason in first_contributor(row)["reasons"]


@pytest.mark.parametrize("effect,uncertainty,reason", [
    (5, 0, "exposure_after_effect"),
    (10, 1, "ambiguous_exposure_before_effect"),
])
def test_effect_before_or_overlapping_exposure_end_is_not_fresh(
    recording, effect, uncertainty, reason,
):
    recording.streams[1].times_ns["effective"] = (effect, 101, 102)
    recording.streams[1].uncertainty_ns = uncertainty
    row = qualify(recording, [association(0, ("wfs", 0))], max_support_age_ns=1000)["commands"][0]
    contributor = first_contributor(row)
    assert contributor["oldest_support_age_interval_ns"] is not None
    assert contributor["budget"] == row["budget"] == "not_established"
    assert reason in contributor["reasons"]


def test_equal_exposure_end_and_effect_are_consistent_when_available(recording):
    source, target = recording.streams
    source.times_ns["available"] = (10, 96)
    target.times_ns["effective"] = (10, 101, 102)
    row = qualify(recording, [association(0, ("wfs", 0))], max_support_age_ns=10)["commands"][0]
    assert first_contributor(row)["support_age_hull_ns"] == [0, 10]
    assert row["budget"] == "met"


def test_missing_availability_keeps_descriptive_age_but_withholds_command(recording):
    recording.streams[0].times_ns["available"] = (None, 96)
    row = qualify(recording)["commands"][0]
    assert first_contributor(row)["budget"] == "exceeded"
    assert row["oldest_support_age_interval_ns"] == [100, 100]
    assert row["budget"] == "not_established"
    assert row["reasons"] == ["command_timing_not_consistent"]


def test_multiple_streams_and_repeated_support_keep_explicit_reference_counts(recording):
    camera = Stream(
        "camera", "wfs_pixels", np.ones((1, 2, 2)), "electrons", clock_id="clock",
        uncertainty_ns=0, times_ns={"exposure_start": (80,), "exposure_end": (85,),
                                    "available": (86,)},
    )
    recording.streams = (*recording.streams, camera)
    derived = conditioned(recording, [0.5, 0.5])
    row = qualify(derived, [association(0, ("wfs", 0), ("filtered", 0), ("camera", 0))])["commands"][0]
    assert [obs["stream_id"] for obs in row["observations"]] == ["camera", "filtered", "wfs"]
    assert row["coverage"]["contributor_references"] == 4
    assert row["oldest_support_age_interval_ns"] == [100, 100]
    assert row["budget"] == "exceeded"


def test_maximum_age_bounds_can_come_from_different_contributors(recording):
    recording.streams[0].uncertainty_ns = 1
    camera = Stream(
        "camera", "wfs_pixels", np.ones((1, 2, 2)), "electrons", clock_id="clock",
        uncertainty_ns=5, times_ns={"exposure_start": (1,), "exposure_end": (20,),
                                    "available": (90,)},
    )
    recording.streams = (*recording.streams, camera)
    row = qualify(
        recording, [association(0, ("wfs", 0), ("camera", 0))], max_support_age_ns=100,
    )["commands"][0]
    intervals = [first["contributors"][0]["oldest_support_age_interval_ns"]
                 for first in row["observations"]]
    assert intervals == [[94, 104], [99, 101]]
    assert row["oldest_support_age_interval_ns"] == [99, 104]
    assert row["budget"] == "ambiguous"


@pytest.mark.parametrize("mutation", ["support", "exposure", "source_values"])
def test_changed_conditioned_lineage_is_rejected_before_report(recording, mutation):
    derived = conditioned(recording, [0.5, 0.5])
    descriptor = derived.streams[-1].metadata["conditioning"]
    if mutation == "support":
        descriptor["support_indices"][0] = [1]
    elif mutation == "exposure":
        descriptor["contributing_exposures"][0][0]["exposure_start_ns"] = 90
    else:
        derived.streams[0].values[0, 0] += 1
    with pytest.raises(ValueError):
        qualify(derived, [association(0, ("filtered", 0))])


@pytest.mark.parametrize("bound", [None, True, np.bool_(False), -1, 1.5, "20", float("nan")])
def test_invalid_age_bound_rejected(recording, bound):
    with pytest.raises(ValueError, match="Maximum support age"):
        qualify(recording, max_support_age_ns=bound)


def test_zero_age_bound_is_valid_and_does_not_mean_unlimited(recording):
    assert qualify(recording, max_support_age_ns=0)["commands"][0]["budget"] == "exceeded"


@pytest.mark.parametrize("kind", ["command_requested", "command_recorded", "other"])
def test_only_explicitly_applied_commands_are_admitted(recording, kind):
    recording.streams[1].kind = kind
    with pytest.raises(ValueError, match="command_applied"):
        qualify(recording)


@pytest.mark.parametrize("associations", [
    [], [association(), association()], [association(0, ("wfs", -1))],
    [association(0, ("wfs", 0), ("wfs", 0))],
])
def test_association_contract_is_the_existing_timing_contract(recording, associations):
    if not associations:
        report = qualify(recording, associations)
        assert report["summary"]["budget_counts"]["not_established"] == 3
        return
    with pytest.raises(ValueError):
        qualify(recording, associations)


def test_qualification_does_not_promote_calibration_or_measurement_values(recording):
    assert recording.calibrations == ()
    recording.streams[0].values[0, 0] = np.nan
    report = qualify(recording, max_support_age_ns=200)
    assert report["commands"][0]["budget"] == "met"
    assert not any("accuracy" in key or "calibration" in key for key in report["commands"][0])
    assert any("measurement accuracy" in text for text in report["limitations"])


def test_recording_mutation_during_qualification_rejected(recording, monkeypatch):
    original = freshness._contributor

    def mutate(*args, **kwargs):
        value = original(*args, **kwargs)
        recording.streams[0].values[0, 0] += 1
        return value

    monkeypatch.setattr(freshness, "_contributor", mutate)
    with pytest.raises(ValueError, match="Recording changed during freshness"):
        qualify(recording)


@pytest.mark.parametrize("bound", [100, 2000])
def test_oversized_report_rejects_without_truncation(recording, monkeypatch, bound):
    monkeypatch.setattr(freshness, "MAX_JSON_BYTES", bound)
    with pytest.raises(ValueError, match="Freshness report exceeds"):
        qualify(recording)


def test_final_complete_json_bound_is_enforced(recording, monkeypatch):
    exact_size = len(canonical_json(qualify(recording)))
    monkeypatch.setattr(freshness, "MAX_JSON_BYTES", exact_size - 1)
    with pytest.raises(ValueError, match="Freshness report exceeds"):
        qualify(recording)
    monkeypatch.setattr(freshness, "MAX_JSON_BYTES", exact_size)
    assert len(canonical_json(qualify(recording))) == exact_size


def test_deterministic_hashes_bind_normalized_configuration_and_source(recording):
    before = recording.digest()
    report = qualify(recording)
    assert canonical_json(report) == canonical_json(qualify(copy.deepcopy(recording)))
    reversed_refs = association()
    reversed_refs["observations"].reverse()
    assert canonical_json(report) == canonical_json(qualify(recording, [reversed_refs]))
    expected = hashlib.sha256(canonical_json({
        "recording_sha256": before, "configuration": report["configuration"],
    })).hexdigest()
    assert report["configuration_sha256"] == expected
    assert qualify(recording, max_support_age_ns=21)["configuration_sha256"] != expected
    assert recording.digest() == before
