"""Analytical public-API checks; no instrument synchronization is asserted."""

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
    return Recording(
        "two-clock-analytical-fixture",
        (
            Stream(
                "wfs", "wfs_measurements", np.array([[1.0]]), "rad", (7,), "camera",
                {"available": (1000,)}, 3,
            ),
            Stream(
                "dm", "command_applied", np.array([[2.0]]), "m", (19,), "actuator",
                {"effective": (5100,)}, 2,
            ),
        ),
        provenance={"source": "analytical fixture", "clock_instances": "fixture-boot-1"},
    )


def receipt(recording, **overrides):
    return {
        "schema": "noether-replay-clock-conversion/1",
        "recording_sha256": recording.digest(),
        "source_clock_id": "camera",
        "target_clock_id": "actuator",
        "source_valid_from_ns": 900,
        "source_valid_until_ns": 1100,
        "offset_ns": 4000,
        "max_mapping_error_ns": 4,
        "bound_scope": "uniform_over_source_validity_interval",
        "evidence_sha256": {"analytical-contract": "a" * 64},
        "provenance": {
            "source": "synthetic coordinate construction; no instrument evidence",
            "method": "asserted bounded translation",
            "clock_instances": ["fixture-camera-boot-1", "fixture-actuator-boot-1"],
            "bound_rationale": "uniform total residual bound, including drift",
        },
        **overrides,
    }


def association(index=0, *streams):
    return {
        "applied_index": index,
        "observations": [
            {"stream_id": stream, "sample_index": 0} for stream in (streams or ("wfs",))
        ],
    }


def qualify(recording, conversion, **kwargs):
    return qualify_command_timing(
        recording, applied_stream_id="dm",
        associations=kwargs.pop("associations", [association()]),
        clock_conversion=conversion, **kwargs,
    )


@pytest.mark.parametrize("budget,verdict", [(110, "met"), (100, "ambiguous"), (90, "exceeded")])
def test_external_bound_propagates_to_exact_latency_and_budget(recording, budget, verdict):
    report = qualify(recording, receipt(recording), latency_budget_ns=budget)
    row = report["commands"][0]
    observation = row["observations"][0]
    assert report["schema"] == "noether-replay-command-timing/2"
    assert report["mode"] == "observational"
    assert observation["clock_id"] == "camera"
    assert observation["available_ns"] == 1000
    assert observation["uncertainty_ns"] == 3
    assert observation["availability_interval_ns"] == [997, 1003]
    assert observation["comparison_clock_id"] == row["comparison_clock_id"] == "actuator"
    assert observation["comparison_availability_interval_ns"] == [4993, 5007]
    assert row["latency_interval_ns"] == [91, 109]
    assert row["causal_order"] == "consistent"
    assert row["budget"] == verdict
    assert report["summary"]["clock_conversion"] == {
        "reference_count": 1, "converted_interval_count": 1, "withheld_interval_count": 0,
    }
    assert qualify(recording, None)["commands"][0]["reasons"] == ["cross_clock_reference"]


@pytest.mark.parametrize("offset", [4000, -4000])
def test_large_epochs_and_signed_offsets_remain_exact(recording, offset):
    epoch = 2**100 + 17
    recording.streams[0].times_ns["available"] = (epoch + 1000,)
    recording.streams[1].times_ns["effective"] = (epoch + 1100 + offset,)
    mapping = receipt(
        recording, source_valid_from_ns=epoch + 900, source_valid_until_ns=epoch + 1100,
        offset_ns=offset,
    )
    row = qualify(recording, mapping)["commands"][0]
    assert row["latest_availability_interval_ns"] == [epoch + 993 + offset, epoch + 1007 + offset]
    assert row["latency_interval_ns"] == [91, 109]
    assert all(type(value) is int for value in row["latest_availability_interval_ns"])


@pytest.mark.parametrize("start,end,valid", [(997, 1004, True), (998, 1004, False), (997, 1003, False)])
def test_entire_uncertainty_support_must_fit_half_open_validity(recording, start, end, valid):
    report = qualify(recording, receipt(recording, source_valid_from_ns=start, source_valid_until_ns=end))
    row = report["commands"][0]
    assert report["summary"]["command_count"] == 1
    assert row["causal_order"] == ("consistent" if valid else "not_established")
    assert row["reasons"] == ([] if valid else ["clock_conversion_outside_validity"])
    assert report["summary"]["clock_conversion"]["withheld_interval_count"] == int(not valid)


def test_all_dependencies_share_comparison_coordinates_without_hiding_unmapped_input(recording):
    native = Stream(
        "native", "wfs_pixels", np.zeros((1, 1, 1)), clock_id="actuator",
        times_ns={"available": (5010,)}, uncertainty_ns=20,
    )
    foreign = Stream(
        "unmapped", "wfs_measurements", np.zeros((1, 1)), clock_id="third-clock",
        times_ns={"available": (1,)}, uncertainty_ns=0,
    )
    recording.streams += (native, foreign)
    mapping = receipt(recording)
    report = qualify(recording, mapping, associations=[association(0, "wfs", "native")])
    # Converted lower bound wins; native upper bound wins. Neither nominal
    # timestamp alone nor one arbitrarily selected input establishes readiness.
    assert report["commands"][0]["latest_availability_interval_ns"] == [4993, 5030]
    assert report["commands"][0]["latency_interval_ns"] == [68, 109]
    row = qualify(
        recording, mapping, associations=[association(0, "wfs", "native", "unmapped")],
    )["commands"][0]
    assert len(row["observations"]) == 3
    assert row["reasons"] == ["cross_clock_reference"]
    assert row["causal_order"] == "not_established"
    assert row["latency_interval_ns"] is None
    unknown = next(item for item in row["observations"] if item["stream_id"] == "unmapped")
    assert unknown["comparison_availability_interval_ns"] is None
    assert unknown["clock_conversion_sha256"] is None


@pytest.mark.parametrize("stage", ["completed", "issued"])
@pytest.mark.parametrize(
    "time,reason", [(4980, "input_unavailable_at_"), (5000, "ambiguous_input_availability_at_")],
)
def test_earlier_command_stages_use_the_same_converted_interval(recording, stage, time, reason):
    recording.streams[1].times_ns[stage] = (time,)
    report = qualify(recording, receipt(recording), latency_budget_ns=110)
    row = report["commands"][0]
    assert row["latency_interval_ns"] == [91, 109]
    assert row["reasons"] == [reason + stage]
    assert row["causal_order"] == row["budget"] == "not_established"
    # The count describes interval conversion, not a qualified command.
    assert report["summary"]["clock_conversion"]["converted_interval_count"] == 1


def test_native_stage_contradiction_still_withholds_conversion(recording):
    recording.streams[0].times_ns["exposure_end"] = (1001,)
    row = qualify(recording, receipt(recording))["commands"][0]
    assert row["reasons"] == ["event_sequence_violation"]
    assert row["latency_interval_ns"] is None


@pytest.mark.parametrize("missing,reason", [("time", "missing_availability"), ("bound", "unknown_observation_uncertainty")])
def test_missing_native_evidence_is_counted_without_inventing_zero(recording, missing, reason):
    if missing == "time":
        recording.streams[0].times_ns["available"] = (None,)
    else:
        recording.streams[0].uncertainty_ns = None
    report = qualify(recording, receipt(recording))
    row = report["commands"][0]
    assert row["reasons"] == [reason]
    assert row["observations"][0]["comparison_availability_interval_ns"] is None
    assert report["summary"]["clock_conversion"] == {
        "reference_count": 1, "converted_interval_count": 0, "withheld_interval_count": 1,
    }


def test_expired_and_unassociated_commands_remain_in_the_cohort(recording):
    recording.streams[0].values = np.zeros((2, 1))
    recording.streams[0].frame_ids = (7, 8)
    recording.streams[0].times_ns["available"] = (1000, 1200)
    recording.streams[1].values = np.zeros((3, 1))
    recording.streams[1].frame_ids = (19, 20, 21)
    recording.streams[1].times_ns["effective"] = (5100, 5300, 5400)
    refs = [association(0), association(1)]
    refs[1]["observations"][0]["sample_index"] = 1
    report = qualify(recording, receipt(recording), associations=refs)
    assert [row["reasons"] for row in report["commands"]] == [
        [], ["clock_conversion_outside_validity"], ["missing_association"],
    ]
    assert report["summary"]["command_count"] == 3
    assert report["summary"]["causal_order_counts"]["not_established"] == 2
    assert "pass" not in report["summary"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "unknown"), ("recording_sha256", "f" * 64),
        ("source_clock_id", "actuator"), ("source_clock_id", ""),
        ("target_clock_id", "camera"), ("target_clock_id", None),
        ("source_valid_from_ns", True), ("source_valid_until_ns", 900),
        ("offset_ns", 1.5), ("offset_ns", np.bool_(False)),
        ("max_mapping_error_ns", -1), ("max_mapping_error_ns", False),
        ("bound_scope", "at_calibration_time_only"), ("evidence_sha256", {}),
        ("evidence_sha256", {"test": "A" * 64}), ("evidence_sha256", {"": "a" * 64}),
        ("evidence_sha256", {"test": "a" * 64 + "\n"}),
        ("provenance", {}), ("provenance", []), ("provenance", {"x": float("nan")}),
        ("provenance", {"x": object()}), ("provenance", {1: "invalid key"}),
        ("provenance", {"x": "\ud800"}),
    ],
)
def test_malformed_receipt_is_rejected_before_any_partial_report(recording, field, value):
    with pytest.raises(ValueError):
        qualify(recording, receipt(recording, **{field: value}))


@pytest.mark.parametrize("mutation", ["missing", "extra", "list", "unused", "no_command_clock"])
def test_strict_fields_and_receipt_use(recording, mutation):
    if mutation == "no_command_clock":
        recording.streams[1].clock_id = None
        recording.streams[1].times_ns = {}
    mapping = receipt(recording)
    if mutation == "missing":
        del mapping["max_mapping_error_ns"]
    elif mutation == "extra":
        mapping["inverse_allowed"] = True
    elif mutation == "list":
        mapping = [mapping]
    elif mutation == "unused":
        mapping["source_clock_id"] = "unreferenced-clock"
    with pytest.raises(ValueError):
        qualify(recording, mapping)


def test_content_binding_rejects_separately_identified_reset_even_with_reused_timestamps(recording):
    mapping = receipt(recording)
    recording.provenance["clock_instances"] = "fixture-boot-2"
    with pytest.raises(ValueError, match="does not bind"):
        qualify(recording, mapping)


def test_receipt_is_detached_normalized_and_bound_to_configuration(recording):
    mapping = receipt(recording, offset_ns=np.int64(4000), max_mapping_error_ns=np.int64(4))
    mapping["provenance"]["extra"] = ("µ", "\n", np.int64(12))
    report = qualify(recording, mapping)
    retained = report["configuration"]["clock_conversion"]
    assert type(retained["offset_ns"]) is int
    assert retained["provenance"]["extra"] == ["µ", "\n", 12]
    digest = hashlib.sha256(canonical_json(retained)).hexdigest()
    assert report["configuration"]["clock_conversion_sha256"] == digest
    assert report["commands"][0]["observations"][0]["clock_conversion_sha256"] == digest
    assert report["configuration_sha256"] == hashlib.sha256(canonical_json({
        "recording_sha256": recording.digest(), "configuration": report["configuration"],
    })).hexdigest()
    before = canonical_json(report)
    mapping["offset_ns"] = -123
    mapping["provenance"]["clock_instances"].append("other")
    mapping["evidence_sha256"]["extra"] = "b" * 64
    assert canonical_json(report) == before


def test_raw_bytes_calibration_metadata_and_native_clocks_are_preserved(recording):
    recording.streams[0].values = np.array([[0x7FF8000000000123]], dtype=np.uint64).view(np.float64)
    recording.calibrations = ({
        "id": "native-calibration", "clock_id": "camera", "start_ns": 900, "end_ns": 1100,
        "stream_ids": ["wfs"], "products": {"response": "c" * 64},
    },)
    recording.metadata["opaque"] = {"clock_conversion": "must not be executed"}
    before = canonical_json(recording.manifest())
    raw = recording.streams[0].values.tobytes()
    report = qualify(recording, receipt(recording))
    assert canonical_json(recording.manifest()) == before
    assert recording.streams[0].values.tobytes() == raw
    assert report["recording_sha256"] == hashlib.sha256(before).hexdigest()
    assert qualify(recording, None)["commands"][0]["reasons"] == ["cross_clock_reference"]


def test_conditioned_lineage_is_preserved_and_tampering_is_still_rejected(recording):
    derived = condition_observations(
        recording, stream_id="wfs", output_id="filtered", operator=[[1.0]],
        covariance=[[1.0]], covariance_provenance={"source": "analytical fixture"},
        available_ns=[1010],
    )
    before = derived.digest()
    report = qualify(derived, receipt(derived), associations=[association(0, "filtered")])
    assert report["commands"][0]["latency_interval_ns"] == [81, 99]
    assert derived.digest() == before
    mapping = receipt(derived)
    derived.streams[-1].values[0, 0] += 1
    with pytest.raises(ValueError):
        qualify(derived, mapping)


@pytest.mark.parametrize("kind", ["string", "expansion", "cycle", "depth", "negative_integer"])
def test_excessive_metadata_rejects_before_canonical_expansion(recording, monkeypatch, kind):
    import noether_replay._clock as clock
    import noether_replay.timing as timing

    mapping = receipt(recording)
    if kind == "string":
        mapping["provenance"]["extra"] = "\x00" * 2000
    elif kind == "expansion":
        value = [0]
        for _ in range(20):
            value = [value, value]
        mapping["provenance"]["extra"] = value
    elif kind == "cycle":
        mapping["provenance"]["extra"] = mapping
    elif kind == "negative_integer":
        mapping["provenance"]["extra"] = -(1 << 20000)
    else:
        value = []
        for _ in range(65):
            value = [value]
        mapping["provenance"]["extra"] = value
    monkeypatch.setattr(timing, "MAX_JSON_BYTES", 4096)

    def unexpected_expansion(_):
        pytest.fail("Oversized, cyclic or deeply nested receipt reached canonical normalization")

    monkeypatch.setattr(clock, "canonical_json", unexpected_expansion)
    with pytest.raises(ValueError):
        qualify(recording, mapping)


def test_combined_configuration_is_bounded_before_canonical_expansion(recording, monkeypatch):
    import noether_replay._clock as clock
    import noether_replay.timing as timing

    mapping = receipt(recording)
    payload = {
        "recording_sha256": recording.digest(),
        "configuration": {
            "applied_stream_id": "dm", "latency_budget_ns": None,
            "associations": [association()], "clock_conversion": mapping,
            "clock_conversion_sha256": hashlib.sha256(canonical_json(mapping)).hexdigest(),
        },
    }
    limit = len(canonical_json(payload)) - 1
    assert len(canonical_json(mapping)) < limit
    assert len(canonical_json([association()])) < limit
    monkeypatch.setattr(timing, "MAX_JSON_BYTES", limit)

    def guarded_encoder(value):
        if isinstance(value, dict) and "configuration" in value:
            pytest.fail("Oversized combined configuration reached canonical expansion")
        return canonical_json(value)

    monkeypatch.setattr(clock, "canonical_json", guarded_encoder)
    monkeypatch.setattr(timing, "canonical_json", guarded_encoder)
    with pytest.raises(ValueError, match="metadata bound"):
        qualify(recording, mapping)


def test_complete_report_still_obeys_aggregate_size_bound(recording, monkeypatch):
    import noether_replay.timing as timing

    mapping = receipt(recording)
    report = qualify(recording, mapping)
    monkeypatch.setattr(timing, "MAX_JSON_BYTES", len(canonical_json(report)) - 1)
    with pytest.raises(ValueError, match="Timing report exceeds"):
        qualify(recording, mapping)


def test_evidence_references_are_not_physical_authentication(recording):
    mapping = receipt(recording)
    mapping["provenance"]["source"] = "https://invalid.example/never-resolved"
    report = qualify(recording, mapping)
    assert report["mode"] == "observational"
    assert any("do not authenticate" in line for line in report["limitations"])
    assert any("calibration epochs, probe phase and shadow execution are unchanged" in line
               for line in report["limitations"])


def test_explicit_none_is_byte_identical_to_default(recording):
    args = {"applied_stream_id": "dm", "associations": [association()]}
    assert canonical_json(qualify_command_timing(recording, **args)) == canonical_json(
        qualify_command_timing(recording, clock_conversion=None, **args),
    )
    report = qualify_command_timing(recording, **args)
    assert report["schema"] == "noether-replay-command-timing/1"
    assert "clock_conversion" not in report["configuration"]
    assert "clock_conversion" not in report["summary"]
    assert "comparison_clock_id" not in report["commands"][0]


def test_receipt_does_not_disable_source_mutation_guard(recording, monkeypatch):
    mapping = receipt(recording)
    digest = recording.digest
    calls = 0

    def changed_digest():
        nonlocal calls
        calls += 1
        if calls == 2:
            recording.metadata["changed"] = True
        return digest()

    monkeypatch.setattr(recording, "digest", changed_digest)
    with pytest.raises(ValueError, match="changed during"):
        qualify(recording, copy.deepcopy(mapping))
