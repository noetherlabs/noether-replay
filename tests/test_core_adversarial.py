"""Adversarial regressions for telemetry evidence and causal replay boundaries."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from noether_replay.model import SCHEMA, Recording, Stream, load_recording
from noether_replay.replay import run_shadow
from noether_replay.report import build_report


def observation(**overrides: object) -> Stream:
    fields = {
        "id": "wfs",
        "kind": "wfs_measurements",
        "values": np.array([[1.0], [2.0]]),
        "units": "rad",
        "frame_ids": (0, 1),
        "clock_id": "instrument",
        "times_ns": {
            "exposure_start": (0, 10),
            "exposure_end": (1, 11),
            "available": (2, 12),
        },
        "uncertainty_ns": 0,
    }
    fields.update(overrides)
    return Stream(**fields)


@pytest.mark.parametrize(
    "times",
    [
        {"exposure_start": (0, 10), "exposure_end": (3, 13), "available": (2, 12)},
        {"exposure_start": (2, 12), "exposure_end": (1, 11), "available": (3, 13)},
    ],
    ids=["available-before-exposure-completes", "negative-exposure-duration"],
)
def test_shadow_rejects_known_acquisition_order_contradictions(times: dict) -> None:
    recording = Recording("contradictory-acquisition", (observation(times_ns=times),))
    calls = []

    def policy(value: np.ndarray) -> np.ndarray:
        calls.append(value)
        return -value

    with pytest.raises(ValueError):
        run_shadow(recording, "wfs", policy, policy_id="test", policy_version="1")
    assert calls == [], "Invalid causal input must be rejected before invoking a stateful policy"


def test_all_unknown_timestamp_entries_are_reported_as_missing() -> None:
    stream = Stream(
        "temperature", "housekeeping", np.array([[280.0]]), units="K", frame_ids=(0,),
        clock_id="instrument", times_ns={"recorded": (None,)}, uncertainty_ns=0,
    )
    report = build_report(Recording("untimed", (stream,)))
    findings = {finding["code"]: finding for finding in report["findings"]}
    assert report["event_count"] == 0
    assert findings["missing_event_times"]["count"] == 1
    assert report["status"] == "evidence_gaps"


def test_report_provenance_is_a_snapshot_bound_to_its_digest() -> None:
    recording = Recording(
        "source-provenance", (observation(),),
        provenance={"source": {"sha256": "a" * 64}},
    )
    report = build_report(recording)
    snapshot = copy.deepcopy(report)
    recording.provenance["source"]["sha256"] = "b" * 64
    assert report == snapshot, "An already-issued report must not silently acquire new source provenance"
    assert report["recording_sha256"] != recording.digest()


@pytest.mark.parametrize("field", ["times_ns", "frame_ids", "uncertainty_ns"])
def test_numpy_integer_timing_and_identity_survive_manifest(field: str) -> None:
    stream = observation()
    if field == "times_ns":
        stream.times_ns = {key: tuple(np.int64(x) for x in times) for key, times in stream.times_ns.items()}
    elif field == "frame_ids":
        stream.frame_ids = tuple(np.int64(x) for x in stream.frame_ids)
    else:
        stream.uncertainty_ns = np.int64(0)
    stream.validate()
    assert Recording("r", (stream,)).digest() == Recording("r", (observation(),)).digest()


@pytest.mark.parametrize(
    ("dtype", "values"),
    [("int64", [[1.9], [2.1]]), ("bool", [[0], [2]]), ("float64", [["1.0"], ["2.0"]])],
    ids=["fractional-counts-truncated", "nonbinary-data-coerced-to-boolean", "numeric-strings-promoted"],
)
def test_fixture_loader_rejects_lossful_or_nonnumeric_value_coercion(tmp_path, dtype, values) -> None:
    payload = {
        "schema": SCHEMA, "id": "fixture", "streams": [{
            "id": "wfs", "kind": "wfs_measurements", "values": values, "dtype": dtype,
            "units": "count", "frame_ids": [0, 1], "clock_id": None, "times_ns": {},
            "uncertainty_ns": None, "metadata": {},
        }], "calibrations": [], "metadata": {}, "provenance": {},
    }
    path = tmp_path / "recording.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_recording(path)


def test_wide_numpy_clock_uncertainty_cannot_overflow_overlap_guard() -> None:
    stream = observation(uncertainty_ns=np.int64(2**62))
    recording = Recording("ambiguous-clock", (stream,))
    calls = []

    def policy(value: np.ndarray) -> np.ndarray:
        calls.append(value)
        return -value

    with pytest.raises(ValueError):
        run_shadow(recording, "wfs", policy, policy_id="test", policy_version="1")
    assert calls == []


def test_partially_missing_timestamps_remain_evidence_gaps() -> None:
    stream = Stream(
        "temperature", "housekeeping", np.array([[280.0], [281.0]]), units="K",
        frame_ids=(0, 1), clock_id="instrument", times_ns={"recorded": (1, None)},
        uncertainty_ns=0,
    )
    report = build_report(Recording("partly-untimed", (stream,)))
    counts = {item["code"]: item["count"] for item in report["findings"]}
    assert report["event_count"] == 1
    assert counts["missing_event_times"] == 1
    assert report["status"] == "evidence_gaps"


@pytest.fixture
def json_fixture_payload() -> dict:
    return {
        "schema": SCHEMA,
        "id": "fixture",
        "streams": [{
            "id": "wfs", "kind": "wfs_measurements", "values": [[1.0]], "dtype": "float64",
            "units": "rad", "frame_ids": [0], "clock_id": "instrument",
            "times_ns": {"available": [1]}, "uncertainty_ns": 0, "metadata": {},
        }],
        "calibrations": [], "metadata": {}, "provenance": {},
    }


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ('"id": "fixture"', '"id": "contradictory-source", "id": "fixture"'),
        ('"metadata": {}', '"metadata": {"operator": "A", "operator": "B"}'),
        ('"available": [1]', '"available": [0], "available": [1]'),
    ],
    ids=["duplicate-recording-id", "duplicate-nested-provenance", "duplicate-event-times"],
)
def test_fixture_rejects_duplicate_json_keys(tmp_path, json_fixture_payload, original, replacement):
    text = json.dumps(json_fixture_payload)
    assert original in text
    path = tmp_path / "ambiguous.json"
    path.write_text(text.replace(original, replacement, 1), encoding="utf-8")
    with pytest.raises(ValueError):
        load_recording(path)


@pytest.mark.parametrize(
    ("location", "key", "replacement"),
    [
        ("recording", "streams", {}),
        ("recording", "calibrations", {}),
        ("recording", "metadata", None),
        ("recording", "provenance", []),
        ("stream", "metadata", "not-an-object"),
        ("stream", "times_ns", []),
        ("stream", "frame_ids", {"0": 0}),
        ("stream", "times_ns", {"available": "1"}),
    ],
)
def test_fixture_rejects_malformed_container_types(
    tmp_path, json_fixture_payload, location, key, replacement,
):
    target = json_fixture_payload if location == "recording" else json_fixture_payload["streams"][0]
    target[key] = replacement
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(json_fixture_payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_recording(path)
