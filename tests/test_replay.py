import copy
import hashlib
import json

import numpy as np
import pytest

from noether_replay import Recording, Stream, load_recording
from noether_replay.cli import main
from noether_replay.model import array_digest
from noether_replay.replay import calibration_assignment, events, run_shadow, timeline_digest
from noether_replay.report import build_report, markdown_report


@pytest.fixture
def recording():
    stream = Stream(
        "wfs",
        "wfs_measurements",
        np.array([[1.0], [2.0], [3.0]]),
        "m",
        (10, 11, 12),
        "instrument",
        {
            "exposure_start": (0, 100, 200),
            "exposure_end": (10, 110, 210),
            "available": (20, 120, 220),
        },
        0,
    )
    epoch = {
        "id": "cal-1",
        "clock_id": "instrument",
        "start_ns": -1,
        "end_ns": 300,
        "stream_ids": ["wfs"],
        "products": {"response": hashlib.sha256(b"synthetic response").hexdigest()},
    }
    return Recording("test", (stream,), (epoch,), provenance={"kind": "synthetic"})


def test_exact_events_and_content_binding(recording):
    assert [e["time_ns"] for e in events(recording)] == [0, 10, 20, 100, 110, 120, 200, 210, 220]
    before = timeline_digest(recording)
    assert timeline_digest(copy.deepcopy(recording)) == before
    recording.streams[0].values[0, 0] += 1
    assert timeline_digest(recording) != before


def test_array_hash_binds_dtype_shape_and_noncontiguous_bytes():
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert array_digest(values[:, ::2]) == array_digest(values[:, ::2].copy())
    assert array_digest(values) != array_digest(values.astype(np.float32))
    assert array_digest(values) != array_digest(values.reshape(4, 3))
    assert array_digest(np.array([0.0])) != array_digest(np.array([-0.0]))


def test_calibration_boundary_and_uncertainty(recording):
    stream = recording.streams[0]
    assert calibration_assignment(recording, stream, 2) == "cal-1"
    recording.calibrations[0]["end_ns"] = 210
    assert calibration_assignment(recording, stream, 2) is None
    recording.calibrations[0]["end_ns"] = 211
    stream.uncertainty_ns = 1
    assert calibration_assignment(recording, stream, 2) is None


def test_partial_epoch_overlap_withholds_assignment(recording):
    other = copy.deepcopy(recording.calibrations[0])
    other.update(id="cal-2", start_ns=5, end_ns=9)
    recording.calibrations += (other,)
    assert calibration_assignment(recording, recording.streams[0], 0) is None


def test_missing_timing_never_synthesized_from_frequency(recording):
    stream = recording.streams[0]
    stream.times_ns = {}
    stream.metadata = {"frame_rate_hz": 1000}
    assert events(recording) == []
    report = build_report(recording)
    assert report["event_count"] == 0
    assert {f["code"] for f in report["findings"]} >= {
        "missing_availability",
        "unqualified_calibration_interval",
    }
    with pytest.raises(ValueError, match="availability"):
        run_shadow(recording, "wfs", lambda x: -x, policy_id="test", policy_version="1")


def test_mixed_clocks_and_ambiguous_order(recording):
    stream = recording.streams[0]
    stream.uncertainty_ns = 20
    recording.streams += (
        Stream(
            "dm",
            "command_recorded",
            np.zeros((1, 1)),
            "m",
            (0,),
            "other-clock",
            {"recorded": (5,)},
            0,
        ),
    )
    codes = {f["code"] for f in build_report(recording)["findings"]}
    assert {"multiple_clock_domains", "ambiguous_event_order", "unknown_command_semantics"} <= codes


def test_frame_gaps_are_not_silently_dropped(recording):
    recording.streams[0].frame_ids = (10, 13, 13)
    result = build_report(recording)
    counts = {f["code"]: f["count"] for f in result["findings"]}
    assert counts["frame_id_gaps"] == 2
    assert counts["duplicate_frame_ids"] == 1


def test_nonfinite_observations_retained_but_shadow_withheld(recording):
    recording.streams[0].values[0] = np.nan
    assert build_report(recording)["streams"][0]["nonfinite_values"] == 1
    assert np.isnan(recording.streams[0].values[0, 0])
    with pytest.raises(ValueError, match="missing-observation"):
        run_shadow(recording, "wfs", lambda x: -x, policy_id="test", policy_version="1")


def test_shadow_actions_have_no_attributed_outcomes(recording):
    before = recording.digest()
    result = run_shadow(recording, "wfs", lambda x: -x, policy_id="test", policy_version="1")
    assert result["samples"][0]["hypothetical_command"] == [-1.0]
    assert result["outcome_assessed"] is False
    assert recording.digest() == before
    assert "wfe" not in result
    with pytest.raises(ValueError, match="counterfactual"):
        build_report(recording, mode="counterfactual")


def test_shadow_policy_receives_copy(recording):
    def policy(observation):
        observation[:] = 42
        return observation

    run_shadow(recording, "wfs", policy, policy_id="test", policy_version="1")
    np.testing.assert_array_equal(recording.streams[0].values[:, 0], [1, 2, 3])


@pytest.mark.parametrize("times", [(20, 20, 220), (20, None, 220)])
def test_shadow_rejects_unknown_causal_order(recording, times):
    recording.streams[0].times_ns["available"] = times
    with pytest.raises(ValueError):
        run_shadow(recording, "wfs", lambda x: x, policy_id="test", policy_version="1")


def test_json_fixture_and_cli_never_overwrite(tmp_path):
    from pathlib import Path

    source = Path(__file__).parents[1] / "examples" / "synthetic.json"
    recording = load_recording(source)
    assert recording.provenance["kind"] == "synthetic"
    output = tmp_path / "report.json"
    assert main([str(source), "--output", str(output)]) == 0
    original = output.read_bytes()
    assert json.loads(original)["mode"] == "observational"
    with pytest.raises(SystemExit) as exc:
        main([str(source), "--output", str(output)])
    assert exc.value.code == 2
    assert output.read_bytes() == original


def test_markdown_has_explicit_limits(recording):
    text = markdown_report(build_report(recording))
    assert "not instrument qualification" in text
    assert "cannot evaluate a different controller" in text


def test_no_object_arrays_or_float_nanosecond_times(recording):
    stream = recording.streams[0]
    stream.times_ns["available"] = (20.0, 120, 220)
    with pytest.raises(ValueError, match="integer"):
        recording.validate()
    stream.times_ns = {}
    stream.values = np.array([object()], dtype=object)
    with pytest.raises(ValueError, match="numeric"):
        stream.validate()


def test_report_publication_failure_leaves_no_partial_artifact(tmp_path, monkeypatch):
    from pathlib import Path

    import noether_replay.cli as cli

    source = Path(__file__).parents[1] / "examples" / "synthetic.json"
    output = tmp_path / "report.json"

    def fail_publication(*args):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(cli.os, "link", fail_publication)
    with pytest.raises(SystemExit) as exc:
        cli.main([str(source), "--output", str(output)])
    assert exc.value.code == 2
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_policy_identity_is_required_and_retained(recording):
    result = run_shadow(
        recording, "wfs", lambda x: x, policy_id="baseline", policy_version="abc123"
    )
    assert result["policy"] == {"id": "baseline", "version": "abc123"}
    with pytest.raises(ValueError, match="ID and version"):
        run_shadow(recording, "wfs", lambda x: x, policy_id="", policy_version="1")


def test_array_hash_preserves_big_endian_nan_payload_bits():
    from noether_replay.model import canonical_json

    bits = np.array([0x7FF8000000000001, 0x8000000000000000], dtype=">u8")
    values = bits.view(">f8")
    expected = hashlib.sha256(
        canonical_json({"dtype": ">f8", "shape": [2]}) + bits.tobytes()
    ).hexdigest()
    assert array_digest(values) == expected


def test_untrusted_recording_labels_cannot_embed_markdown_images(recording):
    recording.id = "![tracking](https://invalid.example/pixel)"
    text = markdown_report(build_report(recording))
    assert r"\!\[tracking\]\(https://invalid.example/pixel\)" in text
    assert "**![tracking]" not in text
