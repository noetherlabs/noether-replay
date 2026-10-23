"""Observational qualification reports with explicit evidence gaps."""

from __future__ import annotations

import hashlib
import html
import json
import re

import numpy as np

from . import __version__
from .model import Recording, canonical_json
from .replay import calibration_assignment, events


def build_report(recording: Recording, *, mode: str = "observational") -> dict:
    if mode != "observational":
        raise ValueError(
            "This report assesses recorded evidence only. Use run_shadow for hypothetical actions; counterfactual outcomes require a validated simulator or HIL backend."
        )
    manifest = recording.manifest()
    source_digest = hashlib.sha256(canonical_json(manifest)).hexdigest()
    timeline = events(recording)
    findings = []
    summaries = []

    def finding(code: str, stream_id: str | None, count: int, detail: str) -> None:
        if count:
            findings.append(
                {"code": code, "stream_id": stream_id, "count": int(count), "detail": detail}
            )

    for stream in sorted(recording.streams, key=lambda s: s.id):
        size = len(stream.values)
        nonfinite = int(np.count_nonzero(~np.isfinite(stream.values)))
        finding(
            "nonfinite_values",
            stream.id,
            nonfinite,
            "Values are retained; no zero imputation is performed.",
        )
        finding(
            "unknown_units",
            stream.id,
            int(stream.units is None),
            "Physical units are not established by the source.",
        )
        finding(
            "unknown_clock_uncertainty",
            stream.id,
            int(stream.uncertainty_ns is None),
            "Nominal timestamps cannot establish an exact causal ordering.",
        )
        stages = {key: sum(x is not None for x in times) for key, times in stream.times_ns.items()}
        finding(
            "missing_event_times",
            stream.id,
            sum(
                not any(times[i] is not None for times in stream.times_ns.values())
                for i in range(size)
            ),
            "These samples have no exposure, availability or actuation event times.",
        )
        if stream.kind in {"wfs_measurements", "wfs_pixels"}:
            finding(
                "missing_availability",
                stream.id,
                size - stages.get("available", 0),
                "Causal controller replay requires observation availability, not just sampling frequency.",
            )
        if stream.kind == "command_recorded":
            finding(
                "unknown_command_semantics",
                stream.id,
                size,
                "The source does not establish whether these commands were requested or physically applied.",
            )
        if stream.kind == "command_applied":
            finding(
                "missing_effective_time",
                stream.id,
                size - stages.get("effective", 0),
                "Actuator effect time is required to reconstruct applied-command timing.",
            )
        if stream.kind == "command_requested":
            finding(
                "missing_issue_time",
                stream.id,
                size - stages.get("issued", 0),
                "Command issue time is not established for these requests.",
            )
        if stream.frame_ids is None:
            finding(
                "unknown_frame_identity",
                stream.id,
                size,
                "Array position is retained; missing frames cannot be inferred from it.",
            )
            missing = None
        else:
            ids = [int(x) for x in stream.frame_ids]
            finding(
                "duplicate_frame_ids",
                stream.id,
                len(ids) - len(set(ids)),
                "Repeated frame identifiers require source-specific interpretation.",
            )
            finding(
                "nonmonotonic_frame_ids",
                stream.id,
                sum(b < a for a, b in zip(ids, ids[1:])),
                "The stored frame sequence is not monotonic.",
            )
            missing = sum(max(0, b - a - 1) for a, b in zip(ids, ids[1:]))
            finding(
                "frame_id_gaps",
                stream.id,
                missing,
                "Unrepresented integer frame IDs; this is not proof of detector frame loss without the counter contract.",
            )
        for stage, times in sorted(stream.times_ns.items()):
            known = [x for x in times if x is not None]
            finding(
                "nonmonotonic_event_times",
                stream.id,
                sum(b < a for a, b in zip(known, known[1:])),
                f"Stored {stage} timestamps move backwards.",
            )
        ordered_stages = [
            key
            for key in (
                "exposure_start",
                "exposure_end",
                "available",
                "completed",
                "issued",
                "effective",
            )
            if key in stream.times_ns
        ]
        violations = 0
        for index in range(size):
            known = [
                stream.times_ns[key][index]
                for key in ordered_stages
                if stream.times_ns[key][index] is not None
            ]
            violations += any(b < a for a, b in zip(known, known[1:]))
        finding(
            "event_sequence_violation",
            stream.id,
            violations,
            "A sample's nominal event timestamps contradict acquisition-to-actuation order.",
        )
        applicable = stream.kind in {"wfs_measurements", "wfs_pixels", "science"}
        coverage = (
            sum(calibration_assignment(recording, stream, i) is not None for i in range(size))
            if applicable
            else None
        )
        if applicable:
            finding(
                "unqualified_calibration_interval",
                stream.id,
                size - coverage,
                "No unique calibration epoch covers the complete exposure and timing uncertainty.",
            )
        summaries.append(
            {
                "id": stream.id,
                "kind": stream.kind,
                "samples": size,
                "shape": list(stream.values.shape),
                "dtype": stream.values.dtype.str,
                "units": stream.units,
                "nonfinite_values": nonfinite,
                "unrepresented_frame_ids": missing,
                "known_event_times": stages,
                "calibration_covered_samples": coverage,
            }
        )
    clocks = sorted({event["clock_id"] for event in timeline})
    finding(
        "multiple_clock_domains",
        None,
        int(len(clocks) > 1),
        "No cross-clock synchronization is inferred; replay order is defined within each clock only.",
    )
    overlaps = 0
    latest_upper = {}
    for event in timeline:
        uncertainty = event["uncertainty_ns"]
        if uncertainty is None:
            continue
        uncertainty = int(uncertainty)
        clock = event["clock_id"]
        lower, upper = event["time_ns"] - uncertainty, event["time_ns"] + uncertainty
        if clock in latest_upper and lower <= latest_upper[clock]:
            overlaps += 1
        latest_upper[clock] = max(upper, latest_upper.get(clock, upper))
    finding(
        "ambiguous_event_order",
        None,
        overlaps,
        "These event intervals overlap an earlier interval; canonical tie-breaking is not physical causality.",
    )
    finding(
        "empty_recording",
        None,
        int(not recording.streams or not any(len(s.values) for s in recording.streams)),
        "There are no telemetry samples to qualify.",
    )
    return {
        "schema": "noether-replay-report/1",
        "software_version": __version__,
        "mode": "observational",
        "recording_id": recording.id,
        "recording_sha256": source_digest,
        "timeline_sha256": hashlib.sha256(
            canonical_json({"recording_sha256": source_digest, "events": timeline})
        ).hexdigest(),
        "source": json.loads(canonical_json(recording.provenance)),
        "event_count": len(timeline),
        "clock_domains": clocks,
        "status": "evidence_gaps" if findings else "recording_consistent",
        "streams": summaries,
        "findings": findings,
        "limitations": [
            "Recording consistency is not instrument qualification or controller validation.",
            "Calibration hashes bind references; calibration accuracy is not tested.",
            "Recorded closed-loop outcomes cannot evaluate a different controller.",
            "Hashes establish content consistency, not source authenticity.",
        ],
    }


def markdown_report(report: dict) -> str:
    def cell(value: object) -> str:
        text = html.escape(str(value).replace("\n", " ").replace("\r", " "), quote=False)
        return re.sub(r"([\\|`*_{}\[\]()#+!])", r"\\\1", text)

    lines = [
        "# Telemetry evidence report",
        "",
        f"Recording: **{cell(report['recording_id'])}**  ",
        f"Mode: `{report['mode']}` · Status: `{report['status']}`",
        "",
        f"Reconstructed events: {report['event_count']}. Clock domains: {len(report['clock_domains'])}.",
        "",
        "| Stream | Samples | Units | Calibration-covered samples |",
        "| --- | ---: | --- | ---: |",
    ]
    for stream in report["streams"]:
        lines.append(
            f"| {cell(stream['id'])} | {stream['samples']} | {cell(stream['units'] or 'unknown')} | {stream['calibration_covered_samples'] if stream['calibration_covered_samples'] is not None else 'n/a'} |"
        )
    lines += ["", "## Findings", ""]
    if not report["findings"]:
        lines.append(
            "No inconsistency was found within the supplied metadata. This is not a control-performance result."
        )
    for item in report["findings"]:
        lines.append(
            f"- `{item['code']}` — {cell(item['stream_id'] or 'recording')} ({item['count']}): {cell(item['detail'])}"
        )
    lines += ["", "## Interpretation", ""] + [f"- {text}" for text in report["limitations"]]
    lines += [
        "",
        "## Reproducibility",
        "",
        f"Recording SHA-256: `{report['recording_sha256']}`  ",
        f"Timeline SHA-256: `{report['timeline_sha256']}`",
        "",
    ]
    return "\n".join(lines)
