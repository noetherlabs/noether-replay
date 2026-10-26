"""Bounded latency evidence from explicitly associated recorded events."""

from __future__ import annotations

import hashlib

from .model import MAX_JSON_BYTES, Recording, Stream, _integer, canonical_json

SCHEMA = "noether-replay-command-timing/1"
_CAUSAL = ("consistent", "violated", "ambiguous", "not_established")
_BUDGET = ("met", "exceeded", "ambiguous", "not_established", "not_requested")


def _time(stream: Stream, stage: str, index: int) -> int | None:
    series = stream.times_ns.get(stage)
    value = None if series is None else series[index]
    return None if value is None else int(value)


def _uncertainty(stream: Stream) -> int | None:
    return None if stream.uncertainty_ns is None else int(stream.uncertainty_ns)


def _interval(time: int | None, uncertainty: int | None) -> list[int] | None:
    if time is None or uncertainty is None:
        return None
    return [time - uncertainty, time + uncertainty]


def _sequence_violation(stream: Stream, index: int) -> bool:
    stages = (
        "exposure_start", "exposure_end", "available", "completed", "issued", "effective",
    )
    known = [time for stage in stages if (time := _time(stream, stage, index)) is not None]
    return any(after < before for before, after in zip(known, known[1:]))


def _associations(associations: list, streams: dict[str, Stream], count: int) -> list[dict]:
    if not isinstance(associations, list) or len(associations) > count:
        raise ValueError("Associations must be a list with at most one entry per applied command")
    normalized = []
    applied_indices = set()
    for item in associations:
        if not isinstance(item, dict) or set(item) != {"applied_index", "observations"}:
            raise ValueError("Association fields must be applied_index and observations")
        index = item["applied_index"]
        if not _integer(index) or not 0 <= index < count:
            raise ValueError("Applied indices must be in-range integers, not booleans")
        index = int(index)
        if index in applied_indices:
            raise ValueError("Applied command associations must be unique")
        applied_indices.add(index)
        observations = item["observations"]
        if not isinstance(observations, list) or not observations:
            raise ValueError("Associated observations must be a nonempty list")
        references = []
        seen = set()
        for reference in observations:
            if not isinstance(reference, dict) or set(reference) != {"stream_id", "sample_index"}:
                raise ValueError("Observation reference fields must be stream_id and sample_index")
            stream_id, sample = reference["stream_id"], reference["sample_index"]
            if not isinstance(stream_id, str) or stream_id not in streams:
                raise ValueError("Observation references require an existing stream ID")
            source = streams[stream_id]
            if source.kind not in {"wfs_measurements", "wfs_pixels"}:
                raise ValueError("Observation references must identify WFS streams")
            if not _integer(sample) or not 0 <= sample < len(source.values):
                raise ValueError("Observation sample indices must be in-range integers, not booleans")
            sample = int(sample)
            if (stream_id, sample) in seen:
                raise ValueError("Observation references within an association must be unique")
            seen.add((stream_id, sample))
            references.append({"stream_id": stream_id, "sample_index": sample})
        normalized.append({
            "applied_index": index,
            "observations": sorted(references, key=lambda ref: (ref["stream_id"], ref["sample_index"])),
        })
    if len(canonical_json(normalized)) > MAX_JSON_BYTES:
        raise ValueError("Normalized associations exceed the 16 MiB metadata bound")
    return sorted(normalized, key=lambda item: item["applied_index"])


def qualify_command_timing(
    recording: Recording,
    *,
    applied_stream_id: str,
    associations: list,
    latency_budget_ns: int | None = None,
) -> dict:
    """Assess availability-to-actuator latency for all recorded applied commands.

    Associations are caller assertions, using original array indices; frame IDs,
    array positions and nominal frequencies never establish a correspondence.
    Each association is ``{applied_index, observations: [{stream_id, sample_index}]}``.
    Missing associations remain unestablished rows. Empty or malformed mappings
    are rejected. The target must explicitly have kind ``command_applied``.

    For input availability intervals [l_i, u_i], the latest availability is in
    [max(l_i), max(u_i)]. Subtract this interval from the applied command's
    effective-time interval using exact integer arithmetic. No independence of
    clock errors is assumed. Bounds can be conservative for correlated errors.
    Nonnegative lower latency is timing-consistent, a negative upper bound is
    violated, and an interval spanning zero is ambiguous. A latency budget is
    assessed only for timing-consistent associations; equality meets the budget.

    This assesses asserted event relationships, not calibrated measurements,
    computation time, observed deadline frequency or alternative-loop outcomes.
    Contradictory known nominal acquisition-to-actuation stages withhold the row;
    logging time (``recorded``) does not participate in that stage order.
    Supplied command completion/issue intervals must also be no earlier than the
    latest associated input availability. Contradictory or overlapping intervals
    withhold qualification; missing optional stages do not acquire timestamps.
    Unknown calibration and nonfinite measurement values do not alter the event
    arithmetic. Validated conditioned observations use their supplied availability,
    never a readiness bound substituted for processing completion.
    The complete JSON report is limited to 16 MiB. Oversized results raise rather
    than truncate command coverage; the caller must select a bounded recording.
    The source digest is checked again before return; this is not a concurrent lock.
    """
    source_digest = recording.digest()
    if not isinstance(applied_stream_id, str) or not applied_stream_id:
        raise ValueError("An applied command stream ID is required")
    streams = {stream.id: stream for stream in recording.streams}
    target = streams.get(applied_stream_id)
    if target is None or target.kind != "command_applied":
        raise ValueError("Timing qualification requires a command_applied stream")
    if not len(target.values):
        raise ValueError("Timing qualification requires at least one applied command")
    if latency_budget_ns is not None:
        if not _integer(latency_budget_ns) or latency_budget_ns < 0:
            raise ValueError("Latency budget must be nonnegative integer nanoseconds or None")
        latency_budget_ns = int(latency_budget_ns)
    normalized = _associations(associations, streams, len(target.values))
    by_index = {item["applied_index"]: item["observations"] for item in normalized}
    configuration = {
        "applied_stream_id": applied_stream_id,
        "latency_budget_ns": latency_budget_ns,
        "associations": normalized,
    }
    configuration_bytes = canonical_json({
        "recording_sha256": source_digest, "configuration": configuration,
    })
    report_bytes = len(configuration_bytes)
    if report_bytes > MAX_JSON_BYTES:
        raise ValueError("Timing report exceeds the 16 MiB JSON bound; select a bounded recording")
    commands = []
    for index in range(len(target.values)):
        reasons = []
        effective = _time(target, "effective", index)
        completed = _time(target, "completed", index)
        issued = _time(target, "issued", index)
        uncertainty = _uncertainty(target)
        effective_interval = _interval(effective, uncertainty)
        completed_interval = _interval(completed, uncertainty)
        issued_interval = _interval(issued, uncertainty)
        if target.clock_id is None:
            reasons.append("missing_command_clock")
        if uncertainty is None:
            reasons.append("unknown_command_uncertainty")
        if effective is None:
            reasons.append("missing_effective_time")
        if _sequence_violation(target, index):
            reasons.append("event_sequence_violation")
        references = by_index.get(index, [])
        if not references:
            reasons.append("missing_association")
        observations = []
        for reference in references:
            source = streams[reference["stream_id"]]
            sample = reference["sample_index"]
            available = _time(source, "available", sample)
            source_uncertainty = _uncertainty(source)
            if source.clock_id is None:
                reasons.append("missing_observation_clock")
            elif target.clock_id is not None and source.clock_id != target.clock_id:
                reasons.append("cross_clock_reference")
            if source_uncertainty is None:
                reasons.append("unknown_observation_uncertainty")
            if available is None:
                reasons.append("missing_availability")
            if _sequence_violation(source, sample):
                reasons.append("event_sequence_violation")
            observations.append({
                **reference,
                "clock_id": source.clock_id,
                "available_ns": available,
                "uncertainty_ns": source_uncertainty,
                "availability_interval_ns": _interval(available, source_uncertainty),
            })
        latest = None
        latency = None
        causal = "not_established"
        budget = "not_requested" if latency_budget_ns is None else "not_established"
        if not reasons:
            # Explicit lineage -> latest availability -> effective-time interval.
            latest = [
                max(item["availability_interval_ns"][bound] for item in observations)
                for bound in (0, 1)
            ]
            latency = [effective_interval[0] - latest[1], effective_interval[1] - latest[0]]
            for stage, interval in (("completed", completed_interval), ("issued", issued_interval)):
                if interval is not None:
                    if interval[1] < latest[0]:
                        reasons.append(f"input_unavailable_at_{stage}")
                    elif interval[0] < latest[1]:
                        reasons.append(f"ambiguous_input_availability_at_{stage}")
        if not reasons:
            if latency[0] >= 0:
                causal = "consistent"
                if latency_budget_ns is not None:
                    budget = (
                        "met" if latency[1] <= latency_budget_ns
                        else "exceeded" if latency[0] > latency_budget_ns else "ambiguous"
                    )
            elif latency[1] < 0:
                causal = "violated"
            else:
                causal = "ambiguous"
        row = {
            "applied_index": index,
            "frame_id": None if target.frame_ids is None else int(target.frame_ids[index]),
            "clock_id": target.clock_id,
            "observations": observations,
            "completed_ns": completed,
            "issued_ns": issued,
            "effective_ns": effective,
            "uncertainty_ns": uncertainty,
            "completed_interval_ns": completed_interval,
            "issued_interval_ns": issued_interval,
            "effective_interval_ns": effective_interval,
            "latest_availability_interval_ns": latest,
            "latency_interval_ns": latency,
            "causal_order": causal,
            "budget": budget,
            "reasons": sorted(set(reasons)),
        }
        report_bytes += len(canonical_json(row))
        if report_bytes > MAX_JSON_BYTES:
            raise ValueError("Timing report exceeds the 16 MiB JSON bound; select a bounded recording")
        commands.append(row)
    if recording.digest() != source_digest:
        raise ValueError("Recording changed during timing qualification")
    report = {
        "schema": SCHEMA,
        "mode": "observational",
        "recording_sha256": source_digest,
        "configuration": configuration,
        "configuration_sha256": hashlib.sha256(configuration_bytes).hexdigest(),
        "commands": commands,
        "summary": {
            "command_count": len(commands),
            "associated_count": len(normalized),
            "causal_order_counts": {
                status: sum(row["causal_order"] == status for row in commands) for status in _CAUSAL
            },
            "budget_counts": {
                status: sum(row["budget"] == status for row in commands) for status in _BUDGET
            },
        },
        "limitations": [
            "Associations are caller assertions; timing consistency does not establish causation.",
            "Association coverage does not prove that all physical command dependencies were recorded.",
            "Missing completion or issue times remain unknown; qualification does not establish every stage.",
            "Latency starts at the latest associated observation availability, not exposure time.",
            "These intervals are uncertainty bounds, not statistical confidence intervals.",
            "No calibration accuracy, computation time, controller performance or real-time guarantee is assessed.",
            "Unassociated observations do not establish skipped frames or a dropped-command rate.",
            "Digests bind source content and declared associations, not their authenticity.",
        ],
    }
    if len(canonical_json(report)) > MAX_JSON_BYTES:
        raise ValueError("Timing report exceeds the 16 MiB JSON bound; select a bounded recording")
    return report
