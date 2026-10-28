"""Exposure-support age for every declared input to a recorded applied command."""

from __future__ import annotations

import hashlib

from .model import MAX_JSON_BYTES, Recording, Stream, _integer, canonical_json
from .timing import _interval, _sequence_violation, _time, _uncertainty, qualify_command_timing

SCHEMA = "noether-replay-input-freshness/1"
_BUDGET = ("met", "exceeded", "ambiguous", "not_established")


def _budget(interval: list[int], maximum: int) -> str:
    if interval[1] <= maximum:
        return "met"
    return "exceeded" if interval[0] > maximum else "ambiguous"


def _contributor(source: Stream, index: int, target: Stream, applied_index: int,
                 maximum: int) -> dict:
    start, end = _time(source, "exposure_start", index), _time(source, "exposure_end", index)
    uncertainty = _uncertainty(source)
    effect = _interval(_time(target, "effective", applied_index), _uncertainty(target))
    start_interval, end_interval = _interval(start, uncertainty), _interval(end, uncertainty)
    reasons = []
    for name, value in (("exposure_start", start), ("exposure_end", end)):
        if value is None:
            reasons.append(f"missing_{name}")
    if uncertainty is None:
        reasons.append("unknown_exposure_uncertainty")
    if source.clock_id is None:
        reasons.append("missing_exposure_clock")
    if target.clock_id is None:
        reasons.append("missing_command_clock")
    if source.clock_id is not None and target.clock_id is not None:
        if source.clock_id != target.clock_id:
            reasons.append("cross_clock_reference")
    if effect is None:
        reasons.append("missing_effective_time_or_uncertainty")
    if _sequence_violation(source, index):
        reasons.append("exposure_event_sequence_violation")
    if _sequence_violation(target, applied_index):
        reasons.append("command_event_sequence_violation")
    oldest, hull = None, None
    if not reasons:
        oldest = [effect[0] - start_interval[1], effect[1] - start_interval[0]]
        hull = [effect[0] - end_interval[1], effect[1] - start_interval[0]]
        # Every point in the exposure support must precede actuator effect.
        if effect[1] < end_interval[0]:
            reasons.append("exposure_after_effect")
        elif effect[0] < end_interval[1]:
            reasons.append("ambiguous_exposure_before_effect")
    return {
        "stream_id": source.id,
        "sample_index": index,
        "clock_id": source.clock_id,
        "exposure_start_ns": start,
        "exposure_end_ns": end,
        "uncertainty_ns": uncertainty,
        "exposure_start_interval_ns": start_interval,
        "exposure_end_interval_ns": end_interval,
        "support_age_hull_ns": hull,
        "oldest_support_age_interval_ns": oldest,
        "budget": _budget(oldest, maximum) if not reasons else "not_established",
        "reasons": reasons,
    }


def qualify_input_freshness(
    recording: Recording,
    *,
    applied_stream_id: str,
    associations: list,
    max_support_age_ns: int,
    latency_budget_ns: int | None = None,
) -> dict:
    """Bound the oldest exposure support used by each recorded applied command.

    Associations and command timing use ``qualify_command_timing`` unchanged.
    A raw observation contributes its own exposure; a conditioned observation
    contributes every raw sample with nonzero operator weight. Conditioning
    lineage must reproduce before any result is produced. No association is
    inferred from matching frame IDs, timestamps or positions.

    With effect interval [L, U], exposure start s and clock bound u, the oldest
    support age is in [L - (s + u), U - (s - u)]. The maximum over all contributors
    bounds the oldest support for the command. Its upper endpoint at most the
    requested bound is ``met``; its lower endpoint above the bound is ``exceeded``;
    otherwise it is ``ambiguous``. Missing/contradictory evidence or command timing
    not qualified as consistent yields ``not_established``. All arithmetic uses
    integer nanoseconds, without independent-clock-error assumptions.

    Contributor age verdicts describe their own exposure-to-effect evidence.
    A known stale contributor remains visible when another dependency withholds
    command qualification. They do not establish input availability at decision
    time. A long integration is represented by its full support, not its midpoint,
    end, filtered-output availability or a fabricated effective sample time.

    This is an observational report, not a deadline guarantee, causal benefit,
    calibration validation or controller-performance assessment. It has a 16 MiB
    canonical JSON bound and never truncates dependencies to fit. The recording
    digest is rechecked before return; this does not provide a concurrent lock.
    """
    if not _integer(max_support_age_ns) or max_support_age_ns < 0:
        raise ValueError("Maximum support age must be nonnegative integer nanoseconds")
    maximum = int(max_support_age_ns)
    timing = qualify_command_timing(
        recording, applied_stream_id=applied_stream_id,
        associations=associations, latency_budget_ns=latency_budget_ns,
    )
    source_digest = timing["recording_sha256"]
    streams = {stream.id: stream for stream in recording.streams}
    target = streams[applied_stream_id]
    configuration = {**timing["configuration"], "max_support_age_ns": maximum}
    configuration_bytes = canonical_json({
        "recording_sha256": source_digest, "configuration": configuration,
    })
    report_bytes = len(canonical_json(timing)) + len(configuration_bytes)

    def account(size: int) -> None:
        nonlocal report_bytes
        report_bytes += size
        if report_bytes > MAX_JSON_BYTES:
            raise ValueError("Freshness report exceeds the 16 MiB JSON bound; select a bounded recording")

    account(0)
    commands = []
    contributor_counts = dict.fromkeys(_BUDGET, 0)
    for command in timing["commands"]:
        observations = []
        contributors = []
        counted_bytes = 0
        for reference in command["observations"]:
            source = streams[reference["stream_id"]]
            index = reference["sample_index"]
            descriptor = source.metadata.get("conditioning")
            raw_source = source if descriptor is None else streams[descriptor["source_stream_id"]]
            support = [index] if descriptor is None else descriptor["support_indices"][index]
            evidence = []
            for raw_index in support:
                item = _contributor(raw_source, int(raw_index), target,
                                    command["applied_index"], maximum)
                size = len(canonical_json(item))
                account(size)
                counted_bytes += size
                evidence.append(item)
                contributors.append(item)
                contributor_counts[item["budget"]] += 1
            observations.append({
                "stream_id": source.id,
                "sample_index": index,
                "contributors": evidence,
            })
        reasons = []
        if command["causal_order"] != "consistent":
            reasons.append("command_timing_not_consistent")
        if not observations:
            reasons.append("missing_association")
        qualified = [item for item in contributors if item["budget"] != "not_established"]
        if len(qualified) != len(contributors):
            reasons.append("unqualified_exposure_support")
        oldest = None
        if contributors and len(qualified) == len(contributors):
            oldest = [
                max(item["oldest_support_age_interval_ns"][bound] for item in contributors)
                for bound in (0, 1)
            ]
        row = {
            "applied_index": command["applied_index"],
            "frame_id": command["frame_id"],
            "observations": observations,
            "oldest_support_age_interval_ns": oldest,
            "budget": _budget(oldest, maximum) if not reasons else "not_established",
            "reasons": reasons,
            "coverage": {
                "declared_observations": len(observations),
                "contributor_references": len(contributors),
                "qualified_contributor_references": len(qualified),
            },
        }
        # Count each contributor once while bounding construction incrementally.
        account(len(canonical_json(row)) - counted_bytes)
        commands.append(row)
    if recording.digest() != source_digest:
        raise ValueError("Recording changed during freshness qualification")
    report = {
        "schema": SCHEMA,
        "mode": "observational",
        "recording_sha256": source_digest,
        "configuration": configuration,
        "configuration_sha256": hashlib.sha256(configuration_bytes).hexdigest(),
        "timing": timing,
        "commands": commands,
        "summary": {
            "command_count": len(commands),
            "associated_count": timing["summary"]["associated_count"],
            "budget_counts": {
                status: sum(row["budget"] == status for row in commands) for status in _BUDGET
            },
            "contributor_reference_budget_counts": contributor_counts,
        },
        "limitations": [
            "Age is measured from exposure support to actuator effect, not from output availability.",
            "Every nonzero conditioned contributor counts; weights do not reduce its temporal age.",
            "Contributor counts count references, including repeated uses, not independent measurements.",
            "Associations are caller assertions and need not include every physical dependency.",
            "Contributor age does not establish availability at a policy decision or command issue.",
            "Uncertainty intervals are supplied bounds, not statistical confidence intervals.",
            "Calibration validity, measurement accuracy and policy use of observations are not assessed.",
            "Meeting the age bound establishes no causal benefit, controller performance or safety.",
            "Digests bind supplied content, not association authenticity or physical execution.",
        ],
    }
    if len(canonical_json(report)) > MAX_JSON_BYTES:
        raise ValueError("Freshness report exceeds the 16 MiB JSON bound; select a bounded recording")
    return report
