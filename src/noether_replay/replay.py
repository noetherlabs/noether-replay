"""Deterministic event reconstruction without inventing physical causality."""

from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np

from .model import Recording, Stream, canonical_json


def events(recording: Recording) -> list[dict]:
    """Return canonical order within each named clock, including all known stages.

    Sorting clock names is only a serialization convention. Equal nominal times
    or intersecting uncertainty intervals do not establish a physical order.
    """
    recording.validate()
    result = []
    for stream in recording.streams:
        for stage, times in stream.times_ns.items():
            for index, time in enumerate(times):
                if time is not None:
                    result.append(
                        {
                            "clock_id": stream.clock_id,
                            "time_ns": int(time),
                            "uncertainty_ns": stream.uncertainty_ns,
                            "stream_id": stream.id,
                            "sample_index": index,
                            "frame_id": None
                            if stream.frame_ids is None
                            else int(stream.frame_ids[index]),
                            "stage": stage,
                        }
                    )
    return sorted(
        result,
        key=lambda e: (e["clock_id"], e["time_ns"], e["stream_id"], e["sample_index"], e["stage"]),
    )


def timeline_digest(recording: Recording) -> str:
    """Bind the canonical event sequence and all underlying recording arrays."""
    return hashlib.sha256(
        canonical_json({"recording_sha256": recording.digest(), "events": events(recording)})
    ).hexdigest()


def calibration_assignment(recording: Recording, stream: Stream, index: int) -> str | None:
    """Assign only a unique epoch covering the complete exposure uncertainty.

    An epoch product digest is an asserted reference, not evidence that the
    calibration is accurate. This operation establishes temporal coverage only.
    """
    if stream.uncertainty_ns is None:
        return None
    start = stream.times_ns.get("exposure_start")
    end = stream.times_ns.get("exposure_end")
    if start is None or end is None or start[index] is None or end[index] is None:
        return None
    lo = int(start[index]) - int(stream.uncertainty_ns)
    hi = int(end[index]) + int(stream.uncertainty_ns)
    if lo > hi or start[index] > end[index]:
        return None
    matches = [
        epoch["id"]
        for epoch in recording.calibrations
        if stream.id in epoch["stream_ids"]
        and epoch["clock_id"] == stream.clock_id
        and epoch["start_ns"] <= lo
        and hi < epoch["end_ns"]
    ]
    # Reject partially overlapping epochs as well: full coverage by one epoch
    # does not eliminate ambiguity when another overlaps part of the exposure.
    overlaps = [
        epoch["id"]
        for epoch in recording.calibrations
        if stream.id in epoch["stream_ids"]
        and epoch["clock_id"] == stream.clock_id
        and epoch["start_ns"] <= hi
        and epoch["end_ns"] > lo
    ]
    return matches[0] if len(matches) == len(overlaps) == 1 else None


def run_shadow(
    recording: Recording,
    stream_id: str,
    policy: Callable[[np.ndarray], np.ndarray],
    *,
    policy_id: str,
    policy_version: str,
) -> dict:
    """Evaluate hypothetical actions on causally ordered recorded observations.

    The policy receives a copy of each observation in availability order. It
    cannot change the stored plant trajectory. Outcome metrics are deliberately
    absent; a different action's closed-loop effect requires a validated plant.
    """
    recording.validate()
    if not all(isinstance(value, str) and value.strip() for value in (policy_id, policy_version)):
        raise ValueError("Shadow policy requires a nonempty ID and version")
    streams = [s for s in recording.streams if s.id == stream_id]
    if len(streams) != 1 or streams[0].kind not in {"wfs_measurements", "wfs_pixels"}:
        raise ValueError("Shadow mode requires one WFS observation stream")
    stream = streams[0]
    if not len(stream.values):
        raise ValueError("Shadow replay requires at least one observation")
    times = stream.times_ns.get("available")
    if times is None or any(t is None for t in times) or stream.uncertainty_ns is None:
        raise ValueError(
            "Shadow replay requires complete availability times and known clock uncertainty"
        )
    times = tuple(int(time) for time in times)
    for index in range(len(times)):
        stages = [
            stream.times_ns[key][index]
            for key in ("exposure_start", "exposure_end", "available")
            if key in stream.times_ns and stream.times_ns[key][index] is not None
        ]
        if any(after < before for before, after in zip(stages, stages[1:])):
            raise ValueError(
                "Observation event order contradicts exposure and availability semantics"
            )
    if not np.isfinite(stream.values).all():
        raise ValueError(
            "Shadow replay requires an explicit missing-observation policy for nonfinite data"
        )
    order = sorted(range(len(times)), key=lambda i: times[i])
    for first, second in zip(order, order[1:]):
        if times[second] - times[first] <= 2 * int(stream.uncertainty_ns):
            raise ValueError("Availability intervals overlap; causal shadow ordering is ambiguous")
    samples = []
    shape = None
    before = recording.digest()
    for index in order:
        action = np.asarray(policy(stream.values[index].copy()))
        if (
            action.dtype.kind not in "biuf"
            or action.ndim != 1
            or not action.size
            or not np.isfinite(action).all()
        ):
            raise ValueError("Shadow policy must return a finite one-dimensional command")
        if shape is not None and action.shape != shape:
            raise ValueError("Shadow command shape changed during replay")
        shape = action.shape
        samples.append(
            {
                "sample_index": index,
                "available_ns": int(times[index]),
                "hypothetical_command": action.tolist(),
            }
        )
    if recording.digest() != before:
        raise ValueError("Recording changed during shadow evaluation")
    return {
        "mode": "shadow",
        "policy": {"id": policy_id, "version": policy_version},
        "recording_sha256": before,
        "stream_id": stream_id,
        "clock_id": stream.clock_id,
        "samples": samples,
        "outcome_assessed": False,
        "limitation": "Actions are hypothetical; recorded outcomes are not attributed to this policy.",
    }
