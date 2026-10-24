"""Deterministic event reconstruction without inventing physical causality."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

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
    For a conditioned window this is the common contributing source epoch;
    it does not make an untransformed calibration a model for filtered data.
    """
    if "conditioning" in stream.metadata:
        recording.validate()
        if not any(member is stream for member in recording.streams):
            raise ValueError("Conditioned calibration assignment requires the owning recording")
    return _calibration_assignment(recording, stream, index)


def _calibration_assignment(recording: Recording, stream: Stream, index: int) -> str | None:
    """Internal path for consumers that have already validated the recording."""
    if "conditioning" in stream.metadata:
        return stream.metadata["conditioning"]["calibration_epoch_ids"][index]
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


@dataclass(frozen=True, slots=True, eq=False)
class ShadowObservation:
    """Current shadow input with immutable metadata and an isolated value copy.

    ``values`` is writable for policy-local processing; it never shares memory
    with the recording. Other fields and nested mappings/tuples are immutable.
    Calibration products are temporally applicable references, not validated
    physical models. Conditioning metadata describes only this output's support;
    it is not a covariance, whitening or likelihood contract.

    This is an interface for trusted Python callbacks, not a sandbox. A callback
    that separately holds the source recording can still access or mutate it.
    """

    values: np.ndarray
    stream_id: str
    sample_index: int
    frame_id: int | None
    units: str | None
    clock_id: str
    available_ns: int
    uncertainty_ns: int
    exposure_start_ns: int | None
    exposure_end_ns: int | None
    calibration_epoch_id: str | None
    calibration_products: Mapping[str, str]
    conditioning: Mapping[str, Any] | None


def _freeze_metadata(value: Any) -> Any:
    """Detach the selected metadata while making every nested container immutable."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _shadow_observation(recording: Recording, stream: Stream, index: int) -> ShadowObservation:
    """Build only the current input after the complete recording passed admission."""
    epoch_id = _calibration_assignment(recording, stream, index)
    products = next(
        (epoch["products"] for epoch in recording.calibrations if epoch["id"] == epoch_id), {}
    )
    descriptor = stream.metadata.get("conditioning")
    conditioning = None
    if descriptor is not None:
        conditioning = _freeze_metadata(
            {
                "source_recording_sha256": descriptor["source_recording_sha256"],
                "source_stream_id": descriptor["source_stream_id"],
                "support_indices": descriptor["support_indices"][index],
                "channel_indices": descriptor["channel_indices"],
                "weights": [value for value in descriptor["operator"][index] if value != 0],
                "contributing_exposures": descriptor["contributing_exposures"][index],
                "ready_not_before_ns": descriptor["ready_not_before_ns"][index],
            }
        )

    def time(stage: str) -> int | None:
        times = stream.times_ns.get(stage)
        value = None if times is None else times[index]
        return None if value is None else int(value)

    return ShadowObservation(
        values=stream.values[index].copy(),
        stream_id=stream.id,
        sample_index=index,
        frame_id=None if stream.frame_ids is None else int(stream.frame_ids[index]),
        units=stream.units,
        clock_id=stream.clock_id,
        available_ns=int(stream.times_ns["available"][index]),
        uncertainty_ns=int(stream.uncertainty_ns),
        exposure_start_ns=time("exposure_start"),
        exposure_end_ns=time("exposure_end"),
        calibration_epoch_id=epoch_id,
        calibration_products=_freeze_metadata(products),
        conditioning=conditioning,
    )


def run_shadow(
    recording: Recording,
    stream_id: str,
    policy: Callable[[np.ndarray], np.ndarray] | Callable[[ShadowObservation], np.ndarray],
    *,
    policy_id: str,
    policy_version: str,
    policy_input: Literal["array", "event"] = "array",
) -> dict:
    """Evaluate hypothetical actions on causally ordered recorded observations.

    The default policy receives an array copy. With ``policy_input="event"`` it
    receives a ShadowObservation with current timing, identity and calibration
    context. Both use the same admission gates and availability order. No future
    values or full Recording are passed through either interface. Outcome metrics
    are absent; a different action's closed-loop effect requires a validated plant.
    """
    recording.validate()
    if not isinstance(policy_input, str) or policy_input not in ("array", "event"):
        raise ValueError("Shadow policy input must be 'array' or 'event'")
    if not all(isinstance(value, str) and value.strip() for value in (policy_id, policy_version)):
        raise ValueError("Shadow policy requires a nonempty ID and version")
    streams = [s for s in recording.streams if s.id == stream_id]
    if len(streams) != 1 or streams[0].kind not in {"wfs_measurements", "wfs_pixels"}:
        raise ValueError("Shadow mode requires one WFS observation stream")
    stream = streams[0]
    conditioning = stream.metadata.get("conditioning")
    if conditioning is not None and any(
        epoch is None for epoch in conditioning["calibration_epoch_ids"]
    ):
        raise ValueError("Conditioned shadow observations require one calibration epoch per window")
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
        observation = (
            _shadow_observation(recording, stream, index)
            if policy_input == "event"
            else stream.values[index].copy()
        )
        action = np.asarray(policy(observation))
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
    result = {
        "mode": "shadow",
        "policy": {"id": policy_id, "version": policy_version},
        "recording_sha256": before,
        "stream_id": stream_id,
        "clock_id": stream.clock_id,
        "samples": samples,
        "outcome_assessed": False,
        "limitation": "Actions are hypothetical; recorded outcomes are not attributed to this policy.",
    }
    if policy_input == "event":
        result["policy_input"] = "event"
    return result
