"""Bind finite linear observation windows to their raw telemetry and timing.

Covariance is a supplied noise model, not an estimate or physical certification.
Unknown processing completion stays unknown. Derived observations cannot create
new measurements or evidence about alternative closed-loop trajectories.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .conditioning import MAX_DIMENSION, _matrix, propagate_covariance, validate_covariance
from .model import MAX_JSON_BYTES, Recording, Stream, _integer, canonical_json
from .replay import calibration_assignment

SCHEMA = "noether-replay/conditioning/1"
FIELDS = {
    "schema",
    "source_recording_sha256",
    "source_stream_id",
    "sample_indices",
    "channel_indices",
    "operator",
    "input_covariance",
    "output_covariance",
    "covariance_provenance",
    "support_indices",
    "input_calibration_epoch_ids",
    "calibration_epoch_ids",
    "ready_not_before_ns",
    "contributing_exposures",
}


def _raw_recording(recording: Recording) -> Recording:
    derived = {s.id for s in recording.streams if "conditioning" in s.metadata}
    return Recording(
        recording.id,
        tuple(s for s in recording.streams if s.id not in derived),
        recording.calibrations,
        recording.metadata,
        recording.provenance,
    )


def _indices(values, size: int, name: str) -> tuple[int, ...]:
    if values is None:
        if not 1 <= size <= MAX_DIMENSION:
            raise ValueError(f"Select an explicit bounded {name} window")
        values = range(size)
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} indices must be a sequence") from exc
    if (
        not 1 <= len(values) <= MAX_DIMENSION
        or any(not _integer(i) or not 0 <= i < size for i in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{name} indices must be unique, bounded, in-range integers")
    return tuple(int(i) for i in values)


def _build(
    base: Recording,
    source: Stream,
    operator,
    covariance,
    *,
    output_id: str,
    covariance_provenance: dict,
    sample_indices,
    channel_indices,
    available_ns,
    source_digest: str,
) -> Stream:
    if source.kind not in {"wfs_measurements", "wfs_pixels"} or not source.units:
        raise ValueError("Conditioning requires a WFS stream with declared units")
    samples = _indices(sample_indices, len(source.values), "sample")
    channel_count = int(np.prod(source.values.shape[1:]))
    channels = _indices(channel_indices, channel_count, "channel")
    operator = np.ascontiguousarray(_matrix(operator, "Operator"))
    rows, columns = operator.shape
    if columns != len(samples) or max(rows, columns) * len(channels) > MAX_DIMENSION:
        raise ValueError("Joint sample/channel dimensions must fit the bounded covariance window")
    if np.any(np.count_nonzero(operator, axis=1) == 0):
        raise ValueError("Every derived observation requires nonempty source support")
    if not isinstance(covariance_provenance, dict) or not covariance_provenance:
        raise ValueError("Supplied covariance requires explicit provenance")
    canonical_json(covariance_provenance)
    # Preserve the supplied numeric matrix, before PSD roundoff projection, so
    # reconstruction does not repeatedly project an already projected matrix.
    covariance = np.ascontiguousarray(_matrix(covariance, "Input covariance"))
    if covariance.shape != (columns * len(channels),) * 2:
        raise ValueError("Covariance must cover the complete sample-major selected window")
    joint = np.kron(operator, np.eye(len(channels)))
    transformed_covariance = propagate_covariance(covariance, joint)
    # Gather only the bounded window. Reshaping a noncontiguous full recording
    # could otherwise copy hundreds of MiB before selecting a few channels.
    if source.values.ndim == 1:
        selected = source.values[np.asarray(samples)][:, None]
    else:
        coordinates = np.unravel_index(channels, source.values.shape[1:], order="C")
        selected = source.values[
            (np.asarray(samples)[:, None], *(axis[None, :] for axis in coordinates))
        ]
    with np.errstate(over="ignore", invalid="ignore"):
        selected = np.ascontiguousarray(selected, dtype=np.float64)
    if not np.isfinite(selected).all():
        raise ValueError("Conditioning requires an explicit policy for nonfinite source values")
    with np.errstate(over="ignore", invalid="ignore"):
        values = operator @ selected
    if not np.isfinite(values).all():
        raise ValueError("Conditioned observations exceed the float64 numerical range")
    try:
        available = (None,) * rows if available_ns is None else tuple(available_ns)
    except TypeError as exc:
        raise ValueError("Availability must be a sequence") from exc
    if len(available) != rows or any(t is not None and not _integer(t) for t in available):
        raise ValueError("Availability must contain one integer or None per output")
    available = tuple(None if t is None else int(t) for t in available)
    uncertainty = source.uncertainty_ns
    support, starts, ends, readiness, exposures = [], [], [], [], []
    input_epochs = [calibration_assignment(base, source, i) for i in samples]
    for row in operator:
        positions = np.flatnonzero(row)
        indices = [samples[int(i)] for i in positions]
        support.append(indices)
        timing = {}
        for key in ("exposure_start", "exposure_end", "available"):
            series = source.times_ns.get(key)
            timing[key] = [None if series is None else series[i] for i in indices]
        exposures.append(
            [
                {"sample_index": i, "exposure_start_ns": first, "exposure_end_ns": last}
                for i, first, last in zip(indices, timing["exposure_start"], timing["exposure_end"])
            ]
        )
        for first, last, ready in zip(*timing.values()):
            known = [t for t in (first, last, ready) if t is not None]
            if any(b < a for a, b in zip(known, known[1:])):
                raise ValueError("Source exposure and availability ordering is contradictory")
        starts.append(None if None in timing["exposure_start"] else min(timing["exposure_start"]))
        ends.append(None if None in timing["exposure_end"] else max(timing["exposure_end"]))
        lower = (
            None
            if uncertainty is None or None in timing["available"]
            else max(int(t) + int(uncertainty) for t in timing["available"])
        )
        readiness.append(lower)
        supplied = available[len(support) - 1]
        if supplied is not None and (
            source.clock_id is None or lower is None or supplied - int(uncertainty) < lower
        ):
            raise ValueError("Output availability is unknown or precedes a contributing input")
    times = {} if source.clock_id is None else {"available": available}
    # Use the source ID only to qualify the complete exposure hull against its
    # epochs. The output never inherits a last-frame or nearest calibration.
    hull = Stream(
        source.id,
        source.kind,
        values,
        source.units,
        clock_id=source.clock_id,
        times_ns={"exposure_start": tuple(starts), "exposure_end": tuple(ends)},
        uncertainty_ns=uncertainty,
    )
    epochs = []
    for row_index, row in enumerate(operator):
        assigned = {input_epochs[int(i)] for i in np.flatnonzero(row)}
        covering = calibration_assignment(base, hull, row_index)
        epochs.append(covering if len(assigned) == 1 and covering in assigned else None)
    descriptor = {
        "schema": SCHEMA,
        "source_recording_sha256": source_digest,
        "source_stream_id": source.id,
        "sample_indices": list(samples),
        "channel_indices": list(channels),
        "operator": operator.tolist(),
        "input_covariance": covariance.tolist(),
        "output_covariance": transformed_covariance.tolist(),
        "covariance_provenance": deepcopy(covariance_provenance),
        "support_indices": support,
        "input_calibration_epoch_ids": input_epochs,
        "calibration_epoch_ids": epochs,
        "ready_not_before_ns": readiness,
        "contributing_exposures": exposures,
    }
    if len(canonical_json(descriptor)) > MAX_JSON_BYTES:
        raise ValueError("Conditioning metadata exceeds 16 MiB")
    return Stream(
        output_id,
        source.kind,
        values,
        source.units,
        clock_id=source.clock_id,
        times_ns=times,
        uncertainty_ns=uncertainty,
        metadata={"conditioning": descriptor},
    )


def condition_observations(
    recording: Recording,
    stream_id: str,
    operator,
    covariance,
    *,
    output_id: str,
    covariance_provenance: dict,
    sample_indices=None,
    channel_indices=None,
    available_ns=None,
) -> Recording:
    """Append a provenance-bound conditioned window without mutating the input.

    Dimensionless A acts on samples, identically for each selected channel. Covariance covers
    the full sample-major window, including cross-channel terms. Source channel
    indices refer to C-order flattening of each observation. Original streams
    remain shared; subsequent source mutation invalidates the derived binding.

    Availability is explicitly supplied in the source clock or remains unknown.
    Conditioning chains must be composed into one operator against raw data.
    Every model prediction compared with this output must use the same operator.
    """
    recording.validate()
    if (
        not isinstance(output_id, str)
        or not output_id
        or any(s.id == output_id for s in recording.streams)
    ):
        raise ValueError("Output requires a new nonempty stream ID")
    base = _raw_recording(recording)
    sources = [s for s in base.streams if s.id == stream_id]
    if len(sources) != 1:
        raise ValueError("Select one raw observation stream; compose conditioning operators first")
    output = _build(
        base,
        sources[0],
        operator,
        covariance,
        output_id=output_id,
        covariance_provenance=covariance_provenance,
        sample_indices=sample_indices,
        channel_indices=channel_indices,
        available_ns=available_ns,
        source_digest=base.digest(),
    )
    result = Recording(
        recording.id,
        (*recording.streams, output),
        deepcopy(recording.calibrations),
        deepcopy(recording.metadata),
        deepcopy(recording.provenance),
    )
    result.validate()
    return result


def _consistent(actual, expected, name: str) -> None:
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (
        actual.shape != expected.shape
        or actual.dtype.kind not in "fiu"
        or not np.isfinite(actual).all()
    ):
        raise ValueError(f"Invalid reconstructed {name}")
    # Retained JSON covariance may contain signed integers. abs(INT64_MIN)
    # overflows in integer arithmetic, so normalize before computing scales.
    with np.errstate(over="ignore", invalid="ignore"):
        actual = actual.astype(np.float64)
    if not np.isfinite(actual).all():
        raise ValueError(f"Reconstructed {name} exceeds the float64 numerical range")
    scale = np.maximum(np.abs(expected), np.abs(actual))
    tolerance = 128 * MAX_DIMENSION * np.finfo(float).eps
    # Compare each coordinate at its own scale so a large channel cannot hide
    # corruption in a small one. Exact zero must reconstruct as zero. Scale
    # before subtraction to avoid overflow; there is no absolute SI noise floor.
    nonzero = scale != 0
    if np.any(
        np.abs(actual[nonzero] / scale[nonzero] - expected[nonzero] / scale[nonzero]) > tolerance
    ):
        raise ValueError(f"Conditioned {name} does not reproduce from its source")


def validate_conditioned(recording: Recording) -> None:
    """Enforce reserved conditioning metadata before replay/report consumers."""
    derived = [s for s in recording.streams if "conditioning" in s.metadata]
    if not derived:
        return
    metadata_bytes = 0
    for stream in derived:
        descriptor = stream.metadata["conditioning"]
        if (
            not isinstance(descriptor, dict)
            or set(descriptor) != FIELDS
            or descriptor["schema"] != SCHEMA
        ):
            raise ValueError("Conditioning descriptor fields do not match schema")
        metadata_bytes += len(canonical_json(descriptor))
        if metadata_bytes > MAX_JSON_BYTES:
            raise ValueError("Conditioning metadata exceeds 16 MiB in total")
    base = _raw_recording(recording)
    source_digest = base.digest()
    sources = {s.id: s for s in base.streams}
    for stream in derived:
        desc = stream.metadata["conditioning"]
        if (
            desc["source_recording_sha256"] != source_digest
            or not isinstance(desc["source_stream_id"], str)
            or desc["source_stream_id"] not in sources
        ):
            raise ValueError("Conditioning source recording changed or is unavailable")
        expected = _build(
            base,
            sources[desc["source_stream_id"]],
            desc["operator"],
            desc["input_covariance"],
            output_id=stream.id,
            covariance_provenance=desc["covariance_provenance"],
            sample_indices=desc["sample_indices"],
            channel_indices=desc["channel_indices"],
            available_ns=stream.times_ns.get("available"),
            source_digest=source_digest,
        )
        _consistent(stream.values, expected.values, "observations")
        # A reconstruction tolerance must not admit an indefinite or asymmetric
        # supplied covariance, especially along a null direction.
        validate_covariance(desc["output_covariance"])
        _consistent(
            desc["output_covariance"],
            expected.metadata["conditioning"]["output_covariance"],
            "covariance",
        )
        for key in FIELDS - {"output_covariance"}:
            if canonical_json(desc[key]) != canonical_json(expected.metadata["conditioning"][key]):
                raise ValueError(f"Conditioning lineage does not reproduce: {key}")
        if (
            stream.kind != expected.kind
            or stream.units != expected.units
            or stream.frame_ids is not None
            or stream.clock_id != expected.clock_id
            or stream.uncertainty_ns != expected.uncertainty_ns
            or canonical_json(stream.times_ns) != canonical_json(expected.times_ns)
            or stream.values.dtype != np.dtype("float64")
        ):
            raise ValueError("Conditioned stream coordinates or timing differ from their source")
