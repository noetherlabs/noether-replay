"""Small, explicit in-memory contract for observational optical telemetry.

Times are integer nanoseconds in a named clock. A missing time is not inferred
from sampling frequency. Source timestamps with unspecified event semantics may
be retained in metadata without being promoted to observation availability.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "noether-replay/1"
MAX_ARRAY_BYTES = 256 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
KINDS = frozenset(
    {
        "wfs_measurements",
        "wfs_pixels",
        "command_recorded",
        "command_requested",
        "command_applied",
        "science",
        "housekeeping",
        "other",
    }
)
TIME_FIELDS = (
    "exposure_start",
    "exposure_end",
    "available",
    "completed",
    "issued",
    "effective",
    "recorded",
)


def canonical_json(value: Any) -> bytes:
    """Canonical UTF-8 JSON; reject nonfinite metadata instead of changing it."""

    def normalize(item: Any) -> Any:
        if isinstance(item, np.integer) and not isinstance(item, np.bool_):
            return int(item)
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("Metadata object keys must be strings")
            return {key: normalize(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [normalize(val) for val in item]
        return item

    return json.dumps(
        normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(values: np.ndarray) -> str:
    """Bind dtype, shape and exact C-order bytes, including nonfinite values."""
    digest = hashlib.sha256(
        canonical_json({"dtype": values.dtype.str, "shape": list(values.shape)})
    )
    # Buffered iteration avoids a second full array for noncontiguous FITS views.
    for chunk in np.nditer(
        values, flags=["external_loop", "buffered", "zerosize_ok"], order="C", buffersize=65536
    ):
        digest.update(chunk.tobytes())
    return digest.hexdigest()


def _integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


@dataclass
class Stream:
    id: str
    kind: str
    values: np.ndarray
    units: str | None = None
    frame_ids: tuple[int, ...] | None = None
    clock_id: str | None = None
    times_ns: dict[str, tuple[int | None, ...]] = field(default_factory=dict)
    uncertainty_ns: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if (
            not isinstance(self.id, str)
            or not self.id
            or not isinstance(self.kind, str)
            or self.kind not in KINDS
        ):
            raise ValueError("Stream requires a nonempty ID and a supported kind")
        if not isinstance(self.values, np.ndarray) or self.values.dtype.kind not in "biuf":
            raise ValueError("Stream values must be a real numeric NumPy array")
        if self.values.ndim < 1 or self.values.ndim > 6 or self.values.nbytes > MAX_ARRAY_BYTES:
            raise ValueError("Stream array shape or byte size exceeds the supported bound")
        if self.units is not None and (not isinstance(self.units, str) or not self.units):
            raise ValueError("Units must be nonempty text or None")
        if self.clock_id is not None and (not isinstance(self.clock_id, str) or not self.clock_id):
            raise ValueError("Clock ID must be nonempty text or None")
        if self.uncertainty_ns is not None and (
            not _integer(self.uncertainty_ns) or self.uncertainty_ns < 0
        ):
            raise ValueError("Clock uncertainty must be nonnegative integer nanoseconds or None")
        size = len(self.values)
        if self.frame_ids is not None:
            if (
                not isinstance(self.frame_ids, (tuple, list))
                or len(self.frame_ids) != size
                or any(not _integer(x) for x in self.frame_ids)
            ):
                raise ValueError("Frame IDs must contain one integer per sample")
        if not isinstance(self.times_ns, dict) or set(self.times_ns) - set(TIME_FIELDS):
            raise ValueError("Unsupported event-time field")
        if self.times_ns and self.clock_id is None:
            raise ValueError("Event times require a named clock")
        for times in self.times_ns.values():
            if (
                not isinstance(times, (tuple, list))
                or len(times) != size
                or any(x is not None and not _integer(x) for x in times)
            ):
                raise ValueError("Event times must contain one integer or None per sample")
        if not isinstance(self.metadata, dict):
            raise ValueError("Stream metadata must be an object")
        canonical_json(self.metadata)

    def manifest(self) -> dict[str, Any]:
        self.validate()
        return {
            "id": self.id,
            "kind": self.kind,
            "units": self.units,
            "shape": list(self.values.shape),
            "dtype": self.values.dtype.str,
            "array_sha256": array_digest(self.values),
            "frame_ids": self.frame_ids,
            "clock_id": self.clock_id,
            "times_ns": self.times_ns,
            "uncertainty_ns": self.uncertainty_ns,
            "metadata": self.metadata,
        }


@dataclass
class Recording:
    id: str
    streams: tuple[Stream, ...]
    calibrations: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Recording requires a nonempty ID")
        if not isinstance(self.streams, (tuple, list)) or any(
            not isinstance(s, Stream) for s in self.streams
        ):
            raise ValueError("Recording streams must be a sequence of Stream objects")
        if not isinstance(self.calibrations, (tuple, list)):
            raise ValueError("Calibration epochs must be a sequence")
        if not isinstance(self.metadata, dict) or not isinstance(self.provenance, dict):
            raise ValueError("Recording metadata and provenance must be objects")
        for stream in self.streams:
            stream.validate()
        ids = [stream.id for stream in self.streams]
        if len(ids) != len(set(ids)):
            raise ValueError("Stream IDs must be unique")
        if sum(stream.values.nbytes for stream in self.streams) > 2 * MAX_ARRAY_BYTES:
            raise ValueError("Recording exceeds the supported total array size")
        epoch_ids = set()
        for epoch in self.calibrations:
            required = {"id", "clock_id", "start_ns", "end_ns", "stream_ids", "products"}
            if not isinstance(epoch, dict) or set(epoch) != required:
                raise ValueError("Calibration epoch fields do not match schema")
            if not isinstance(epoch["id"], str) or not epoch["id"] or epoch["id"] in epoch_ids:
                raise ValueError("Calibration epoch IDs must be unique nonempty strings")
            epoch_ids.add(epoch["id"])
            if not isinstance(epoch["clock_id"], str) or not epoch["clock_id"]:
                raise ValueError("Calibration epoch requires a named clock")
            if (
                not all(_integer(epoch[key]) for key in ("start_ns", "end_ns"))
                or epoch["start_ns"] >= epoch["end_ns"]
            ):
                raise ValueError(
                    "Calibration interval must be a nonempty half-open integer interval"
                )
            if (
                not isinstance(epoch["stream_ids"], list)
                or not epoch["stream_ids"]
                or any(x not in ids for x in epoch["stream_ids"])
            ):
                raise ValueError("Calibration must reference existing streams")
            products = epoch["products"]
            if not isinstance(products, dict) or not products:
                raise ValueError("Calibration products require named SHA-256 digests")
            for name, digest in products.items():
                if (
                    not name
                    or not isinstance(digest, str)
                    or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                ):
                    raise ValueError("Invalid calibration product digest")
        canonical_json(self.calibrations)
        canonical_json(self.metadata)
        canonical_json(self.provenance)

    def manifest(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": SCHEMA,
            "id": self.id,
            "streams": [s.manifest() for s in sorted(self.streams, key=lambda s: s.id)],
            "calibrations": sorted(self.calibrations, key=lambda e: e["id"]),
            "metadata": self.metadata,
            "provenance": self.provenance,
        }

    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.manifest())).hexdigest()


def load_recording(path: str | Path) -> Recording:
    """Load a bounded, self-contained JSON fixture. No paths or URLs are resolved."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Expected a regular JSON file no larger than 16 MiB")
    with path.open("rb") as handle:
        payload = handle.read(MAX_JSON_BYTES + 1)
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError("JSON fixture exceeds 16 MiB")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    raw = json.loads(
        payload,
        object_pairs_hook=unique_object,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant: {x}")),
    )
    canonical_json(raw)
    expected = {"schema", "id", "streams", "calibrations", "metadata", "provenance"}
    if not isinstance(raw, dict) or set(raw) != expected or raw["schema"] != SCHEMA:
        raise ValueError("Unsupported recording schema or fields")
    if not isinstance(raw["streams"], list) or not isinstance(raw["calibrations"], list):
        raise ValueError("Streams and calibrations must be JSON arrays")
    streams = []
    allowed = {
        "id",
        "kind",
        "values",
        "dtype",
        "units",
        "frame_ids",
        "clock_id",
        "times_ns",
        "uncertainty_ns",
        "metadata",
    }
    for item in raw["streams"]:
        if not isinstance(item, dict) or set(item) != allowed:
            raise ValueError("Unexpected stream fields")
        if not isinstance(item["times_ns"], dict):
            raise ValueError("Event-time fields must be an object")
        if any(not isinstance(times, list) for times in item["times_ns"].values()):
            raise ValueError("Event-time samples must be JSON arrays")
        if item["frame_ids"] is not None and not isinstance(item["frame_ids"], list):
            raise ValueError("Frame identifiers must be a JSON array or null")
        dtype = np.dtype(item["dtype"])
        if dtype.kind not in "biuf":
            raise ValueError("Only real numeric array types are supported")

        def validate_values(value: Any) -> None:
            if isinstance(value, list):
                for member in value:
                    validate_values(member)
            elif dtype.kind == "b":
                if not isinstance(value, bool):
                    raise ValueError("Boolean arrays require JSON boolean values")
            elif dtype.kind in "iu":
                bounds = np.iinfo(dtype)
                if not _integer(value) or not bounds.min <= value <= bounds.max:
                    raise ValueError("Integer arrays require in-range JSON integers")
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Floating arrays require JSON numeric values")

        validate_values(item["values"])
        values = np.asarray(item["values"], dtype=dtype)
        if not np.isfinite(values).all():
            raise ValueError("JSON numeric values overflow the declared dtype")
        streams.append(
            Stream(
                item["id"],
                item["kind"],
                values,
                item["units"],
                None if item["frame_ids"] is None else tuple(item["frame_ids"]),
                item["clock_id"],
                {k: tuple(v) for k, v in item["times_ns"].items()},
                item["uncertainty_ns"],
                item["metadata"],
            )
        )
    recording = Recording(
        raw["id"], tuple(streams), tuple(raw["calibrations"]), raw["metadata"], raw["provenance"]
    )
    recording.validate()
    return recording
