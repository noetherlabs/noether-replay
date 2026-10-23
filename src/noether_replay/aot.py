"""Bounded local AOT FITS import with conservative event semantics."""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
import stat
import struct
import warnings
from datetime import date, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from .model import MAX_ARRAY_BYTES, MAX_JSON_BYTES, Recording, Stream, array_digest, canonical_json

ADAPTER_VERSION = "1"
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_HDUS = 1024
MAX_TABLE_ELEMENTS = 2_000_000
_TFORM = re.compile(r"(\d*)([PQ]?)([LXBIJKAEDCM])(?:\((\d*)\))?", re.IGNORECASE)
_WIDTHS = {"L": 1, "B": 1, "I": 2, "J": 4, "K": 8, "A": 1, "E": 4, "D": 8, "C": 8, "M": 16}


def _snapshot(path: str | Path) -> bytes:
    """Read one regular local file once; parse and hash precisely these bytes."""
    path = Path(path)
    if path.suffix.lower() not in {".fits", ".fit"}:
        raise ValueError("AOT input must be a local uncompressed .fits or .fit file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_FILE_BYTES:
            raise ValueError("AOT input must be a regular file no larger than 256 MiB")
        payload = handle.read(MAX_FILE_BYTES + 1)
        after = os.fstat(handle.fileno())
    if (
        len(payload) != before.st_size
        or len(payload) > MAX_FILE_BYTES
        or (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    ):
        raise ValueError("AOT source changed while its byte snapshot was acquired")
    if not payload.startswith(b"SIMPLE  ="):
        raise ValueError(
            "Only uncompressed FITS files are supported; compressed inputs are rejected"
        )
    return payload


def _integer(header: Any, key: str, default: int | None = None) -> int:
    value = header.get(key, default)
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"FITS {key} must be an integer")
    return int(value)


def _preflight(payload: bytes) -> None:
    """Inspect headers and variable-array descriptors before aotpy loads data.

    This includes unused images. Heap aliases count repeatedly toward the table
    budget, preventing a small on-disk heap from expanding into many copies.
    """
    from astropy.io import fits

    image_bytes = 0
    table_elements = 0
    with fits.open(
        io.BytesIO(payload),
        memmap=False,
        lazy_load_hdus=True,
        do_not_scale_image_data=True,
        disable_image_compression=True,
    ) as hdus:
        for index, hdu in enumerate(hdus):
            if index >= MAX_HDUS:
                raise ValueError("FITS HDU count exceeds the supported bound")
            header = hdu.header
            if header.get("ZIMAGE", False) or isinstance(hdu, fits.CompImageHDU):
                raise ValueError("Compressed FITS images are unsupported")
            if not isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.BinTableHDU)):
                raise ValueError("Only ordinary FITS images and binary tables are supported")
            # Structural duplicates make the layout ambiguous to different readers.
            structural = [
                key
                for key in header
                if key in {"BITPIX", "NAXIS", "PCOUNT", "GCOUNT", "THEAP", "TFIELDS"}
                or re.fullmatch(r"(?:NAXIS|TFORM)\d+", key)
            ]
            if len(structural) != len(set(structural)):
                raise ValueError("Duplicate FITS structural keywords")
            bitpix = _integer(header, "BITPIX")
            naxis = _integer(header, "NAXIS")
            if bitpix not in {8, 16, 32, 64, -32, -64} or not 0 <= naxis <= 6:
                raise ValueError("Unsupported FITS array type or dimension")
            dims = [_integer(header, f"NAXIS{i}") for i in range(1, naxis + 1)]
            if any(dimension < 0 for dimension in dims):
                raise ValueError("FITS dimensions cannot be negative")
            count = math.prod(dims) if dims else 0
            pcount = _integer(header, "PCOUNT", 0)
            gcount = _integer(header, "GCOUNT", 1)
            if pcount < 0 or gcount != 1:
                raise ValueError("Unsupported FITS heap size or random groups")
            data_size = count * (abs(bitpix) // 8) + pcount
            info = hdu.fileinfo()
            data_start = info["datLoc"]
            if data_size > MAX_FILE_BYTES or data_start + data_size > len(payload):
                raise ValueError("FITS data extent exceeds the source file or supported bound")
            if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU)):
                if pcount:
                    raise ValueError("Image heaps are unsupported")
                # Scaling integer data can produce float64: admit that upper bound.
                expanded = count * max(8, abs(bitpix) // 8)
                image_bytes += expanded
                if expanded > MAX_ARRAY_BYTES or image_bytes > MAX_FILE_BYTES:
                    raise ValueError("FITS decoded image allocation exceeds the supported bound")
                continue
            if naxis != 2 or bitpix != 8:
                raise ValueError("Malformed FITS binary table dimensions")
            row_size, rows = dims
            fields = _integer(header, "TFIELDS")
            if not 0 <= fields <= 1000 or rows * fields > MAX_TABLE_ELEMENTS:
                raise ValueError("FITS table exceeds the supported element bound")
            heap_start = _integer(header, "THEAP", row_size * rows)
            if not row_size * rows <= heap_start <= data_size:
                raise ValueError("FITS heap starts outside the declared table extent")
            column_offset = 0
            for field in range(1, fields + 1):
                match = _TFORM.fullmatch(str(header.get(f"TFORM{field}", "")).strip())
                if match is None:
                    raise ValueError("Unsupported FITS table column format")
                repeat = int(match[1] or 1)
                pointer, code = match[2].upper(), match[3].upper()
                if pointer:
                    if repeat != 1:
                        raise ValueError(
                            "Only one FITS variable-array descriptor per cell is supported"
                        )
                    descriptor_size = 8 if pointer == "P" else 16
                    width = repeat * descriptor_size
                    if column_offset + width > row_size:
                        raise ValueError("FITS column exceeds the declared row width")
                    for row in range(rows):
                        for item in range(repeat):
                            offset = (
                                data_start + row * row_size + column_offset + item * descriptor_size
                            )
                            length, start = struct.unpack_from(
                                ">ii" if pointer == "P" else ">qq", payload, offset
                            )
                            if length < 0 or start < 0:
                                raise ValueError(
                                    "FITS variable-array descriptor cannot be negative"
                                )
                            byte_count = (
                                (length + 7) // 8 if code == "X" else length * _WIDTHS[code]
                            )
                            if heap_start + start + byte_count > data_size:
                                raise ValueError("FITS variable-array descriptor exceeds its heap")
                            table_elements += length
                            if table_elements > MAX_TABLE_ELEMENTS:
                                raise ValueError(
                                    "FITS expanded table exceeds the supported element bound"
                                )
                else:
                    width = (repeat + 7) // 8 if code == "X" else repeat * _WIDTHS[code]
                    table_elements += repeat * rows
                    if table_elements > MAX_TABLE_ELEMENTS:
                        raise ValueError("FITS expanded table exceeds the supported element bound")
                column_offset += width
                if column_offset > row_size:
                    raise ValueError("FITS column exceeds the declared row width")
            if column_offset != row_size:
                raise ValueError("FITS columns do not match the declared row width")


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite": str(value)}
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, complex):
        return {"real": _json_value(value.real), "imaginary": _json_value(value.imag)}
    raise ValueError(f"Unsupported AOT metadata value type: {type(value).__name__}")


def _metadata(items: Any) -> list[dict]:
    return [
        {"key": item.key, "value": _json_value(item.value), "comment": item.comment}
        for item in items
    ]


def _image_reference(image: Any) -> dict | None:
    if image is None:
        return None
    values = np.asarray(image.data)
    reference = {
        "name": image.name,
        "dtype": values.dtype.str,
        "shape": list(values.shape),
        "array_sha256": array_digest(values),
        "unit": image.unit,
    }
    # Small filter coefficients are useful collection-controller metadata.
    if values.size <= 4096:
        reference["values"] = _json_value(values.tolist())
    return reference


def _stream(image: Any, stream_id: str, kind: str, context: dict) -> Stream:
    values = np.asarray(image.data)
    if values.ndim < 1 or values.ndim > 6 or values.dtype.kind not in "biuf":
        raise ValueError("AOT telemetry must be a real numeric array with a sample axis")
    metadata = {
        "aot_image_name": image.name,
        "source_metadata": _metadata(image.metadata),
        **context,
    }
    frame_ids = None
    if image.time is not None:
        time = image.time
        metadata["aot_time"] = {
            "uid": time.uid,
            "timestamps_seconds": _json_value(time.timestamps),
            "frame_numbers": _json_value(time.frame_numbers),
            "event_semantics": "unspecified",
        }
        numbers = time.frame_numbers
        if (
            numbers
            and len(numbers) == len(values)
            and all(
                isinstance(number, (int, float, np.integer, np.floating))
                and not isinstance(number, (bool, np.bool_))
                and math.isfinite(number)
                and abs(number) <= 2**53
                and int(number) == number
                for number in numbers
            )
        ):
            frame_ids = tuple(int(number) for number in numbers)
        elif numbers:
            metadata["frame_identity_issue"] = (
                "Source frame numbers are incomplete or not exact integers"
            )
    stream = Stream(
        stream_id, kind, values.copy(), image.unit or None, frame_ids, metadata=metadata
    )
    stream.validate()
    return stream


def read_aot(path: str | Path) -> Recording:
    """Import local AOT FITS through aotpy without inferring event times.

    Source timestamps remain seconds in metadata. Calibration matrices and loop
    filters do not establish a validity interval or physical command application.
    """
    import aotpy
    from aotpy.io.fits._file import AOTVerificationException

    payload = _snapshot(path)
    _preflight(payload)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            system = aotpy.AOSystem.read_from_file(
                io.BytesIO(payload),
                file_type="fits",
                externals="disallow",
                memmap=False,
                lazy_load_hdus=False,
            )
        except (
            ValueError,
            RuntimeError,
            TypeError,
            KeyError,
            IndexError,
            OverflowError,
            AOTVerificationException,
        ) as exc:
            raise ValueError(f"AOT parsing failed: {exc}") from exc

    def telemetry_images():
        for sensor in system.wavefront_sensors:
            if sensor.measurements is not None:
                yield sensor.measurements
            if sensor.detector is not None and sensor.detector.pixel_intensities is not None:
                yield sensor.detector.pixel_intensities
        for loop in system.loops:
            if loop.commands is not None:
                yield loop.commands
        for camera in system.scoring_cameras:
            if camera.detector is not None and camera.detector.pixel_intensities is not None:
                yield camera.detector.pixel_intensities

    # AOT allows many objects to reference the same internal image. Count every
    # prospective stream before copying any array, including those aliases.
    retained_bytes = 0
    for image in telemetry_images():
        values = np.asarray(image.data)
        if values.nbytes > MAX_ARRAY_BYTES:
            raise ValueError("AOT stream exceeds the supported array byte bound")
        retained_bytes += values.nbytes
        if retained_bytes > 2 * MAX_ARRAY_BYTES:
            raise ValueError("AOT retained streams exceed the aggregate array byte bound")
    streams = []
    controllers = []
    metadata_bytes = 0

    def charge_metadata(value: Any) -> None:
        nonlocal metadata_bytes
        metadata_bytes += len(canonical_json(value))
        if metadata_bytes > MAX_JSON_BYTES:
            raise ValueError("AOT expanded metadata exceeds the 16 MiB output bound")

    def append_stream(stream: Stream) -> None:
        charge_metadata(stream.manifest())
        streams.append(stream)

    for sensor in system.wavefront_sensors:
        context = {"sensor_uid": sensor.uid, "sensor_type": type(sensor).__name__}
        if sensor.measurements is not None:
            append_stream(
                _stream(
                    sensor.measurements,
                    f"wfs:{sensor.uid}:measurements",
                    "wfs_measurements",
                    context,
                )
            )
        if sensor.detector is not None and sensor.detector.pixel_intensities is not None:
            detector = sensor.detector
            append_stream(
                _stream(
                    detector.pixel_intensities,
                    f"wfs:{sensor.uid}:pixels",
                    "wfs_pixels",
                    {
                        **context,
                        "detector_uid": detector.uid,
                        "frame_rate_hz": _json_value(detector.frame_rate),
                        "integration_time_seconds": _json_value(detector.integration_time),
                    },
                )
            )
    for loop in system.loops:
        controller = {
            "loop_uid": loop.uid,
            "loop_type": type(loop).__name__,
            "closed": loop.closed,
            "commanded_corrector_uid": loop.commanded_corrector.uid,
            "input_sensor_uid": getattr(getattr(loop, "input_sensor", None), "uid", None),
            "frame_rate_hz": _json_value(loop.framerate),
            "delay_frames": _json_value(loop.delay),
            "time_filter_numerator": _image_reference(loop.time_filter_num),
            "time_filter_denominator": _image_reference(loop.time_filter_den),
        }
        charge_metadata(controller)
        controllers.append(controller)
        if loop.commands is not None:
            append_stream(
                _stream(
                    loop.commands,
                    f"loop:{loop.uid}:commands",
                    "command_recorded",
                    {
                        "collection_controller": controller,
                        "command_semantics": "source command sequence; actuator effect unestablished",
                    },
                )
            )
    for camera in system.scoring_cameras:
        if camera.detector is not None and camera.detector.pixel_intensities is not None:
            append_stream(
                _stream(
                    camera.detector.pixel_intensities,
                    f"science:{camera.uid}:pixels",
                    "science",
                    {"camera_uid": camera.uid, "detector_uid": camera.detector.uid},
                )
            )
    source_hash = hashlib.sha256(payload).hexdigest()
    metadata = {
        "ao_mode": system.ao_mode,
        "configuration": system.config,
        "date_beginning": _json_value(system.date_beginning),
        "date_end": _json_value(system.date_end),
        "source_metadata": _metadata(system.metadata),
        "reader_warnings": [str(item.message) for item in caught],
    }
    charge_metadata(metadata)
    metadata["collection_controllers"] = controllers
    recording = Recording(
        system.name or f"aot-{source_hash[:16]}",
        tuple(streams),
        metadata=metadata,
        provenance={
            "format": "AOT FITS",
            "source_sha256": source_hash,
            "source_bytes": len(payload),
            "adapter": "noether_replay.aot",
            "adapter_version": ADAPTER_VERSION,
            "aotpy_version": version("aotpy"),
            "timing_policy": "Source timestamps retained; event times and clock uncertainty not inferred",
        },
    )
    recording.validate()
    return recording
