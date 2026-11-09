"""Content-bound external clock assertions; no estimation or synchronization."""

from __future__ import annotations

import hashlib
import json
import math
import re

from .model import _integer, canonical_json

SCHEMA = "noether-replay-clock-conversion/1"
_FIELDS = {
    "schema", "recording_sha256", "source_clock_id", "target_clock_id",
    "source_valid_from_ns", "source_valid_until_ns", "offset_ns",
    "max_mapping_error_ns", "bound_scope", "evidence_sha256", "provenance",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _bounded_json(value: object, limit: int) -> bytes:
    """Reject excessive metadata before canonical normalization allocates a copy.

    Count canonical UTF-8 bytes incrementally, with a 64-level nesting bound.
    Only containers on the current path are tracked: shared acyclic values are
    valid, but each occurrence consumes the budget of its serialized expansion.
    """
    remaining = limit
    active = set()

    def charge(size: int) -> None:
        nonlocal remaining
        remaining -= size
        if remaining < 0:
            raise ValueError("Clock conversion exceeds the JSON metadata bound")

    def string(text: str) -> None:
        charge(2)
        if len(text) > remaining:
            raise ValueError("Clock conversion exceeds the JSON metadata bound")
        for char in text:
            point = ord(char)
            if char in '\\"\b\t\n\f\r':
                charge(2)
            elif point < 0x20:
                charge(6)
            elif point < 0x80:
                charge(1)
            elif point < 0x800:
                charge(2)
            elif 0xD800 <= point <= 0xDFFF:
                raise ValueError("Clock conversion strings must contain valid Unicode")
            else:
                charge(3 if point < 0x10000 else 4)

    def visit(item: object, depth: int) -> None:
        if depth > 64:
            raise ValueError("Clock conversion metadata exceeds 64 nesting levels")
        if isinstance(item, str):
            string(item)
        elif item is None:
            charge(4)
        elif isinstance(item, bool):
            charge(4 if item else 5)
        elif _integer(item):
            # A bit-length lower bound avoids expanding enormous integer text.
            number = int(item)
            if number.bit_length() > (remaining + 1) * 4:
                raise ValueError("Clock conversion exceeds the JSON metadata bound")
            try:
                charge(len(str(number)))
            except ValueError as exc:
                raise ValueError("Clock conversion integer exceeds JSON limits") from exc
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("Clock conversion metadata must be finite JSON")
            charge(len(json.dumps(item, allow_nan=False)))
        elif isinstance(item, (dict, list, tuple)):
            identity = id(item)
            if identity in active:
                raise ValueError("Clock conversion metadata must not contain cycles")
            charge(2 + max(0, len(item) - 1))
            active.add(identity)
            try:
                if isinstance(item, dict):
                    charge(len(item))  # Colons.
                    for key, child in item.items():
                        if not isinstance(key, str):
                            raise ValueError("Clock conversion JSON object keys must be strings")
                        string(key)
                        visit(child, depth + 1)
                else:
                    for child in item:
                        visit(child, depth + 1)
            finally:
                active.remove(identity)
        else:
            raise ValueError("Clock conversion must contain only canonical JSON values")

    visit(value, 0)
    result = canonical_json(value)
    if len(result) > limit:
        raise ValueError("Clock conversion exceeds the JSON metadata bound")
    return result


def normalize_conversion(
    receipt: dict, recording_sha256: str, target_clock_id: str | None, limit: int,
) -> tuple[dict, str]:
    """Validate and detach an assertion without authenticating its physical bound."""
    encoded = _bounded_json(receipt, limit)
    detached = json.loads(encoded)
    if not isinstance(detached, dict) or set(detached) != _FIELDS:
        raise ValueError("Clock conversion receipt fields do not match its schema")
    if detached["schema"] != SCHEMA:
        raise ValueError("Unsupported clock conversion schema")
    if detached["recording_sha256"] != recording_sha256:
        raise ValueError("Clock conversion does not bind this recording")
    source, target = detached["source_clock_id"], detached["target_clock_id"]
    if not all(isinstance(clock, str) and clock for clock in (source, target)):
        raise ValueError("Clock conversion requires two named clocks")
    if source == target or target != target_clock_id:
        raise ValueError("Clock conversion must map a distinct source to the applied-command clock")
    for key in ("source_valid_from_ns", "source_valid_until_ns", "offset_ns", "max_mapping_error_ns"):
        if not _integer(detached[key]):
            raise ValueError("Clock conversion bounds must be integer nanoseconds, not booleans")
    if detached["source_valid_until_ns"] <= detached["source_valid_from_ns"]:
        raise ValueError("Clock conversion validity must be a nonempty half-open interval")
    if detached["max_mapping_error_ns"] < 0:
        raise ValueError("Clock conversion mapping error must be nonnegative")
    if detached["bound_scope"] != "uniform_over_source_validity_interval":
        raise ValueError("Clock conversion requires a uniform total mapping-error bound")
    evidence = detached["evidence_sha256"]
    if not isinstance(evidence, dict) or not evidence or any(
        not name or not isinstance(digest, str) or not _SHA256.fullmatch(digest)
        for name, digest in evidence.items()
    ):
        raise ValueError("Clock conversion evidence must be a nonempty named SHA-256 map")
    if not isinstance(detached["provenance"], dict) or not detached["provenance"]:
        raise ValueError("Clock conversion requires nonempty provenance")
    return detached, hashlib.sha256(encoded).hexdigest()


def convert_interval(interval: list[int], receipt: dict) -> list[int] | None:
    """Translate only when the entire native uncertainty support is covered."""
    lower, upper = interval
    if lower < receipt["source_valid_from_ns"] or upper >= receipt["source_valid_until_ns"]:
        return None
    offset, error = receipt["offset_ns"], receipt["max_mapping_error_ns"]
    return [lower + offset - error, upper + offset + error]
