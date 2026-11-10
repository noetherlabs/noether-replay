"""Readable timing decisions from a synthetic, explicitly bounded two-clock fixture."""

from __future__ import annotations

import hashlib

import numpy as np

from noether_replay.model import Recording, Stream
from noether_replay.timing import qualify_command_timing


def synthetic_inputs():
    """Return asserted analytical inputs, not measured instrument synchronization."""
    recording = Recording(
        "synthetic-instrument-timing",
        (
            Stream(
                "wfs", "wfs_measurements", np.zeros((5, 1)), "rad",
                (10, 11, 12, 13, 14), "camera",
                {"available": (1000, 1100, 1200, None, 1500)}, 3,
            ),
            Stream(
                "dm", "command_applied", np.zeros((6, 1)), "m",
                (20, 21, 22, 23, 24, 25), "actuator",
                {"effective": (5100, 5220, 5310, 5420, 5520, 5600)}, 2,
            ),
        ),
        provenance={"source": "synthetic analytical fixture; no instrument measurement"},
    )
    contract = b"Synthetic coordinate contract: offset 4000 ns, total error <= 4 ns."
    conversion = {
        "schema": "noether-replay-clock-conversion/1",
        "recording_sha256": recording.digest(),
        "source_clock_id": "camera",
        "target_clock_id": "actuator",
        "source_valid_from_ns": 900,
        "source_valid_until_ns": 1400,
        "offset_ns": 4000,
        "max_mapping_error_ns": 4,
        "bound_scope": "uniform_over_source_validity_interval",
        "evidence_sha256": {"synthetic-contract": hashlib.sha256(contract).hexdigest()},
        "provenance": {
            "source": contract.decode("ascii"),
            "clock_instances": ["synthetic-camera-1", "synthetic-actuator-1"],
            "bound_rationale": "asserted total mapping error over the whole validity interval",
        },
    }
    associations = [
        {"applied_index": command,
         "observations": [{"stream_id": "wfs", "sample_index": sample}]}
        for command, sample in ((0, 0), (1, 1), (2, 2), (3, 3), (5, 4))
    ]
    return recording, {
        "applied_stream_id": "dm",
        "associations": associations,
        "latency_budget_ns": 110,
        "clock_conversion": conversion,
    }


def render_review(report):
    """Present every command in this example's existing timing report."""
    configuration = report["configuration"]
    conversion = configuration["clock_conversion"]
    lines = [
        "# Synthetic instrument timing review",
        "",
        "Analytical fixture only; no hardware timing or synchronization was measured.",
        "The decision is whether each availability-to-effect interval meets the",
        "illustrative budget. This is not a full control-loop deadline or safety decision.",
        "",
        f"Recording SHA-256: `{report['recording_sha256']}`",
        f"Configuration SHA-256: `{report['configuration_sha256']}`",
        f"Clock receipt SHA-256: `{configuration['clock_conversion_sha256']}`",
        "",
        f"Budget: {configuration['latency_budget_ns']} ns. "
        f"Clock mapping: {conversion['source_clock_id']} to {conversion['target_clock_id']}, "
        f"offset {conversion['offset_ns']} ns, total mapping error at most "
        f"{conversion['max_mapping_error_ns']} ns.",
        f"Mapping validity: [{conversion['source_valid_from_ns']}, "
        f"{conversion['source_valid_until_ns']}) ns in the source clock.",
        "",
        "| Command index (frame ID) | Latency interval (ns) | Causal order | Budget | Evidence gaps |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["commands"]:
        latency = row["latency_interval_ns"]
        interval = "not established" if latency is None else f"[{latency[0]}, {latency[1]}]"
        reasons = ", ".join(row["reasons"]) or "none"
        lines.append(
            f"| {row['applied_index']} ({row['frame_id']}) | {interval} | "
            f"{row['causal_order']} | {row['budget']} | {reasons} |"
        )
    counts = report["summary"]["budget_counts"]
    lines.extend([
        "",
        f"All {report['summary']['command_count']} commands are retained: "
        + ", ".join(f"{count} {status}" for status, count in counts.items() if count)
        + ". These hand-constructed cases do not estimate instrument failure rates.",
        "",
        "For the ambiguous budget, obtain tighter independently supported timing bounds",
        "before deciding. For missing availability or associations, obtain the native",
        "event meaning and dependency record. For expired conversion evidence, obtain",
        "a bound covering that clock epoch; do not extrapolate or silently drop the row.",
        "",
        "Generic AOT timestamps and command records do not establish availability or",
        "actuator effect. A clock receipt cannot promote them to applied events.",
        "",
        "## Interpretation limits",
        "",
        *[f"- {limitation}" for limitation in report["limitations"]],
    ])
    return "\n".join(lines) + "\n"


def main():
    recording, arguments = synthetic_inputs()
    # Native event evidence -> bounded conversion -> existing qualifier -> all rows.
    report = qualify_command_timing(recording, **arguments)
    print(render_review(report), end="")


if __name__ == "__main__":
    main()
