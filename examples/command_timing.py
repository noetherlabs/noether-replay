"""Analytical timing cases; no instrument or controller performance is measured."""

import json

import numpy as np

from noether_replay.model import Recording, Stream
from noether_replay.timing import qualify_command_timing


def main():
    epoch = 10**18
    available = tuple(epoch + offset for offset in (100, 200, 300, 400, 520, 600))
    effective = tuple(
        None if offset is None else epoch + offset
        for offset in (110, 245, 320, 400, 500, None, 710)
    )
    recording = Recording(
        id="synthetic-command-timing",
        streams=(
            Stream(
                id="wfs",
                kind="wfs_measurements",
                values=np.zeros((6, 2)),
                clock_id="fixture-clock",
                times_ns={"available": available},
                uncertainty_ns=5,
            ),
            Stream(
                id="dm_applied",
                kind="command_applied",
                values=np.zeros((7, 2)),
                clock_id="fixture-clock",
                times_ns={"effective": effective},
                uncertainty_ns=5,
            ),
        ),
        provenance={"source": "synthetic analytical fixture"},
    )
    associations = [
        {
            "applied_index": index,
            "observations": [{"stream_id": "wfs", "sample_index": index}],
        }
        for index in range(6)
    ]
    report = qualify_command_timing(
        recording,
        applied_stream_id="dm_applied",
        associations=associations,
        latency_budget_ns=20,
    )
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
