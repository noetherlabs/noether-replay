"""A fast command can still depend on an old exposure; analytical fixture only."""

import json

import numpy as np

from noether_replay.conditioned import condition_observations
from noether_replay.freshness import qualify_input_freshness
from noether_replay.model import Recording, Stream


def main():
    recording = Recording(
        "synthetic-fast-command-stale-input",
        (
            Stream(
                "wfs", "wfs_measurements", np.array([[1.0], [2.0]]), "rad",
                clock_id="fixture-clock", uncertainty_ns=0,
                times_ns={"exposure_start": (0, 90), "exposure_end": (10, 95),
                          "available": (11, 96)},
            ),
            Stream(
                "dm", "command_applied", np.zeros((2, 1)),
                clock_id="fixture-clock", uncertainty_ns=0,
                times_ns={"effective": (100, 100)},
            ),
        ),
        provenance={"source": "synthetic analytical fixture; times in nanoseconds"},
    )
    recording = condition_observations(
        recording, "wfs", [[0.5, 0.5]], np.eye(2), output_id="mean",
        covariance_provenance={"source": "analytical unit noise"}, available_ns=(97,),
    )
    report = qualify_input_freshness(
        recording, applied_stream_id="dm", max_support_age_ns=20, latency_budget_ns=5,
        associations=[
            {"applied_index": 0, "observations": [
                {"stream_id": "wfs", "sample_index": 0},
                {"stream_id": "wfs", "sample_index": 1},
            ]},
            {"applied_index": 1, "observations": [{"stream_id": "mean", "sample_index": 0}]},
        ],
    )
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
