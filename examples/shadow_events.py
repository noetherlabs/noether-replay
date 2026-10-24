"""Synthetic timing-aware shadow integration; no optical performance evaluation."""

import hashlib
import json

import numpy as np

from noether_replay import Recording, Stream
from noether_replay.replay import ShadowObservation, run_shadow


def main() -> dict:
    starts = (0, 1_000_000, 4_000_000, 5_000_000)
    stream = Stream(
        "wfs", "wfs_measurements", np.array([[0.001], [0.002], [0.002], [0.003]]),
        units="rad", frame_ids=(0, 1, 4, 5), clock_id="synthetic-instrument",
        times_ns={
            "exposure_start": starts,
            "exposure_end": tuple(t + 100_000 for t in starts),
            "available": tuple(t + 200_000 for t in starts),
        },
        uncertainty_ns=0,
    )
    epochs = tuple(
        {
            "id": name, "clock_id": stream.clock_id, "start_ns": start, "end_ns": end,
            "stream_ids": [stream.id],
            "products": {"response": hashlib.sha256(name.encode()).hexdigest()},
        }
        for name, start, end in (("synthetic-a", 0, 5_000_000), ("synthetic-b", 5_000_000, 6_000_000))
    )
    recording = Recording("irregular-events", (stream,), epochs, provenance={"kind": "synthetic"})
    previous_time = None
    previous_epoch = None
    integral = np.zeros(1)
    decisions = []

    def policy(observation: ShadowObservation) -> np.ndarray:
        nonlocal previous_time, previous_epoch, integral
        elapsed_ns = None if previous_time is None else observation.available_ns - previous_time
        reset = observation.calibration_epoch_id != previous_epoch
        if reset:
            integral[:] = 0
        elif elapsed_ns is not None:
            # Declared software example: right-endpoint integration, gain 1/s.
            integral += observation.values * (elapsed_ns * 1e-9)
        decisions.append(
            {
                "sample_index": observation.sample_index,
                "frame_id": observation.frame_id,
                "elapsed_available_ns": elapsed_ns,
                "calibration_epoch_id": observation.calibration_epoch_id,
                "state_reset": reset,
            }
        )
        previous_time = observation.available_ns
        previous_epoch = observation.calibration_epoch_id
        return -integral.copy()

    result = run_shadow(
        recording, stream.id, policy, policy_id="synthetic-time-integral",
        policy_version="1", policy_input="event",
    )
    return {
        "evidence": "Synthetic callback example; no instrument or controller-performance result",
        "policy_decisions": decisions,
        "shadow": result,
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
