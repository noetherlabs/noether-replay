"""Synthetic multi-channel conditioning with explicit timing and source epochs."""

import json
from pathlib import Path

import numpy as np

from noether_replay import load_recording
from noether_replay.conditioned import condition_observations
from noether_replay.conditioning import compose_operators, decimation_operator, fir_operator
from noether_replay.replay import run_shadow


def main() -> dict:
    recording = load_recording(Path(__file__).with_name("synthetic.json"))
    source = next(stream for stream in recording.streams if stream.id == "wfs")
    samples = tuple(range(6))
    times = np.array(source.times_ns["exposure_start"][:6])
    # A declared synthetic model indexed by time, including the fixture's gap.
    # It is not a noise estimate from the deterministic observation values.
    temporal = 0.8 ** (np.abs(times[:, None] - times[None, :]) / 1_000_000)
    channel_covariance = 4e-18 * np.array([[1.0, 0.25], [0.25, 2.0]])
    covariance = np.kron(temporal, channel_covariance)
    operator = compose_operators(fir_operator(6, np.array([0.5, 0.5])), decimation_operator(5, 2))
    parameters = dict(
        output_id="pair-means",
        covariance_provenance={
            "kind": "declared synthetic model; not estimated from the recording",
            "temporal_correlation": "0.8 ** (absolute exposure-start separation / 1000000 ns)",
            "channel_covariance_m2": channel_covariance.tolist(),
        },
        sample_indices=samples,
        channel_indices=(0, 1),
    )
    # Declared fixture completion times, not inferred or measured filter latency.
    conditioned = condition_observations(
        recording,
        "wfs",
        operator,
        covariance,
        available_ns=(1_250_000, 3_250_000, 6_250_000),
        **parameters,
    )
    output = conditioned.streams[-1]
    shadow = run_shadow(
        conditioned,
        output.id,
        lambda observation: -0.1 * observation,
        policy_id="synthetic-filtered-coordinate-policy",
        policy_version="1",
    )

    def withheld(candidate) -> dict:
        calls = []

        def policy(observation):
            calls.append(True)
            return observation

        try:
            run_shadow(
                candidate,
                candidate.streams[-1].id,
                policy,
                policy_id="must-not-run",
                policy_version="1",
            )
        except ValueError:
            assert not calls
            return {"status": "withheld", "policy_calls": len(calls)}
        raise AssertionError("An unqualified fixture reached the policy")

    unknown = condition_observations(recording, "wfs", operator, covariance, **parameters)
    expired = condition_observations(
        recording,
        "wfs",
        np.array([[0.5, 0.5]]),
        np.eye(4) * 4e-18,
        output_id="calibration-expiry-window",
        sample_indices=(7, 8),
        covariance_provenance={"kind": "declared independent synthetic noise"},
        available_ns=(9_250_000,),
    )
    descriptor = output.metadata["conditioning"]
    return {
        "evidence": "synthetic software example; no instrument or controller-performance result",
        "source_recording_sha256": recording.digest(),
        "conditioned_recording_sha256": conditioned.digest(),
        "input_samples": list(samples),
        "support_indices": descriptor["support_indices"],
        "output_values_m": output.values.tolist(),
        "output_covariance_m2": descriptor["output_covariance"],
        "ready_not_before_ns": descriptor["ready_not_before_ns"],
        "declared_output_available_ns": output.times_ns["available"],
        "source_calibration_epoch_ids": descriptor["calibration_epoch_ids"],
        "unknown_output_availability": withheld(unknown),
        "calibration_expiry_window": withheld(expired),
        "shadow": shadow,
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
