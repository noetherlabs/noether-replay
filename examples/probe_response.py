"""Deterministic exposure-integrated fit and retained missing-timing exclusion."""

import json
from copy import deepcopy

import numpy as np

from noether_replay.model import Recording, Stream
from noether_replay.probe import qualify_probe_response


def main():
    starts = np.array([0, 2, 5, 7, 10, 13, 15, 19, 22, 25, 29, 31]) * 1_000_000
    ends = starts + 3_000_000
    omega = 2 * np.pi * 125
    first, last = starts * 1e-9, ends * 1e-9
    # Analytical integral evaluated independently through antiderivatives.
    cosine = (np.sin(omega * last) - np.sin(omega * first)) / (omega * (last - first))
    sine = (np.cos(omega * first) - np.cos(omega * last)) / (omega * (last - first))
    values = (2 + 3 * cosine + 4 * sine)[:, None]
    recording = Recording(
        "synthetic-probe-response",
        (Stream("wfs", "wfs_measurements", values, "rad", clock_id="fixture-clock",
                times_ns={"exposure_start": tuple(int(t) for t in starts),
                          "exposure_end": tuple(int(t) for t in ends)}, uncertainty_ns=0),),
        ({"id": "fixture-calibration", "clock_id": "fixture-clock", "start_ns": -1,
          "end_ns": int(ends[-1]) + 1, "stream_ids": ["wfs"],
          "products": {"synthetic-model": "a" * 64}},),
        provenance={"source": "analytical uniform-exposure fixture"},
    )
    kwargs = {
        "stream_id": "wfs", "sample_indices": list(range(12)), "channel_indices": [0],
        "reference": {
            "clock_id": "fixture-clock", "origin_ns": 0, "origin_uncertainty_ns": 0,
            "frequency_numerator_hz": 125, "frequency_denominator": 1,
            "frequency_uncertainty_hz": 0.0, "phase_at_origin_rad": 0.0,
            "phase_uncertainty_rad": 0.0,
            "provenance": {"source": "exact fixed synthetic oscillator"},
        },
        "covariance": np.eye(12) * 0.04,
        "covariance_provenance": {"source": "declared noise model; no noise realized here"},
        "observation_model": "uniform_exposure_average", "max_phase_error_rad": 0.0,
    }
    fitted = qualify_probe_response(recording, **kwargs)
    missing = deepcopy(recording)
    missing.streams[0].times_ns["exposure_end"] = (None, *tuple(int(t) for t in ends[1:]))
    withheld = qualify_probe_response(missing, **kwargs)
    midpoint = (first + last) / 2
    point_design = np.column_stack((np.ones(12), np.cos(omega * midpoint),
                                   np.sin(omega * midpoint)))
    midpoint_coefficients = np.linalg.lstsq(point_design, values, rcond=None)[0]
    print(json.dumps({
        "fixture": "analytical, not hardware evidence",
        "truth": {"offset": 2, "cosine": 3, "sine": 4, "amplitude": 5},
        "integrated": {"status": fitted["status"], "coefficients": fitted["fit"]["coefficients"],
                       "amplitude": fitted["fit"]["amplitude"],
                       "recording_sha256": fitted["recording_sha256"],
                       "configuration_sha256": fitted["configuration_sha256"]},
        "midpoint_comparator_coefficients": midpoint_coefficients.ravel().tolist(),
        "missing_endpoint": {"status": withheld["status"], "reasons": withheld["reasons"],
                             "retained_samples": len(withheld["exposures"])},
    }, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
