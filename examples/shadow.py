"""Hypothetical commands on a synthetic recording; no closed-loop outcome claim."""

import json
from pathlib import Path

from noether_replay import load_recording
from noether_replay.replay import run_shadow

recording = load_recording(Path(__file__).with_name("synthetic.json"))
result = run_shadow(
    recording,
    "wfs",
    lambda measurement: -0.3 * measurement,
    policy_id="synthetic-modal-gain",
    policy_version="1",
)
print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
