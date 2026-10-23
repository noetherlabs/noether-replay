# Noether Replay

Offline reconstruction and evidence qualification for adaptive-optics telemetry.

Noether Replay inspects what a recording can establish about timing, calibration
and command provenance. It reads the existing [Adaptive Optics Telemetry (AOT)
format](https://doi.org/10.1051/0004-6361/202348486), reconstructs known events,
and produces a reproducible report of inconsistencies and missing evidence.

Recorded closed-loop measurements describe the controller that generated them.
They cannot establish the performance of a different controller. Replay keeps
that distinction explicit.

## Quick start

Python 3.11 or later is required. The core package depends only on NumPy.

```sh
git clone https://github.com/noetherlabs/noether-replay.git
cd noether-replay
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
noether-replay examples/synthetic.json --format markdown
```

The synthetic example contains separate requested and applied commands, an
omitted frame, and an expired calibration interval. Its report identifies the
missing frame ID and the observations outside the calibration's validity.
The values are a software fixture, not optical performance measurements.

To inspect an instrument recording:

```sh
python -m pip install -e '.[aot]'
noether-replay /path/to/recording.fits --output report.json
```

Inputs remain local. The reader does not fetch external files or URLs referenced
inside FITS files. Output files are created exclusively; an existing report is
never overwritten.

## Capabilities

| Component | Behavior |
| --- | --- |
| AOT import | Retains source arrays, units, frame identifiers and available metadata through `aotpy`. Missing event semantics remain unknown. |
| Event reconstruction | Separates exposure, observation availability, controller completion, command issue and actuator effect. Reports ambiguous order and unsynchronized clocks. |
| Calibration epochs | Assigns an epoch only when it uniquely covers the complete exposure and its timing uncertainty. |
| Evidence binding | Hashes array dtype, shape and exact bytes together with timing, calibration and source metadata. |
| Shadow policies | Computes hypothetical commands from ordered observations without assigning recorded outcomes to those commands. |
| Noise conditioning | Propagates full temporal covariance through finite linear filtering, selection and decimation operators. |

See the [data contract](docs/data-contract.md), [evaluation modes](docs/evaluation-modes.md)
and [reproducible examples](docs/examples.md).

## Scientific scope

This version qualifies the completeness and internal consistency of telemetry.
It does not certify an instrument, validate a calibration's physical accuracy,
rank controllers, or establish safe recovery. There is no hardware-control
interface and no counterfactual simulator backend.

Canonical replay order is reproducible even when physical ordering is unknown.
An unknown exposure or actuation time is never reconstructed from nominal frame
rate. A source command is not labelled as physically applied unless that meaning
is established by its data contract.

The covariance implementation evaluates the established transformation
$R' = A R A^\mathsf{T}$. It retains off-diagonal temporal correlations. Dense
operators are bounded to small analysis windows; they are not intended to
materialize an entire instrument recording's covariance.

## Development

```sh
python -m pip install -e '.[aot,dev]'
python -m pytest -q
ruff check .
python examples/conditioning.py
```

Tests cover actual AOT FITS round trips, missing metadata, calibration boundaries,
clock ambiguity, data-integrity changes, causal shadow evaluation and analytic
covariance cases. Instrument-data downloads are separate from the test suite.

## Citation and license

Use [CITATION.cff](CITATION.cff) to cite this software. AOT and `aotpy` are developed
by their respective authors; Noether Replay builds on their exchange format.
Public example data retain their original attribution and license.

Code and synthetic fixtures are licensed under [Apache 2.0](LICENSE).
