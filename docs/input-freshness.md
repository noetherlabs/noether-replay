# Qualifying the age of command inputs

`qualify_input_freshness` checks the age of **every declared exposure support**
at a recorded command's actuator effect. A short processing delay is compatible
with stale inputs: a command can follow its latest observation promptly while
also depending on an older measurement.

The function reuses [command timing](command-timing.md) and adds a separate age
bound. It consumes recorded evidence and explicit input associations. It neither
runs a policy nor changes a command.

```python
from noether_replay.freshness import qualify_input_freshness

report = qualify_input_freshness(
    recording,
    applied_stream_id="dm_applied",
    associations=[{
        "applied_index": 0,
        "observations": [
            {"stream_id": "wfs", "sample_index": 3},
            {"stream_id": "wfs", "sample_index": 4},
        ],
    }],
    max_support_age_ns=2_000_000,
    latency_budget_ns=1_000_000,
)
```

`max_support_age_ns` is a required nonnegative integer bound in nanoseconds.
Zero is a real bound, not an instruction to disable the check. The optional
latency budget has the existing availability-to-effect semantics. Both results
are retained separately; neither one substitutes for the other.

## Exposure support and uncertainty

An exposure integrates over an interval. Replacing it with its end or midpoint
would conceal its older support. Let its declared start and end be \(s_j,e_j\),
with timing uncertainty bound \(u_j\). Let the command's actuator-effect interval
be \([E_L,E_U]\). The report retains the full exposure-support age hull:

\[
[E_L-(e_j+u_j),\;E_U-(s_j-u_j)].
\]

The age of that exposure's **oldest** support has the narrower interval:

\[
[L_j,U_j]=[E_L-(s_j+u_j),\;E_U-(s_j-u_j)].
\]

For all declared contributors, the oldest-support age is bounded by
\([\max_j L_j,\max_j U_j]\). This is the interval assessed against the requested
age limit. A long exposure can exceed that limit even if its end was recent.

These operations use exact integer nanoseconds, including epochs larger than
64-bit integers. They assume no independence between timing errors. Shared clock
errors can make the bounds conservative. Supplied uncertainties are bounds,
not automatically standard deviations or confidence intervals. This is temporal
support accounting, not a model of how much each instant contributes to a
measurement or how quickly the physical wavefront changes.

## Filtered observations

For a raw reference, the contributor is that observation itself. For a conditioned
reference, the qualifier expands every nonzero operator coefficient into its raw
source observation. Existing recording validation first reconstructs the
conditioning operator, source digest, values, covariance and exposure lineage.
It rejects changed or forged descriptors.

Positive, negative and arbitrarily small nonzero weights retain their source
support. Exact zero weights do not contribute. There is no weight threshold or
weighted-average timestamp. Filtered-output availability describes when the
output could be consumed; it does not replace the acquisition intervals of its
inputs. Conditioning chains must already be composed into one operator against
raw data, as required by the existing conditioning contract.

Every declaration remains visible in `commands[].observations[].contributors`.
If a raw sample is referenced directly and through a filtered observation, both
paths remain visible. Contributor counts count references, including repeated
uses; they are not counts of independent measurements.

## Verdicts and missing evidence

| Age-budget result | Meaning |
| --- | --- |
| `met` | The entire oldest-support age interval is at most the requested bound. Equality is permitted. |
| `exceeded` | The entire oldest-support age interval exceeds the requested bound. |
| `ambiguous` | The interval overlaps the bound. |
| `not_established` | Required evidence is missing or contradictory, clocks differ, or command timing is not qualified as consistent. |

Both exposure endpoints, their named clock and uncertainty, and the command's
effect time and uncertainty are required. The whole exposure must be no later
than the effect under the supplied bounds. Missing endpoints are not filled from
frame rates, nearby frames, output availability or calibration intervals.
Reversed known stages and overlapping exposure-end/effect intervals withhold
qualification. The underlying command-timing report also checks observation
availability and any supplied command completion or issue times.

A contributor's local verdict concerns its exposure-to-effect evidence. For
example, one known old exposure can remain `exceeded` while another lacks an
exposure start. The command result is then `not_established`, and its complete
oldest-support interval is absent. The stale witness is retained. Likewise,
known exposure/effect times can describe an age even when missing availability
withholds the command verdict. Contributor results do not establish that a
policy could consume the observation at decision time.

Every applied command appears, including those omitted from `associations`.
Missing associations remain unestablished rows. Duplicate associations or
references, invalid indices, unsupported command kinds and malformed bounds
are rejected using the existing timing contract. Unknown or requested commands
cannot be promoted to physically applied commands by this function.

The report includes separate command and contributor-reference counts, the
original timing report, normalized configuration and its digest, and the
recording digest. The source digest is checked again before return. This detects
retained changes during evaluation, but is not a concurrent lock or a guarantee
against mutate-and-restore behavior by other trusted code.

The complete canonical JSON report is limited to 16 MiB. Oversized reports are
rejected, not truncated. Select an explicit bounded recording window for larger
sessions; do not silently omit inconvenient dependencies.

## Reproduce the analytical example

```sh
python examples/input_freshness.py
```

Two exposures span `[0, 10]` and `[90, 95]` ns and become available at 11 and
96 ns. The first command declares both inputs and becomes effective at 100 ns.
Its availability-to-effect latency is 4 ns, meeting a 5 ns budget. Its oldest
exposure support is 100 ns old, exceeding a 20 ns age bound.

The second command uses a mean of those observations, available at 97 ns.
Its latency is 3 ns, but the same raw old support still exceeds the age bound.
The example writes a deterministic report to standard output; it acquires no
instrument data and measures no controller performance.

## What this establishes

The assessment answers whether the supplied temporal evidence meets a declared
input-age requirement. Associations remain caller assertions: they may omit
physical dependencies and do not prove what a policy actually consumed. Digests
bind content, not its authenticity or physical execution.

Calibration coverage, calibration-product availability, measurement validity and
physical accuracy remain separate. Missing calibration and nonfinite raw values
do not change otherwise valid temporal arithmetic. A temporal result must not
be interpreted as qualification of those measurements. A filtered observation's
existing conditioning validation still applies before this calculation.

This is neither an online anomaly detector nor a controller promotion rule.
There is no automatic age threshold, causal attribution, closed-loop performance
score or safety guarantee. An instrument-specific age requirement needs its own
physical justification and validation.
