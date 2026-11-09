# Qualifying recorded command timing

`qualify_command_timing` checks the interval from the latest declared input's
availability to a recorded command's actuator effect. It answers a specific
question: **does the supplied timing evidence establish that this applied command
followed its observations and met a declared latency budget?**

The function consumes an existing `Recording`. It neither runs a controller nor
changes a command. Its associations are explicit source assertions; matching
timestamps, frame IDs or array positions does not establish which observations
a controller used.

```python
from noether_replay.timing import qualify_command_timing

report = qualify_command_timing(
    recording,
    applied_stream_id="dm_applied",
    associations=[
        {
            "applied_index": 0,
            "observations": [{"stream_id": "wfs", "sample_index": 3}],
        },
    ],
    latency_budget_ns=1_000_000,
)
```

Indices refer to the original arrays, before any sorting. A command can declare
multiple observations from one or more WFS streams. Every applied command appears
in the output, including commands omitted from `associations`. Omission produces
an unestablished result; it does not remove the command from the denominator.
Duplicate command associations, duplicate references within an association,
out-of-range indices and empty observation lists are rejected. A recorded command
whose requested/applied meaning is unknown cannot be used as `command_applied`.

## Timing calculation

For observation \(i\), let its availability be \(t_i\) and its declared timing
uncertainty be \(u_i\). The latest input's availability lies in

\[
[L,U] = [\max_i(t_i-u_i),\;\max_i(t_i+u_i)].
\]

For actuator effect time \(e\) with uncertainty \(u_e\), the latency interval is

\[
[e-u_e-U,\;e+u_e-L].
\]

All subtraction uses integer nanoseconds. Large absolute epochs are not converted
to floating point. The bounds use the declared timestamp intervals without
assuming that timing errors are statistically independent. They can therefore
be conservative when clocks share correlated errors. The supplied uncertainties
are treated as bounds, not automatically as standard deviations or confidence
intervals; their validity must be established by the instrument's clock contract.

| Causal-order result | Meaning |
| --- | --- |
| `consistent` | The entire latency interval is nonnegative. Equality is permitted. |
| `violated` | The entire interval is negative: the effect precedes a declared required input. |
| `ambiguous` | The interval includes both negative and nonnegative values. |
| `not_established` | Required evidence is absent, differing clocks have no applicable conversion, known stages contradict their order, or input availability at command completion/issue is not established. |

Equality establishes only consistency with a nonnegative dependency delay. It
does not establish strict precedence between equal-time events; canonical replay
order remains a separate serialization convention.

The latency budget is assessed only for causally consistent rows. An interval
whose upper endpoint is at most the budget is `met`; one whose lower endpoint
exceeds it is `exceeded`; an interval crossing the budget is `ambiguous`.
Otherwise the budget is `not_established`. With no budget, it is `not_requested`.
The summary retains separate counts rather than turning missing evidence into
an overall pass.

Known timestamps must also respect the declared exposure-to-actuation stage
order. An observation available before its exposure ends, or an actuator effect
before command issue, withholds qualification for the affected command. Missing
optional stages do not create an ordering contradiction. Logging timestamps
(`recorded`) do not define physical stage order.

When command `completed` or `issued` timestamps are supplied, their uncertainty
intervals must also lie entirely at or after the latest input-availability
interval. A stage entirely before the inputs yields
`input_unavailable_at_completed` or `input_unavailable_at_issued`; overlapping
intervals yield the corresponding `ambiguous_input_availability_at_…` reason.
Both withhold causal-order and budget qualification. Exact boundary equality is
permitted. The report retains the descriptive availability-to-effect latency
interval in these cases, but that interval does not override contradictory or
ambiguous earlier stages. Missing completion or issue timestamps remain unknown;
a qualified availability-to-effect interval does not establish those stages.

## Instrument integration

Supply an applied-command stream with physical `effective` timestamps and an
observation stream with `available` timestamps. Establish their common clock and
uncertainty bounds, or supply the external conversion described below, and export
the controller's input associations. Multiple clocks are not synchronized by this
API. A nominal sampling rate is insufficient.
Conditioned observations use their validated availability, after every supporting
input is available; the conditioning operator's physical accuracy is a separate
question.

The report retains the recording digest, normalized associations, timing budget
and configuration digest. These bind the supplied evidence but do not authenticate
the association history or prove the actuator executed a logged value.

Reports and association metadata are bounded to 16 MiB of canonical JSON. An
oversized report is rejected in full, never truncated to a favorable subset.
Use an explicitly selected recording window when processing larger sessions.

This latency starts at **observation availability**, not exposure start or end.
It includes whatever happened between the latest input becoming available and
the reported actuator effect. It is not isolated controller compute time, total
sensor-to-actuator latency, the age of every historical input, or a real-time
execution benchmark. Meeting it establishes neither calibration accuracy nor
closed-loop stability, optical improvement or safe controller transfer.

## External clock-conversion receipts

An optional `clock_conversion` argument admits one externally established mapping
from an observation clock to the selected applied-command clock. The default
`None` retains the existing `noether-replay-command-timing/1` report, including its
canonical bytes. Supplying a receipt produces `noether-replay-command-timing/2`.
The recording format and raw timestamps do not change.

The receipt has exactly these fields:

| Field | Contract |
| --- | --- |
| `schema` | `noether-replay-clock-conversion/1` |
| `recording_sha256` | Exact input `recording.digest()` |
| `source_clock_id` | Nonempty observation clock name, distinct from the target |
| `target_clock_id` | Selected applied-command stream's clock name |
| `source_valid_from_ns`, `source_valid_until_ns` | Native source-clock validity `[a,b)`, with `b > a` |
| `offset_ns` | Signed integer offset `d`, in nanoseconds: target = source + d |
| `max_mapping_error_ns` | Nonnegative integer total mapping-error bound `E` |
| `bound_scope` | `uniform_over_source_validity_interval` |
| `evidence_sha256` | Nonempty named map of lowercase SHA-256 references to external evidence |
| `provenance` | Nonempty JSON object explaining source, method, clock instances and bound rationale |

All bounds use exact integer nanoseconds; booleans and floating-point timestamps
are rejected. Evidence hashes are references only: the API does not read files,
resolve URLs or authenticate the referenced evidence. Provenance is retained,
not judged for physical sufficiency.

The external assertion is that **every applicable true source coordinate**
`s` in `[a,b)` maps into `[s+d-E, s+d+E]` in the target clock. `E` must uniformly
cover the total residual error over the interval, including offset error, drift,
skew, synchronization error and any allowed clock steps. A constant offset is a
bounded approximation, not an assertion of zero oscillator drift. A single offset
measurement, matching counters or a nominal sample rate does not establish this
uniform bound. When a reset or unbounded rate error invalidates it, withhold the
receipt or supply a separately identified recording and a new supported bound.

For a native availability timestamp `t` with bound `u`, conversion requires
`a <= t-u` and `t+u < b`. The comparison interval is then
`[t+d-u-E, t+d+u+E]`. Native target-clock observations retain `[t-u,t+u]`.
These resolved intervals feed the same latest-availability, completion, issue,
effect and budget checks. Errors are added conservatively without independence
assumptions or removing potentially shared uncertainty.

For example, an **analytical fixture** with `t=1000`, `u=3`, `d=4000`, `E=4`,
and validity `[900,1100)` produces `[4993,5007]`. An actuator effect of `5100`
with uncertainty `2` yields latency `[91,109]`: budgets `110`, `100` and `90`
are respectively met, ambiguous and exceeded. This is interval arithmetic,
not an instrument synchronization result.

```python
report = qualify_command_timing(
    recording,
    applied_stream_id="dm_applied",
    associations=associations,
    latency_budget_ns=1_000_000,
    clock_conversion=externally_established_receipt,
)
```

A malformed receipt, mismatched recording digest, wrong target clock or unused
source clock rejects the call. “Unused” means that no associated observation
references the source clock. A referenced observation with missing availability
or uncertainty remains a withheld row. A complete uncertainty interval outside
validity yields `clock_conversion_outside_validity`. An entirely expired cohort
therefore remains a report, not an unused-receipt error. Other clock domains keep
`cross_clock_reference`; there is no chaining, inversion or automatic selection.
Within-stream stage contradictions still withhold the row.

Each observation retains its native `clock_id`, `available_ns`, `uncertainty_ns`
and `availability_interval_ns`. Schema `/2` adds `comparison_clock_id`,
`comparison_availability_interval_ns` and `clock_conversion_sha256`. The digest
identifies the receipt for source-clock references, including those withheld;
it is `null` for native or unmapped references. A missing comparison interval
is `null`. Commands label their latest-availability and latency comparison clock.
The complete detached receipt and its digest appear once in `configuration`.

`summary.clock_conversion` counts associated source-clock references, converted
availability intervals and withheld availability intervals. Repeated use of an
observation counts per association. A converted interval is **not** a qualified
command: another dependency or a stage contradiction may still withhold it.
Every applied command remains in the report.

Receipt metadata is checked before canonical expansion, with a 64-level nesting
limit and the existing 16 MiB JSON budget; cycles and unsupported JSON values are
rejected. The complete configuration and report must also fit the aggregate
budget. The receipt is detached from caller-owned containers and bound into the
configuration digest. A recording digest binds content, not an authenticated
physical clock instance. Reset identities must actually be retained in the
recording ID or provenance to distinguish otherwise identical recordings.

This extension converts **availability only**. It does not transfer calibration
epochs, qualify exposure freshness, alter probe phase, synchronize event replay,
or enable physical shadow execution. It establishes neither the external bound's
physical truth nor calibration accuracy. Existing public recordings without that
evidence cannot acquire it from this API.

## Reproduce the example

```sh
python examples/command_timing.py
```

The seven-command synthetic fixture exercises a met budget, an exceeded budget,
budget ambiguity, causal-order ambiguity, reversed causal order, a missing effect
time and a missing association. These are analytical software cases, not instrument
measurements. The retained PAPYRUS example has no event timestamps and unknown
command semantics; it cannot support this assessment without additional evidence.
