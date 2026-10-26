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
| `not_established` | Required evidence is absent, clocks differ, known stages contradict their order, or input availability at command completion/issue is not established. |

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
uncertainty bounds, and export the controller's input associations. Multiple
clocks are not synchronized by this API. A nominal sampling rate is insufficient.
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

## Reproduce the example

```sh
python examples/command_timing.py
```

The seven-command synthetic fixture exercises a met budget, an exceeded budget,
budget ambiguity, causal-order ambiguity, reversed causal order, a missing effect
time and a missing association. These are analytical software cases, not instrument
measurements. The retained PAPYRUS example has no event timestamps and unknown
command semantics; it cannot support this assessment without additional evidence.
