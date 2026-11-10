# Review a command-timing decision

Run the [synthetic instrument-timing example](../examples/instrument_timing.py)
from an installed checkout:

```sh
python examples/instrument_timing.py
```

It prints a Markdown review of six commands, using the existing
[`qualify_command_timing`](command-timing.md) API and an explicit clock-conversion
receipt. It writes no files, downloads nothing and connects to no instrument.
Every timestamp, uncertainty, association and clock assertion is synthetic.
The 110 ns budget illustrates the arithmetic; it is not an instrument requirement.

The report retains one met budget, one exceeded budget, one ambiguous budget and
three unestablished cases. Missing availability, a missing association and an
expired conversion remain separate rows. It includes recording/configuration/receipt
digests and the API's interpretation limits. These six cases are not independent
instrument trials or an estimated deadline-failure rate.

For command 0, camera availability is 1000 +/- 3 ns. The asserted mapping adds
4000 ns with a total error of at most 4 ns, giving [4993, 5007] ns in the actuator
clock. The effect time is 5100 +/- 2 ns, so latency lies in
[5098 - 5007, 5102 - 4993] = **[91, 109] ns**, meeting the 110 ns budget.
Commands 1 and 2 instead give [111, 129] and [101, 119] ns: exceeded and ambiguous.
These are deterministic bounds, not confidence intervals. Missing completion
and issue times remain unknown even when the availability-to-effect budget is met.

## Use with an instrument recording

The example is a small Python integration pattern, not a new input format or a
hardware adapter. Keep the existing `Recording`, associations and clock-conversion
contract. Before substituting an instrument recording, establish:

- Native observation-availability and actuator-effective event meanings, clock
  identities/epochs, units and uncertainty bounds. A command request or software
  acknowledgement is not physical readback.
- Each command's actual observation dependencies, using original sample indices.
  Matching frame counters alone do not establish those dependencies.
- Independently supported clock conversion, including its validity interval and
  total residual error bound across that interval. Content hashes do not prove
  synchronization or authenticate the supplied clock evidence.
- The operator's latency criterion and what action, if any, follows a result.

Do not rewrite missing fields with the example's defaults. Supply only established
event semantics and let missing evidence remain visible. An ambiguous budget
requires tighter supported bounds; it is neither a pass nor a demonstrated miss.
An exceeded bound identifies this particular timing relationship, not its physical
cause or an unsafe instrument. A met bound does not establish calibrated input,
freshness, complete dependency coverage or controller performance.

For a local AOT file, first use `noether-replay recording.fits --format markdown`
to inspect what the source actually supplies. The [CIAO example](ciao-example.md)
shows why generic timestamps remain metadata and its commands remain
`command_recorded`. The timing API rejects those commands even if a conversion
receipt is supplied; converting clock coordinates cannot create event semantics.
Malformed input raises an error rather than producing a partial timing report.
