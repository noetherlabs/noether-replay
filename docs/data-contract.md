# Telemetry data contract

`noether-replay/1` is the package's analysis manifest, not a replacement for the
AOT exchange standard. A `Recording` contains numeric `Stream` arrays, explicit
event metadata, calibration references and source provenance. AOT remains the
instrument interchange format.

## Arrays and integrity

Each stream retains its decoded real numeric array. FITS scaling follows
`aotpy`/Astropy semantics; the source-file digest binds the original FITS bytes.
The array digest binds the decoded array's
NumPy dtype string, shape and exact C-order bytes. Integer width, byte order,
signed zero and nonfinite bit patterns are not normalized away. Nonfinite source
values are retained and reported; they are not silently interpolated or replaced.

The recording digest additionally binds units, identifiers, event times,
uncertainty, calibration epochs and metadata. The timeline digest binds that
recording digest and the canonical event sequence. These establish consistency,
not authenticity or experimental validity. A digest is not a signature.

Individual arrays are limited to 256 MiB and a recording to 512 MiB. JSON fixtures
are self-contained and limited to 16 MiB; they do not resolve paths or URLs.
The AOT reader accepts regular local files up to 256 MiB, with at most 1,024
HDUs, two million expanded table elements and a conservative 256 MiB total
decoded-image budget. Repeated references count toward retained-array and
metadata limits before their contents are copied. Expanded output metadata is
limited to 16 MiB. Ordinary single-descriptor variable-length table columns are
supported; compressed inputs and random-group HDUs are rejected. These bounds
limit this reader's scope and are not restrictions of the AOT standard.

## Timing

| Field | Meaning |
| --- | --- |
| `exposure_start` | Start of the measurement's integration interval. |
| `exposure_end` | End of that interval. |
| `available` | Time the controller could first access the observation. |
| `completed` | Completion of the controller calculation. |
| `issued` | Time the command was issued. |
| `effective` | Time the command affected the actuator. |
| `recorded` | A timestamp whose source establishes recording time only. |

Times are integer nanoseconds in a named clock. `None` means unknown. Clock
uncertainty is an explicit nonnegative bound in nanoseconds; `None` is not zero.
AOT floating-point source timestamps remain source metadata unless their event
semantics and clock conversion are established. Nominal frequency does not fill
missing event times.

Events are serialized by clock, nominal time, stream ID, sample index and stage.
This ordering does not synchronize different clocks. Intersecting closed time
uncertainty intervals have ambiguous physical order, including ties with zero
uncertainty. The source array order and sample indices remain available.

Frame-ID gaps are reported as absent identifiers. They are not automatically
classified as acquisition losses: wraparound, deliberate decimation and counter
semantics require instrument-specific evidence.

## Calibration

An epoch supplies a unique ID, clock, half-open validity interval `[start, end)`,
stream IDs and named calibration-product SHA-256 references. Assignment requires
one epoch to cover the full exposure and its uncertainty; the upper uncertainty
bound must be strictly below the epoch's end. Missing exposure times, unknown
clock uncertainty, gaps, cross-clock epochs and overlapping assignments remain
unqualified. The nearest epoch is never borrowed.

Product hashes are references. This version does not load those products, check
their dimensions or determine whether the calibration is physically correct.
A temporally covered measurement is not a validated measurement.

## Commands and conditioning

`command_requested`, `command_applied` and `command_recorded` are distinct kinds.
Ordinary AOT command arrays use `command_recorded` when actuator application is
not established. Collection-controller identity, gains and injected probes can
be retained in metadata; absence must not be interpreted as zero excitation.

The conditioning module accepts explicit finite linear operators. Operator
composition follows application order. FIR filters use valid support, retaining
the induced cross-sample covariance. Transforming values without transforming
their covariance changes the observation model and must not be treated as neutral
preprocessing.
