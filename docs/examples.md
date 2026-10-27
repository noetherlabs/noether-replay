# Reproducible examples

The examples cover reconstruction with complete synthetic event metadata,
qualification of incomplete instrument telemetry, command timing, and the effect
of noise correlations on preprocessing uncertainty. None measures the
performance of an alternative closed-loop controller.

Retained JSON and Markdown outputs are in [examples/reports](../examples/reports).
Their [original manifest](../examples/reports/manifest.json) and the
[conditioned-window manifest](../examples/reports/conditioned-observations-manifest.json)
bind their outputs to source digests, implementation commits and dependency versions.
Those artifacts were generated twice from the installed package and compared
byte-for-byte in the recorded environment. The later
[CIAO manifest](../examples/reports/ciao-manifest.json) separately records its
source-based execution and verification. Cross-platform floating-point bitwise
identity is not assumed.

## Synthetic event reconstruction

```sh
noether-replay examples/synthetic.json --format markdown
python examples/shadow.py
```

The fixture contains 12 observations and separate requested/applied command
streams. It declares 72 event timestamps in one clock, intentionally omits frame
ID 5, and ends the calibration interval before the last four WFS exposures.
Replay preserves all records, reports the absent identifier in each stream, and
assigns temporal calibration coverage to eight of the twelve observations.
See the [retained report](../examples/reports/synthetic.md) and its
[machine-readable output](../examples/reports/synthetic.json).

The shadow example produces hypothetical commands from those recorded
observations. It does not change their subsequent values or claim an improved
wavefront error.
The [retained shadow result](../examples/reports/shadow.json) includes policy
identity and explicitly marks its outcome as unassessed.

## Public PAPYRUS recording

The source is the pyramid-WFS example in Gomes, Garcia, Correia and Morujão's
[Proof-of-concept AO telemetry data using the AOT standard format](https://doi.org/10.5281/zenodo.8192742)
(2023, CC BY 4.0). The original file remains hosted by the dataset authors; it is
not redistributed here.

| Property | Value |
| --- | --- |
| File | `PAPYRUS_PWFS_20230525_2304_20B1FBD7.fits` |
| Size | 75,666,240 bytes |
| Source MD5 | `74dd68f6395b0ebf4f5046a67b5f043f` |
| Verified SHA-256 | `4e1568033171ad2d17b6e20d9c05be02a29c00894f4e6a6962075cbcdaa074d3` |
| Telemetry | 200 WFS measurements, 200 detector frames, 200 command records |
| Event timing | Empty `AOT_TIME` table; no loop time reference |

Download the named file from the dataset page, then run:

```sh
python -m pip install -e '.[aot]' -r requirements-reproduce.txt
noether-replay /path/to/PAPYRUS_PWFS_20230525_2304_20B1FBD7.fits --format markdown
```

The file supports array inspection and source-content verification. It does not
establish exposure, availability, command-issue or actuator-effective times.
No exact event sequence is reconstructed from its nominal observation interval,
array positions or a presumed loop frequency. Command records remain
`command_recorded`; the file alone does not establish requested-versus-applied
semantics.

This is a finding about the metadata in this particular demonstration file, not
a defect claim about PAPYRUS or the AOT standard. AOT permits partial recordings;
additional instrument metadata may establish relationships absent here.
The [retained report](../examples/reports/papyrus.md) and
[machine-readable output](../examples/reports/papyrus.json) contain 17 findings
about missing evidence across the three streams; these are not 17 demonstrated
instrument faults.

## Public CIAO multirate recording

The CIAO1 example from the same public AOT dataset contains 15,000 loop samples
and 316 pixel frames. Of the pixel counter values, 299 also occur in the loop
record and 17 exceed its final counter. Pixel counters advance by 50; the
intervening integers are not evidence of dropped exposures without the counter
contract. Counter overlap alone does not establish observation-to-command lineage.

Generic loop timestamps are retained, but exposure, availability and actuator
stages remain unknown. Commands remain `command_recorded`, so applied-command
timing qualification is refused. See the [source attribution and bounded
reproduction](ciao-example.md), [retained report](../examples/reports/ciao.md),
[JSON report](../examples/reports/ciao.json) and
[counter coverage](../examples/reports/ciao-coverage.json).

## Correlated-noise conditioning

```sh
python examples/conditioning.py
```

For a stationary synthetic AR(1) process with correlation $\rho=0.8$ and variance
$\sigma^2=4\times10^{-18}\,\mathrm{m}^2$, a two-sample mean has variance
$\sigma^2(1+\rho)/2=3.6\times10^{-18}\,\mathrm{m}^2$. Replacing the input
covariance with its diagonal gives $2.0\times10^{-18}\,\mathrm{m}^2$, an
underestimate of 44.44%. Factor-two decimation retains nonzero adjacent-output
covariance of $2.592\times10^{-18}\,\mathrm{m}^2$.

These values follow analytically from the declared synthetic covariance and are
also checked by deterministic numerical tests. They demonstrate why conditioning
must propagate temporal correlations. They are not measurements of PAPYRUS noise,
a newly discovered covariance law, or evidence of controller superiority.
The [retained numerical output](../examples/reports/conditioning.json) records
the input parameters, covariance terms and analytic formula.

## Conditioning with timing and calibration context

```sh
python examples/conditioned_observations.py
```

This example selects six two-channel observations from the synthetic recording
and forms three disjoint pair means. It supplies a separable covariance with
temporal and cross-channel correlations. Temporal correlation uses actual fixture
timestamp separations, preserving the deliberately omitted frame interval.
The source values and covariance model are synthetic; no noise parameters are
estimated from instrument data.

Each mean retains its contributing indices, propagated joint covariance,
availability admission bound and source calibration epoch. Explicit fixture
completion times allow three hypothetical policy commands. A second window with
unknown completion times and a window spanning calibration expiry are both
withheld before policy execution. Neither withholding nor successful shadow
execution measures closed-loop performance.
The [retained output](../examples/reports/conditioned-observations.json) includes
the full joint covariance, source support and both withheld cases.

## Applied-command timing

```sh
python examples/command_timing.py
```

This analytical fixture declares observation-to-command associations and exact
integer timestamp bounds. Its seven applied commands include met and exceeded
budgets, ambiguous timing, contradictory causal order, a missing effect time and
an omitted association. Every command remains in the report. The 20 ns budget
is a fixture value, not an instrument requirement or measured execution latency.
See the [timing contract](command-timing.md) for interval arithmetic, admission
requirements and the distinction from total sensor-to-actuator latency.

## Exposure-integrated probe response

```sh
python examples/probe_response.py
```

This analytical fixture recovers a fixed-frequency response with offset 2,
cosine coefficient 3 and sine coefficient 4, giving amplitude 5. A midpoint-only
comparator shows the predicted finite-exposure attenuation. Removing an exposure
endpoint withholds the complete twelve-sample window and retains its identity.
The supplied noise covariance is synthetic; timing bounds remain distinct from
conditional coefficient uncertainty. See the [probe-response contract](probe-response.md)
for assumptions, numerical admission and the limits of closed-loop identification.
