# When scalar optical evidence does not identify a unique explanation

[The runnable example](../examples/decision_evidence.py) contains two exact
analytical constructions: sensor gain versus disturbance, and requested command
versus actual applied value. Each has an independent-measurement control and a
degenerate control. These are mathematical examples, not a sampled diagnostic
performance study.

```sh
python examples/decision_evidence.py
```

The script produces a deterministic JSON case table and descriptive counts.
It makes no model calls, acquires no observations and writes no files. It uses
the existing `Recording` and `Stream` types. Exact rational values in public
stream metadata are the analytical measurement contract; the numeric arrays
are their finite float64 renderings. The exact numbers are synthetic fixture
assumptions, not precision claims about a physical instrument.

## Sensor gain and disturbance

For one modal coordinate, let

\[
y=g(x-a),\qquad g>0,
\]

where \(a\) is the known applied command, \(x\) is the incoming disturbance,
and \(y\) is the sensor reading in nominal OPD coordinates. Without an
independently established disturbance model, every positive gain \(c\) has a
compatible disturbance

\[
x'=a+y/c.
\]

The applied command and sensor reading are exactly the same for every such
world. When \(y\ne0\), their physical residuals \(x'-a=y/c\) differ. At zero
reading, all these worlds have zero residual but gain remains unidentified.
Two displayed worlds are witnesses to the ambiguity; they are not its complete extent. The compatible
gain set is **all positive real numbers**, not the two gains selected for display.

For the example reading \(y=12\) and applied value \(a=3\), unit gain is
compatible with disturbance 15 and residual 12; gain one-half is compatible with
disturbance 27 and residual 24. These values are exact arithmetic witnesses,
not operating points or measured optical results.

An independent exact measurement \(r=x-a\) changes the question. If \(r\ne0\),
then \(g=y/r\) is identified under this model. When \(r=0\) and \(y=0\), every
positive gain remains compatible even with the independent measurement. The
example retains this zero-reference case. Contradictory exact readings, such as
zero residual with a nonzero sensor reading, do not yield a gain estimate.

The reference is a declared independent physical residual measurement in the
synthetic model. A second copy of the nominal sensor reading would not supply
that information. No calibration accuracy or reference-instrument performance
is established by this construction.

## Requested command and actual applied value

Suppose the visible software trace contains a request \(q\), a successful tool
response and a calibrated sensor reading

\[
y=x-a.
\]

With unrestricted disturbance, any real applied value \(b\) has a compatible
disturbance \(x'=y+b\). A request and success response therefore do not identify
the actual applied value. The compatible set is **all real applied values**,
not only zero and the request.

The example uses a request of 5 and a reading of 7. Applied value 5 with
disturbance 12 and applied value zero with disturbance 7 have identical visible
request/status/sensor evidence. The request stream remains
`command_requested`; it is not promoted to `command_applied`.

The control adds an independent exact physical readback. This identifies the
applied value and whether it equals the request under the scalar contract. The
readback is an assumed measurement of physical state, not an echo of a software
register. It carries no effect timestamp: a known value does not establish
when that value was applied. Existing command-timing qualification consequently
remains unestablished for these examples.

### A zero command is a useful distinction

An exact zero readback after a zero request establishes value equality. It
does **not** establish that the transaction executed: an executed no-op and an
unchanged zero state can produce the same evidence. The example constructs both
worlds and deliberately does not score transaction causation as identified.

Applied-value estimation, value equality and transaction execution are separate
questions. A wrong numeric estimate can still correctly say that the actual
value differs from the request. Their scores are therefore kept separate.

## Comparators and scoring

All methods receive the same detached `Recording` in an evidence condition:

| Method | Behavior |
| --- | --- |
| Constraint baseline | Solves the declared scalar constraints. Returns the full compatible domain without an identifying measurement; uses independent reference/readback when it resolves the target. |
| Nominal guess | Assumes unit sensor gain or that the requested value was applied. It deliberately exposes the difference between a plausible guess and identification. |
| Always withhold | Returns no target set or unique conclusion. It is a coverage control, not a competitive diagnostic method. |

The constraint method is a conventional analytical baseline. The example does
not claim that a new algorithm outperforms it. There are no fitted thresholds,
confidence scores, trained policies or selected operating points.

The case table retains the realized target, the analytically compatible set,
each method's claimed set and separate scores:

- **Resolution coverage:** unique target conclusions divided by all applicable
  cases. Failed and withheld decisions remain in the denominator.
- **Selective realized error:** wrong unique conclusions divided by unique
  conclusions. With no unique conclusions, this quantity is undefined (`null`),
  not zero.
- **Unsupported uniqueness:** a unique conclusion that fails to cover all
  target values compatible with the evidence. A lucky guess is still unsupported.
- **Compatible-set coverage:** whether a claimed set includes the full
  analytically compatible target set. An unrestricted set receives no unique
  resolution credit.
- **Resolution on identifiable controls:** correct unique conclusions on cases
  whose independent evidence determines the target. Always withholding cannot
  succeed by avoiding this part of the task.
- **Requested-value equality:** resolution, correctness and evidential support
  for equality, separately from numerical applied-value estimation.

No result claims transaction causation. Invalid method output, exceptions or
changes to the supplied evidence are retained as terminal failures. Exception
types are recorded without publishing arbitrary exception-message contents.
The complete report is bounded to 16 MiB and rejects oversized output rather
than truncating cases.

## Evidence separation and limitations

The physical truth, case labels, compatible-set oracle and construction-group
IDs belong to the evaluator. A method receives only a detached visible
`Recording`. Within an alias condition, all visible arrays, IDs and metadata
are identical. Independent controls add only their declared measurement stream.
All positive gains or real applied values remain possible when the nuisance
disturbance is unrestricted; the example does not pretend the displayed worlds
exhaust those alternatives.

This is an interface for trusted deterministic callbacks, not a Python sandbox
or sealed agent evaluation. Its source, fixtures and truth are public. There is
no claim that an adversarial callback cannot inspect other process state.

Aliases, independent-measurement controls and degenerate variants retain their
base construction group. The case counts describe this hand-constructed table;
they are not independent experimental replications or estimates of an observatory
failure distribution. No statistical confidence intervals are reported.

A broader noisy comparison would require justified disturbance/noise models,
instrument-derived limits, grouped sampling, a precision or power design and
separate development and evaluation evidence. None of those experiments is
reported here. The examples establish identification boundaries under explicit
scalar assumptions, not detector validity, controller superiority, causal
benefit or hardware safety.
