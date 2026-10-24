# Evaluation modes

| Mode | Input | Permitted result | Implementation |
| --- | --- | --- | --- |
| Observational | Recorded observations and commands | Recording consistency, metadata coverage and retained event reconstruction | `build_report` and the command-line interface |
| Shadow | Recorded observations and an alternative policy | Hypothetical commands on the recorded observation sequence | `run_shadow` |
| Counterfactual | An independently validated plant or authorized HIL experiment | Alternate closed-loop trajectories within the model's validated scope | No backend in this release |

## Observational reconstruction

The report can identify missing observation availability, ambiguous timing,
unrepresented frame IDs and missing calibration coverage. It does not attribute
the recorded residual wavefront or science image to any alternative controller.
Calling the observational report with another mode is an error.

## Shadow evaluation

`run_shadow(recording, stream_id, policy, policy_id=..., policy_version=...)`
passes copies of WFS observations to a
Python callable in availability order. It requires complete availability times,
known uncertainty and nonoverlapping availability intervals. Nonfinite inputs
require an explicit missing-observation policy and are rejected by this interface.
Commands must be finite vectors with a fixed shape. The recording digest is
checked before and after evaluation. Policy identity and version are mandatory
and retained with the hypothetical actions.

The result records hypothetical commands and sets `outcome_assessed` to `false`.
It does not feed those commands back into the recording. A stateful policy should
be constructed anew for each independent replay. The caller is responsible for
recording the policy's source revision and configuration with the result.

This is sequential offline evaluation, not a real-time execution benchmark.
Recorded availability does not measure policy execution latency.

### Policies that use event context

Set `policy_input="event"` to receive a `ShadowObservation` instead of an array:

```python
def policy(observation):
    # A stateful policy can retain the previous available_ns and epoch ID.
    return -0.3 * observation.values

result = run_shadow(
    recording, "wfs", policy,
    policy_id="modal-gain", policy_version="1", policy_input="event",
)
```

The event contains the current observation's original `sample_index`, optional
`frame_id`, units, clock, availability and uncertainty in integer nanoseconds,
and separate optional exposure start/end times. Subtract integer timestamps
before converting a duration to floating-point seconds. Availability is when an
observation could be consumed; it is not the time at which the wavefront was
sampled. Missing acquisition times and frame identity remain `None`.

`calibration_epoch_id` and `calibration_products` identify only the epoch assigned
to this observation. Unqualified raw observations retain `None` and an empty
mapping; opting into event input adds no calibration qualification. A policy
decides how to handle missing calibration or a transition, for example by
raising an exception to stop replay or resetting its own state. Epoch coverage does not verify
the physical calibration or prove when its products became available.

The event's fields are frozen. Nested metadata uses immutable mappings and
tuples, detached from the recording. `values` is a separate, writable array copy,
so a policy may modify it without altering recorded measurements. No arbitrary
recording metadata, future measurements or full `Recording` is passed. This is
an interface for trusted Python callbacks, not a sandbox: a callback that already
holds the recording can access it outside this interface. The existing final
digest check rejects a changed recording, but does not undo callback side effects.

Event mode uses the same admission gates, action checks and hypothetical-only
result as array mode. Its result additionally records `policy_input: "event"`.
The default array input and result format are unchanged. See the
[irregular-time example](../examples/shadow_events.py) for a stateful synthetic
policy using elapsed availability time and calibration transitions.

### Conditioned observations

For conditioned observations, validation also reconstructs source dependencies
and covariance. Every output requires a single qualifying source calibration
epoch and explicit availability after all contributing inputs. Missing or mixed
calibration context is rejected before invoking the policy. These checks do not
validate a policy's use of filtered coordinates: its observation model must apply
the same sample/channel selection and linear operator. The policy remains a
hypothetical action generator, including when all conditioning checks pass.

In event mode, `conditioning` contains only the current output's source recording
digest, source stream ID, nonzero `support_indices`, matching `weights`, selected
`channel_indices`, individual `contributing_exposures` and conservative
`ready_not_before_ns`. The support/weight order follows the original declared
selection; it is not silently sorted. Filtered outputs have no fabricated
exposure timestamps or detector frame ID.

This lineage is not a noise likelihood or whitening interface. It exposes no
marginal covariance that could conceal correlations between filtered outputs.
The complete supplied and transformed covariance remains in the validated
recording's conditioning descriptor. A policy requiring a noise model must be
configured separately for that model and the same transformed coordinates.

## Counterfactual evaluation

Changing a command generally changes subsequent wavefront measurements. An
alternate closed-loop result therefore needs a validated simulator or an
authorized physical experiment. A retained recording alone cannot supply that
trajectory. This package deliberately has no operation that converts a shadow
action sequence into a controller-performance score.
