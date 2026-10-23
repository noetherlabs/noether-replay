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

For conditioned observations, validation also reconstructs source dependencies
and covariance. Every output requires a single qualifying source calibration
epoch and explicit availability after all contributing inputs. Missing or mixed
calibration context is rejected before invoking the policy. These checks do not
validate a policy's use of filtered coordinates: its observation model must apply
the same sample/channel selection and linear operator. The policy remains a
hypothetical action generator, including when all conditioning checks pass.

## Counterfactual evaluation

Changing a command generally changes subsequent wavefront measurements. An
alternate closed-loop result therefore needs a validated simulator or an
authorized physical experiment. A retained recording alone cannot supply that
trajectory. This package deliberately has no operation that converts a shadow
action sequence into a controller-performance score.
