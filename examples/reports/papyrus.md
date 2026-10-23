# Telemetry evidence report

Recording: **aot-4e1568033171ad2d**  
Mode: `observational` · Status: `evidence_gaps`

Reconstructed events: 0. Clock domains: 0.

| Stream | Samples | Units | Calibration-covered samples |
| --- | ---: | --- | ---: |
| loop:20230525\_2304\_20B1FBD7:commands | 200 | unknown | n/a |
| wfs:ACE.PYRAMID.REMOTE.Wfs.25:measurements | 200 | unknown | 0 |
| wfs:ACE.PYRAMID.REMOTE.Wfs.25:pixels | 200 | unknown | 0 |

## Findings

- `unknown_units` — loop:20230525\_2304\_20B1FBD7:commands (1): Physical units are not established by the source.
- `unknown_clock_uncertainty` — loop:20230525\_2304\_20B1FBD7:commands (1): Nominal timestamps cannot establish an exact causal ordering.
- `missing_event_times` — loop:20230525\_2304\_20B1FBD7:commands (200): These samples have no exposure, availability or actuation event times.
- `unknown_command_semantics` — loop:20230525\_2304\_20B1FBD7:commands (200): The source does not establish whether these commands were requested or physically applied.
- `unknown_frame_identity` — loop:20230525\_2304\_20B1FBD7:commands (200): Array position is retained; missing frames cannot be inferred from it.
- `unknown_units` — wfs:ACE.PYRAMID.REMOTE.Wfs.25:measurements (1): Physical units are not established by the source.
- `unknown_clock_uncertainty` — wfs:ACE.PYRAMID.REMOTE.Wfs.25:measurements (1): Nominal timestamps cannot establish an exact causal ordering.
- `missing_event_times` — wfs:ACE.PYRAMID.REMOTE.Wfs.25:measurements (200): These samples have no exposure, availability or actuation event times.
- `missing_availability` — wfs:ACE.PYRAMID.REMOTE.Wfs.25:measurements (200): Causal controller replay requires observation availability, not just sampling frequency.
- `unknown_frame_identity` — wfs:ACE.PYRAMID.REMOTE.Wfs.25:measurements (200): Array position is retained; missing frames cannot be inferred from it.
- `unqualified_calibration_interval` — wfs:ACE.PYRAMID.REMOTE.Wfs.25:measurements (200): No unique calibration epoch covers the complete exposure and timing uncertainty.
- `unknown_units` — wfs:ACE.PYRAMID.REMOTE.Wfs.25:pixels (1): Physical units are not established by the source.
- `unknown_clock_uncertainty` — wfs:ACE.PYRAMID.REMOTE.Wfs.25:pixels (1): Nominal timestamps cannot establish an exact causal ordering.
- `missing_event_times` — wfs:ACE.PYRAMID.REMOTE.Wfs.25:pixels (200): These samples have no exposure, availability or actuation event times.
- `missing_availability` — wfs:ACE.PYRAMID.REMOTE.Wfs.25:pixels (200): Causal controller replay requires observation availability, not just sampling frequency.
- `unknown_frame_identity` — wfs:ACE.PYRAMID.REMOTE.Wfs.25:pixels (200): Array position is retained; missing frames cannot be inferred from it.
- `unqualified_calibration_interval` — wfs:ACE.PYRAMID.REMOTE.Wfs.25:pixels (200): No unique calibration epoch covers the complete exposure and timing uncertainty.

## Interpretation

- Recording consistency is not instrument qualification or controller validation.
- Calibration hashes bind references; calibration accuracy is not tested.
- Recorded closed-loop outcomes cannot evaluate a different controller.
- Hashes establish content consistency, not source authenticity.

## Reproducibility

Recording SHA-256: `8d6dc9fa1c8e3b384b7667beaa741934eab217e85270687d10f00640ebc82f09`  
Timeline SHA-256: `6b3b649711496a8cf3b8a1a750a5fe1636bb15cdb3953d1df44671f4e98ccd78`
