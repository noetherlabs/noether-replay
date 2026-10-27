# Telemetry evidence report

Recording: **CIAO**  
Mode: `observational` · Status: `evidence_gaps`

Reconstructed events: 0. Clock domains: 0.

| Stream | Samples | Units | Calibration-covered samples |
| --- | ---: | --- | ---: |
| loop:HO Loop:commands | 15000 | unknown | n/a |
| loop:TT Loop:commands | 15000 | unknown | n/a |
| wfs:WFS:measurements | 15000 | unknown | 0 |
| wfs:WFS:pixels | 316 | unknown | 0 |

## Findings

- `unknown_units` — loop:HO Loop:commands (1): Physical units are not established by the source.
- `unknown_clock_uncertainty` — loop:HO Loop:commands (1): Nominal timestamps cannot establish an exact causal ordering.
- `missing_event_times` — loop:HO Loop:commands (15000): These samples have no exposure, availability or actuation event times.
- `unknown_command_semantics` — loop:HO Loop:commands (15000): The source does not establish whether these commands were requested or physically applied.
- `unknown_units` — loop:TT Loop:commands (1): Physical units are not established by the source.
- `unknown_clock_uncertainty` — loop:TT Loop:commands (1): Nominal timestamps cannot establish an exact causal ordering.
- `missing_event_times` — loop:TT Loop:commands (15000): These samples have no exposure, availability or actuation event times.
- `unknown_command_semantics` — loop:TT Loop:commands (15000): The source does not establish whether these commands were requested or physically applied.
- `unknown_units` — wfs:WFS:measurements (1): Physical units are not established by the source.
- `unknown_clock_uncertainty` — wfs:WFS:measurements (1): Nominal timestamps cannot establish an exact causal ordering.
- `missing_event_times` — wfs:WFS:measurements (15000): These samples have no exposure, availability or actuation event times.
- `missing_availability` — wfs:WFS:measurements (15000): Causal controller replay requires observation availability, not just sampling frequency.
- `unqualified_calibration_interval` — wfs:WFS:measurements (15000): No unique calibration epoch covers the complete exposure and timing uncertainty.
- `unknown_units` — wfs:WFS:pixels (1): Physical units are not established by the source.
- `unknown_clock_uncertainty` — wfs:WFS:pixels (1): Nominal timestamps cannot establish an exact causal ordering.
- `missing_event_times` — wfs:WFS:pixels (316): These samples have no exposure, availability or actuation event times.
- `missing_availability` — wfs:WFS:pixels (316): Causal controller replay requires observation availability, not just sampling frequency.
- `frame_id_gaps` — wfs:WFS:pixels (15435): Unrepresented integer frame IDs; this is not proof of detector frame loss without the counter contract.
- `unqualified_calibration_interval` — wfs:WFS:pixels (316): No unique calibration epoch covers the complete exposure and timing uncertainty.

## Interpretation

- Recording consistency is not instrument qualification or controller validation.
- Calibration hashes bind references; calibration accuracy is not tested.
- Recorded closed-loop outcomes cannot evaluate a different controller.
- Hashes establish content consistency, not source authenticity.

## Reproducibility

Recording SHA-256: `e50cf64d4dd348f26239f17d777570df2792393f1f7232b79d4317a9370751a2`  
Timeline SHA-256: `126db27f51d6e0de5bca26d7ee75434e5f6d40efa071457b4e2ecc0c0befb836`
