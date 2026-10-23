# Telemetry evidence report

Recording: **synthetic-calibration-expiry**  
Mode: `observational` · Status: `evidence_gaps`

Reconstructed events: 72. Clock domains: 1.

| Stream | Samples | Units | Calibration-covered samples |
| --- | ---: | --- | ---: |
| dm-applied | 12 | m | n/a |
| dm-requested | 12 | m | n/a |
| wfs | 12 | m | 8 |

## Findings

- `frame_id_gaps` — dm-applied (1): Unrepresented integer frame IDs; this is not proof of detector frame loss without the counter contract.
- `frame_id_gaps` — dm-requested (1): Unrepresented integer frame IDs; this is not proof of detector frame loss without the counter contract.
- `frame_id_gaps` — wfs (1): Unrepresented integer frame IDs; this is not proof of detector frame loss without the counter contract.
- `unqualified_calibration_interval` — wfs (4): No unique calibration epoch covers the complete exposure and timing uncertainty.

## Interpretation

- Recording consistency is not instrument qualification or controller validation.
- Calibration hashes bind references; calibration accuracy is not tested.
- Recorded closed-loop outcomes cannot evaluate a different controller.
- Hashes establish content consistency, not source authenticity.

## Reproducibility

Recording SHA-256: `4cd38b1958e79fb1b74086edd6d043659f855b13f09cb38eff319a1969decdfa`  
Timeline SHA-256: `b0ab6dbe39b446d1ad1f4cd53d868cc8288f0a147ab7fe7ff7f2af57848ebfa8`
