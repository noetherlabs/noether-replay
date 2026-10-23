"""Analytic synthetic AR(1) example; no instrument-performance claim."""

import json

import numpy as np

from noether_replay.conditioning import (
    compose_operators,
    decimation_operator,
    fir_operator,
    propagate_covariance,
)

rho = 0.8
variance = 4e-18
indices = np.arange(6)
covariance = variance * rho ** np.abs(indices[:, None] - indices[None, :])
operator = compose_operators(fir_operator(6, np.array([0.5, 0.5])), decimation_operator(5, 2))
full = propagate_covariance(covariance, operator)
diagonal = propagate_covariance(np.diag(np.diag(covariance)), operator)
print(
    json.dumps(
        {
            "evidence": "synthetic analytic AR(1)",
            "rho": rho,
            "input_variance_m2": variance,
            "operator": "two-sample mean followed by factor-two decimation; valid support",
            "full_output_variance_m2": float(full[0, 0]),
            "diagonal_only_output_variance_m2": float(diagonal[0, 0]),
            "relative_variance_underestimate": float(1 - diagonal[0, 0] / full[0, 0]),
            "adjacent_output_covariance_m2": float(full[0, 1]),
            "formula": "Var[(x_t + x_(t-1))/2] = sigma^2 (1 + rho) / 2",
        },
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
)
