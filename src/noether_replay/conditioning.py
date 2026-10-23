"""Finite-window linear conditioning with full temporal covariance.

For y = A x and Cov[x] = R, Cov[y] = A R A.T. Selection, overlapping
filters and decimation therefore retain induced cross-sample correlations.
Dense operators are limited to 512 rows/columns; whole-session covariance
requires a structured representation outside this module's scope.
"""

from __future__ import annotations

import numpy as np

MAX_DIMENSION = 512


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if (
        raw.dtype.kind not in "biuf"
        or raw.ndim != 2
        or any(size < 1 or size > MAX_DIMENSION for size in raw.shape)
    ):
        raise ValueError(f"{name} must be a real matrix with dimensions in [1, {MAX_DIMENSION}]")
    result = raw.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


def validate_covariance(covariance: np.ndarray) -> np.ndarray:
    """Validate symmetry/PSD at scale-relative float64 roundoff tolerance.

    Only negative eigenvalues within 64*n*eps of the scaled spectral magnitude
    are projected to zero. No diagonal jitter or physical noise floor is added.
    The returned array is independent of the input. Units may be arbitrarily
    small; a fixed absolute tolerance would incorrectly admit invalid SI data.
    """
    result = _matrix(covariance, "Covariance")
    if result.shape[0] != result.shape[1]:
        raise ValueError("Covariance must be square")
    scale = float(np.max(np.abs(result)))
    if scale == 0:
        return result
    scaled = result / scale
    tolerance = 64 * result.shape[0] * np.finfo(np.float64).eps
    if np.max(np.abs(scaled - scaled.T)) > tolerance:
        raise ValueError("Covariance must be symmetric")
    scaled = 0.5 * scaled + 0.5 * scaled.T
    eigenvalues, eigenvectors = np.linalg.eigh(scaled)
    bound = tolerance * max(1.0, float(np.max(np.abs(eigenvalues))))
    if eigenvalues[0] < -bound:
        raise ValueError("Covariance must be positive semidefinite")
    if eigenvalues[0] < 0:
        scaled = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
    with np.errstate(over="ignore", invalid="ignore"):
        result = (0.5 * scaled + 0.5 * scaled.T) * scale
    if not np.isfinite(result).all():
        raise ValueError("Covariance projection exceeds float64 numerical range")
    return result


def propagate_covariance(covariance: np.ndarray, operator: np.ndarray) -> np.ndarray:
    """Propagate complete covariance; never discard temporal off-diagonal terms."""
    covariance = validate_covariance(covariance)
    operator = _matrix(operator, "Operator")
    if operator.shape[1] != covariance.shape[0]:
        raise ValueError("Operator input dimension must match covariance")
    with np.errstate(over="ignore", invalid="ignore"):
        result = operator @ covariance @ operator.T
    return validate_covariance(result)


def compose_operators(*operators: np.ndarray) -> np.ndarray:
    """Compose in application order: compose_operators(A, B) returns B @ A."""
    if not operators:
        raise ValueError("At least one operator is required")
    result = _matrix(operators[0], "Operator")
    for item in operators[1:]:
        next_operator = _matrix(item, "Operator")
        if next_operator.shape[1] != result.shape[0]:
            raise ValueError("Operator dimensions do not compose")
        with np.errstate(over="ignore", invalid="ignore"):
            result = next_operator @ result
        if not np.isfinite(result).all():
            raise ValueError("Operator composition overflowed")
    return result


def _dimension(size: int) -> int:
    if (
        not isinstance(size, (int, np.integer))
        or isinstance(size, (bool, np.bool_))
        or not 1 <= size <= MAX_DIMENSION
    ):
        raise ValueError(f"Window size must be an integer in [1, {MAX_DIMENSION}]")
    return int(size)


def selection_operator(size: int, indices: list[int] | tuple[int, ...]) -> np.ndarray:
    """Select explicit samples, preserving order and repeated-sample covariance."""
    size = _dimension(size)
    indices = tuple(indices)
    _dimension(len(indices))
    if any(
        not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_)) or not 0 <= i < size
        for i in indices
    ):
        raise ValueError("Selection indices must be in-range integers")
    result = np.zeros((len(indices), size))
    result[np.arange(len(indices)), indices] = 1
    return result


def decimation_operator(size: int, factor: int, phase: int = 0) -> np.ndarray:
    """Select phase, phase+factor, ...; anti-alias filtering is a separate operator."""
    size = _dimension(size)
    factor = _dimension(factor)
    if (
        not isinstance(phase, (int, np.integer))
        or isinstance(phase, (bool, np.bool_))
        or not 0 <= phase < min(factor, size)
    ):
        raise ValueError("Decimation phase must select a sample within the first factor")
    return selection_operator(size, tuple(range(int(phase), size, factor)))


def fir_operator(size: int, taps: np.ndarray) -> np.ndarray:
    """Valid causal FIR; taps[0] multiplies the newest sample.

    The first output belongs to input index len(taps)-1. No padding or implicit
    interpolation is performed, and no timestamp shift is silently applied.
    """
    size = _dimension(size)
    taps = np.asarray(taps)
    if (
        taps.dtype.kind not in "biuf"
        or taps.ndim != 1
        or not 1 <= taps.size <= size
        or not np.isfinite(taps).all()
    ):
        raise ValueError("FIR taps must be a finite nonempty real vector no longer than the window")
    with np.errstate(over="ignore", invalid="ignore"):
        taps = taps.astype(np.float64)
    if not np.isfinite(taps).all():
        raise ValueError("FIR taps exceed float64 numerical range")
    result = np.zeros((size - len(taps) + 1, size))
    for row in range(result.shape[0]):
        result[row, row : row + len(taps)] = taps[::-1]
    return result
