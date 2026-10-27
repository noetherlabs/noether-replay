# Qualifying a fixed-frequency probe response

`qualify_probe_response` estimates an offset and two sinusoidal quadratures from
an explicitly selected WFS window. It integrates the reference oscillator over
each finite exposure and uses the complete supplied noise covariance. Missing
timing, ambiguous calibration coverage or an unidentifiable design withholds the
whole window; the report retains the original indices and exclusion reasons.

This is an offline regression and evidence-qualification operation. It does not
estimate a controller gain, update calibration, infer missing timestamps or run
a controller. A fitted response is not evidence of improved control.

```python
from noether_replay.probe import qualify_probe_response

report = qualify_probe_response(
    recording,
    stream_id="wfs",
    sample_indices=[0, 2, 3, 5, 6, 8],
    channel_indices=[0, 1],
    reference={
        "clock_id": "instrument-clock",
        "origin_ns": 0,
        "origin_uncertainty_ns": 0,
        "frequency_numerator_hz": 80,
        "frequency_denominator": 1,
        "frequency_uncertainty_hz": 0.0,
        "phase_at_origin_rad": 0.0,
        "phase_uncertainty_rad": 0.0,
        "provenance": {"source": "independently characterized probe oscillator"},
    },
    covariance=noise_covariance,  # 12 by 12, in exactly the selected order
    covariance_provenance={"source": "separate noise characterization"},
    observation_model="uniform_exposure_average",
    max_phase_error_rad=0.0,
)
```

The zero uncertainty and zero phase-error budget above are appropriate only when
the source contract establishes exact timing, as in an analytical fixture. They
are not suggested defaults for hardware. Unknown reference origin, clock, phase
or uncertainty is supplied as `None` and yields a withheld result. Frequency is
an externally fixed positive rational number of hertz; it is never selected from
these observations. `provenance` is a nonempty JSON object and can include the
probe direction, basis, generator configuration and independent frequency source.
These are asserted source references, not independently authenticated facts.

## Exposure model and covariance

The supported model is a **uniform exposure average** of a raw
`wfs_measurements` stream in declared units. It is not an integrated photon count,
a rolling-shutter model or a filtered observation. Conditioned observations must
be modelled with their full conditioning operator; this API withholds them rather
than assigning a fictitious single exposure.

For nominal endpoints \(a_i,b_i\), duration \(\Delta_i=b_i-a_i\), midpoint
\(m_i=(a_i+b_i)/2\) and phase \(\theta(t)=2\pi f(t-t_0)+\phi_0\), the design is

\[
c_i=\operatorname{sinc}(f\Delta_i)\cos\theta(m_i),\qquad
s_i=\operatorname{sinc}(f\Delta_i)\sin\theta(m_i),
\]
\[
y_i=b_0+c_i b_c+s_i b_s+\epsilon_i,
\qquad \operatorname{sinc}(x)=\frac{\sin(\pi x)}{\pi x}.
\]

All times are converted to seconds only after subtracting integer epochs or
reducing rational cycles. A large common timestamp origin therefore does not
erase sub-microsecond phase information. A full integer number of periods has
exactly zero quadrature support; the offset alone cannot identify the response.

For \(C\) selected channels, flatten observations in **sample-major** order:
all selected channels of the first selected sample, then those of the next.
With \(X=[1,c,s]\), the joint design is \(H=X\otimes I_C\). The supplied matrix
\(R\) covers every temporal and cross-channel covariance in that same order:

\[
\hat\beta=(H^T R^{-1}H)^{-1}H^T R^{-1}y,
\qquad \operatorname{Cov}(\hat\beta)=(H^T R^{-1}H)^{-1}.
\]

The implementation uses Cholesky whitening and an SVD solve, without explicitly
forming an inverse. Covariance validation uses Replay's scale-relative symmetry
and positive-semidefinite checks. A relative eigenvalue-rank gate is applied
before whitening: covariance eigenvalues must all exceed `rank_rtol` times the
largest eigenvalue. Singular or numerically unresolved covariance withholds the fit; no
diagonal jitter, ridge, noise floor or covariance rescaling from fitted residuals
is introduced. The default relative rank tolerance is `1e-12`, applies also to
the whitened design singular values, and is recorded in the configuration.
The covariance eigenspectrum and numerical rank are retained. Rank failure is numerical non-identifiability,
not evidence of zero physical response.

The coefficient order is term-major: offsets for all selected channels, then
cosine coefficients, then sine coefficients. The report retains their full
covariance, residuals, singular values and the integrated design. This covariance
is conditional on the **nominal timing, correct linear mean model and supplied
measurement-noise covariance**. It is not an unconditional error bar under drift
or an estimate of noise covariance from this same window. With Gaussian noise,
the coefficient error is Gaussian under those assumptions; the function does
not claim that real telemetry satisfies them.

Amplitude is `hypot(cosine, sine)`. The reported phase lag is
`atan2(sine, cosine)`, so the response can be written
\(A\cos(\theta-\varphi)\). It is `None` at exactly zero amplitude. Nonzero noisy
amplitude is positively biased, and phase can be unstable near zero; no amplitude
or phase confidence interval is supplied. An in-phase coefficient alone can be
small because of rotation, rather than attenuation.

## Admission and timing bounds

Every selected exposure requires both endpoints, a known stream uncertainty, and
strictly positive duration even after expanding both endpoint bounds. Measurement
and oscillator clocks must have the same explicit name. No synchronization is
inferred. The selected exposures and their entire intervening time hull must fall
within one unambiguous calibration epoch, including timestamp uncertainty. An
epoch overlap in a gap still withholds the fit. A calibration digest establishes
only an asserted product identity and temporal applicability, not accuracy.

For stream timestamp bound \(u\), reference-origin bound \(u_0\), frequency
bound \(\delta f\), phase bound \(\delta\phi\), and
\(T_i=\max(|a_i-t_0|,|b_i-t_0|)\), use seconds in

\[
\delta t=u+u_0,\qquad
\delta\theta_i\leq\delta\phi+2\pi\{f\delta t+\delta f(T_i+\delta t)\}.
\]

Each averaged quadrature's absolute design error is at most
\(\min(2,\delta\theta_i)\). This follows by parameterizing a uniform exposure
between its two uncertain endpoints and using the unit Lipschitz bound for sine
and cosine. The bound permits correlated errors; it can be conservative. Unknown
bounds or a bound exceeding the caller's explicit `max_phase_error_rad` withhold
the fit. Equality meets that declared budget. This budget does not bound response
coefficient error: conditioning of the design can amplify small timing errors.

**These are deterministic timing bounds, not standard deviations.** They remain
separate from the measurement-noise covariance even when they meet the budget.
No clock-error distribution or independence is assumed. A single frequency cannot
separate an oscillator/clock phase offset from a physical response lag.

Availability is not required for an offline exposure fit and stays unknown when
absent. Contradictory known exposure-to-actuation stage order does withhold the
window. The result does not establish when it could have been computed online.

Selectors must be unique, explicit original array indices. Their order is
preserved; channel indices use C-order flattening within an observation. The joint
window has at most 512 values. Malformed arguments or invalid covariance raise an
error. Scientifically unusable but well-formed windows return `status="withheld"`
and reasons, with no fitted coefficients. No row is dropped, repaired or
interpolated. Configuration and reports are limited to 16 MiB of canonical JSON.

The report binds the recording digest, raw and normalized covariance identities,
configuration, reference provenance and original sample coordinates. Returned
metadata is detached from input objects. The source digest is checked again before
return; this detects changes during ordinary use but is not a concurrent lock.

## Scope of an identified response

A sinusoidal probe in one direction \(q\) measures a response along that
direction, at that frequency. It does not determine a full interaction matrix.
The response retains measurement units; it is not divided by an asserted command
amplitude or labelled an optical gain.

For the simplified frequency-domain feedback convention
\(u=p-Ky\), \(y=Hu+d+n\), the response from requested probe to measurement is
\((I+HK)^{-1}H\), rather than the physical plant \(H\). Actual applied commands,
the controller, delays, probe injection point and disturbance assumptions are
needed for stronger identification claims. This module does not supply them or
establish disturbance independence, instrument validity, safe control transfer
or hardware performance.

The motivation includes [SPRINT2026](https://arxiv.org/abs/2607.27941), which reports
skipped WFS telemetry frames affecting demodulation and describes interpolation
and frequency scanning around an existing probe. This module's fixed-frequency,
exposure-integrated regression is not a reproduction of that instrument study.
Floating-mean sine/cosine fitting is established methodology; see
[Zechmeister and Kürster (2009)](https://arxiv.org/abs/0901.2573). No new
identification method is claimed.

## Analytical example

Run `python examples/probe_response.py`. The deterministic fixture prints a
two-quadrature fit with a nonzero nuisance offset, the predictable finite-exposure
bias of a midpoint comparator, and a withheld window whose exposure endpoint is
missing. It writes no files and performs no acquisition, frequency selection or
parameter tuning. These are analytical software checks, not a benchmark of
observatory calibration or controller quality.
