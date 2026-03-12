"""
BCF (Burton-Cabrera-Frank) kinetic model with dead zone.

Model: R(σ) = β * (σ - σ_d)^2 / (σ_1 + σ - σ_d)

where:
  σ    — supersaturation (fraction)
  σ_d  — dead zone (critical supersaturation below which no growth)
  β    — kinetic coefficient
  σ_1  — transition parameter (determines curve shape)

At low σ: R ≈ β*(σ-σ_d)^2/σ_1  (parabolic, spiral growth)
At high σ: R ≈ β*(σ-σ_d)        (linear, rough growth)

Reference: Burton, Cabrera, Frank (1951); Cabrera & Vermilyea (1958) for dead zone.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit, minimize


@dataclass
class BCFResult:
    """Result of BCF model fitting."""
    beta: float          # kinetic coefficient
    sigma_d: float       # dead zone (fraction)
    sigma_1: float       # transition parameter (fraction)
    residual: float      # sum of squared residuals
    sigma_percent: np.ndarray   # input σ (%)
    rate_measured: np.ndarray   # measured R (μm/min)
    rate_fitted: np.ndarray     # fitted R (μm/min)


def bcf_model(sigma_percent: np.ndarray, beta: float,
              sigma_d: float, sigma_1: float) -> np.ndarray:
    """
    BCF growth rate model with dead zone.

    R(σ) = β * (σ - σ_d)^2 / (σ_1 + σ - σ_d)  for σ > σ_d
    R(σ) = 0                                     for σ ≤ σ_d

    Parameters
    ----------
    sigma_percent : array
        Supersaturation in %.
    beta : float
        Kinetic coefficient.
    sigma_d : float
        Dead zone in % supersaturation.
    sigma_1 : float
        Transition parameter in %.

    Returns
    -------
    rate : array
        Growth rate (μm/min).
    """
    sigma = sigma_percent
    x = sigma - sigma_d
    x = np.maximum(x, 0)

    denom = sigma_1 + x
    denom = np.maximum(denom, 1e-10)

    return beta * x**2 / denom


def bcf_model_fraction(sigma_frac: np.ndarray, beta: float,
                       sigma_d_frac: float,
                       sigma_1_frac: float) -> np.ndarray:
    """BCF model using fractional supersaturation (not percent)."""
    return bcf_model(sigma_frac * 100, beta, sigma_d_frac * 100,
                     sigma_1_frac * 100)


def fit_bcf(sigma_percent: np.ndarray, rate: np.ndarray,
            initial_guess: Optional[dict] = None) -> BCFResult:
    """
    Fit BCF model to measured data.

    Parameters
    ----------
    sigma_percent : array
        Supersaturation in %.
    rate : array
        Measured growth rate (μm/min).
    initial_guess : dict, optional
        Initial parameters: {"beta": ..., "sigma_d": ..., "sigma_1": ...}

    Returns
    -------
    BCFResult
    """
    # Filter valid data
    valid = np.isfinite(sigma_percent) & np.isfinite(rate) & (rate > 0)
    sigma_v = sigma_percent[valid]
    rate_v = rate[valid]

    if len(sigma_v) < 4:
        return BCFResult(
            beta=0, sigma_d=0, sigma_1=1,
            residual=np.inf,
            sigma_percent=sigma_v,
            rate_measured=rate_v,
            rate_fitted=np.zeros_like(rate_v),
        )

    # Initial guess
    if initial_guess is None:
        sigma_min = np.min(sigma_v)
        rate_max = np.max(rate_v)
        sigma_max = np.max(sigma_v)

        initial_guess = {
            "beta": rate_max / (sigma_max - sigma_min + 0.1),
            "sigma_d": max(sigma_min - 0.5, 0),
            "sigma_1": (sigma_max - sigma_min) / 2,
        }

    p0 = [initial_guess["beta"], initial_guess["sigma_d"],
          initial_guess["sigma_1"]]

    try:
        popt, pcov = curve_fit(
            bcf_model, sigma_v, rate_v,
            p0=p0,
            bounds=([0, -5, 0.01], [np.inf, np.max(sigma_v), 100]),
            maxfev=10000,
        )
        beta, sigma_d, sigma_1 = popt
        rate_fitted = bcf_model(sigma_v, beta, sigma_d, sigma_1)
        residual = float(np.sum((rate_v - rate_fitted) ** 2))
    except (RuntimeError, ValueError):
        # Fallback: manual optimization
        def objective(params):
            b, sd, s1 = params
            if b <= 0 or s1 <= 0:
                return 1e20
            fitted = bcf_model(sigma_v, b, sd, s1)
            return np.sum((rate_v - fitted) ** 2)

        result = minimize(objective, p0, method="Nelder-Mead",
                          options={"maxiter": 10000})
        beta, sigma_d, sigma_1 = result.x
        rate_fitted = bcf_model(sigma_v, beta, sigma_d, sigma_1)
        residual = float(result.fun)

    return BCFResult(
        beta=beta,
        sigma_d=sigma_d,
        sigma_1=sigma_1,
        residual=residual,
        sigma_percent=sigma_v,
        rate_measured=rate_v,
        rate_fitted=rate_fitted,
    )


def bcf_to_power_law_comparison(bcf: BCFResult,
                                sigma_range: Optional[np.ndarray] = None
                                ) -> dict:
    """
    Compare BCF fit with equivalent power law parameters.

    At low σ: R ≈ (β/σ₁)*(σ-σ_d)^2  → power law with w=2
    At high σ: R ≈ β*(σ-σ_d)        → power law with w=1

    Returns dict with comparison metrics.
    """
    if sigma_range is None:
        sigma_range = np.linspace(0, 10, 200)

    bcf_rate = bcf_model(sigma_range, bcf.beta, bcf.sigma_d, bcf.sigma_1)

    # Effective power law exponent at each point
    # d(ln R)/d(ln(σ-σ_d))
    x = sigma_range - bcf.sigma_d
    x = np.maximum(x, 1e-10)
    denom = bcf.sigma_1 + x
    effective_w = 2.0 * bcf.sigma_1 / denom  # ranges from 2 (low σ) to 1 (high σ)

    return {
        "sigma_range": sigma_range,
        "bcf_rate": bcf_rate,
        "effective_exponent": effective_w,
        "low_sigma_w": 2.0,
        "high_sigma_w": 1.0,
        "transition_sigma": bcf.sigma_d + bcf.sigma_1,
    }
