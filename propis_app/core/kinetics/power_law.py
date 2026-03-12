"""
Classic kinetic curve fitting: power law with dead zone.

Model: G(σ) = s0 * (σ*0.01 - s1)^w

where:
  σ  — supersaturation (%)
  s0 — kinetic coefficient
  s1 — dead zone (fraction, usually negative)
  w  — power exponent (usually 1 or 2)

Optimization: grid search over s1, then least squares for s0
(reproducing the Mathcad algorithm).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class PowerLawResult:
    """Result of power law fitting."""
    s0: float           # kinetic coefficient
    s1: float           # dead zone (fraction of supersaturation)
    w: float            # power exponent
    residual: float     # sum of squared residuals
    sigma_percent: np.ndarray   # σ (%) — input data used for fit
    rate_measured: np.ndarray   # measured rate (μm/min)
    rate_fitted: np.ndarray     # fitted rate (μm/min)
    sig035: float       # σ at which normalized F(σ) > 0.35
    s2: float           # shape parameter (slope of linearized curve)


def power_law_model(sigma_percent: np.ndarray, s0: float, s1: float,
                    w: float = 1.0) -> np.ndarray:
    """
    G(σ) = s0 * (σ*0.01 - s1)^w  for σ*0.01 > s1, else 0.

    Parameters
    ----------
    sigma_percent : array
        Supersaturation in %.
    s0 : float
        Kinetic coefficient.
    s1 : float
        Dead zone threshold (fraction).
    w : float
        Power exponent.
    """
    sigma_frac = sigma_percent * 0.01
    x = sigma_frac - s1
    x = np.maximum(x, 0)
    return s0 * np.power(x, w)


def _grid_search_s1(sigma_percent: np.ndarray, rate: np.ndarray,
                    w: float = 1.0,
                    s1_range: tuple = (-0.02, 0.0),
                    s1_steps: int = 200) -> tuple[float, float, float]:
    """
    Grid search over s1, solving for s0 analytically at each step.

    This reproduces the Mathcad fitting approach.

    Returns
    -------
    s0, s1, residual
    """
    sigma_frac = sigma_percent * 0.01
    best_s0, best_s1, best_residual = 0.0, 0.0, np.inf

    s1_values = np.linspace(s1_range[0], s1_range[1], s1_steps)

    for s1 in s1_values:
        x = sigma_frac - s1
        mask = x > 0
        if np.sum(mask) < 3:
            continue

        xw = np.power(x[mask], w)
        y = rate[mask]

        # Weighted least squares: s0 = Σ(y * xw) / Σ(xw^2)
        xw2 = np.power(x[mask], 2 * w)
        denom = np.sum(xw2)
        if denom < 1e-20:
            continue

        s0 = np.sum(y * xw) / denom

        if s0 <= 0:
            continue

        # Residual
        fitted = s0 * xw
        residual = np.sum((y - fitted) ** 2)

        if residual < best_residual:
            best_s0 = s0
            best_s1 = s1
            best_residual = residual

    return best_s0, best_s1, best_residual


def compute_sig035(sigma_percent: np.ndarray, rate: np.ndarray,
                   s0: float, s1: float, w: float) -> float:
    """
    Find σ at which normalized F(σ) > 0.35.

    F(σ) is the cumulative fraction of rate relative to fitted maximum.
    """
    if len(rate) == 0 or s0 == 0:
        return 0.0

    # Sort by sigma
    sort_idx = np.argsort(sigma_percent)
    sigma_sorted = sigma_percent[sort_idx]
    rate_sorted = rate[sort_idx]

    # Compute fitted values
    fitted = power_law_model(sigma_sorted, s0, s1, w)
    max_fitted = np.max(fitted)
    if max_fitted < 1e-20:
        return 0.0

    # Normalized: F(σ) = fitted / max_fitted
    F = fitted / max_fitted

    for i in range(len(F)):
        if F[i] > 0.35:
            return float(sigma_sorted[i])

    return float(sigma_sorted[-1])


def compute_s2(sigma_percent: np.ndarray, rate: np.ndarray) -> float:
    """
    Compute shape parameter s2 from linear regression of log-log plot.

    ln(R) = ln(s0) + w * ln(σ - s1)
    s2 corresponds to the slope parameter.
    """
    if len(rate) < 3:
        return 0.0

    mask = (rate > 0) & (sigma_percent > 0)
    if np.sum(mask) < 3:
        return 0.0

    log_r = np.log(rate[mask])
    log_s = np.log(sigma_percent[mask])

    # Linear regression
    coeffs = np.polyfit(log_s, log_r, 1)
    return float(coeffs[0])  # slope


def fit_power_law(sigma_percent: np.ndarray, rate: np.ndarray,
                  w: float = 1.0,
                  s1_range: Optional[tuple] = None,
                  use_scipy: bool = True) -> PowerLawResult:
    """
    Fit power law G(σ) = s0 * (σ*0.01 - s1)^w to measured data.

    Parameters
    ----------
    sigma_percent : array
        Supersaturation in %.
    rate : array
        Measured growth rate (μm/min).
    w : float
        Power exponent (fixed). Default 1.0.
    s1_range : tuple, optional
        Range for s1 search. If None, auto-determined from data.
    use_scipy : bool
        If True, refine with scipy.optimize after grid search.

    Returns
    -------
    PowerLawResult
    """
    # Filter valid data
    valid = np.isfinite(sigma_percent) & np.isfinite(rate) & (rate > 0)
    sigma_v = sigma_percent[valid]
    rate_v = rate[valid]

    if len(sigma_v) < 3:
        return PowerLawResult(
            s0=0, s1=0, w=w, residual=np.inf,
            sigma_percent=sigma_v, rate_measured=rate_v,
            rate_fitted=np.zeros_like(rate_v),
            sig035=0, s2=0,
        )

    # Auto s1 range
    if s1_range is None:
        sigma_min = np.min(sigma_v) * 0.01
        s1_range = (sigma_min - 0.01, sigma_min + 0.005)

    # Grid search
    s0, s1, residual = _grid_search_s1(sigma_v, rate_v, w=w, s1_range=s1_range)

    # Refine with scipy
    if use_scipy and s0 > 0:
        try:
            def model(sigma, s0_opt, s1_opt):
                return power_law_model(sigma, s0_opt, s1_opt, w)

            popt, pcov = curve_fit(model, sigma_v, rate_v,
                                   p0=[s0, s1 * 100],
                                   bounds=([0, -10], [np.inf, 10]),
                                   maxfev=5000)
            s0_opt, s1_opt_pct = popt
            s1_opt = s1_opt_pct  # s1 in the model is already used as fraction
            fitted = model(sigma_v, s0_opt, s1_opt_pct)
            res = np.sum((rate_v - fitted) ** 2)
            if res < residual:
                s0, s1, residual = s0_opt, s1_opt_pct * 0.01, res
        except (RuntimeError, ValueError):
            pass

    # Compute fitted values
    rate_fitted = power_law_model(sigma_v, s0, s1, w)

    # Sig035
    sig035 = compute_sig035(sigma_v, rate_v, s0, s1, w)

    # s2 parameter
    s2 = compute_s2(sigma_v, rate_v)

    return PowerLawResult(
        s0=s0, s1=s1, w=w, residual=residual,
        sigma_percent=sigma_v,
        rate_measured=rate_v,
        rate_fitted=rate_fitted,
        sig035=sig035,
        s2=s2,
    )
