"""
Classic kinetic curve fitting: power law with dead zone.

Model: G(sigma) = s0 * (sigma - s1)^w   for sigma > s1, else 0

where:
  sigma — supersaturation (%)  [NOT fraction!]
  s0    — kinetic coefficient
  s1    — dead zone threshold (%, can be negative)
  w     — power exponent (usually 1)

Mathcad fitting: grid search over s1 (stepping through actual VX values
with step 0.01), analytical s0 at each step, full residual including
zero-rate points below threshold.

s2: linear regression sqrt(R) vs sigma%, for 0 < R < 0.3.
    s2 = -intercept / slope  (sigma% where sqrt(R) = 0).

Sig035: LOESS smoothing of R(sigma%), then first sigma% where LOESS > 0.35.

Reference: docs/mathcad_classic_algorithm_strict.md
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PowerLawResult:
    """Result of power law fitting."""
    s0: float           # kinetic coefficient
    s1: float           # dead zone (% supersaturation)
    w: float            # power exponent
    residual: float     # sum of squared residuals
    sigma_percent: np.ndarray   # sigma (%) — input data used for fit
    rate_measured: np.ndarray   # measured rate (mm/day)
    rate_fitted: np.ndarray     # fitted rate (mm/day)
    sig035: float       # sigma at which LOESS(R) > 0.35
    s2: float           # shape parameter: -intercept/slope of sqrt(R) vs sigma%


def power_law_model(sigma_percent: np.ndarray, s0: float, s1: float,
                    w: float = 1.0) -> np.ndarray:
    """
    G(sigma) = s0 * (sigma - s1)^w  for sigma > s1, else 0.

    sigma and s1 are both in % units.
    """
    x = sigma_percent - s1
    x = np.maximum(x, 0)
    return s0 * np.power(x, w)


def _grid_search_s1_mathcad(sigma_percent: np.ndarray, rate: np.ndarray,
                             w: float = 1.0) -> tuple:
    """
    Mathcad-style grid search over s1.

    Iterates through consecutive VX values with step 0.01.
    For each candidate threshold x:
      - Points with sigma > x: fit s0 = sum(rate * (sigma-x)^w) / sum((sigma-x)^(2w))
      - Points with sigma <= x: contribute rate^2 to residual (model = 0)
      - Total residual includes both parts

    Returns (s0, s1, residual).
    """
    n = len(sigma_percent)
    if n < 3:
        return 0.0, 0.0, np.inf

    # Sort by sigma
    sort_idx = np.argsort(sigma_percent)
    VX = sigma_percent[sort_idx]
    VY = rate[sort_idx]

    best_s0, best_s1, best_res = 0.0, 0.0, np.inf

    # Iterate through all intervals between consecutive VX values
    for i in range(n - 1):
        # Step through x from VX[i] to VX[i+1] with step 0.01
        x = VX[i]
        while x < VX[i + 1]:
            # Points above threshold (k >= i means sigma >= x approximately)
            # Find first index where VX >= x
            above = VX >= x
            if np.sum(above) < 2:
                x += 0.01
                continue

            VX_above = VX[above]
            VY_above = VY[above]

            # Analytical s0
            xw = np.power(VX_above - x, w)
            xw2 = np.power(VX_above - x, 2 * w)
            denom = np.sum(xw2)
            if denom < 1e-20:
                x += 0.01
                continue

            q1 = np.sum(xw * VY_above)
            s0_cand = q1 / denom

            # Full residual (Mathcad formula):
            # f = sum(VY[below]^2) + sum((s0*(VX[above]-x)^w - VY[above])^2)
            below = ~above
            f = np.sum(VY[below] ** 2)
            f += np.sum((s0_cand * xw - VY_above) ** 2)

            if f < best_res:
                best_s0 = s0_cand
                best_s1 = x
                best_res = f

            x += 0.01

    return best_s0, best_s1, best_res


def compute_sig035_loess(sigma_percent: np.ndarray, rate: np.ndarray,
                         span: float = 0.2) -> float:
    """
    Sig035: first sigma% where LOESS-smoothed R(sigma) > 0.504 mm/day (= 0.35 мкм/мин).

    Mathcad uses loess() + interp() with span=0.2 (or span1).

    Parameters
    ----------
    sigma_percent : array
        Supersaturation values (%), sorted by sigma.
    rate : array
        Growth rate (mm/day), corresponding to sigma_percent.
    span : float
        LOESS span parameter.

    Returns
    -------
    sig035 : float
        Supersaturation (%) at which smoothed R first exceeds 0.35.
    """
    if len(sigma_percent) < 5:
        return 0.0

    # Sort by sigma
    sort_idx = np.argsort(sigma_percent)
    VX = sigma_percent[sort_idx]
    VY = rate[sort_idx]

    # Use statsmodels LOWESS if available, otherwise simple moving average
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        # lowess returns (x, y_smooth) sorted by x
        result = lowess(VY, VX, frac=span, return_sorted=True)
        VX_smooth = result[:, 0]
        VY_smooth = result[:, 1]
    except ImportError:
        # Fallback: simple moving average
        window = max(3, int(len(VX) * span))
        if window % 2 == 0:
            window += 1
        kernel = np.ones(window) / window
        VY_smooth = np.convolve(VY, kernel, mode='same')
        VX_smooth = VX

    # Find first sigma where smoothed R > 0.504 mm/day (= 0.35 мкм/мин * 1.44)
    for i in range(len(VX_smooth)):
        if VY_smooth[i] > 0.504:
            return float(VX_smooth[i])

    return float(VX_smooth[-1]) if len(VX_smooth) > 0 else 0.0


def compute_s2_mathcad(sigma_percent: np.ndarray, rate: np.ndarray) -> float:
    """
    Mathcad s2: linear regression sqrt(R) vs sigma%, for 0 < R < 0.3.

    sqrt(R) = Q1 + Q2 * sigma%
    s2 = -Q1 / Q2  (x-intercept where sqrt(R) = 0)

    Data is taken from z (unsorted, in original order), and we stop
    at the first negative R (transition to dissolution).

    Parameters
    ----------
    sigma_percent : array
        Supersaturation (%), in original order (not sorted).
    rate : array
        Growth rate (mm/day), in original order.

    Returns
    -------
    s2 : float
    """
    if len(rate) < 3:
        return 0.0

    # Mathcad algorithm: stop at first R < 0, collect 0 < R < 0.3.
    # Additional sigma > 0 filter compensates for LOWESS smoothing artifacts
    # that produce tiny non-zero rates in the dead zone (sigma ≈ 0).
    sigma_sel = []
    sqrt_r_sel = []

    for i in range(len(rate)):
        if rate[i] < 0:
            break
        if 0 < rate[i] < 0.432 and sigma_percent[i] > 0:  # 0.3 мкм/мин * 1.44 = 0.432 мм/день
            sigma_sel.append(sigma_percent[i])
            sqrt_r_sel.append(np.sqrt(rate[i]))

    if len(sigma_sel) < 3:
        return 0.0

    x = np.array(sigma_sel)
    y = np.array(sqrt_r_sel)

    # Linear regression: y = Q1 + Q2 * x
    # Using numpy polyfit (degree 1): coeffs[0] = Q2 (slope), coeffs[1] = Q1 (intercept)
    try:
        coeffs = np.polyfit(x, y, 1)
        Q2 = coeffs[0]  # slope
        Q1 = coeffs[1]  # intercept

        if abs(Q2) < 1e-15:
            return 0.0

        s2 = -Q1 / Q2
        return float(s2)
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


# Legacy API — kept for backward compatibility
def compute_sig035(sigma_percent: np.ndarray, rate: np.ndarray,
                   s0: float, s1: float, w: float) -> float:
    """Legacy: analytical Sig035 from power law. Use compute_sig035_loess instead."""
    return compute_sig035_loess(sigma_percent, rate)


def compute_s2(sigma_percent: np.ndarray, rate: np.ndarray,
               s1: float = 0.0) -> float:
    """Legacy: compute s2. Now delegates to Mathcad formula."""
    return compute_s2_mathcad(sigma_percent, rate)


def fit_dissolution(sigma_percent: np.ndarray, rate: np.ndarray,
                    w: float = 1.0) -> PowerLawResult:
    """
    Fit dissolution branch: R < 0, sigma < 0.

    Takes raw (negative) sigma and rate, mirrors to positive,
    fits the same power law, then stores dissolution parameters.

    Model: R(σ) = -s0_d * (|σ| - s1_d)^w   for σ < -s1_d

    Returns PowerLawResult with s0_d, s1_d (positive values).
    """
    # Select dissolution data: sigma < 0 AND rate < 0
    mask = np.isfinite(sigma_percent) & np.isfinite(rate) & (sigma_percent < 0) & (rate < 0)
    sigma_diss = np.abs(sigma_percent[mask])   # mirror to positive
    rate_diss = np.abs(rate[mask])              # mirror to positive

    if len(sigma_diss) < 3:
        return PowerLawResult(
            s0=0, s1=0, w=w, residual=np.inf,
            sigma_percent=sigma_diss, rate_measured=rate_diss,
            rate_fitted=np.zeros_like(rate_diss),
            sig035=0, s2=0,
        )

    # Same grid search on mirrored data
    s0_d, s1_d, residual = _grid_search_s1_mathcad(sigma_diss, rate_diss, w=w)

    rate_fitted = power_law_model(sigma_diss, s0_d, s1_d, w)

    return PowerLawResult(
        s0=s0_d, s1=s1_d, w=w, residual=residual,
        sigma_percent=sigma_diss,
        rate_measured=rate_diss,
        rate_fitted=rate_fitted,
        sig035=0, s2=0,
    )


def fit_power_law(sigma_percent: np.ndarray, rate: np.ndarray,
                  w: float = 1.0,
                  s1_range: Optional[tuple] = None,
                  use_scipy: bool = False) -> PowerLawResult:
    """
    Fit power law G(sigma) = s0 * (sigma - s1)^w to measured data.

    Uses Mathcad-style grid search: iterate through actual data values
    with step 0.01, include zero-rate points in residual.

    Parameters
    ----------
    sigma_percent : array
        Supersaturation in %.
    rate : array
        Measured growth rate (mm/day).
    w : float
        Power exponent (fixed). Default 1.0.
    s1_range : tuple, optional
        Not used in Mathcad mode (kept for API compatibility).
    use_scipy : bool
        Not used in Mathcad mode.

    Returns
    -------
    PowerLawResult
    """
    # Filter valid data
    valid = np.isfinite(sigma_percent) & np.isfinite(rate) & (rate >= 0)
    sigma_v = sigma_percent[valid]
    rate_v = rate[valid]

    if len(sigma_v) < 3:
        return PowerLawResult(
            s0=0, s1=0, w=w, residual=np.inf,
            sigma_percent=sigma_v, rate_measured=rate_v,
            rate_fitted=np.zeros_like(rate_v),
            sig035=0, s2=0,
        )

    # Mathcad grid search
    s0, s1, residual = _grid_search_s1_mathcad(sigma_v, rate_v, w=w)

    # Compute fitted values
    rate_fitted = power_law_model(sigma_v, s0, s1, w)

    # Sig035 via LOESS
    sig035 = compute_sig035_loess(sigma_v, rate_v, span=0.2)

    # s2 via sqrt(R) regression (use unsorted data!)
    s2 = compute_s2_mathcad(sigma_percent, rate)

    return PowerLawResult(
        s0=s0, s1=s1, w=w, residual=residual,
        sigma_percent=sigma_v,
        rate_measured=rate_v,
        rate_fitted=rate_fitted,
        sig035=sig035,
        s2=s2,
    )
