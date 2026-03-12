"""
Determination of saturation temperature (tn) and dead zone (Td).

Physical meaning:
  - tn: temperature where growth rate = 0 (transition from growth to dissolution)
  - Td: dead zone = tn - T_onset (°C), where T_onset is the temperature
    at which visible growth first appears
  - s1: dead zone in % supersaturation

Method (classic):
  1. Smooth the envelope of the signal in the saturation region
  2. Find zero-crossing of the derivative (growth rate = 0)
  3. tn = temperature at that point

Method (modern):
  1. Use instantaneous frequency from Hilbert transform
  2. Find where frequency → 0 (no more fringes = no growth)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal as sig
from scipy.interpolate import UnivariateSpline

from .solubility import SolubilitySet, supersaturation_percent


@dataclass
class SaturationResult:
    """Result of saturation temperature determination."""
    tn: float                    # saturation temperature (°C)
    tn_index: int                # sample index of saturation point
    td: float                    # dead zone (°C)
    s1: float                    # dead zone in % supersaturation
    t_onset: float               # temperature of growth onset (°C)
    t_onset_index: int           # sample index of growth onset
    envelope: np.ndarray         # smoothed envelope used for detection
    derivative: np.ndarray       # derivative of envelope


def compute_envelope_loess(signal: np.ndarray, span: float = 0.15) -> np.ndarray:
    """
    Compute smoothed envelope approximating Mathcad's loess smoothing.

    Uses Savitzky-Golay filter as a local polynomial approximation.
    """
    # Compute analytic signal envelope
    from scipy.signal import hilbert as scipy_hilbert
    analytic = scipy_hilbert(signal)
    env = np.abs(analytic)

    # Smooth with large window (like loess with span)
    window = int(len(env) * span)
    if window % 2 == 0:
        window += 1
    window = max(window, 5)
    window = min(window, len(env))

    return sig.savgol_filter(env, window, polyorder=2)


def find_saturation_classic(signal: np.ndarray, temp_c: np.ndarray,
                            im: int, isat: int,
                            span: float = 0.15,
                            smooth_factor: float = 0.01) -> tuple[float, int]:
    """
    Find saturation temperature using classic method (envelope derivative zero).

    Parameters
    ----------
    signal : array
        Interferometric signal in the saturation region.
    temp_c : array
        Temperature array, same length as signal.
    im : int
        Left boundary of the search region (sample index, relative to signal start).
    isat : int
        Right boundary of the search region.
    span : float
        LOESS span parameter for smoothing.
    smooth_factor : float
        Additional smoothing factor.

    Returns
    -------
    tn : float
        Saturation temperature (°C).
    tn_idx : int
        Index within signal array.
    """
    region = signal[im:isat]
    temp_region = temp_c[im:isat]

    if len(region) < 10:
        raise ValueError("Saturation region too small")

    # Compute envelope
    env = compute_envelope_loess(region, span=span)

    # Compute derivative of envelope
    deriv = np.gradient(env)

    # Smooth derivative
    w = max(int(len(deriv) * smooth_factor), 5)
    if w % 2 == 0:
        w += 1
    w = min(w, len(deriv))
    deriv_smooth = sig.savgol_filter(deriv, w, polyorder=2)

    # Find zero crossing of derivative (from positive to negative or near zero)
    # This corresponds to the point where growth rate = 0
    zero_crossings = []
    for i in range(1, len(deriv_smooth)):
        if deriv_smooth[i-1] > 0 and deriv_smooth[i] <= 0:
            # Linear interpolation for exact position
            frac = deriv_smooth[i-1] / (deriv_smooth[i-1] - deriv_smooth[i])
            zero_crossings.append(i - 1 + frac)

    if not zero_crossings:
        # Fallback: find minimum of |derivative|
        min_idx = np.argmin(np.abs(deriv_smooth))
        zero_crossings = [float(min_idx)]

    # Take the first zero crossing (growth → dissolution transition)
    zc = zero_crossings[0]
    zc_int = int(round(zc))
    zc_int = min(zc_int, len(temp_region) - 1)

    tn = temp_region[zc_int]
    tn_idx = im + zc_int

    return tn, tn_idx


def find_growth_onset(signal: np.ndarray, temp_c: np.ndarray,
                      tn_index: int, threshold_factor: float = 0.1,
                      search_backward: int = 5000) -> tuple[float, int]:
    """
    Find the onset of growth (where oscillations first appear).

    Searches backward from saturation point.

    Parameters
    ----------
    signal : array
        Full interferometric signal.
    temp_c : array
        Temperature array.
    tn_index : int
        Index of saturation point.
    threshold_factor : float
        Fraction of max envelope for onset detection.
    search_backward : int
        How many samples before tn_index to search.

    Returns
    -------
    t_onset : float
        Temperature of growth onset (°C).
    onset_idx : int
        Sample index.
    """
    start = max(0, tn_index - search_backward)
    region = signal[start:tn_index]

    if len(region) < 10:
        return temp_c[start], start

    # Envelope
    from scipy.signal import hilbert as scipy_hilbert
    analytic = scipy_hilbert(region)
    env = np.abs(analytic)

    # Smooth
    w = min(101, len(env))
    if w % 2 == 0:
        w += 1
    env_smooth = sig.savgol_filter(env, w, polyorder=2)

    # Threshold: where envelope first exceeds threshold_factor * max
    threshold = threshold_factor * np.max(env_smooth)
    above = env_smooth > threshold

    # Find first index above threshold
    onset_rel = np.argmax(above)
    if not above[onset_rel]:
        onset_rel = 0

    onset_idx = start + onset_rel
    t_onset = temp_c[min(onset_idx, len(temp_c) - 1)]

    return t_onset, onset_idx


def determine_saturation(signal: np.ndarray, temp_c: np.ndarray,
                         sol: SolubilitySet,
                         im: int, isat: int,
                         span: float = 0.15,
                         smooth_factor: float = 0.01,
                         search_backward: int = 5000) -> SaturationResult:
    """
    Full saturation determination: tn, Td, s1.

    Parameters
    ----------
    signal : array
        Interferometric signal (one channel).
    temp_c : array
        Temperature in °C.
    sol : SolubilitySet
        Solubility coefficients.
    im : int
        Left boundary of saturation search region.
    isat : int
        Right boundary of saturation search region.
    span : float
        LOESS span.
    smooth_factor : float
        Smoothing factor for derivative.
    search_backward : int
        Samples to search backward for growth onset.

    Returns
    -------
    SaturationResult
    """
    # Find tn
    tn, tn_idx = find_saturation_classic(signal, temp_c, im, isat, span, smooth_factor)

    # Find growth onset
    t_onset, onset_idx = find_growth_onset(signal, temp_c, tn_idx,
                                           search_backward=search_backward)

    # Dead zone
    td = tn - t_onset

    # Dead zone in % supersaturation
    s1 = float(supersaturation_percent(t_onset, tn, sol))

    # Compute envelope for visualization
    region = signal[im:isat]
    env = compute_envelope_loess(region, span=span)
    deriv = np.gradient(env)

    return SaturationResult(
        tn=tn,
        tn_index=tn_idx,
        td=td,
        s1=s1,
        t_onset=t_onset,
        t_onset_index=onset_idx,
        envelope=env,
        derivative=deriv,
    )
