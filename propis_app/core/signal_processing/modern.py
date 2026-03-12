"""
Modern signal processing: Hilbert transform + instantaneous frequency.

Instead of detecting individual extrema, uses:
  1. Analytic signal via Hilbert transform
  2. Instantaneous phase φ(t) = unwrap(angle(analytic))
  3. Instantaneous frequency f(t) = dφ/dt / (2π)
  4. Growth rate R(t) = f(t) * Δλ (converted to μm/min)

Advantages over classic method:
  - Continuous R(t) instead of discrete points at extrema
  - Better resolution (sub-fringe)
  - No need for extremum detection
  - Automatic — no manual peak finding

Requirements:
  - Signal should be bandpass-filtered (monocomponent) for reliable Hilbert transform
  - Use preprocessing pipeline before this module
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import hilbert, savgol_filter, medfilt

from .classic import FACE_COEFFICIENTS, GrowthRateData


@dataclass
class InstantaneousData:
    """Instantaneous signal properties from Hilbert transform."""
    amplitude: np.ndarray     # instantaneous amplitude (envelope)
    phase: np.ndarray         # unwrapped instantaneous phase (radians)
    frequency: np.ndarray     # instantaneous frequency (Hz)
    growth_rate: np.ndarray   # growth rate (μm/min)
    growth_rate_mm_day: np.ndarray  # growth rate (mm/day)


def compute_analytic_signal(signal: np.ndarray) -> np.ndarray:
    """Compute analytic signal using Hilbert transform."""
    return hilbert(signal)


def compute_instantaneous_phase(analytic: np.ndarray) -> np.ndarray:
    """
    Compute unwrapped instantaneous phase.

    Returns phase in radians, monotonically increasing (for a growing crystal).
    """
    phase = np.angle(analytic)
    return np.unwrap(phase)


def compute_instantaneous_frequency(phase: np.ndarray,
                                    fs: float = 1.0,
                                    smooth_window: int = 51) -> np.ndarray:
    """
    Compute instantaneous frequency from unwrapped phase.

    f(t) = (1/2π) * dφ/dt

    Parameters
    ----------
    phase : array
        Unwrapped phase (radians).
    fs : float
        Sampling frequency (Hz).
    smooth_window : int
        Smoothing window for frequency estimate.

    Returns
    -------
    freq : array
        Instantaneous frequency (Hz).
    """
    # Derivative of phase
    dphase = np.gradient(phase) * fs
    freq = dphase / (2.0 * np.pi)

    # Smooth to remove noise
    if smooth_window > 1:
        w = min(smooth_window, len(freq))
        if w % 2 == 0:
            w += 1
        if w >= 5:
            freq = savgol_filter(freq, w, polyorder=2)

    # Frequency should be non-negative for growth
    return np.abs(freq)


def frequency_to_growth_rate(freq: np.ndarray, gran: float,
                             dt_seconds: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert instantaneous frequency to growth rate.

    Each full cycle (frequency=1/period) corresponds to thickness change = gran/2.
    R = freq * gran/2 (μm/s) → convert to μm/min and mm/day.

    Parameters
    ----------
    freq : array
        Instantaneous frequency (Hz = cycles/second).
    gran : float
        λ/(2n) coefficient (μm).
    dt_seconds : float
        Not used directly, but documents the time base.

    Returns
    -------
    rate_um_min : array
        Growth rate in μm/min.
    rate_mm_day : array
        Growth rate in mm/day.
    """
    # Each cycle = thickness change of gran (μm)
    # For half-cycle extremum-to-extremum: gran/2
    # But frequency counts full cycles, so:
    rate_um_s = freq * gran  # μm/s
    rate_um_min = rate_um_s * 60.0  # μm/min
    rate_mm_day = rate_um_min * 1.44  # mm/day

    return rate_um_min, rate_mm_day


def process_channel_modern(signal: np.ndarray, temp_c: np.ndarray,
                           face: int = 0, channel: int = 1,
                           fs: float = 1.0,
                           smooth_window: int = 51,
                           time_seconds: Optional[np.ndarray] = None,
                           ) -> tuple[InstantaneousData, GrowthRateData]:
    """
    Full modern processing pipeline for one channel.

    Parameters
    ----------
    signal : array
        Interferometric signal (should be preprocessed/bandpass filtered).
    temp_c : array
        Temperature in °C, same length as signal.
    face : int
        0 = prism, 1 = pyramid.
    channel : int
        1 or 2.
    fs : float
        Sampling frequency (Hz).
    smooth_window : int
        Smoothing window for frequency estimation.
    time_seconds : array, optional
        Actual time stamps.

    Returns
    -------
    inst : InstantaneousData
        Full instantaneous signal analysis.
    growth : GrowthRateData
        Growth rate data (compatible with classic format).
    """
    # Analytic signal
    analytic = compute_analytic_signal(signal)

    # Amplitude (envelope)
    amplitude = np.abs(analytic)

    # Phase
    phase = compute_instantaneous_phase(analytic)

    # Frequency
    freq = compute_instantaneous_frequency(phase, fs=fs,
                                           smooth_window=smooth_window)

    # Gran coefficient
    gran1, gran2 = FACE_COEFFICIENTS.get(face, (1.0, 1.0))
    gran = gran1 if channel == 1 else gran2

    # Growth rate
    rate_um_min, rate_mm_day = frequency_to_growth_rate(freq, gran)

    inst = InstantaneousData(
        amplitude=amplitude,
        phase=phase,
        frequency=freq,
        growth_rate=rate_um_min,
        growth_rate_mm_day=rate_mm_day,
    )

    # Time in hours
    if time_seconds is not None:
        time_hours = time_seconds / 3600.0
    else:
        time_hours = np.arange(len(signal)) / (fs * 3600.0)

    # Package as GrowthRateData for compatibility
    growth = GrowthRateData(
        rate=rate_um_min,
        rate_mm_day=rate_mm_day,
        temperature=temp_c,
        supercooling=np.zeros_like(temp_c),
        sigma_percent=np.zeros_like(temp_c),
        time_hours=time_hours,
        channel=channel,
    )

    return inst, growth


def find_saturation_modern(freq: np.ndarray, temp_c: np.ndarray,
                           threshold_hz: float = 0.001,
                           smooth_window: int = 201) -> tuple[float, int]:
    """
    Find saturation temperature using instantaneous frequency.

    When frequency → 0, growth stops → saturation point.

    Parameters
    ----------
    freq : array
        Instantaneous frequency.
    temp_c : array
        Temperature.
    threshold_hz : float
        Frequency threshold for "no growth".
    smooth_window : int
        Smoothing window.

    Returns
    -------
    tn : float
        Saturation temperature.
    tn_idx : int
        Sample index.
    """
    # Smooth frequency
    w = min(smooth_window, len(freq))
    if w % 2 == 0:
        w += 1
    if w >= 5:
        freq_smooth = savgol_filter(freq, w, polyorder=2)
    else:
        freq_smooth = freq

    # Find where frequency drops below threshold
    # Search from the region where temperature is increasing
    # (transition: growth → dissolution)
    below_threshold = freq_smooth < threshold_hz

    # Find the first sustained drop below threshold
    min_sustained = 50  # minimum consecutive samples
    count = 0
    for i in range(len(below_threshold)):
        if below_threshold[i]:
            count += 1
            if count >= min_sustained:
                tn_idx = i - min_sustained + 1
                tn = float(temp_c[min(tn_idx, len(temp_c) - 1)])
                return tn, tn_idx
        else:
            count = 0

    # Fallback: minimum frequency point
    tn_idx = int(np.argmin(freq_smooth))
    tn = float(temp_c[min(tn_idx, len(temp_c) - 1)])
    return tn, tn_idx
