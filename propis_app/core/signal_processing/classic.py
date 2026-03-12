"""
Classic signal processing: extremum detection (reproducing Mathcad algorithm).

Finds minima and maxima of interferometric fringes from two LED channels,
then converts inter-extremum intervals to growth rate R.

Growth rate formula:
  ΔL = gran / 2  (λ/(2n) for the given face, encoded in gran1/gran2)
  R = ΔL / Δt

Temperature at each point: average of temperatures at adjacent extrema.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal as sig


# Face-dependent coefficients (normalized λ/(2n) in micrometers)
# For prism {100}: gran1=0.47854 (LED1), gran2=0.9969 (LED2)
# For pyramid {101}: gran1=1.0, gran2=1.0 (placeholder)
FACE_COEFFICIENTS = {
    0: (0.47854, 0.9969),   # prism {100}
    1: (1.0, 1.0),          # pyramid {101}
}


@dataclass
class ExtremumData:
    """Detected extrema for one channel."""
    positions: np.ndarray     # sample indices of extrema
    values: np.ndarray        # signal values at extrema
    types: np.ndarray         # +1 for max, -1 for min
    temperatures: np.ndarray  # temperature at each extremum


@dataclass
class GrowthRateData:
    """Growth rate computed from extrema."""
    rate: np.ndarray          # growth rate (μm/min)
    rate_mm_day: np.ndarray   # growth rate (mm/day)
    temperature: np.ndarray   # temperature at each rate point (°C)
    supercooling: np.ndarray  # ΔT = tn - T (°C), filled after tn is known
    sigma_percent: np.ndarray # supersaturation (%), filled after fitting
    time_hours: np.ndarray    # time from start (hours)
    channel: int              # 1 or 2


def smooth_signal(signal: np.ndarray, method: str = "median",
                  window: int = 5) -> np.ndarray:
    """
    Smooth signal using median or Savitzky-Golay filter.

    Parameters
    ----------
    signal : array
    method : str
        "median" or "savgol"
    window : int
        Window size (must be odd).
    """
    if window % 2 == 0:
        window += 1
    window = min(window, len(signal))
    if window < 3:
        return signal

    if method == "median":
        return sig.medfilt(signal, kernel_size=window)
    elif method == "savgol":
        return sig.savgol_filter(signal, window, polyorder=2)
    else:
        return signal


def find_extrema(signal: np.ndarray, min_prominence: Optional[float] = None,
                 min_distance: int = 10) -> ExtremumData:
    """
    Find maxima and minima of the interferometric signal.

    Parameters
    ----------
    signal : array
        Interferometric signal (one channel).
    min_prominence : float, optional
        Minimum prominence for peak detection. If None, auto-estimated.
    min_distance : int
        Minimum distance between extrema (samples).

    Returns
    -------
    ExtremumData
        Note: temperatures field is empty, needs to be filled by caller.
    """
    if min_prominence is None:
        # Auto-estimate: fraction of signal range
        min_prominence = 0.1 * (np.max(signal) - np.min(signal))

    # Find maxima
    max_pos, max_props = sig.find_peaks(
        signal, prominence=min_prominence, distance=min_distance
    )

    # Find minima (invert signal)
    min_pos, min_props = sig.find_peaks(
        -signal, prominence=min_prominence, distance=min_distance
    )

    # Merge and sort
    positions = np.concatenate([max_pos, min_pos])
    types = np.concatenate([
        np.ones(len(max_pos), dtype=np.int8),
        -np.ones(len(min_pos), dtype=np.int8),
    ])
    sort_idx = np.argsort(positions)
    positions = positions[sort_idx]
    types = types[sort_idx]
    values = signal[positions]

    return ExtremumData(
        positions=positions,
        values=values,
        types=types,
        temperatures=np.empty(0),  # filled by caller
    )


def fill_temperatures(extrema: ExtremumData, temp_c: np.ndarray) -> ExtremumData:
    """Assign temperature values to each extremum from the temperature array."""
    valid = extrema.positions < len(temp_c)
    positions = extrema.positions[valid]
    extrema.temperatures = temp_c[positions]
    if np.sum(~valid) > 0:
        extrema.positions = positions
        extrema.values = extrema.values[valid]
        extrema.types = extrema.types[valid]
    return extrema


def extrema_to_growth_rate(extrema: ExtremumData, gran: float,
                           dt_seconds: float = 1.0,
                           time_seconds: Optional[np.ndarray] = None,
                           channel: int = 1) -> GrowthRateData:
    """
    Convert extrema positions to growth rate.

    Each pair of same-type adjacent extrema (max-max or min-min) gives one
    growth rate point. We also use half-period (max-min) for higher resolution.

    Parameters
    ----------
    extrema : ExtremumData
        Detected extrema with temperatures filled.
    gran : float
        λ/(2n) coefficient for this channel/face (μm).
    dt_seconds : float
        Time step between samples (seconds).
    time_seconds : array, optional
        Actual time array; if provided, used instead of dt_seconds * index.
    channel : int
        Channel number (1 or 2).

    Returns
    -------
    GrowthRateData
    """
    pos = extrema.positions
    temps = extrema.temperatures
    types = extrema.types

    if len(pos) < 2:
        empty = np.array([])
        return GrowthRateData(
            rate=empty, rate_mm_day=empty, temperature=empty,
            supercooling=empty, sigma_percent=empty,
            time_hours=empty, channel=channel,
        )

    rates = []
    rate_temps = []
    rate_times = []

    # Use consecutive extrema (half-period: λ/(4n) per half-period)
    # Full period (same type): λ/(2n) per period
    for i in range(len(pos) - 1):
        if time_seconds is not None:
            t1 = time_seconds[pos[i]] if pos[i] < len(time_seconds) else pos[i] * dt_seconds
            t2 = time_seconds[pos[i+1]] if pos[i+1] < len(time_seconds) else pos[i+1] * dt_seconds
            delta_t = t2 - t1
        else:
            delta_t = (pos[i+1] - pos[i]) * dt_seconds

        if delta_t <= 0:
            continue

        # Half-period: thickness change = gran/2
        # (each extremum transition = λ/(4n))
        delta_L = gran / 2.0  # μm

        rate_um_min = delta_L / (delta_t / 60.0)  # μm/min
        rates.append(rate_um_min)

        # Temperature: average between two extrema
        T_avg = (temps[i] + temps[i+1]) / 2.0
        rate_temps.append(T_avg)

        # Time: midpoint
        if time_seconds is not None:
            t_mid = (t1 + t2) / 2.0
        else:
            t_mid = ((pos[i] + pos[i+1]) / 2.0) * dt_seconds
        rate_times.append(t_mid / 3600.0)  # hours

    rates = np.array(rates)
    rate_temps = np.array(rate_temps)
    rate_times = np.array(rate_times)

    # Convert μm/min to mm/day: 1 μm/min = 1.44 mm/day
    rates_mm_day = rates * 1.44

    return GrowthRateData(
        rate=rates,
        rate_mm_day=rates_mm_day,
        temperature=rate_temps,
        supercooling=np.zeros_like(rates),  # filled later
        sigma_percent=np.zeros_like(rates),  # filled later
        time_hours=rate_times,
        channel=channel,
    )


def process_channel_classic(signal: np.ndarray, temp_c: np.ndarray,
                            face: int = 0, channel: int = 1,
                            smooth_window: int = 5,
                            min_prominence: Optional[float] = None,
                            min_distance: int = 10,
                            dt_seconds: float = 1.0,
                            time_seconds: Optional[np.ndarray] = None,
                            n_parasitic: int = 0) -> GrowthRateData:
    """
    Full classic processing pipeline for one channel.

    Parameters
    ----------
    signal : array
        Interferometric signal.
    temp_c : array
        Temperature in °C, same length as signal.
    face : int
        0 = prism {100}, 1 = pyramid {101}.
    channel : int
        1 or 2 (LED channel).
    smooth_window : int
        Smoothing window for median filter.
    min_prominence : float, optional
        Minimum prominence for peak detection.
    min_distance : int
        Minimum distance between peaks.
    dt_seconds : float
        Time step.
    time_seconds : array, optional
        Actual time array.
    n_parasitic : int
        Number of initial fringes to exclude (from прочитай.txt).

    Returns
    -------
    GrowthRateData
    """
    # Smooth
    smoothed = smooth_signal(signal, method="median", window=smooth_window)

    # Find extrema
    extrema = find_extrema(smoothed, min_prominence=min_prominence,
                           min_distance=min_distance)

    # Fill temperatures
    extrema = fill_temperatures(extrema, temp_c)

    # Exclude parasitic fringes
    if n_parasitic > 0 and len(extrema.positions) > n_parasitic:
        extrema.positions = extrema.positions[n_parasitic:]
        extrema.values = extrema.values[n_parasitic:]
        extrema.types = extrema.types[n_parasitic:]
        extrema.temperatures = extrema.temperatures[n_parasitic:]

    # Get gran coefficient
    gran1, gran2 = FACE_COEFFICIENTS.get(face, (1.0, 1.0))
    gran = gran1 if channel == 1 else gran2

    # Convert to growth rate
    growth = extrema_to_growth_rate(
        extrema, gran=gran, dt_seconds=dt_seconds,
        time_seconds=time_seconds, channel=channel,
    )

    return growth
