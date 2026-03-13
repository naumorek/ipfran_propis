"""
Automatic detection of cycles, dead zone, and saturation temperature.

Algorithm:
  1. Segment PRN data into temperature cycles (sawtooth pattern)
  2. For each cycle, find the dead zone using rolling standard deviation
  3. Determine tn (saturation temperature) at the minimum of local variance
  4. Track crystal growth across cycles and apply tn correction

Key insight: the signal has oscillations during growth and dissolution,
but is flat (low variance) in the dead zone near saturation.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter


@dataclass
class CycleInfo:
    """Information about one temperature cycle (iteration)."""
    cycle_index: int
    start: int                    # absolute sample index in PRN
    end: int                      # absolute sample index
    t_min: float                  # minimum temperature in cycle (°C)
    t_max: float                  # maximum temperature in cycle (°C)

    # Dead zone detection
    plateau_start: int            # absolute index: beginning of flat region
    plateau_end: int              # absolute index: end of flat region
    t_onset: float                # T where growth stops (left edge of plateau)
    t_dissolve: float             # T where dissolution starts (right edge)
    tn: float                     # saturation temperature (at minimum std)
    tn_index: int                 # absolute index of tn
    td: float                     # dead zone width (°C)
    min_std: float                # minimum rolling std in this cycle

    # Growth tracking
    n_fringes_growth: int = 0     # number of fringes in growth phase
    n_fringes_dissolve: int = 0   # number of fringes in dissolution phase
    crystal_growth_um: float = 0  # net crystal growth this cycle (μm)

    # Correction
    tn_predicted: float = 0.0     # predicted tn from previous cycle
    tn_correction: float = 0.0    # applied correction (°C)


@dataclass
class AutoDetectResult:
    """Result of automatic detection for entire PRN file."""
    cycles: list[CycleInfo]
    total_growth_um: float = 0.0        # total crystal growth (μm)
    tn_drift_per_cycle: float = 0.2     # estimated tn drift (°C/cycle)


def _rolling_std(signal: np.ndarray, window: int = 500) -> np.ndarray:
    """
    Fast rolling standard deviation using cumulative sums.

    Parameters
    ----------
    signal : array
    window : int
        Window size in samples.

    Returns
    -------
    std : array
        Rolling std, same length as signal.
    """
    n = len(signal)
    if n < window:
        return np.full(n, np.std(signal))

    cs = np.cumsum(signal)
    cs2 = np.cumsum(signal ** 2)

    # Pad with zero at start for diff
    cs = np.concatenate([[0], cs])
    cs2 = np.concatenate([[0], cs2])

    # Rolling mean and variance
    mean = (cs[window:] - cs[:-window]) / window
    mean2 = (cs2[window:] - cs2[:-window]) / window
    var = np.maximum(mean2 - mean ** 2, 0)
    std_arr = np.sqrt(var)

    # Pad to original length (center the window)
    hw = window // 2
    result = np.zeros(n)
    result[hw:hw + len(std_arr)] = std_arr
    result[:hw] = std_arr[0]
    result[hw + len(std_arr):] = std_arr[-1]

    return result


def find_cycle_boundaries(temp: np.ndarray,
                          drop_window: int = 500,
                          drop_threshold: float = -2.0,
                          recovery_threshold: float = 0.3) -> list[tuple[int, int]]:
    """
    Segment temperature data into cycles.

    Each cycle: monotonic rise from T_min to T_max, followed by rapid drop.
    Sawtooth pattern with ~5-6 cycles per PRN file.

    Parameters
    ----------
    temp : array
        Temperature in °C.
    drop_window : int
        Window for detecting temperature drops (samples).
    drop_threshold : float
        Threshold for dT over drop_window to detect cycle boundary (°C).
    recovery_threshold : float
        dT over drop_window to detect recovery after drop (°C).

    Returns
    -------
    cycles : list of (start, end) tuples
        Each tuple = (start_index, end_index) for one cycle.
    """
    n = len(temp)
    if n < drop_window * 2:
        return [(0, n)]

    # dT over window
    dT = temp[drop_window:] - temp[:-drop_window]

    # Find drop regions (rapid cooling between cycles)
    in_drop = False
    drop_starts = []
    drop_ends = []

    for i in range(len(dT)):
        if dT[i] < drop_threshold and not in_drop:
            drop_starts.append(i)
            in_drop = True
        elif dT[i] > recovery_threshold and in_drop:
            drop_ends.append(i + drop_window)
            in_drop = False

    # Handle case: still in drop at end of data
    if in_drop:
        drop_ends.append(n)

    # Build cycle list
    cycles = []

    if not drop_starts:
        # No drops found — single cycle (entire file)
        return [(0, n)]

    # First cycle: from start to first drop
    cycles.append((0, drop_starts[0]))

    # Cycles between drops
    for i in range(len(drop_ends)):
        start = drop_ends[i]
        if i + 1 < len(drop_starts):
            end = drop_starts[i + 1]
        else:
            end = n
        if end > start + 500:  # minimum cycle length
            cycles.append((start, end))

    return cycles


def find_dead_zone(signal: np.ndarray, temp: np.ndarray,
                   std_window: int = 500,
                   std_threshold: float = 0.08,
                   min_plateau_length: int = 200) -> dict:
    """
    Find the dead zone (plateau) in one cycle using rolling std.

    The dead zone is the region where the interferometric signal is flat
    (no fringes = no growth and no dissolution).

    Parameters
    ----------
    signal : array
        Interferometric signal for one cycle.
    temp : array
        Temperature for one cycle.
    std_window : int
        Window for rolling std calculation.
    std_threshold : float
        Maximum rolling std to consider signal "flat" (Volts).
    min_plateau_length : int
        Minimum plateau length in samples.

    Returns
    -------
    dict with keys:
        plateau_start, plateau_end : int (relative to cycle start)
        tn_index : int (relative to cycle start)
        tn : float (°C)
        t_onset : float (°C)
        t_dissolve : float (°C)
        td : float (°C)
        min_std : float
        rolling_std : array
    """
    rstd = _rolling_std(signal, window=std_window)

    # Find flat regions (below threshold)
    flat = rstd < std_threshold

    # Find all contiguous flat runs
    runs = []
    start = None
    for i in range(len(flat)):
        if flat[i] and start is None:
            start = i
        elif not flat[i] and start is not None:
            if i - start >= min_plateau_length:
                runs.append((start, i))
            start = None
    if start is not None and len(flat) - start >= min_plateau_length:
        runs.append((start, len(flat)))

    if not runs:
        # Adaptive: relax threshold
        for factor in [1.5, 2.0, 3.0]:
            relaxed_flat = rstd < std_threshold * factor
            start = None
            for i in range(len(relaxed_flat)):
                if relaxed_flat[i] and start is None:
                    start = i
                elif not relaxed_flat[i] and start is not None:
                    if i - start >= min_plateau_length // 2:
                        runs.append((start, i))
                    start = None
            if start is not None:
                runs.append((start, len(relaxed_flat)))
            if runs:
                break

    if not runs:
        # Fallback: use minimum std point
        min_idx = np.argmin(rstd)
        hw = std_window
        p_start = max(0, min_idx - hw)
        p_end = min(len(signal), min_idx + hw)
        runs = [(p_start, p_end)]

    # Select the longest plateau
    longest = max(runs, key=lambda r: r[1] - r[0])
    p_start, p_end = longest

    # tn: temperature at minimum std within the plateau
    plateau_std = rstd[p_start:p_end]
    min_std_rel = np.argmin(plateau_std)
    tn_index = p_start + min_std_rel
    tn = float(temp[min(tn_index, len(temp) - 1)])

    # Onset and dissolution temperatures
    t_onset = float(temp[min(p_start, len(temp) - 1)])
    t_dissolve = float(temp[min(p_end - 1, len(temp) - 1)])

    # Dead zone: distance from onset to tn
    td = abs(tn - t_onset)

    return {
        "plateau_start": p_start,
        "plateau_end": p_end,
        "tn_index": tn_index,
        "tn": tn,
        "t_onset": t_onset,
        "t_dissolve": t_dissolve,
        "td": td,
        "min_std": float(rstd[tn_index]),
        "rolling_std": rstd,
    }


def count_fringes(signal: np.ndarray, std_threshold: float = 0.08,
                  std_window: int = 500,
                  plateau_start: int = 0,
                  plateau_end: int = 0) -> tuple[int, int]:
    """
    Count interference fringes in growth and dissolution phases.

    Growth phase: before the plateau.
    Dissolution phase: after the plateau.

    Returns
    -------
    (n_growth, n_dissolve) : number of fringes in each phase
    """
    from scipy.signal import find_peaks

    # Growth phase: signal before plateau
    growth_sig = signal[:plateau_start]
    n_growth = 0
    if len(growth_sig) > 10:
        prominence = 0.1 * (np.max(growth_sig) - np.min(growth_sig))
        if prominence > 0.01:
            peaks, _ = find_peaks(growth_sig, prominence=prominence, distance=10)
            n_growth = len(peaks)

    # Dissolution phase: signal after plateau
    dissolve_sig = signal[plateau_end:]
    n_dissolve = 0
    if len(dissolve_sig) > 10:
        prominence = 0.1 * (np.max(dissolve_sig) - np.min(dissolve_sig))
        if prominence > 0.01:
            peaks, _ = find_peaks(dissolve_sig, prominence=prominence, distance=10)
            n_dissolve = len(peaks)

    return n_growth, n_dissolve


def estimate_crystal_growth(n_fringes_growth: int, n_fringes_dissolve: int,
                            gran: float = 0.47854) -> float:
    """
    Estimate net crystal growth from fringe counts.

    Each fringe (peak-to-peak) = thickness change of gran μm.
    Net growth = growth fringes - dissolution fringes.

    Parameters
    ----------
    n_fringes_growth : int
    n_fringes_dissolve : int
    gran : float
        λ/(2n) in μm for the given face/channel.

    Returns
    -------
    growth_um : float
        Net crystal growth in μm.
    """
    net_fringes = n_fringes_growth - n_fringes_dissolve
    return net_fringes * gran


def estimate_tn_correction(crystal_growth_um: float, tn: float,
                           sol_set=None) -> float:
    """
    Estimate tn correction due to crystal growth.

    As the crystal grows, it removes salt from solution, lowering tn.
    Approximate: δtn ≈ 0.2°C per cycle (empirical).

    For more precise calculation: use solubility coefficients and
    estimated solution volume.

    Parameters
    ----------
    crystal_growth_um : float
        Crystal growth in μm.
    tn : float
        Current saturation temperature.
    sol_set : SolubilitySet, optional
        For precise calculation.

    Returns
    -------
    delta_tn : float
        Expected tn decrease (positive = tn goes down).
    """
    # Empirical approximation: ~0.2°C per cycle
    # TODO: precise calculation from crystal volume, solution volume,
    # and solubility curve dC/dT
    return 0.2


def auto_detect(led_signal: np.ndarray, temp: np.ndarray,
                face: int = 0, channel: int = 1,
                std_window: int = 500,
                std_threshold: float = 0.08,
                drop_threshold: float = -2.0,
                min_plateau_length: int = 200,
                tn_drift_per_cycle: float = 0.2) -> AutoDetectResult:
    """
    Full automatic detection: segment cycles, find dead zones, track growth.

    Parameters
    ----------
    led_signal : array
        Interferometric signal (one LED channel).
    temp : array
        Temperature in °C.
    face : int
        0 = prism {100}, 1 = pyramid {101}.
    channel : int
        1 or 2.
    std_window : int
        Rolling std window (samples).
    std_threshold : float
        Threshold for flat signal (Volts).
    drop_threshold : float
        Temperature drop threshold for cycle detection (°C).
    min_plateau_length : int
        Minimum plateau length (samples).
    tn_drift_per_cycle : float
        Expected tn decrease per cycle (°C).

    Returns
    -------
    AutoDetectResult
    """
    from .signal_processing.classic import FACE_COEFFICIENTS

    gran1, gran2 = FACE_COEFFICIENTS.get(face, (1.0, 1.0))
    gran = gran1 if channel == 1 else gran2

    # Step 1: Segment into cycles
    cycle_bounds = find_cycle_boundaries(temp, drop_threshold=drop_threshold)

    cycles = []
    total_growth = 0.0
    prev_tn = None

    for ci, (c_start, c_end) in enumerate(cycle_bounds):
        cyc_signal = led_signal[c_start:c_end]
        cyc_temp = temp[c_start:c_end]

        if len(cyc_signal) < std_window * 2:
            continue

        # Step 2: Find dead zone
        dz = find_dead_zone(
            cyc_signal, cyc_temp,
            std_window=std_window,
            std_threshold=std_threshold,
            min_plateau_length=min_plateau_length,
        )

        # Step 3: Count fringes
        n_growth, n_dissolve = count_fringes(
            cyc_signal,
            plateau_start=dz["plateau_start"],
            plateau_end=dz["plateau_end"],
        )

        # Step 4: Crystal growth
        growth_um = estimate_crystal_growth(n_growth, n_dissolve, gran=gran)
        total_growth += growth_um

        # Step 5: tn correction
        tn_predicted = 0.0
        tn_correction = 0.0
        if prev_tn is not None:
            tn_predicted = prev_tn - tn_drift_per_cycle
            tn_correction = dz["tn"] - tn_predicted

        cycle = CycleInfo(
            cycle_index=ci,
            start=c_start,
            end=c_end,
            t_min=float(cyc_temp.min()),
            t_max=float(cyc_temp.max()),
            plateau_start=c_start + dz["plateau_start"],
            plateau_end=c_start + dz["plateau_end"],
            t_onset=dz["t_onset"],
            t_dissolve=dz["t_dissolve"],
            tn=dz["tn"],
            tn_index=c_start + dz["tn_index"],
            td=dz["td"],
            min_std=dz["min_std"],
            n_fringes_growth=n_growth,
            n_fringes_dissolve=n_dissolve,
            crystal_growth_um=growth_um,
            tn_predicted=tn_predicted,
            tn_correction=tn_correction,
        )
        cycles.append(cycle)
        prev_tn = dz["tn"]

    return AutoDetectResult(
        cycles=cycles,
        total_growth_um=total_growth,
        tn_drift_per_cycle=tn_drift_per_cycle,
    )
