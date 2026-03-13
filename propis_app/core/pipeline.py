"""
Unified processing pipeline for interferometric data.

Two modes:
  - Classic: exact reproduction of Mathcad 2000 algorithm
    (zero-crossings extrema, quadratic refinement, phase-based rate,
     Mathcad grid search, LOESS Sig035, sqrt(R) s2)
  - Modern: automatic detection, improved processing, same output format

Both modes produce PipelineResult with identical fields for comparison.

Reference: docs/mathcad_classic_algorithm_strict.md
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .prn_reader import PrnData, read_prn
from .solubility import (
    SolubilitySet, get_solubility_set, solubility,
    supersaturation_percent, supercooling,
)
from .signal_processing.classic import (
    smooth_signal, find_extrema, fill_temperatures,
    extrema_to_growth_rate, FACE_COEFFICIENTS, OPTICAL_COEFFICIENTS,
    GrowthRateData, ExtremumData,
    find_extrema_mathcad, filter_extrema_edges, refine_extrema_quadratic,
    build_phase_dstep,
)
from .kinetics.power_law import (
    fit_power_law, power_law_model,
    compute_sig035_loess, compute_s2_mathcad,
    PowerLawResult,
)


@dataclass
class CycleParams:
    """Parameters for one cycle (from MCD or auto-detected)."""
    n1: int           # start index in PRN array
    n2: int           # end index
    im: int           # precise dead zone start (RELATIVE to n1)
    isat: int         # precise dissolution start (RELATIVE to n1)
    im1: int          # rough dead zone start (RELATIVE to n1)
    isat1: int        # rough dissolution start (RELATIVE to n1)


@dataclass
class PipelineResult:
    """Result of processing one cycle."""
    # Identification
    filename: str = ""
    cycle_index: int = 0
    mode: str = "classic"     # "classic" or "modern"

    # Key output parameters (same as Mathcad Mat vector)
    te: float = 0.0           # temperature at isat point (C)
    tn: float = 0.0           # saturation temperature (C)
    Td: float = 0.0           # dead zone width (C)
    Sigm: float = 0.0         # dead zone in relative supersaturation (%)
    s0: float = 0.0           # kinetic coefficient
    s1: float = 0.0           # dead zone from fit (% supersaturation)
    s2: float = 0.0           # shape parameter: -Q1/Q2 from sqrt(R) vs sigma%
    Sig035: float = 0.0       # supersaturation at LOESS(R) = 0.35

    # Input parameters
    Salt: int = 1
    Acid: int = 0
    Face: int = 0
    n1: int = 0
    n2: int = 0
    im: int = 0
    isat: int = 0
    im1: int = 0
    isat1: int = 0
    dtau: float = 0.055
    ww: int = 1
    d: float = 3.3

    # Data arrays for plotting
    growth_rate: Optional[GrowthRateData] = None
    fit_result: Optional[PowerLawResult] = None
    bcf_result: Optional[object] = None  # BCFResult from modern pipeline

    # Dense d-step data (from build_phase_dstep, ~2000+ points including dead zone)
    # Used for Mathcad-style F1 LOESS curve
    dense_rate: Optional[np.ndarray] = None          # R (mm/day)
    dense_temperature: Optional[np.ndarray] = None   # T (°C)
    dense_sigma: Optional[np.ndarray] = None         # σ (%)
    dense_supercooling: Optional[np.ndarray] = None  # dT = tn - T (°C)

    # Diagnostic: number of extrema found
    n_extrema: int = 0

    def mat_vector(self) -> list:
        """Return Mathcad-compatible Mat vector."""
        return [
            self.Salt, self.Acid, self.te, 0, self.tn,
            self.s1, self.Sigm, self.s2, self.Sig035, self.Td,
            self.n1, self.n2, self.ww, self.d, self.dtau, 0,
            self.im1, self.isat1, self.im, self.isat,
        ]


def run_classic(prn: PrnData, params: CycleParams,
                salt: int = 1, acid: int = 0, face: int = 0,
                channel: int = 1, dtau: float = 0.055,
                ww: int = 1, d: float = 3.3, k1: float = 0.15,
                smooth_window: int = 5,
                span1: float = 0.2,
                tn_manual: Optional[float] = None) -> PipelineResult:
    """
    Classic pipeline — exact Mathcad reproduction.

    Algorithm (from Mathcad screenshots _020326_5_1):
    1. Extract sub-array a[n1..n2]
    2. Compute signal = LED1 / LED2 (ratio normalizes common-mode variations)
    3. S1 = medsmooth(signal, 5)
    4. y0 = mean(S1[0..im1]) — baseline in growth zone
    5. te = T[isat] - 273.15
    6. find_extrema_mathcad: zero-crossings with step=2 in GROWTH zone (0..im1)
    7. filter_extrema_edges: remove unreliable edge extrema
    8. refine_extrema_quadratic: parabolic fit for sub-sample precision
    9. Compute growth rate from consecutive extrema
    10. Compute sigma% from temperatures and tn
    11. Grid search s0, s1 (Mathcad style)
    12. s2 = -Q1/Q2 from sqrt(R) vs sigma%
    13. Sig035 via LOESS
    14. Td = te - T[im], Sigm from solubility
    """
    sol = get_solubility_set(salt, acid)

    # Step 1: Extract sub-array
    n1, n2 = params.n1, params.n2
    sub = prn.slice(n1, n2)
    temp_c = sub.temp_c_mathcad  # a[i,4] - 273.15

    im = params.im
    isat = params.isat
    im1 = params.im1
    isat1 = params.isat1

    # Step 2: Signal = LED1 / LED2 (Mathcad: a[i,9] := a[i,1] / a[i,2])
    # This ratio normalizes out common-mode LED/detector variations,
    # leaving only the interference fringes.
    led1 = sub.led1
    led2 = sub.led2
    # Protect against division by zero
    led2_safe = np.where(np.abs(led2) < 1e-10, 1e-10, led2)
    signal = led1 / led2_safe
    m = len(signal)

    # Step 3: medsmooth
    smoothed = smooth_signal(signal, method="median", window=smooth_window)

    # Step 4: Baseline in growth zone (y0, NOT dissolution zone y0s)
    # Mathcad: y0 := mean(S1[0..im1-1])
    if im1 > 0 and im1 < m:
        y0 = float(np.mean(smoothed[:im1]))
    else:
        y0 = float(np.mean(smoothed))

    # Step 5: te = T[isat], tn = saturation temperature (manually set or default to te)
    te = float(temp_c[min(isat, len(temp_c) - 1)])
    tn = tn_manual if tn_manual is not None else te

    # Step 6: Find extrema via zero-crossings in GROWTH zone (0 to im1)
    # Mathcad: for i ∈ 0, 2..im1 using baseline y0
    extrs = find_extrema_mathcad(smoothed, y0, 0, im1)

    if len(extrs.positions) < 3:
        return _empty_result("classic", prn, params, salt, acid, face, dtau, ww, d, te, tn)

    # Step 7: Filter edge extrema
    extrs = filter_extrema_edges(extrs, 0, im1)

    if len(extrs.positions) < 3:
        return _empty_result("classic", prn, params, salt, acid, face, dtau, ww, d, te, tn)

    n_extrema = len(extrs.positions)

    # Step 7: Refine via quadratic regression
    es = refine_extrema_quadratic(smoothed, extrs, k1=k1)

    # Step 8: Compute growth rates from consecutive extrema
    # Mathcad formula: R = ΔL / Δsample * 30 * nn
    # where L = (phase/(π·x) + L0·x0/x - L0) · nn
    # x = co0 - co1·T (optical path factor)
    # For consecutive extrema, Δphase = π/2
    # Simplified (ignoring L0 correction for small ΔT):
    #   ΔL ≈ (π/2) / (π·x) = 1/(2·x) per half-period
    #   R = ΔL / Δsample * 30
    co0, co1_opt = OPTICAL_COEFFICIENTS.get((salt, face), (0.06446, 1.92e-5))

    # nn: direction flag. For growth (temperature decreasing), nn = -1
    # But we take abs for rate magnitude
    nn = -1  # growth phase

    rates = []
    rate_temps = []
    rate_pos = []

    for i in range(len(es) - 1):
        pos1 = es[i, 0]
        pos2 = es[i + 1, 0]

        if pos2 <= pos1:
            continue

        # Integer indices for lookups
        ip1 = int(np.clip(np.round(pos1), 0, len(temp_c) - 1))
        ip2 = int(np.clip(np.round(pos2), 0, len(temp_c) - 1))

        # Average temperature in the interval
        idx_start = max(0, int(np.ceil(pos1)))
        idx_end = min(len(temp_c) - 1, int(np.floor(pos2)))
        if idx_end >= idx_start:
            T_avg = float(np.mean(temp_c[idx_start:idx_end + 1]))
        else:
            T_avg = (temp_c[ip1] + temp_c[ip2]) / 2.0

        # Phase change between consecutive extrema = π/2
        delta_phase = np.pi / 2.0

        # Optical path factor x = co0 - co1*T
        x = co0 - co1_opt * T_avg

        if abs(x) < 1e-15:
            continue

        # ΔL from phase: delta_phase / (π * x)
        delta_L = delta_phase / (np.pi * x)

        # Δsample (position difference in sample units)
        delta_samples = pos2 - pos1

        # Rate in mm/day: R = ΔL / Δsample * 30
        R_mm_day = abs(delta_L / delta_samples * 30)

        rates.append(R_mm_day)
        rate_temps.append(T_avg)
        rate_pos.append((pos1 + pos2) / 2.0)

    rates = np.array(rates)
    rate_temps = np.array(rate_temps)
    rate_pos = np.array(rate_pos)

    if len(rates) < 3:
        return _empty_result("classic", prn, params, salt, acid, face, dtau, ww, d, te, tn)

    # Build GrowthRateData
    rate_times = np.zeros_like(rates)
    if sub.time_seconds is not None and len(sub.time_seconds) > 0:
        for idx, p in enumerate(rate_pos):
            t = np.interp(p, np.arange(len(sub.time_seconds)), sub.time_seconds)
            rate_times[idx] = t / 3600.0

    growth = GrowthRateData(
        rate=rates,              # mm/day (Mathcad native)
        rate_mm_day=rates,       # same
        temperature=rate_temps,
        supercooling=np.zeros_like(rates),
        sigma_percent=np.zeros_like(rates),
        time_hours=rate_times,
        channel=channel,
        positions=rate_pos,
        phase=np.zeros_like(rates),
    )

    # Step 9: Compute sigma%
    sigma = supersaturation_percent(growth.temperature, tn, sol)
    growth.sigma_percent = sigma
    growth.supercooling = supercooling(growth.temperature, tn)

    # Filter: only points with sigma > 0 and rate > 0 (growth phase)
    growth_mask = (sigma > 0) & (growth.rate > 0)
    sigma_growth = sigma[growth_mask]
    rate_growth = rates[growth_mask]

    if len(sigma_growth) < 3:
        return _empty_result("classic", prn, params, salt, acid, face, dtau, ww, d, te, tn)

    # Step 10: Power law fit (Mathcad grid search)
    fit = fit_power_law(sigma_growth, rate_growth, w=1.0)

    # Step 11: Dense d-step rate sampling for s2 computation
    # Mathcad builds phase function at d-step spacing from 0 to isat1
    # (~2000+ points including dead zone), needed for s2 = -Q1/Q2
    dense_rates, dense_temps, dense_pos, _ = build_phase_dstep(
        smoothed, es, temp_c, end_pos=isat1, d=d,
        co0=co0, co1=co1_opt, nn=-1,
        im=im, ww=ww,
    )
    # Store dense data for plotting (Mathcad F1 curve uses this)
    dense_sigma_arr = None
    dense_supercooling_arr = None
    if len(dense_rates) > 10:
        dense_sigma_arr = supersaturation_percent(dense_temps, tn, sol)
        dense_supercooling_arr = supercooling(dense_temps, tn)
        s2 = compute_s2_mathcad(dense_sigma_arr, dense_rates)
    else:
        # Fallback to coarse data
        s2 = compute_s2_mathcad(sigma, rates)

    # Step 12: Sig035 via LOESS (from per-extremum data, which is more reliable)
    Sig035 = compute_sig035_loess(sigma_growth, rate_growth, span=span1)

    # Step 13: Dead zone
    # Td = te - T[im] (temperature difference between dissolution start and dead zone start)
    # Verified against Mathcad: no 0.01*(te-25) correction needed
    t_im = float(temp_c[min(im, len(temp_c) - 1)])
    Td = te - t_im

    # Sigm = 100 * ln(C(te) / C(t_im)) — supersaturation at dead zone boundary
    Cn = solubility(te, sol)
    Cm = solubility(t_im, sol)
    if Cm > 0 and Cn > 0:
        Sigm = float(100.0 * np.log(Cn / Cm))
    else:
        Sigm = float(supersaturation_percent(t_im, tn, sol))

    return PipelineResult(
        filename=str(prn.filepath.name),
        cycle_index=0,
        mode="classic",
        te=te, tn=tn, Td=Td, Sigm=Sigm,
        s0=fit.s0, s1=fit.s1, s2=s2, Sig035=Sig035,
        Salt=salt, Acid=acid, Face=face,
        n1=n1, n2=n2, im=im, isat=isat, im1=im1, isat1=isat1,
        dtau=dtau, ww=ww, d=d,
        growth_rate=growth,
        fit_result=fit,
        dense_rate=dense_rates if len(dense_rates) > 10 else None,
        dense_temperature=dense_temps if len(dense_rates) > 10 else None,
        dense_sigma=dense_sigma_arr,
        dense_supercooling=dense_supercooling_arr,
        n_extrema=n_extrema,
    )


def run_modern(prn: PrnData, params: CycleParams,
               salt: int = 1, acid: int = 0, face: int = 0,
               channel: int = 1, dtau: float = 0.055,
               ww: int = 1, d: float = 3.3, k1: float = 0.15,
               smooth_window: int = 5,
               span1: float = 0.2,
               tn_manual: Optional[float] = None) -> PipelineResult:
    """
    Modern pipeline — classic algorithm enhanced with modern preprocessing.

    Improvements over classic:
    Step 1: Bandpass filter (Butterworth) instead of just medsmooth —
            removes both low-frequency drift and high-frequency noise.
            Adaptive frequency range based on expected fringe frequencies.
    Step 2: Amplitude normalization via Hilbert envelope —
            compensates for signal decay, making baseline y0 more accurate.

    Rest of algorithm identical to classic (zero-crossing extrema,
    quadratic refinement, phase-based rates, dense d-step s2, LOESS Sig035).
    """
    from scipy import signal as scipy_sig
    from scipy.signal import hilbert as scipy_hilbert

    sol = get_solubility_set(salt, acid)

    # Step 1: Extract sub-array
    n1, n2 = params.n1, params.n2
    sub = prn.slice(n1, n2)
    temp_c = sub.temp_c_mathcad

    im = params.im
    isat = params.isat
    im1 = params.im1
    isat1 = params.isat1

    # Step 2: Signal = LED1 / LED2
    led1 = sub.led1
    led2 = sub.led2
    led2_safe = np.where(np.abs(led2) < 1e-10, 1e-10, led2)
    signal = led1 / led2_safe
    m = len(signal)

    # Step 3 (MODERN): Preprocessing — detrend + bandpass + normalize
    # 3a: Remove linear trend (baseline drift)
    preprocessed = scipy_sig.detrend(signal, type="linear")

    # 3b: Bandpass Butterworth filter
    # Fringe frequencies are typically 0.0006–0.005 Hz (fs=1 Hz)
    # Use wider band with margin: 0.0003–0.01 Hz
    fs = 1.0  # 1 sample/second
    f_low = 0.0003
    f_high = 0.01
    nyq = fs / 2.0
    low_norm = max(f_low / nyq, 0.001)
    high_norm = min(f_high / nyq, 0.999)
    if low_norm < high_norm:
        sos = scipy_sig.butter(4, [low_norm, high_norm], btype="band", output="sos")
        preprocessed = scipy_sig.sosfiltfilt(sos, preprocessed)

    # 3c: Amplitude normalization via Hilbert envelope
    analytic = scipy_hilbert(preprocessed)
    envelope = np.abs(analytic)
    # Smooth envelope to avoid local artifacts
    env_window = min(201, len(envelope))
    if env_window % 2 == 0:
        env_window += 1
    envelope_smooth = scipy_sig.savgol_filter(envelope, env_window, polyorder=2)
    envelope_smooth = np.maximum(envelope_smooth, 1e-10)
    normalized = preprocessed / envelope_smooth

    # Step 4 (MODERN): Continuous phase via Hilbert transform
    # Instead of discrete extrema, get continuous phase and instantaneous frequency.
    # This gives ~100x more data points than extrema-based approach.
    te = float(temp_c[min(isat, len(temp_c) - 1)])

    # Step 4a (MODERN): Automatic tn determination when not manually set.
    # Use envelope derivative zero-crossing in the saturation region.
    if tn_manual is not None:
        tn = tn_manual
    else:
        try:
            from .saturation import determine_saturation
            sat_result = determine_saturation(
                signal, temp_c, sol, im, isat,
                span=0.15, smooth_factor=0.01,
            )
            tn = sat_result.tn
            # Sanity check: tn should be near te (within ~2°C)
            if abs(tn - te) > 2.0 or tn > te:
                tn = te  # fallback
        except Exception:
            tn = te

    # Compute analytic signal and unwrapped phase from bandpass-filtered signal
    # (use preprocessed, not normalized — normalization distorts phase)
    analytic_phase = scipy_hilbert(preprocessed)
    phase_raw = np.unwrap(np.angle(analytic_phase))

    # Phase should be monotonically increasing in growth zone (crystal growing)
    # and may decrease in dissolution zone.
    # Count extrema from phase: each π of true phase = one extremum (max→min or min→max)
    phase_growth = phase_raw[:im1]
    total_phase_change = phase_growth[-1] - phase_growth[0] if len(phase_growth) > 1 else 0
    n_extrema = max(0, int(abs(total_phase_change) / np.pi))

    if n_extrema < 3:
        return _empty_result("modern", prn, params, salt, acid, face, dtau, ww, d, te, tn)

    # Step 5 (MODERN): Instantaneous frequency → growth rate
    # freq(t) = (1/2π) * dφ/dt  [Hz = cycles/second]
    # Each full cycle = change in optical path of λ/(2n)
    # R = freq / x * (1/π) * 30  [mm/day], same normalization as classic
    co0, co1_opt = OPTICAL_COEFFICIENTS.get((salt, face), (0.06446, 1.92e-5))

    # Compute instantaneous frequency
    inst_freq = np.gradient(phase_raw) / (2.0 * np.pi)  # Hz

    # Smooth frequency to remove noise (adaptive window)
    freq_window = max(51, min(201, m // 50))
    if freq_window % 2 == 0:
        freq_window += 1
    inst_freq_smooth = scipy_sig.savgol_filter(inst_freq, freq_window, polyorder=2)

    # Convert to growth rate in mm/day
    # Mathcad phase convention: each extremum = π/2 of "Mathcad phase"
    # Hilbert true phase: each extremum (max→min) = π of true phase
    # So true_phase = 2 * mathcad_phase
    # Classic formula: R = d(mathcad_phase) / (π·x) * 30
    # With true phase: R = d(true_phase) / (2·π·x) * 30 = inst_freq * 30 / x
    x_arr = co0 - co1_opt * temp_c
    x_arr = np.where(np.abs(x_arr) < 1e-15, 1e-15, x_arr)
    rate_continuous = np.abs(inst_freq_smooth) * 30.0 / x_arr  # mm/day

    # Zero out rates where envelope is small (dead zone / no oscillations).
    # The envelope of the bandpass signal indicates fringe amplitude.
    # In dead zone, envelope → 0 → rate is noise, not real.
    env_threshold = 0.25 * np.max(envelope_smooth)
    rate_continuous = np.where(envelope_smooth > env_threshold, rate_continuous, 0.0)

    # Subsample at d-step spacing (like classic dense sampling)
    # This gives ~2000 evenly spaced points including dead zone
    g = int(np.floor(isat1 / d))
    if g < 10:
        return _empty_result("modern", prn, params, salt, acid, face, dtau, ww, d, te, tn)

    positions = np.arange(g + 1) * d
    rates = np.interp(positions, np.arange(m), rate_continuous)
    rate_temps = np.interp(positions, np.arange(len(temp_c)), temp_c)

    # Non-negative rates in growth zone
    rates = np.maximum(rates, 0.0)

    # Rate times
    rate_times = np.zeros_like(rates)
    if sub.time_seconds is not None and len(sub.time_seconds) > 0:
        rate_times = np.interp(positions, np.arange(len(sub.time_seconds)),
                               sub.time_seconds) / 3600.0

    growth = GrowthRateData(
        rate=rates,
        rate_mm_day=rates,
        temperature=rate_temps,
        supercooling=np.zeros_like(rates),
        sigma_percent=np.zeros_like(rates),
        time_hours=rate_times,
        channel=channel,
        positions=positions,
        phase=np.interp(positions, np.arange(m), phase_raw),
    )

    # Step 6 (MODERN): Compute sigma% and filter growth zone
    sigma = supersaturation_percent(growth.temperature, tn, sol)
    growth.sigma_percent = sigma
    growth.supercooling = supercooling(growth.temperature, tn)

    growth_mask = (sigma > 0) & (growth.rate > 0)
    sigma_growth = sigma[growth_mask]
    rate_growth = rates[growth_mask]

    if len(sigma_growth) < 3:
        return _empty_result("modern", prn, params, salt, acid, face, dtau, ww, d, te, tn)

    # Step 7 (MODERN): Power law fit (grid search + scipy refinement) + BCF fit
    # Start with Mathcad grid search for initial estimate, then refine with curve_fit
    fit = fit_power_law(sigma_growth, rate_growth, w=1.0)
    # Refine with scipy curve_fit
    try:
        from scipy.optimize import curve_fit as scipy_curve_fit
        from .kinetics.power_law import power_law_model

        def _model(sigma, s0, s1):
            return power_law_model(sigma, s0, s1, w=1.0)

        popt, _ = scipy_curve_fit(
            _model, sigma_growth, rate_growth,
            p0=[fit.s0, fit.s1],
            bounds=([0, -10], [100, np.max(sigma_growth)]),
            maxfev=5000,
        )
        s0_opt, s1_opt = popt
        rate_fitted_opt = power_law_model(sigma_growth, s0_opt, s1_opt, 1.0)
        residual_opt = float(np.sum((rate_growth - rate_fitted_opt) ** 2))
        # Only use if better
        if residual_opt < fit.residual:
            fit = PowerLawResult(
                s0=s0_opt, s1=s1_opt, w=1.0, residual=residual_opt,
                sigma_percent=fit.sigma_percent,
                rate_measured=fit.rate_measured,
                rate_fitted=rate_fitted_opt,
                sig035=fit.sig035, s2=fit.s2,
            )
    except Exception:
        pass  # keep grid search result

    from .kinetics.bcf_model import fit_bcf
    bcf_fit = fit_bcf(sigma_growth, rate_growth)

    # Step 8 (MODERN): s2 via classic dense d-step method (hybrid approach).
    # Hilbert instantaneous frequency is noisy in the dead zone transition,
    # which breaks the sqrt(R) vs σ linear regression for s2.
    # Use classic extrema-based phase for s2 only — it gives clean R→0 transition.
    smoothed_for_s2 = smooth_signal(signal, method="median", window=smooth_window)
    y0_s2 = float(np.mean(smoothed_for_s2[:im1])) if im1 > 0 and im1 < m else float(np.mean(smoothed_for_s2))
    extrs_s2 = find_extrema_mathcad(smoothed_for_s2, y0_s2, 0, im1)
    if len(extrs_s2.positions) >= 3:
        extrs_s2 = filter_extrema_edges(extrs_s2, 0, im1)
    if len(extrs_s2.positions) >= 3:
        es_s2 = refine_extrema_quadratic(smoothed_for_s2, extrs_s2, k1=k1)
        dense_rates, dense_temps, _, _ = build_phase_dstep(
            smoothed_for_s2, es_s2, temp_c, end_pos=isat1, d=d,
            co0=co0, co1=co1_opt, nn=-1,
            im=im, ww=ww,
        )
        if len(dense_rates) > 10:
            dense_sigma = supersaturation_percent(dense_temps, tn, sol)
            s2 = compute_s2_mathcad(dense_sigma, dense_rates)
        else:
            s2 = compute_s2_mathcad(sigma, rates)
    else:
        s2 = compute_s2_mathcad(sigma, rates)

    # Step 9 (MODERN): Sig035 via LOESS
    Sig035 = compute_sig035_loess(sigma_growth, rate_growth, span=span1)

    # Step 14: Dead zone
    t_im = float(temp_c[min(im, len(temp_c) - 1)])
    Td = te - t_im

    Cn = solubility(te, sol)
    Cm = solubility(t_im, sol)
    if Cm > 0 and Cn > 0:
        Sigm = float(100.0 * np.log(Cn / Cm))
    else:
        Sigm = float(supersaturation_percent(t_im, tn, sol))

    return PipelineResult(
        filename=str(prn.filepath.name),
        cycle_index=0,
        mode="modern",
        te=te, tn=tn, Td=Td, Sigm=Sigm,
        s0=fit.s0, s1=fit.s1, s2=s2, Sig035=Sig035,
        Salt=salt, Acid=acid, Face=face,
        n1=n1, n2=n2, im=im, isat=isat, im1=im1, isat1=isat1,
        dtau=dtau, ww=ww, d=d,
        growth_rate=growth,
        fit_result=fit,
        bcf_result=bcf_fit,
        n_extrema=n_extrema,
    )


def _empty_result(mode, prn, params, salt, acid, face, dtau, ww, d,
                  te, tn) -> PipelineResult:
    """Create an empty result when processing fails."""
    return PipelineResult(
        filename=str(prn.filepath.name),
        mode=mode,
        te=te, tn=tn,
        Salt=salt, Acid=acid, Face=face,
        n1=params.n1, n2=params.n2,
        im=params.im, isat=params.isat,
        im1=params.im1, isat1=params.isat1,
        dtau=dtau, ww=ww, d=d,
    )
