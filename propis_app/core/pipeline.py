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
    fit_power_law, fit_dissolution, power_law_model,
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
    s0_d: float = 0.0         # dissolution kinetic coefficient
    s1_d: float = 0.0         # dissolution dead zone (% supersaturation)

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
    diss_fit_result: Optional[PowerLawResult] = None  # Dissolution fit

    # Dense d-step data (from build_phase_dstep, ~2000+ points including dead zone)
    # Used for Mathcad-style F1 LOESS curve
    dense_rate: Optional[np.ndarray] = None          # R (mm/day)
    dense_temperature: Optional[np.ndarray] = None   # T (°C)
    dense_sigma: Optional[np.ndarray] = None         # σ (%)
    dense_supercooling: Optional[np.ndarray] = None  # dT = tn - T (°C)

    # Alternative rate arrays (modern pipeline: method comparison)
    dense_rate_asg: Optional[np.ndarray] = None       # Adaptive Savitzky-Golay
    dense_rate_reg: Optional[np.ndarray] = None       # Sliding linear regression
    dense_rate_reg_r2: Optional[np.ndarray] = None    # R² quality for regression
    dense_rate_pll: Optional[np.ndarray] = None       # Phase-Locked Loop
    dense_rate_cwt: Optional[np.ndarray] = None       # CWT Ridge
    dense_rate_stft: Optional[np.ndarray] = None      # STFT spectral

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
    if tn_manual is not None:
        tn = tn_manual
    else:
        # Mathcad: tn := a[isat,4] - 273.15 - 0.01*(te - 25)
        # Since temp_c[isat] = te by definition, simplifies to:
        tn = te - 0.01 * (te - 25)

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

    rates_coarse = np.array(rates)
    rate_temps_coarse = np.array(rate_temps)
    rate_pos_coarse = np.array(rate_pos)

    if len(rates_coarse) < 3:
        return _empty_result("classic", prn, params, salt, acid, face, dtau, ww, d, te, tn)

    # Step 9: Dense d-step rate sampling (Mathcad zs-formula)
    # Mathcad builds phase function at d-step spacing from 0 to isat1
    # (~2000+ points including dead zone) — this is the PRIMARY data for fitting and plotting
    dense_rates, dense_temps, dense_pos, _ = build_phase_dstep(
        smoothed, es, temp_c, end_pos=isat1, d=d,
        co0=co0, co1=co1_opt, nn=-1,
        im=im, ww=ww,
    )

    if len(dense_rates) > 10:
        # Use dense data as primary — smooth curves (~2000 points)
        dense_sigma_arr = supersaturation_percent(dense_temps, tn, sol)
        dense_supercooling_arr = supercooling(dense_temps, tn)

        # Dead zone suppression: при dT < Td скорость = 0
        tm = float(temp_c[min(im, len(temp_c) - 1)])
        Td_local = te - tm
        dense_rates[dense_supercooling_arr < Td_local] = 0.0

        # Build GrowthRateData from dense data
        dense_times = np.zeros_like(dense_rates)
        if sub.time_seconds is not None and len(sub.time_seconds) > 0:
            dense_times = np.interp(dense_pos, np.arange(len(sub.time_seconds)),
                                    sub.time_seconds) / 3600.0

        growth = GrowthRateData(
            rate=dense_rates,
            rate_mm_day=dense_rates,
            temperature=dense_temps,
            supercooling=dense_supercooling_arr,
            sigma_percent=dense_sigma_arr,
            time_hours=dense_times,
            channel=channel,
            positions=dense_pos,
            phase=np.zeros_like(dense_rates),
        )

        # Filter for fitting: sigma > 0 and rate > 0 (growth zone)
        growth_mask = (dense_sigma_arr > 0) & (dense_rates > 0)
        sigma_growth = dense_sigma_arr[growth_mask]
        rate_growth = dense_rates[growth_mask]

        if len(sigma_growth) < 3:
            return _empty_result("classic", prn, params, salt, acid, face, dtau, ww, d, te, tn)

        # Step 10: Power law fit from dense data
        fit = fit_power_law(sigma_growth, rate_growth, w=1.0)

        # Step 11: s2 from dense data
        s2 = compute_s2_mathcad(dense_sigma_arr, dense_rates)

        # Step 12: Sig035 via LOESS from dense data
        Sig035 = compute_sig035_loess(sigma_growth, rate_growth, span=span1)
    else:
        # Fallback to per-extremum coarse data if dense fails
        dense_sigma_arr = None
        dense_supercooling_arr = None

        coarse_times = np.zeros_like(rates_coarse)
        if sub.time_seconds is not None and len(sub.time_seconds) > 0:
            for idx, p in enumerate(rate_pos_coarse):
                t = np.interp(p, np.arange(len(sub.time_seconds)), sub.time_seconds)
                coarse_times[idx] = t / 3600.0

        growth = GrowthRateData(
            rate=rates_coarse,
            rate_mm_day=rates_coarse,
            temperature=rate_temps_coarse,
            supercooling=np.zeros_like(rates_coarse),
            sigma_percent=np.zeros_like(rates_coarse),
            time_hours=coarse_times,
            channel=channel,
            positions=rate_pos_coarse,
            phase=np.zeros_like(rates_coarse),
        )

        sigma = supersaturation_percent(growth.temperature, tn, sol)
        growth.sigma_percent = sigma
        growth.supercooling = supercooling(growth.temperature, tn)

        growth_mask = (sigma > 0) & (growth.rate > 0)
        sigma_growth = sigma[growth_mask]
        rate_growth = rates_coarse[growth_mask]

        if len(sigma_growth) < 3:
            return _empty_result("classic", prn, params, salt, acid, face, dtau, ww, d, te, tn)

        fit = fit_power_law(sigma_growth, rate_growth, w=1.0)
        s2 = compute_s2_mathcad(sigma, rates_coarse)
        Sig035 = compute_sig035_loess(sigma_growth, rate_growth, span=span1)

    # Step 12b: Dissolution fit — find extrema in dissolution zone (after isat)
    # find_extrema_mathcad doesn't work well here due to baseline drift,
    # so we use scipy.find_peaks on detrended signal.
    from scipy.signal import find_peaks as _find_peaks
    from scipy.signal import detrend as _detrend
    diss_fit = None
    diss_start = isat
    diss_end = m
    if diss_end - diss_start > 50:
        diss_sig = _detrend(smoothed[diss_start:diss_end].astype(np.float64))
        min_dist = max(10, (diss_end - diss_start) // 100)
        peaks_d, _ = _find_peaks(diss_sig, distance=min_dist, prominence=0.003)
        troughs_d, _ = _find_peaks(-diss_sig, distance=min_dist, prominence=0.003)
        # Merge peaks and troughs, sort by position
        all_ext_pos = sorted(
            [int(p) + diss_start for p in peaks_d] +
            [int(t) + diss_start for t in troughs_d]
        )
        if len(all_ext_pos) >= 3:
            diss_rates_list = []
            diss_temps_list = []
            for i in range(len(all_ext_pos) - 1):
                pos1 = all_ext_pos[i]
                pos2 = all_ext_pos[i + 1]
                if pos2 <= pos1:
                    continue
                idx_s = max(0, pos1)
                idx_e = min(len(temp_c) - 1, pos2)
                T_avg = float(np.mean(temp_c[idx_s:idx_e + 1]))
                x = co0 - co1_opt * T_avg
                if abs(x) < 1e-15:
                    continue
                delta_L = (np.pi / 2.0) / (np.pi * x)
                R_mm_day = abs(delta_L / (pos2 - pos1) * 30)
                diss_rates_list.append(R_mm_day)
                diss_temps_list.append(T_avg)
            if len(diss_rates_list) >= 3:
                diss_rates_arr = np.array(diss_rates_list)
                diss_temps_arr = np.array(diss_temps_list)
                diss_sigma_arr = supersaturation_percent(diss_temps_arr, tn, sol)
                diss_rates_signed = -diss_rates_arr  # dissolution = negative rate
                diss_fit = fit_dissolution(diss_sigma_arr, diss_rates_signed, w=1.0)

    if diss_fit is None:
        diss_fit = PowerLawResult(
            s0=0, s1=0, w=1.0, residual=np.inf,
            sigma_percent=np.array([]), rate_measured=np.array([]),
            rate_fitted=np.array([]), sig035=0, s2=0,
        )

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
        s0_d=diss_fit.s0, s1_d=diss_fit.s1,
        Salt=salt, Acid=acid, Face=face,
        n1=n1, n2=n2, im=im, isat=isat, im1=im1, isat1=isat1,
        dtau=dtau, ww=ww, d=d,
        growth_rate=growth,
        fit_result=fit,
        diss_fit_result=diss_fit,
        dense_rate=dense_rates if len(dense_rates) > 10 else None,
        dense_temperature=dense_temps if len(dense_rates) > 10 else None,
        dense_sigma=dense_sigma_arr,
        dense_supercooling=dense_supercooling_arr,
        n_extrema=n_extrema,
    )


def _stft_frequency(signal_channel, fs=1.0, nperseg=256, noverlap=None):
    """
    STFT-based instantaneous frequency: в скользящем окне — FFT,
    доминантная частота = мгновенная частота роста.

    Шум усредняется внутри окна. Результат гладкий по определению.
    В dead zone: нет доминантной частоты → freq ≈ 0.

    Args:
        signal_channel: предобработанный сигнал
        fs: частота дискретизации
        nperseg: размер окна FFT
        noverlap: перекрытие (default: nperseg - 1 для максимального разрешения)

    Returns: inst_freq (Hz) — мгновенная частота, интерполированная на каждую точку
    """
    import scipy.signal as sig
    from scipy.interpolate import interp1d

    N = len(signal_channel)
    nperseg = min(nperseg, N)
    if noverlap is None:
        noverlap = nperseg - 1  # максимальное перекрытие

    f_axis, t_axis, Zxx = sig.stft(signal_channel, fs=fs,
                                     nperseg=nperseg, noverlap=noverlap,
                                     window='hann')

    # Спектрограмма мощности
    power = np.abs(Zxx) ** 2

    # Для каждого момента — частота с максимальной мощностью
    # Исключаем DC (f=0) и очень низкие частоты
    f_min_idx = max(1, int(0.0005 / (f_axis[1] - f_axis[0] + 1e-10)))
    power_trimmed = power.copy()
    power_trimmed[:f_min_idx, :] = 0

    dominant_idx = np.argmax(power_trimmed, axis=0)
    dominant_freq = f_axis[dominant_idx]

    # Мощность на доминантной частоте (мера уверенности)
    dominant_power = np.array([power[dominant_idx[i], i]
                                for i in range(len(dominant_idx))])
    total_power = np.sum(power, axis=0) + 1e-30
    spectral_purity = dominant_power / total_power  # 0..1

    # Интерполировать на исходную сетку (t_axis — центры окон)
    t_samples = t_axis * fs  # в отсчётах
    if len(t_samples) < 2:
        return np.zeros(N), np.zeros(N)

    freq_interp = interp1d(t_samples, dominant_freq, kind='linear',
                           fill_value='extrapolate', bounds_error=False)
    purity_interp = interp1d(t_samples, spectral_purity, kind='linear',
                              fill_value='extrapolate', bounds_error=False)

    inst_freq = freq_interp(np.arange(N))
    purity = purity_interp(np.arange(N))
    purity = np.clip(purity, 0, 1)

    return inst_freq, purity


def _pll_frequency(signal_channel, f_center, fs=1.0, bw=0.002, zeta=0.707):
    """
    Phase-Locked Loop: отслеживает мгновенную частоту сигнала.

    При потере сигнала (dead zone) интегратор удерживает последнюю частоту,
    которая плавно затухает к нулю. Нет скачков фазы.

    Args:
        signal_channel: предобработанный сигнал одного канала
        f_center: начальная оценка частоты (Hz)
        fs: частота дискретизации
        bw: bandwidth петли (Hz) — ширина полосы захвата
        zeta: damping factor (0.707 = критическое демпфирование)

    Returns: inst_freq (Hz) — мгновенная частота в каждой точке
    """
    N = len(signal_channel)
    # Loop filter coefficients (PI controller)
    wn = 2.0 * np.pi * bw / (zeta + 1.0 / (4.0 * zeta))
    Kp = 2.0 * zeta * wn / fs
    Ki = (wn / fs) ** 2

    freq_out = np.zeros(N)
    phase_acc = 0.0
    integrator = f_center  # начинаем с оценки центральной частоты

    for n in range(N):
        # NCO (numerically controlled oscillator)
        nco_phase = phase_acc
        nco_i = np.cos(nco_phase)
        nco_q = -np.sin(nco_phase)

        # Phase detector: перемножение с сигналом
        phase_error = np.arctan2(signal_channel[n] * nco_q,
                                  signal_channel[n] * nco_i)

        # Loop filter (PI)
        integrator += Ki * phase_error
        loop_out = Kp * phase_error + integrator

        # Update NCO phase
        phase_acc += 2.0 * np.pi * loop_out / fs
        freq_out[n] = loop_out

    return freq_out


def _cwt_ridge_frequency(signal_channel, fs=1.0):
    """
    CWT Ridge extraction: мгновенная частота через wavelet ridge.

    При малой амплитуде сигнала гребень ослабевает автоматически →
    dead zone suppression встроена в метод.

    Args:
        signal_channel: предобработанный сигнал одного канала
        fs: частота дискретизации

    Returns: (inst_freq, ridge_energy)
        inst_freq — мгновенная частота (Hz)
        ridge_energy — энергия гребня (мера уверенности)
    """
    from ssqueezepy import cwt, Wavelet
    from ssqueezepy.experimental import phase_cwt

    wavelet = Wavelet('gmw', {'gamma': 3, 'beta': 60})
    Wx, scales = cwt(signal_channel, wavelet, fs=fs)

    # Спектрограмма мощности
    power = np.abs(Wx) ** 2

    # Ridge: для каждого момента — масштаб с максимальной мощностью
    ridge_idx = np.argmax(power, axis=0)
    N = len(signal_channel)

    # Частота на гребне
    freqs_axis = fs * wavelet.center_frequency / scales
    inst_freq = freqs_axis[ridge_idx]

    # Энергия на гребне (мера надёжности)
    ridge_energy = np.array([power[ridge_idx[i], i] for i in range(N)])

    # Нормализуем энергию
    re_max = np.max(ridge_energy)
    if re_max > 0:
        ridge_energy = ridge_energy / re_max

    return inst_freq, ridge_energy


def _adaptive_savgol_rate(phase_raw, envelope_smooth, base_window=51,
                          min_window=21, max_window=501):
    """
    Adaptive Savitzky-Golay: окно обратно пропорционально envelope.

    Высокая амплитуда → короткое окно (хорошее временное разрешение).
    Низкая амплитуда  → длинное окно (шум усредняется, тренд сохраняется).

    Returns: inst_freq (Hz) — мгновенная частота = dφ/dt / (2π).
    """
    import scipy.signal as scipy_sig

    m = len(phase_raw)
    env_max = np.max(envelope_smooth)
    if env_max < 1e-10:
        return np.zeros(m)

    env_norm = envelope_smooth / env_max  # 0..1

    # Сначала вычислим dφ/dt через np.gradient (raw)
    raw_freq = np.gradient(phase_raw) / (2.0 * np.pi)

    # Адаптивное сглаживание: разбиваем сигнал на сегменты по уровню envelope,
    # для каждого сегмента применяем SG с соответствующим окном.
    # Но SG требует одно окно на весь массив → используем поточечное
    # взвешенное усреднение с адаптивным окном.

    result = np.zeros(m)
    # Предвычислим окно для каждой точки
    # window(i) = base_window / max(env_norm(i), 0.05)
    # При env_norm=1: window=base, при env_norm=0.05: window=base/0.05=20*base
    windows = base_window / np.maximum(env_norm, 0.05)
    windows = np.clip(windows, min_window, max_window).astype(int)
    # Делаем нечётными
    windows = windows + (1 - windows % 2)

    # Для эффективности: группируем по уникальным окнам
    unique_windows = np.unique(windows)
    for w in unique_windows:
        w = int(w)
        if w > m:
            w = m if m % 2 == 1 else m - 1
        if w < 5:
            w = 5
        mask = windows == w
        if not np.any(mask):
            continue
        # SG с этим окном на весь массив
        smoothed = scipy_sig.savgol_filter(raw_freq, w, polyorder=2)
        result[mask] = smoothed[mask]

    return result


def _sliding_regression_rate(phase_raw, envelope_smooth, base_window=51,
                             min_window=21, max_window=501):
    """
    Скользящая линейная регрессия фазы: slope = средняя dφ/dt в окне.

    Окно адаптивное: обратно пропорционально envelope.
    При малой амплитуде — длинное окно → шум усредняется,
    но реальный медленный тренд фазы сохраняется.

    Returns: (inst_freq, r_squared)
        inst_freq — мгновенная частота (Hz) = slope / (2π)
        r_squared — коэффициент детерминации (качество линейного фита)
    """
    m = len(phase_raw)
    env_max = np.max(envelope_smooth)
    if env_max < 1e-10:
        return np.zeros(m), np.zeros(m)

    env_norm = envelope_smooth / env_max

    # Адаптивные окна (как в adaptive SG)
    windows = base_window / np.maximum(env_norm, 0.05)
    windows = np.clip(windows, min_window, max_window).astype(int)
    windows = windows + (1 - windows % 2)  # нечётные

    inst_freq = np.zeros(m)
    r_squared = np.zeros(m)

    # Для каждой точки: линейная регрессия phase ~ t в окне
    # Оптимизация: группируем по окнам, используем скользящие суммы
    t_arr = np.arange(m, dtype=np.float64)

    unique_windows = np.unique(windows)
    for w in unique_windows:
        w = int(w)
        half = w // 2
        mask_indices = np.where(windows == w)[0]
        if len(mask_indices) == 0:
            continue

        for i in mask_indices:
            i0 = max(0, i - half)
            i1 = min(m, i + half + 1)
            t_local = t_arr[i0:i1]
            p_local = phase_raw[i0:i1]
            n = len(t_local)
            if n < 3:
                continue

            # Линейная регрессия: slope = (n·Σtp - Σt·Σp) / (n·Σt² - (Σt)²)
            st = np.sum(t_local)
            sp = np.sum(p_local)
            stt = np.sum(t_local * t_local)
            stp = np.sum(t_local * p_local)
            denom = n * stt - st * st
            if abs(denom) < 1e-30:
                continue
            slope = (n * stp - st * sp) / denom
            inst_freq[i] = slope / (2.0 * np.pi)

            # R² = 1 - SS_res / SS_tot
            intercept = (sp - slope * st) / n
            predicted = intercept + slope * t_local
            ss_res = np.sum((p_local - predicted) ** 2)
            ss_tot = np.sum((p_local - sp / n) ** 2)
            if ss_tot > 1e-30:
                r_squared[i] = max(0.0, 1.0 - ss_res / ss_tot)

    return inst_freq, r_squared


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

    # Step 2: Raw signals
    led1 = sub.led1.astype(np.float64)
    led2 = sub.led2.astype(np.float64)
    led2_safe = np.where(np.abs(led2) < 1e-10, 1e-10, led2)
    signal = led1 / led2_safe  # для classic-совместимых частей (s2 extrema)
    m = len(signal)

    # Step 3 (MODERN): Peak-trough normalization каждого канала
    # Находим пики и впадины → интерполируем огибающие → нормализуем
    # После нормализации оба канала = cos(φ), независимо от split ratio
    from scipy.interpolate import interp1d

    # Step 3 (MODERN): Preprocessing каждого канала + peak-trough amplitude
    fs = 1.0
    f_low = 0.0002
    nyq = fs / 2.0
    low_norm = max(f_low / nyq, 0.001)
    sos = scipy_sig.butter(4, low_norm, btype="high", output="sos")

    def _preprocess_channel(ch):
        pp = scipy_sig.detrend(ch, type="linear")
        pp = scipy_sig.sosfiltfilt(sos, pp)
        return pp

    pp1 = _preprocess_channel(led1)
    pp2 = _preprocess_channel(led2)

    # Peak-trough амплитуда каждого канала (для маски dead zone)
    def _peak_trough_amplitude(pp):
        """Амплитуда из пиков/впадин — надёжнее Hilbert envelope."""
        from scipy.interpolate import interp1d
        n = len(pp)
        min_dist = max(30, n // 500)
        peaks, _ = scipy_sig.find_peaks(pp, distance=min_dist)
        troughs, _ = scipy_sig.find_peaks(-pp, distance=min_dist)
        if len(peaks) < 3 or len(troughs) < 3:
            return np.abs(pp), np.ones(n, dtype=bool)
        t = np.arange(n, dtype=np.float64)
        upper = interp1d(peaks, np.abs(pp[peaks]), kind='linear',
                         fill_value='extrapolate', bounds_error=False)(t)
        lower = interp1d(troughs, np.abs(pp[troughs]), kind='linear',
                         fill_value='extrapolate', bounds_error=False)(t)
        amp = (upper + lower) / 2.0
        # Сгладить
        w = min(501, n)
        if w % 2 == 0:
            w += 1
        amp = scipy_sig.savgol_filter(amp, w, polyorder=2)
        amp = np.maximum(amp, 0.0)
        # Маска
        amp_med = float(np.median(amp[amp > 0])) if np.any(amp > 0) else 1.0
        has_signal = amp > amp_med * 0.10  # 10% от медианы
        return amp, has_signal

    amp1_pt, mask1 = _peak_trough_amplitude(pp1)
    amp2_pt, mask2 = _peak_trough_amplitude(pp2)
    has_signal = mask1 | mask2

    # Step 4 (MODERN): Dual-channel Hilbert на исходных (не нормализованных) сигналах
    analytic1 = scipy_hilbert(pp1)
    analytic2 = scipy_hilbert(pp2)
    phase1 = np.unwrap(np.angle(analytic1))
    phase2 = np.unwrap(np.angle(analytic2))
    amp1 = np.abs(analytic1)
    amp2 = np.abs(analytic2)

    # Weighted phase averaging
    w1 = amp1 ** 2
    w2 = amp2 ** 2
    phase_raw = (w1 * phase1 + w2 * phase2) / (w1 + w2 + 1e-10)

    # Комбинированная амплитуда (Hilbert envelope, для совместимости)
    envelope_smooth = np.sqrt((amp1 ** 2 + amp2 ** 2) / 2.0)
    env_window = min(201, m)
    if env_window % 2 == 0:
        env_window += 1
    envelope_smooth = scipy_sig.savgol_filter(envelope_smooth, env_window, polyorder=2)
    envelope_smooth = np.maximum(envelope_smooth, 1e-10)

    te = float(temp_c[min(isat, len(temp_c) - 1)])

    # Automatic tn determination
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
            if abs(tn - te) > 2.0 or tn > te:
                tn = te - 0.01 * (te - 25)
        except Exception:
            tn = te - 0.01 * (te - 25)

    # Count extrema from phase
    phase_growth = phase_raw[:im1]
    total_phase_change = phase_growth[-1] - phase_growth[0] if len(phase_growth) > 1 else 0
    n_extrema = max(0, int(abs(total_phase_change) / np.pi))

    if n_extrema < 3:
        return _empty_result("modern", prn, params, salt, acid, face, dtau, ww, d, te, tn)

    # Step 5 (MODERN): Instantaneous frequency → growth rate (3 метода)
    # freq(t) = (1/2π) * dφ/dt  [Hz = cycles/second]
    # R = inst_freq * 30 / x  [mm/day]
    co0, co1_opt = OPTICAL_COEFFICIENTS.get((salt, face), (0.06446, 1.92e-5))
    x_arr = co0 - co1_opt * temp_c
    x_arr = np.where(np.abs(x_arr) < 1e-15, 1e-15, x_arr)

    base_window = max(51, min(201, m // 50))
    if base_window % 2 == 0:
        base_window += 1

    # Signed rate: знак из физики (температура), амплитуда из Hilbert.
    # Hilbert не различает cos(+ωt) от cos(-ωt) → частота всегда ≥ 0.
    # Знак определяем из dT: рост (dT>0) → R>0, растворение (dT<0) → R<0.
    dT_sign = np.sign(tn - temp_c)  # +1 рост, -1 растворение, 0 на tn

    # Метод 1: Fixed Savitzky-Golay
    inst_freq_fixed = np.gradient(phase_raw) / (2.0 * np.pi)
    inst_freq_fixed = scipy_sig.savgol_filter(inst_freq_fixed, base_window, polyorder=2)
    rate_fixed = np.abs(inst_freq_fixed) * 30.0 / x_arr * dT_sign

    # Метод 2: Adaptive Savitzky-Golay
    inst_freq_asg = _adaptive_savgol_rate(phase_raw, envelope_smooth,
                                          base_window=base_window)
    rate_asg = np.abs(inst_freq_asg) * 30.0 / x_arr * dT_sign

    # Метод 3: Sliding linear regression
    inst_freq_reg, r2_reg = _sliding_regression_rate(
        phase_raw, envelope_smooth, base_window=base_window)
    rate_reg = np.abs(inst_freq_reg) * 30.0 / x_arr * dT_sign

    # Метод 4: PLL
    f_est = n_extrema / (2.0 * im1) if im1 > 0 else 0.01
    pp_avg = (pp1 + pp2) / 2.0
    inst_freq_pll = _pll_frequency(pp_avg, f_center=f_est, fs=1.0,
                                    bw=max(f_est * 0.1, 0.0005))
    rate_pll = np.abs(inst_freq_pll) * 30.0 / x_arr * dT_sign

    # Метод 5: CWT Ridge
    try:
        inst_freq_cwt, ridge_energy = _cwt_ridge_frequency(pp_avg, fs=1.0)
        rate_cwt = np.abs(inst_freq_cwt) * 30.0 / x_arr * dT_sign
    except Exception:
        rate_cwt = rate_asg.copy()

    # Метод 6: STFT
    stft_window = max(512, min(2048, m // 4))
    inst_freq_stft, spectral_purity = _stft_frequency(pp_avg, fs=1.0, nperseg=stft_window)
    rate_stft = np.abs(inst_freq_stft) * 30.0 / x_arr * dT_sign

    # Основной rate = adaptive SG
    rate_continuous = rate_asg

    # Subsample at d-step spacing (like classic dense sampling)
    # This gives ~2000 evenly spaced points including dead zone
    g = int(np.floor(isat1 / d))
    if g < 10:
        return _empty_result("modern", prn, params, salt, acid, face, dtau, ww, d, te, tn)

    positions = np.arange(g + 1) * d
    sample_x = np.arange(m)
    rates = np.interp(positions, sample_x, rate_continuous)
    rates_asg = np.interp(positions, sample_x, rate_asg)
    rates_reg = np.interp(positions, sample_x, rate_reg)
    rates_pll = np.interp(positions, sample_x, rate_pll)
    rates_cwt = np.interp(positions, sample_x, rate_cwt)
    rates_stft = np.interp(positions, sample_x, rate_stft)
    rates_reg_r2 = np.interp(positions, sample_x, r2_reg)
    rate_temps = np.interp(positions, np.arange(len(temp_c)), temp_c)

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

    # Step 7b (MODERN): Dissolution fit — mirror of growth power law

    diss_fit = fit_dissolution(sigma, rates, w=1.0)

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

    # Dense data for F1 LOESS plotting (Hilbert-based rates at d-step spacing)
    dense_sigma_arr = supersaturation_percent(rate_temps, tn, sol)
    dense_supercooling_arr = supercooling(rate_temps, tn)

    return PipelineResult(
        filename=str(prn.filepath.name),
        cycle_index=0,
        mode="modern",
        te=te, tn=tn, Td=Td, Sigm=Sigm,
        s0=fit.s0, s1=fit.s1, s2=s2, Sig035=Sig035,
        s0_d=diss_fit.s0, s1_d=diss_fit.s1,
        Salt=salt, Acid=acid, Face=face,
        n1=n1, n2=n2, im=im, isat=isat, im1=im1, isat1=isat1,
        dtau=dtau, ww=ww, d=d,
        growth_rate=growth,
        fit_result=fit,
        diss_fit_result=diss_fit,
        bcf_result=bcf_fit,
        dense_rate=rates,
        dense_temperature=rate_temps,
        dense_sigma=dense_sigma_arr,
        dense_supercooling=dense_supercooling_arr,
        dense_rate_asg=rates_asg,
        dense_rate_reg=rates_reg,
        dense_rate_reg_r2=rates_reg_r2,
        dense_rate_pll=rates_pll,
        dense_rate_cwt=rates_cwt,
        dense_rate_stft=rates_stft,
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
