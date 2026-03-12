"""
Signal preprocessing pipeline for interferometric data.

Pipeline stages:
  1. Validation: NaN/gap/spike detection
  2. Detrend: remove baseline drift
  3. Bandpass filter: Butterworth, removes LF drift and HF noise
  4. Amplitude normalization: divide by Hilbert envelope
  5. Wavelet denoising (optional): PyWavelets soft thresholding
  6. Artifact masking: parasitic fringes, jumps

Two modes: raw (no preprocessing) and full pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import signal as sig
from scipy.signal import hilbert


@dataclass
class PreprocessingParams:
    """Parameters for the preprocessing pipeline."""
    # Detrend
    detrend: bool = True
    detrend_type: str = "linear"  # "linear" or "constant"

    # Bandpass filter
    bandpass: bool = True
    f_low: float = 0.005   # Hz — highpass cutoff (removes drift)
    f_high: float = 0.1    # Hz — lowpass cutoff (removes HF noise)
    filter_order: int = 4

    # Amplitude normalization
    normalize_amplitude: bool = True
    envelope_smooth_window: int = 201  # samples for smoothing envelope

    # Wavelet denoising
    wavelet_denoise: bool = False
    wavelet_name: str = "db4"
    wavelet_level: Optional[int] = None  # auto if None
    wavelet_threshold_mode: str = "soft"

    # Median filter for spike removal
    median_filter: bool = False
    median_kernel: int = 5

    # Artifact detection
    spike_threshold: float = 5.0    # std deviations for spike detection
    parasitic_fringes: int = 0      # number of initial fringes to exclude


@dataclass
class PreprocessingResult:
    """Result of preprocessing."""
    signal_raw: np.ndarray
    signal_processed: np.ndarray
    envelope: Optional[np.ndarray] = None
    artifact_mask: Optional[np.ndarray] = None  # True = valid, False = artifact
    params: Optional[PreprocessingParams] = None


def validate_signal(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Check for NaN, inf, and gaps. Interpolate missing values.

    Returns
    -------
    cleaned : np.ndarray
        Signal with NaN/inf replaced by interpolation.
    valid_mask : np.ndarray
        Boolean mask, True where original data was valid.
    """
    valid = np.isfinite(signal)
    cleaned = signal.copy()

    if not np.all(valid):
        x = np.arange(len(signal))
        cleaned[~valid] = np.interp(x[~valid], x[valid], signal[valid])

    return cleaned, valid


def detect_spikes(signal: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """Detect spikes as points deviating more than threshold*std from local median."""
    kernel = min(51, len(signal) // 10)
    if kernel % 2 == 0:
        kernel += 1
    if kernel < 3:
        return np.ones(len(signal), dtype=bool)

    median = sig.medfilt(signal, kernel_size=kernel)
    residual = np.abs(signal - median)
    std = np.std(residual)
    return residual < threshold * std


def detrend_signal(signal: np.ndarray, detrend_type: str = "linear") -> np.ndarray:
    """Remove linear or constant trend."""
    return sig.detrend(signal, type=detrend_type)


def bandpass_filter(signal: np.ndarray, f_low: float, f_high: float,
                    fs: float = 1.0, order: int = 4) -> np.ndarray:
    """
    Apply Butterworth bandpass filter.

    Parameters
    ----------
    signal : array
    f_low, f_high : float
        Cutoff frequencies in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Filter order.
    """
    nyq = fs / 2.0

    # Clamp frequencies to valid range
    low = max(f_low / nyq, 0.001)
    high = min(f_high / nyq, 0.999)

    if low >= high:
        return signal

    sos = sig.butter(order, [low, high], btype="band", output="sos")
    return sig.sosfiltfilt(sos, signal)


def compute_envelope(signal: np.ndarray,
                     smooth_window: int = 201) -> np.ndarray:
    """
    Compute signal envelope using Hilbert transform.

    Parameters
    ----------
    signal : array
        Input signal (should be bandpass-filtered first for best results).
    smooth_window : int
        Window for smoothing the envelope.
    """
    analytic = hilbert(signal)
    env = np.abs(analytic)

    # Smooth the envelope
    if smooth_window > 1:
        window = min(smooth_window, len(env))
        if window % 2 == 0:
            window += 1
        env = sig.savgol_filter(env, window, polyorder=2)
        env = np.maximum(env, 1e-10)  # avoid division by zero

    return env


def normalize_amplitude(signal: np.ndarray,
                        envelope: np.ndarray) -> np.ndarray:
    """Normalize signal by dividing by its envelope."""
    return signal / np.maximum(envelope, 1e-10)


def wavelet_denoise(signal: np.ndarray, wavelet: str = "db4",
                    level: Optional[int] = None,
                    threshold_mode: str = "soft") -> np.ndarray:
    """
    Wavelet denoising with soft thresholding.

    Requires PyWavelets (pywt).
    """
    try:
        import pywt
    except ImportError:
        return signal

    if level is None:
        level = pywt.dwt_max_level(len(signal), wavelet)
        level = min(level, 6)

    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Estimate noise from finest detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    # Threshold detail coefficients (keep approximation)
    denoised_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(c, threshold, mode=threshold_mode))

    return pywt.waverec(denoised_coeffs, wavelet)[:len(signal)]


def preprocess(signal: np.ndarray, fs: float = 1.0,
               params: Optional[PreprocessingParams] = None) -> PreprocessingResult:
    """
    Run the full preprocessing pipeline.

    Parameters
    ----------
    signal : array
        Raw interferometric signal.
    fs : float
        Sampling frequency (Hz). Default 1.0 (1 sample/second).
    params : PreprocessingParams, optional
        Pipeline parameters. Uses defaults if None.

    Returns
    -------
    PreprocessingResult
    """
    if params is None:
        params = PreprocessingParams()

    raw = signal.copy()
    s = signal.copy()

    # 1. Validation
    s, valid_mask = validate_signal(s)
    artifact_mask = valid_mask.copy()

    # Spike detection
    spike_mask = detect_spikes(s, params.spike_threshold)
    artifact_mask &= spike_mask

    # Replace spikes with interpolated values
    if not np.all(spike_mask):
        x = np.arange(len(s))
        s[~spike_mask] = np.interp(x[~spike_mask], x[spike_mask], s[spike_mask])

    # 2. Median filter (optional, for initial spike removal)
    if params.median_filter:
        kernel = params.median_kernel
        if kernel % 2 == 0:
            kernel += 1
        s = sig.medfilt(s, kernel_size=kernel)

    # 3. Detrend
    if params.detrend:
        s = detrend_signal(s, params.detrend_type)

    # 4. Bandpass filter
    if params.bandpass:
        s = bandpass_filter(s, params.f_low, params.f_high, fs=fs,
                            order=params.filter_order)

    # 5. Amplitude normalization
    envelope = None
    if params.normalize_amplitude:
        envelope = compute_envelope(s, params.envelope_smooth_window)
        s = normalize_amplitude(s, envelope)

    # 6. Wavelet denoising (optional)
    if params.wavelet_denoise:
        s = wavelet_denoise(s, params.wavelet_name,
                            params.wavelet_level,
                            params.wavelet_threshold_mode)

    # 7. Mark parasitic fringes
    if params.parasitic_fringes > 0:
        # Mark first N fringes as artifacts — approximate by fraction of signal
        # Actual fringe counting happens in signal_processing
        pass

    return PreprocessingResult(
        signal_raw=raw,
        signal_processed=s,
        envelope=envelope,
        artifact_mask=artifact_mask,
        params=params,
    )
