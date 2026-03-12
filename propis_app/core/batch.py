"""
Batch processing of multiple PRN files.

Processes all PRN files in a directory, applies the same algorithm,
and produces a summary table of results.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .prn_reader import PrnData, read_prn
from .solubility import SolubilitySet, get_solubility_set, supersaturation_percent
from .preprocessing import PreprocessingParams, preprocess
from .signal_processing.classic import process_channel_classic
from .saturation import determine_saturation, SaturationResult
from .kinetics.power_law import fit_power_law, PowerLawResult
from .reference_curves import ReferenceManager


@dataclass
class ProcessingParams:
    """All parameters needed for processing one PRN file."""
    # Data selection
    n1: int = 100             # start of working region
    n2: int = 10000           # end of working region
    im: int = 6500            # left boundary of saturation search
    isat: int = 9000          # right boundary of saturation search

    # Crystal/solution type
    salt: int = 1             # 1=KDP, 2=DKDP
    acid: int = 0             # 0=neutral, 1=acid
    face: int = 0             # 0=prism, 1=pyramid

    # Signal processing
    smooth_window: int = 5
    min_prominence: Optional[float] = None
    min_distance: int = 10
    n_parasitic: int = 0      # parasitic fringes to exclude

    # Saturation
    span: float = 0.15
    smooth_factor: float = 0.01

    # Fitting
    w: float = 1.0            # power law exponent

    # Preprocessing
    use_preprocessing: bool = False
    preprocessing: PreprocessingParams = field(default_factory=PreprocessingParams)

    # Algorithm mode
    mode: str = "classic"     # "classic" or "modern"


@dataclass
class ProcessingResult:
    """Result of processing one PRN file."""
    filepath: Path
    params: ProcessingParams

    # Key parameters (Mat row)
    salt: int = 1
    acid: int = 0
    te: float = 0.0           # equilibrium temperature
    tn: float = 0.0           # saturation temperature
    td: float = 0.0           # dead zone (°C)
    s1: float = 0.0           # dead zone (% supersaturation)
    s0: float = 0.0           # kinetic coefficient
    s2: float = 0.0           # shape parameter
    sig035: float = 0.0       # σ at F>0.35
    w: float = 1.0            # power exponent

    # Detailed results
    saturation: Optional[SaturationResult] = None
    fit_ch1: Optional[PowerLawResult] = None
    fit_ch2: Optional[PowerLawResult] = None

    # Status
    success: bool = False
    error: str = ""


def process_single(prn_data: PrnData, params: ProcessingParams) -> ProcessingResult:
    """
    Process a single PRN file with given parameters.

    Parameters
    ----------
    prn_data : PrnData
        Parsed PRN data.
    params : ProcessingParams
        Processing parameters.

    Returns
    -------
    ProcessingResult
    """
    result = ProcessingResult(filepath=prn_data.filepath, params=params)
    result.salt = params.salt
    result.acid = params.acid

    try:
        # Get solubility coefficients
        sol = get_solubility_set(params.salt, params.acid)

        # Slice to working region
        n1 = max(0, params.n1)
        n2 = min(prn_data.n_samples, params.n2)
        data = prn_data.slice(n1, n2)

        # Choose signal (use LED1 by default)
        signal1 = data.led1
        signal2 = data.led2
        temp = data.temp_c

        # Preprocessing (optional)
        if params.use_preprocessing:
            from .preprocessing import preprocess as pp
            res1 = pp(signal1, fs=1.0/data.dt, params=params.preprocessing)
            res2 = pp(signal2, fs=1.0/data.dt, params=params.preprocessing)
            signal1 = res1.signal_processed
            signal2 = res2.signal_processed

        # Adjust im/isat relative to n1
        im_rel = params.im - n1
        isat_rel = params.isat - n1
        im_rel = max(0, min(im_rel, len(signal1) - 1))
        isat_rel = max(im_rel + 10, min(isat_rel, len(signal1)))

        # Determine saturation
        sat = determine_saturation(
            signal1, temp, sol,
            im=im_rel, isat=isat_rel,
            span=params.span,
            smooth_factor=params.smooth_factor,
        )
        result.saturation = sat
        result.tn = sat.tn
        result.td = sat.td
        result.s1 = sat.s1

        # Equilibrium temperature (max temperature in working region)
        result.te = float(np.max(temp))

        # Classic signal processing — Channel 1
        growth1 = process_channel_classic(
            signal1, temp, face=params.face, channel=1,
            smooth_window=params.smooth_window,
            min_prominence=params.min_prominence,
            min_distance=params.min_distance,
            dt_seconds=data.dt,
            time_seconds=data.time_seconds,
            n_parasitic=params.n_parasitic,
        )

        # Classic signal processing — Channel 2
        growth2 = process_channel_classic(
            signal2, temp, face=params.face, channel=2,
            smooth_window=params.smooth_window,
            min_prominence=params.min_prominence,
            min_distance=params.min_distance,
            dt_seconds=data.dt,
            time_seconds=data.time_seconds,
            n_parasitic=params.n_parasitic,
        )

        # Fill supercooling and supersaturation
        for growth in [growth1, growth2]:
            if len(growth.temperature) > 0:
                growth.supercooling = sat.tn - growth.temperature
                growth.sigma_percent = supersaturation_percent(
                    growth.temperature, sat.tn, sol
                )

        # Fit kinetic curves
        if len(growth1.sigma_percent) > 3:
            result.fit_ch1 = fit_power_law(
                growth1.sigma_percent, growth1.rate, w=params.w
            )

        if len(growth2.sigma_percent) > 3:
            result.fit_ch2 = fit_power_law(
                growth2.sigma_percent, growth2.rate, w=params.w
            )

        # Use channel 1 fit for primary parameters
        fit = result.fit_ch1 or result.fit_ch2
        if fit is not None:
            result.s0 = fit.s0
            result.s1 = fit.s1 * 100  # convert to %
            result.s2 = fit.s2
            result.sig035 = fit.sig035
            result.w = fit.w

        result.success = True

    except Exception as e:
        result.success = False
        result.error = str(e)

    return result


def process_batch(directory: str | Path, params: ProcessingParams,
                  pattern: str = "*.prn") -> list[ProcessingResult]:
    """
    Process all PRN files in a directory.

    Parameters
    ----------
    directory : str or Path
        Directory containing PRN files.
    params : ProcessingParams
        Processing parameters (same for all files).
    pattern : str
        Glob pattern for finding PRN files.

    Returns
    -------
    list of ProcessingResult
    """
    directory = Path(directory)
    prn_files = sorted(directory.glob(pattern))

    results = []
    for prn_file in prn_files:
        try:
            data = read_prn(prn_file)
            result = process_single(data, params)
        except Exception as e:
            result = ProcessingResult(
                filepath=prn_file, params=params,
                success=False, error=str(e),
            )
        results.append(result)

    return results


def results_to_mat_table(results: list[ProcessingResult]) -> list[dict]:
    """
    Convert results to Mat-format table (like Mathcad output).

    Returns list of dicts with keys:
    Salt, Acid, te, tn, Td, s1, Sigm, s2, Sig035, s0, w, file
    """
    rows = []
    for r in results:
        rows.append({
            "file": r.filepath.name,
            "Salt": r.salt,
            "Acid": r.acid,
            "te": round(r.te, 2),
            "tn": round(r.tn, 2),
            "Td": round(r.td, 2),
            "s1": round(r.s1, 4),
            "s0": round(r.s0, 4),
            "s2": round(r.s2, 4),
            "Sig035": round(r.sig035, 2),
            "w": round(r.w, 2),
            "success": r.success,
            "error": r.error,
        })
    return rows
