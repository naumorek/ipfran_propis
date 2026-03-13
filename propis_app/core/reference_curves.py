"""
Reference kinetic curves for comparison.

Two reference curves:
  - Cfe=0 ppm (green) — clean solution, ideal kinetics
  - Cfe=16 ppm (red) — contaminated solution, suppressed kinetics

Reference curves are temperature-dependent polynomials:
  Si(j, te) = a[j,0] + a[j,1]*te + a[j,2]*te^2

where j=0..7 are curve parameters and te is equilibrium temperature.

NOTE: The exact polynomial coefficients are embedded in binary MCD files
and are not yet extracted. This module provides:
  1. Placeholder structure for loading coefficients from JSON
  2. Manual curve definition interface (operator can input known values)
  3. Parametric reference curve generation from published kinetic data
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ReferenceCurve:
    """A single reference kinetic curve."""
    name: str
    cfe_ppm: float                    # Fe concentration (ppm)
    supercooling: np.ndarray          # ΔT array (°C)
    rate_mm_day: np.ndarray           # R (mm/day)
    sigma_percent: Optional[np.ndarray] = None  # σ (%)
    rate_um_min: Optional[np.ndarray] = None    # R (μm/min)


def _load_reference_json(filepath: Optional[Path] = None) -> dict:
    """Load reference coefficients from JSON file."""
    if filepath is None:
        filepath = Path(__file__).parent.parent / "data" / "reference_coefficients.json"
    with open(filepath) as f:
        return json.load(f)


def generate_power_law_curve(s0: float, s1: float, w: float,
                             dt_max: float = 5.0,
                             n_points: int = 200,
                             sol_set=None,
                             tn: float = 48.6) -> ReferenceCurve:
    """
    Generate a reference curve from power law parameters.

    Parameters
    ----------
    s0, s1, w : float
        Power law parameters.
    dt_max : float
        Maximum supercooling (°C).
    n_points : int
        Number of points.
    sol_set : SolubilitySet, optional
        For computing σ from ΔT.
    tn : float
        Saturation temperature.

    Returns
    -------
    ReferenceCurve
    """
    dt = np.linspace(0, dt_max, n_points)

    # Compute σ from ΔT if we have solubility coefficients
    sigma = None
    if sol_set is not None:
        from .solubility import supersaturation_percent
        T = tn - dt
        sigma = supersaturation_percent(T, tn, sol_set)

    # Compute rate from power law
    if sigma is not None:
        from .kinetics.power_law import power_law_model
        rate_um = power_law_model(sigma, s0, s1, w)
    else:
        # Approximate: σ ≈ a * ΔT for small ΔT
        sigma_approx = dt * 0.5  # rough approximation
        from .kinetics.power_law import power_law_model
        rate_um = power_law_model(sigma_approx, s0, s1, w)
        sigma = sigma_approx

    rate_mm = rate_um * 1.44  # μm/min → mm/day

    return ReferenceCurve(
        name=f"PowerLaw(s0={s0:.2f}, s1={s1:.4f}, w={w:.1f})",
        cfe_ppm=-1,
        supercooling=dt,
        rate_mm_day=rate_mm,
        sigma_percent=sigma,
        rate_um_min=rate_um,
    )


def create_reference_from_points(name: str, cfe_ppm: float,
                                 dt_points: np.ndarray,
                                 rate_points: np.ndarray,
                                 n_interp: int = 200) -> ReferenceCurve:
    """
    Create a reference curve from manually specified data points.

    Parameters
    ----------
    name : str
        Curve name.
    cfe_ppm : float
        Fe concentration.
    dt_points : array
        Supercooling values (°C).
    rate_points : array
        Growth rate values (mm/day).
    n_interp : int
        Number of interpolation points.

    Returns
    -------
    ReferenceCurve
    """
    from scipy.interpolate import interp1d

    sort_idx = np.argsort(dt_points)
    dt_sorted = dt_points[sort_idx]
    rate_sorted = rate_points[sort_idx]

    dt_interp = np.linspace(dt_sorted[0], dt_sorted[-1], n_interp)
    f = interp1d(dt_sorted, rate_sorted, kind="cubic", fill_value="extrapolate")
    rate_interp = np.maximum(f(dt_interp), 0)

    return ReferenceCurve(
        name=name,
        cfe_ppm=cfe_ppm,
        supercooling=dt_interp,
        rate_mm_day=rate_interp,
    )


def get_placeholder_references(te: float = 48.85) -> tuple[ReferenceCurve, ReferenceCurve]:
    """
    Get placeholder reference curves.

    These are approximate — real coefficients need to be extracted from MCD files
    or provided by the operator.

    Based on typical KDP kinetics from literature:
    - Clean solution (Cfe=0): R ≈ 3-5 mm/day at ΔT=1°C
    - Contaminated (Cfe=16ppm): R ≈ 0.5-1 mm/day at ΔT=1°C

    Returns
    -------
    (clean_curve, contaminated_curve)
    """
    dt = np.linspace(0, 5, 200)

    # Clean solution: faster kinetics, smaller dead zone
    # R = k * (ΔT - 0.1)^1.5 with k adjusted for typical values
    clean_rate = np.where(dt > 0.1, 4.0 * (dt - 0.1) ** 1.2, 0.0)

    # Contaminated: slower kinetics, larger dead zone
    contam_rate = np.where(dt > 0.5, 1.5 * (dt - 0.5) ** 1.0, 0.0)

    clean = ReferenceCurve(
        name="Cfe=0 (placeholder)",
        cfe_ppm=0.0,
        supercooling=dt,
        rate_mm_day=clean_rate,
    )

    contaminated = ReferenceCurve(
        name="Cfe=16ppm (placeholder)",
        cfe_ppm=16.0,
        supercooling=dt,
        rate_mm_day=contam_rate,
    )

    return clean, contaminated


class ReferenceManager:
    """
    Manages reference curves for comparison.

    Supports:
    - Loading from JSON
    - Manual entry of polynomial coefficients
    - Dynamic generation from parameters
    """

    def __init__(self):
        self._curves: dict[str, ReferenceCurve] = {}
        self._poly_coeffs: dict[str, np.ndarray] = {}

    def add_curve(self, key: str, curve: ReferenceCurve):
        self._curves[key] = curve

    def get_curve(self, key: str) -> Optional[ReferenceCurve]:
        return self._curves.get(key)

    def list_curves(self) -> list[str]:
        return list(self._curves.keys())

    def set_polynomial_coefficients(self, key: str, coeffs: np.ndarray):
        """
        Set polynomial coefficients a[j, 0..2] for j=0..7.

        coeffs shape: (8, 3) — Si(j, te) = a[j,0] + a[j,1]*te + a[j,2]*te^2
        """
        self._poly_coeffs[key] = coeffs

    def evaluate_polynomial(self, key: str, te: float) -> Optional[np.ndarray]:
        """Evaluate Si(j, te) for all j."""
        coeffs = self._poly_coeffs.get(key)
        if coeffs is None:
            return None
        return coeffs[:, 0] + coeffs[:, 1] * te + coeffs[:, 2] * te**2

    def load_defaults(self, te: float = 48.85):
        """Load placeholder reference curves."""
        clean, contam = get_placeholder_references(te)
        self._curves["Cfe=0"] = clean
        self._curves["Cfe=16ppm"] = contam


# ============================================================================
# Mathcad polynomial reference curves (from MCD screenshots 19-20)
# ============================================================================

# 4 sets of polynomial coefficients, each 8×4:
#   columns 0-2: Si[j,0] = a[j,0] + a[j,1]*te + a[j,2]*te²  → σ% at given R
#   column 3:    Si[j,1] = a[j,3]                              → R (μm/min)
#
# R values for all sets: 0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6 μm/min

# a — KDP, Cac=9.8%, Cfe=4.5ppm
_COEFF_A = np.array([
    [2.879,  -0.097,   8.996e-4,  0.0],
    [3.411,  -0.102,   8.523e-4,  0.025],
    [4.201,  -0.123,   1.035e-3,  0.05],
    [3.477,  -0.075,   4.397e-4,  0.1],
    [3.19,   -0.039,  -2.029e-5,  0.2],
    [5.351,  -0.102,   5.547e-4,  0.4],
    [10.991, -0.284,   2.219e-3,  0.8],
    [6.374,   0.032,  -1.799e-3,  1.6],
])

# a1 — KDP, Cac=9.8%, Cfe=20.5ppm
_COEFF_A1 = np.array([
    [5.69,   -0.196,   1.779e-3,  0.0],
    [11.001, -0.371,   3.27e-3,   0.025],
    [8.767,  -0.244,   1.736e-3,  0.05],
    [9.146,  -0.236,   1.569e-3,  0.1],
    [5.8,    -0.063,  -3.515e-4,  0.2],
    [4.342,   0.03,   -1.376e-3,  0.4],
    [7.774,  -0.087,  -1.15e-4,   0.8],
    [9.816,  -0.091,  -4.187e-4,  1.6],
])

# b — KDP, Cac=0, Cfe=0
_COEFF_B = np.array([
    [6.603,  -0.174,   9.679e-4,  0.0],
    [8.593,  -0.222,   1.266e-3,  0.025],
    [5.748,  -0.032,  -1.365e-3,  0.05],
    [7.389,  -0.099,  -4.851e-4,  0.1],
    [11.691, -0.32,    2.635e-3,  0.2],
    [15.274, -0.472,   4.564e-3,  0.4],
    [22.789, -0.779,   8.167e-3,  0.8],
    [37.35,  -1.374,   0.015,     1.6],
])

# b1 — KDP, Cac=0, Cfe=16ppm
_COEFF_B1 = np.array([
    [32.216, -1.291,   0.014,     0.0],
    [16.196, -0.321,   5.465e-4,  0.025],
    [11.988, -0.066,  -2.761e-3,  0.05],
    [16.538, -0.302,   6.399e-4,  0.1],
    [19.768, -0.453,   2.792e-3,  0.2],
    [24.752, -0.641,   5.073e-3,  0.4],
    [18.015, -0.206,  -5.911e-4,  0.8],
    [16.58,  -0.033,  -2.706e-3,  1.6],
])

# μm/min → mm/day conversion factor
UM_MIN_TO_MM_DAY = 1.441


def _select_coefficients(salt: int, acid: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Select coefficient sets (Si_clean, Si1_contaminated) based on salt/acid type.

    Mathcad logic:
      Acid < 1 → b (Cfe=0), b1 (Cfe=16ppm)
      Acid >= 1 → a (Cfe=4.5ppm), a1 (Cfe=20.5ppm)

    Currently only KDP coefficients available (salt=1).
    For DKDP (salt=2) falls back to KDP neutral.

    Returns
    -------
    (clean_coeffs, contaminated_coeffs) : tuple of (8, 4) arrays
    """
    if salt == 1 and acid >= 1:
        return _COEFF_A, _COEFF_A1
    else:
        # KDP neutral or DKDP (fallback)
        return _COEFF_B, _COEFF_B1


def compute_Si(te: float, salt: int = 1, acid: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute clean reference curve Si from polynomial coefficients.

    Mathcad formula:
      Si[j, 0] = a[j,0] + a[j,1]*te + a[j,2]*te²  → σ% at given growth rate
      Si[j, 1] = a[j, 3]                             → R in μm/min

    Parameters
    ----------
    te : float
        Equilibrium temperature (°C).
    salt : int
        1=KDP, 2=DKDP.
    acid : int
        0=neutral, 1=acid.

    Returns
    -------
    sigma_percent : ndarray, shape (8,)
        Supersaturation values (%).
    rate_um_min : ndarray, shape (8,)
        Growth rate values (μm/min).
    """
    coeffs, _ = _select_coefficients(salt, acid)
    sigma = coeffs[:, 0] + coeffs[:, 1] * te + coeffs[:, 2] * te**2
    rate = coeffs[:, 3]
    return sigma, rate


def compute_Si1(te: float, salt: int = 1, acid: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute contaminated reference curve Si1 from polynomial coefficients.

    Same as compute_Si but uses the contaminated coefficient set.

    Returns
    -------
    sigma_percent : ndarray, shape (8,)
    rate_um_min : ndarray, shape (8,)
    """
    _, coeffs = _select_coefficients(salt, acid)
    sigma = coeffs[:, 0] + coeffs[:, 1] * te + coeffs[:, 2] * te**2
    rate = coeffs[:, 3]
    return sigma, rate


def sigma_to_dT(sigma_percent: np.ndarray, tn: float,
                salt: int = 1, acid: int = 0) -> np.ndarray:
    """
    Convert σ% to honest dT = tn - T via inverse solubility.

    Instead of the Mathcad approximation dT ≈ σ% * tn/100,
    solves C(T) = C(tn) / exp(σ/100) for T, then dT = tn - T.

    Parameters
    ----------
    sigma_percent : ndarray
        Supersaturation in percent.
    tn : float
        Saturation temperature (°C).
    salt, acid : int
        For selecting solubility coefficients.

    Returns
    -------
    dT : ndarray
        Supercooling in °C.
    """
    from .solubility import get_solubility_set, solubility

    sol = get_solubility_set(salt, acid)
    C_sat = solubility(tn, sol)

    # C(T) = C(tn) / exp(σ/100)
    C_target = C_sat / np.exp(sigma_percent / 100.0)

    # Solve: k0 + k1*T + k2*T² = C_target for T
    if sol.k2 != 0:
        # Quadratic: k2*T² + k1*T + (k0 - C_target) = 0
        discriminant = sol.k1**2 - 4 * sol.k2 * (sol.k0 - C_target)
        discriminant = np.maximum(discriminant, 0.0)
        T = (-sol.k1 + np.sqrt(discriminant)) / (2 * sol.k2)
    else:
        # Linear: k1*T + k0 = C_target
        T = (C_target - sol.k0) / sol.k1

    return tn - T


def get_mathcad_references(te: float, tn: float,
                           salt: int = 1, acid: int = 0,
                           ) -> tuple[ReferenceCurve, ReferenceCurve]:
    """
    Get Mathcad-style reference curves with honest dT conversion.

    Parameters
    ----------
    te : float
        Equilibrium temperature (°C), used for polynomial evaluation.
    tn : float
        Saturation temperature (°C), used for σ% → dT conversion.
    salt, acid : int
        Salt and acid type.

    Returns
    -------
    (Si_clean, Si1_contaminated) : tuple of ReferenceCurve
    """
    # Compute Si (clean)
    sigma_si, rate_si_um = compute_Si(te, salt, acid)
    dT_si = sigma_to_dT(sigma_si, tn, salt, acid)
    rate_si_mm = rate_si_um * UM_MIN_TO_MM_DAY

    if acid >= 1:
        name_clean = f"Cfe=4.5ppm (Cac=9.8%, te={te:.1f})"
        cfe_clean = 4.5
    else:
        name_clean = f"Cfe=0 (Cac=0, te={te:.1f})"
        cfe_clean = 0.0

    si_curve = ReferenceCurve(
        name=name_clean,
        cfe_ppm=cfe_clean,
        supercooling=dT_si,
        rate_mm_day=rate_si_mm,
        sigma_percent=sigma_si,
        rate_um_min=rate_si_um,
    )

    # Compute Si1 (contaminated)
    sigma_si1, rate_si1_um = compute_Si1(te, salt, acid)
    dT_si1 = sigma_to_dT(sigma_si1, tn, salt, acid)
    rate_si1_mm = rate_si1_um * UM_MIN_TO_MM_DAY

    if acid >= 1:
        name_contam = f"Cfe=20.5ppm (Cac=9.8%, te={te:.1f})"
        cfe_contam = 20.5
    else:
        name_contam = f"Cfe=16ppm (Cac=0, te={te:.1f})"
        cfe_contam = 16.0

    si1_curve = ReferenceCurve(
        name=name_contam,
        cfe_ppm=cfe_contam,
        supercooling=dT_si1,
        rate_mm_day=rate_si1_mm,
        sigma_percent=sigma_si1,
        rate_um_min=rate_si1_um,
    )

    return si_curve, si1_curve
