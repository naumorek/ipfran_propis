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
