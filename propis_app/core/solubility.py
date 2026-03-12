"""
Solubility calculations for KDP/DKDP solutions.

Uses polynomial C(T) = k0 + k1*T + k2*T^2 with 7 coefficient sets
for different salt/acid combinations (from SolutionMaker).

Supersaturation: σ = ln(C_actual / C_eq(T))
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Default coefficients — same as in reference_coefficients.json
_DEFAULT_K = [
    [0.123,  2.719e-3, 1.1087e-5, 1.0238, 0.5359],   # 0: KDP neutral
    [0.15,   3.249e-3, 0.0,       1.075,  0.518],      # 1: KDP acid 9.83%
    [0.161,  3.116e-3, 0.0,       1.0966, 0.4962],     # 2: KDP acid 12.5%
    [0.166,  3.7e-3,   0.0,       1.137,  0.532],      # 3: DKDP neutral
    [0.2061, 3.171e-3, 0.0,       1.207,  0.429],      # 4: DKDP acid 9.83%
    [0.2145, 3.079e-3, 0.0,       1.215,  0.444],      # 5: DKDP acid 12.67%
    [0.2303, 2.938e-3, 0.0,       1.265,  0.394],      # 6: DKDP acid 18.5%
]

_ACIDITY = [0.0, 9.83, 12.5, 0.0, 9.83, 12.67, 18.5]
_SALT_TYPE = ["KDP", "KDP", "KDP", "DKDP", "DKDP", "DKDP", "DKDP"]


@dataclass
class SolubilitySet:
    """One set of solubility coefficients."""
    index: int
    salt: str        # "KDP" or "DKDP"
    acidity: float   # acid concentration in %
    k0: float
    k1: float
    k2: float
    rho: float       # solution density
    e_over_d: float  # density correction factor


def get_solubility_set(salt: int, acid: int) -> SolubilitySet:
    """
    Get solubility coefficients for given salt/acid type.

    Parameters
    ----------
    salt : int
        1 = KDP, 2 = DKDP
    acid : int
        0 = neutral, 1 = acid (9.83%), 2 = acid (12.5%/12.67%), 3 = acid (18.5%, DKDP only)

    Returns
    -------
    SolubilitySet
    """
    if salt == 1:  # KDP
        idx = min(acid, 2)  # 0, 1, 2
    elif salt == 2:  # DKDP
        idx = 3 + min(acid, 3)  # 3, 4, 5, 6
    else:
        raise ValueError(f"Unknown salt type: {salt}. Use 1=KDP, 2=DKDP")

    k = _DEFAULT_K[idx]
    return SolubilitySet(
        index=idx,
        salt=_SALT_TYPE[idx],
        acidity=_ACIDITY[idx],
        k0=k[0], k1=k[1], k2=k[2],
        rho=k[3], e_over_d=k[4],
    )


def solubility(T: np.ndarray | float, sol: SolubilitySet) -> np.ndarray | float:
    """
    Calculate equilibrium mass fraction C(T) = k0 + k1*T + k2*T^2.

    Parameters
    ----------
    T : array or float
        Temperature in °C.
    sol : SolubilitySet
        Solubility coefficients.

    Returns
    -------
    C : same type as T
        Equilibrium mass fraction of salt.
    """
    return sol.k0 + sol.k1 * T + sol.k2 * T**2


def supersaturation(T: np.ndarray | float, tn: float,
                    sol: SolubilitySet) -> np.ndarray | float:
    """
    Calculate supersaturation σ = ln(C(tn) / C(T)) as a fraction.

    Positive σ means supersaturated (crystal grows).
    Negative σ means undersaturated (crystal dissolves).

    Parameters
    ----------
    T : array or float
        Actual temperature in °C.
    tn : float
        Saturation temperature in °C.
    sol : SolubilitySet

    Returns
    -------
    sigma : same type as T
        Supersaturation (dimensionless, multiply by 100 for %).
    """
    C_sat = solubility(tn, sol)
    C_eq = solubility(T, sol)
    return np.log(C_sat / C_eq)


def supersaturation_percent(T: np.ndarray | float, tn: float,
                            sol: SolubilitySet) -> np.ndarray | float:
    """Supersaturation in percent."""
    return supersaturation(T, tn, sol) * 100.0


def supercooling(T: np.ndarray | float, tn: float) -> np.ndarray | float:
    """
    Calculate supercooling ΔT = tn - T.

    Positive ΔT means supersaturated.
    """
    return tn - T


def temperature_from_supersaturation(sigma: float, sol: SolubilitySet,
                                     tn: float) -> float:
    """
    Inverse: given σ (fraction) and tn, find T such that σ = ln(C(tn)/C(T)).

    Uses Newton's method.
    """
    C_sat = solubility(tn, sol)
    C_target = C_sat / np.exp(sigma)

    # Solve k0 + k1*T + k2*T^2 = C_target
    if sol.k2 != 0:
        # Quadratic: k2*T^2 + k1*T + (k0 - C_target) = 0
        a, b, c = sol.k2, sol.k1, sol.k0 - C_target
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            raise ValueError("No real solution for temperature")
        T = (-b + np.sqrt(discriminant)) / (2*a)
    else:
        # Linear: k1*T + k0 = C_target
        T = (C_target - sol.k0) / sol.k1

    return float(T)
