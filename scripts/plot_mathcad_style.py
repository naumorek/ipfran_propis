#!/usr/bin/env python3
"""
Plot R(dT) kinetic curves in Mathcad style (Base_grath).

Traces:
  - Black solid: LOESS F1 smoothed curve (span=0.2)
  - Green solid + diamonds: Si reference (Cfe=0 / Cfe=4.5ppm)
  - Red solid + diamonds: Si1 reference (Cfe=16ppm / Cfe=20.5ppm)

X axis: dT (C) = tn - T  (honest, from temperature data)
Y axis: R (mm/day)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from propis_app.core.prn_reader import read_prn
from propis_app.core.pipeline import CycleParams, run_classic
from propis_app.core.reference_curves import get_mathcad_references


# Output directory
OUTPUT_DIR = Path("/mnt/c/Users/Artem/Desktop/data_ipfran/mathcad_style_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Test cases: (prn_path, mcd_name, n1, n2, im, isat, im1, isat1, tn_manual)
MCD_PARAMS = [
    ("020326 KDP нейтр XXXIII п.15/__020326_2.prn",
     "__020326_2_1", 1, 8000, 5963, 6692, 5120, 7580, 48.60),
    ("020326 KDP нейтр XXXIII п.15/__020326_2.prn",
     "__020326_2_2", 13000, 20000, 4686, 5346, 3720, 6180, 48.60),
    ("020326 KDP нейтр XXXIII п.17/__020326_5.prn",
     "__020326_5_1", 1, 9000, 7227, 7576, 6350, 8090, 48.60),
    ("020326 KDP нейтр XXXIII п.17/__020326_5.prn",
     "__020326_5_2", 13000, 21000, 5890, 6187, 5210, 6610, 48.60),
    ("030326 KDP нейтр XXXIII п.18/__030326_2.prn",
     "__030326_2_2", 13500, 21000, 4596, 5082, 3500, 5690, None),
    ("030326 KDP нейтр XXXIII п.18/__030326_2.prn",
     "__030326_2_3", 25000, 32000, 4722, 5164, 3795, 5575, None),
    ("030326 KDP нейтр XXXIII п.26/__030326_5.prn",
     "__030326_5_1", 1, 10000, 7606, 8193, 6570, 8990, None),
    ("030326 KDP нейтр XXXIII п.26/__030326_5.prn",
     "__030326_5_2", 13000, 22000, 6260, 6824, 5080, 7500, None),
    ("030326 KDP нейтр XXXIII п.26/__030326_6.prn",
     "__030326_6_1", 1, 10000, 6831, 7395, 5930, 7940, None),
    ("030326 KDP нейтр XXXIII п.26/__030326_6.prn",
     "__030326_6_3", 25000, 33000, 5200, 5758, 4240, 6330, None),
    ("040326 KDP нейтр XXXIII п.24/__040326_2.prn",
     "__040326_2_2", 13000, 20000, 5405, 6045, 4660, 6480, 49.60),
    ("040326 KDP нейтр XXXIII п.24/__040326_2.prn",
     "__040326_2_3", 26000, 32000, 4121, 4606, 3220, 5160, 49.60),
    ("040326 KDP нейтр XXXIV п.28/__040326_5.prn",
     "__040326_5_2", 12880, 22000, 6497, 7038, 5515, 7530, None),
    ("040326 KDP нейтр XXXIV п.28/__040326_5.prn",
     "__040326_5_3", 25000, 33000, 5926, 6468, 5070, 6880, None),
    ("040326 KDP нейтр XXXIV п.28/__040326_6.prn",
     "__040326_6_2", 12880, 22000, 6619, 7190, 5665, 7700, None),
]


def loess_F1(x_data, y_data, x_grid, span=0.2):
    """
    LOESS smoothing + interpolation/extrapolation on arbitrary grid.

    Matches Mathcad:
      S3 := loess(VX, VY, span)
      F1(x) := interp(S3, VX, VY, x)

    Parameters
    ----------
    x_data, y_data : array
        Measured data points.
    x_grid : array
        Grid of x values where F1 is evaluated (can extend beyond data range).
    span : float
        LOESS bandwidth (fraction of data used for each local fit).

    Returns
    -------
    y_grid : array
        Smoothed/extrapolated values at x_grid points. Clamped to >= 0.
    """
    from scipy.interpolate import interp1d

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        result = lowess(y_data, x_data, frac=span, return_sorted=True)
        x_smooth = result[:, 0]
        y_smooth = result[:, 1]
    except ImportError:
        from scipy.signal import savgol_filter
        order = np.argsort(x_data)
        x_smooth = x_data[order]
        y_s = y_data[order]
        window = max(11, int(len(y_s) * span))
        if window % 2 == 0:
            window += 1
        window = min(window, len(y_s))
        if window < 5:
            y_smooth = y_s
        else:
            y_smooth = savgol_filter(y_s, window, polyorder=2)

    # Interpolate/extrapolate to the full grid
    # (Mathcad interp extrapolates linearly beyond data range)
    f = interp1d(x_smooth, y_smooth, kind='linear',
                 fill_value='extrapolate', bounds_error=False)
    y_grid = f(x_grid)
    return np.maximum(y_grid, 0.0)


def plot_mathcad_style(result, output_path):
    """
    Create a single R(dT) plot in Mathcad style.

    Matches Base_grath.jpg layout:
    - Black solid line: LOESS F1 curve
    - Green solid + diamonds: Si reference
    - Red solid + diamonds: Si1 reference
    """
    if result.dense_rate is None or len(result.dense_rate) < 10:
        print(f"  SKIP: no dense data")
        return

    te = result.te
    tn = result.tn
    salt = result.Salt
    acid = result.Acid
    Td = result.Td

    # Use dense d-step data from Mathcad zs formula (~2000+ points)
    # Raw rates have arcsin oscillations — LOESS smoothing handles them
    # Include ALL points with dT > 0 (growth zone + dead zone zeros)
    # so LOESS naturally produces F1→0 in the dead zone
    dT_dense = result.dense_supercooling
    R_dense = result.dense_rate

    mask = dT_dense > 0
    if np.sum(mask) < 10:
        print(f"  SKIP: too few dense points ({np.sum(mask)})")
        return

    dT_data = dT_dense[mask]
    R_data = R_dense[mask]

    # LOESS F1 on dense data (Mathcad: S3=loess(VX,VY,0.2); F1(x)=interp)
    # Grid from 0 to data max (no extrapolation beyond data range)
    dT_grid = np.linspace(0, dT_data.max(), 300)
    R_loess = loess_F1(dT_data, R_data, dT_grid, span=0.2)
    dT_loess = dT_grid

    # For axis limits, use only positive-rate data
    pos_mask = R_data > 0
    dT_growth = dT_data[pos_mask] if np.any(pos_mask) else dT_data
    R_growth = R_data[pos_mask] if np.any(pos_mask) else R_data

    # Reference curves Si, Si1 with honest dT
    si_curve, si1_curve = get_mathcad_references(te, tn, salt, acid)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))

    # Trace 1: LOESS F1 — black solid (main experimental curve)
    ax.plot(dT_loess, R_loess, color="black", linewidth=2.0, zorder=5,
            label="F1 (LOESS)")

    # Trace 2: Si — green solid + diamonds
    # Filter valid points (σ% > 0 → dT > 0)
    si_mask = si_curve.supercooling > 0
    if np.any(si_mask):
        ax.plot(si_curve.supercooling[si_mask], si_curve.rate_mm_day[si_mask],
                color="green", linewidth=1.5, marker="D", markersize=6,
                zorder=4, label=si_curve.name)

    # Trace 3: Si1 — red solid + diamonds
    si1_mask = si1_curve.supercooling > 0
    if np.any(si1_mask):
        ax.plot(si1_curve.supercooling[si1_mask], si1_curve.rate_mm_day[si1_mask],
                color="red", linewidth=1.5, marker="D", markersize=6,
                zorder=4, label=si1_curve.name)

    # Axis labels (Mathcad style)
    ax.set_xlabel("dT (C)", fontsize=13)
    ax.set_ylabel("R (mm/day)", fontsize=13)

    # Axis limits
    x_max_data = dT_growth.max() * 1.1
    y_max_data = R_growth.max() * 1.15
    x_max = max(
        x_max_data,
        si_curve.supercooling.max() * 1.05 if np.any(si_mask) else 4.0,
    )
    y_max = max(
        y_max_data,
        si_curve.rate_mm_day.max() * 1.1 if np.any(si_mask) else 3.0,
    )
    ax.set_xlim(0, min(x_max, 8.0))
    ax.set_ylim(0, min(y_max, 5.0))

    # Grid (like Mathcad green grid)
    ax.grid(True, color="green", alpha=0.3, linewidth=0.5)

    # Title with parameters
    short_name = result.filename.replace("__", "").replace(".prn", "")
    title = f"{short_name}  te={te:.2f}  tn={tn:.2f}  Td={result.Td:.2f}"
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Parameter box (s0, s1, Sigm, Sig035, s2)
    info_text = (
        f"s0={result.s0:.3f}\n"
        f"s1={result.s1:.3f}\n"
        f"Sigm={result.Sigm:.3f}\n"
        f"Sig035={result.Sig035:.3f}\n"
        f"s2={result.s2:.3f}\n"
        f"Td={result.Td:.3f}"
    )
    ax.text(0.03, 0.97, info_text, transform=ax.transAxes,
            fontsize=9, va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def main():
    base = project_root
    plt.rcParams["font.size"] = 11

    print(f"Output: {OUTPUT_DIR}\n")

    for prn_rel, mcd_name, n1, n2, im, isat, im1, isat1, tn in MCD_PARAMS:
        prn_path = base / prn_rel
        if not prn_path.exists():
            print(f"SKIP: {prn_path}")
            continue

        try:
            prn = read_prn(prn_path)
        except Exception as e:
            print(f"ERROR reading {prn_path}: {e}")
            continue

        params = CycleParams(n1=n1, n2=n2, im=im, isat=isat, im1=im1, isat1=isat1)
        short = mcd_name.replace("__", "")
        print(f"Processing {short}...")

        try:
            result = run_classic(
                prn, params, salt=1, acid=0, face=0,
                channel=1, tn_manual=tn,
            )
        except Exception as e:
            print(f"  Classic ERROR: {e}")
            continue

        out_file = OUTPUT_DIR / f"R_dT_{short}.png"
        plot_mathcad_style(result, out_file)

    print(f"\nDone! Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
