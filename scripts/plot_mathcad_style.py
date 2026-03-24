#!/usr/bin/env python3
"""
Plot R(dT) kinetic curves: Classic vs Modern side-by-side + overlay.

For each test case, generates one image with 3 subplots:
  Left:   Classic (Mathcad zs formula)
  Center: Modern (Hilbert instantaneous frequency)
  Right:  Overlay (both F1 curves on same axes)

Traces:
  - Black solid: LOESS F1 smoothed curve (span=0.2)
  - Green solid + diamonds: Si reference (Cfe=0)
  - Red solid + diamonds: Si1 reference (Cfe=16ppm)

X axis: dT (C) = tn - T
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
from propis_app.core.pipeline import CycleParams, run_classic, run_modern
from propis_app.core.reference_curves import get_mathcad_references
from propis_app.core.kinetics.power_law import power_law_model


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
    """LOESS smoothing + interpolation on arbitrary grid."""
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

    f = interp1d(x_smooth, y_smooth, kind='linear',
                 fill_value='extrapolate', bounds_error=False)
    y_grid = f(x_grid)
    return y_grid


def get_f1_curve(result, span=0.2):
    """Extract LOESS F1 curve from a pipeline result. Returns (dT_grid, R_loess) or None."""
    if result is None or result.dense_rate is None or len(result.dense_rate) < 10:
        return None

    dT_dense = result.dense_supercooling
    R_dense = result.dense_rate

    # Включаем ВСЕ данные: рост (R>0) + растворение (R<0) + dead zone (R≈0)
    # Фильтр только по конечности значений
    mask = np.isfinite(dT_dense) & np.isfinite(R_dense)
    if np.sum(mask) < 10:
        return None

    dT_data = dT_dense[mask]
    R_data = R_dense[mask]

    dT_min = dT_data.min()
    dT_max = dT_data.max()
    dT_grid = np.linspace(dT_min, dT_max, 300)
    R_loess = loess_F1(dT_data, R_data, dT_grid, span=span)
    return dT_grid, R_loess


def plot_single_panel(ax, result, mode_label, si_curve, si1_curve, show_refs=True):
    """Plot one R(dT) panel on given axes."""
    f1 = get_f1_curve(result)
    if f1 is None:
        ax.text(0.5, 0.5, "NO DATA", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="red")
        ax.set_title(mode_label, fontsize=10)
        return

    dT_loess, R_loess = f1

    # F1 curve
    ax.plot(dT_loess, R_loess, color="black", linewidth=2.0, zorder=5,
            label="F1 (LOESS)")

    # References
    if show_refs:
        si_mask = si_curve.supercooling > 0
        if np.any(si_mask):
            ax.plot(si_curve.supercooling[si_mask], si_curve.rate_mm_day[si_mask],
                    color="green", linewidth=1.5, marker="D", markersize=5,
                    zorder=4, label=si_curve.name)

        si1_mask = si1_curve.supercooling > 0
        if np.any(si1_mask):
            ax.plot(si1_curve.supercooling[si1_mask], si1_curve.rate_mm_day[si1_mask],
                    color="red", linewidth=1.5, marker="D", markersize=5,
                    zorder=4, label=si1_curve.name)

    ax.set_xlabel("dT (C)", fontsize=10)
    ax.set_ylabel("R (mm/day)", fontsize=10)
    ax.grid(True, color="green", alpha=0.3, linewidth=0.5)

    # Parameter box
    info = (f"s0={result.s0:.3f}\ns1={result.s1:.3f}\n"
            f"Sigm={result.Sigm:.3f}\nSig035={result.Sig035:.3f}\n"
            f"s2={result.s2:.3f}\nTd={result.Td:.3f}")
    if hasattr(result, 's0_d') and result.s0_d > 0:
        info += f"\ns0_d={result.s0_d:.3f}\ns1_d={result.s1_d:.3f}"
    ax.text(0.03, 0.97, info, transform=ax.transAxes, fontsize=7,
            va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    ax.set_title(mode_label, fontsize=10, fontweight="bold")


def get_f1_curve_from_arrays(dT_dense, R_dense, Td=0.0, span=0.2):
    """Build LOESS F1 curve from raw dT/R arrays (for method comparison)."""
    if dT_dense is None or R_dense is None or len(R_dense) < 10:
        return None

    mask = (dT_dense > 0) & (R_dense > 0.01)
    if np.sum(mask) < 10:
        return None

    dT_data = dT_dense[mask]
    R_data = R_dense[mask]

    # Dead zone anchors
    if Td > 0:
        n_anchors = 10
        dT_anchors = np.linspace(0, Td, n_anchors)
        R_anchors = np.zeros(n_anchors)
        dT_data = np.concatenate([dT_anchors, dT_data])
        R_data = np.concatenate([R_anchors, R_data])

    dT_grid = np.linspace(0, dT_data.max(), 300)
    R_loess = loess_F1(dT_data, R_data, dT_grid, span=span)
    R_loess[dT_grid <= Td] = 0.0
    return dT_grid, R_loess


def plot_methods_comparison(result_modern, si_curve, si1_curve, output_path):
    """3-panel comparison of Modern rate extraction methods:
    Fixed SG | Adaptive SG | Sliding Regression.
    + 4th panel: overlay of all 3 + R² from regression.
    """
    if result_modern is None or result_modern.dense_rate is None:
        return

    dT = result_modern.dense_supercooling
    Td = result_modern.Td if hasattr(result_modern, 'Td') else 0.0

    methods = [
        ("Fixed SG (baseline)", result_modern.dense_rate, "black"),
        ("Adaptive SG", result_modern.dense_rate_asg, "blue"),
        ("Sliding Regression", result_modern.dense_rate_reg, "darkred"),
    ]

    # Common axis limits
    all_dT_max = 0
    all_R_max = 0
    for label, rate_arr, _ in methods:
        if rate_arr is not None:
            f1 = get_f1_curve_from_arrays(dT, rate_arr, Td)
            if f1 is not None:
                all_dT_max = max(all_dT_max, f1[0].max())
                all_R_max = max(all_R_max, f1[1].max())

    si_mask = si_curve.supercooling > 0
    if np.any(si_mask):
        all_dT_max = max(all_dT_max, si_curve.supercooling[si_mask].max())
        all_R_max = max(all_R_max, si_curve.rate_mm_day[si_mask].max())

    x_lim = min(all_dT_max * 1.1, 8.0) if all_dT_max > 0 else 8.0
    y_lim = min(all_R_max * 1.15, 5.0) if all_R_max > 0 else 5.0

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Panels 1-3: individual methods
    for idx, (label, rate_arr, color) in enumerate(methods):
        ax = axes[idx]
        if rate_arr is not None:
            f1 = get_f1_curve_from_arrays(dT, rate_arr, Td)
            if f1 is not None:
                ax.plot(f1[0], f1[1], color=color, linewidth=2.0,
                        zorder=5, label="F1 (LOESS)")
            else:
                ax.text(0.5, 0.5, "NO DATA", transform=ax.transAxes,
                        ha="center", va="center", fontsize=14, color="red")
        else:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, color="gray")

        # References
        if np.any(si_mask):
            ax.plot(si_curve.supercooling[si_mask], si_curve.rate_mm_day[si_mask],
                    color="green", linewidth=1.0, marker="D", markersize=4,
                    alpha=0.6, zorder=3, label=si_curve.name)
        si1_mask = si1_curve.supercooling > 0
        if np.any(si1_mask):
            ax.plot(si1_curve.supercooling[si1_mask], si1_curve.rate_mm_day[si1_mask],
                    color="red", linewidth=1.0, marker="D", markersize=4,
                    alpha=0.6, zorder=3, label=si1_curve.name)

        ax.set_xlabel("dT (C)", fontsize=10)
        ax.set_ylabel("R (mm/day)", fontsize=10)
        ax.grid(True, color="green", alpha=0.3, linewidth=0.5)
        ax.set_xlim(0, x_lim)
        ax.set_ylim(0, y_lim)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9)

    # Panel 4: Overlay all 3 methods
    ax = axes[3]
    for label, rate_arr, color in methods:
        if rate_arr is not None:
            f1 = get_f1_curve_from_arrays(dT, rate_arr, Td)
            if f1 is not None:
                ls = "--" if label != "Fixed SG (baseline)" else "-"
                ax.plot(f1[0], f1[1], color=color, linewidth=2.0,
                        linestyle=ls, label=label, zorder=5)

    # R² на вторичной оси (для regression)
    if result_modern.dense_rate_reg_r2 is not None:
        ax2 = ax.twinx()
        # R² vs dT: усредняем R² по бинам dT
        r2_arr = result_modern.dense_rate_reg_r2
        valid = dT > 0
        if np.any(valid):
            ax2.scatter(dT[valid], r2_arr[valid], s=1, alpha=0.15,
                       color="orange", zorder=1)
            ax2.set_ylabel("R² (regression)", fontsize=8, color="orange")
            ax2.set_ylim(0, 1.05)
            ax2.tick_params(axis='y', labelcolor='orange', labelsize=7)

    if np.any(si_mask):
        ax.plot(si_curve.supercooling[si_mask], si_curve.rate_mm_day[si_mask],
                color="green", linewidth=1.0, marker="D", markersize=4,
                alpha=0.4, zorder=3, label=si_curve.name)

    ax.set_xlabel("dT (C)", fontsize=10)
    ax.set_ylabel("R (mm/day)", fontsize=10)
    ax.grid(True, color="green", alpha=0.3, linewidth=0.5)
    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, y_lim)
    ax.set_title("Overlay + R²", fontsize=10, fontweight="bold")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)

    short_name = result_modern.filename.replace("__", "").replace(".prn", "")
    fig.suptitle(f"Methods comparison: {short_name}  (Modern pipeline)",
                 fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_comparison(result_classic, result_modern, output_path):
    """Create 3-panel comparison: Classic | Modern | Overlay."""
    te = result_classic.te
    tn = result_classic.tn
    salt = result_classic.Salt
    acid = result_classic.Acid

    si_curve, si1_curve = get_mathcad_references(te, tn, salt, acid)

    # Determine common axis limits
    all_dT_max = 0
    all_R_max = 0
    for res in [result_classic, result_modern]:
        f1 = get_f1_curve(res)
        if f1 is not None:
            all_dT_max = max(all_dT_max, f1[0].max())
            all_R_max = max(all_R_max, f1[1].max())

    si_mask = si_curve.supercooling > 0
    if np.any(si_mask):
        all_dT_max = max(all_dT_max, si_curve.supercooling[si_mask].max())
        all_R_max = max(all_R_max, si_curve.rate_mm_day[si_mask].max())

    x_lim = min(all_dT_max * 1.1, 8.0)
    y_lim = min(all_R_max * 1.15, 5.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Classic
    plot_single_panel(axes[0], result_classic, "Classic (zs formula)", si_curve, si1_curve)
    axes[0].set_xlim(0, x_lim)
    axes[0].set_ylim(0, y_lim)
    axes[0].legend(loc="lower right", fontsize=7, framealpha=0.9)

    # Panel 2: Modern
    plot_single_panel(axes[1], result_modern, "Modern (Hilbert)", si_curve, si1_curve)
    axes[1].set_xlim(0, x_lim)
    axes[1].set_ylim(0, y_lim)
    axes[1].legend(loc="lower right", fontsize=7, framealpha=0.9)

    # Panel 3: Overlay
    f1_c = get_f1_curve(result_classic)
    f1_m = get_f1_curve(result_modern)

    if f1_c is not None:
        axes[2].plot(f1_c[0], f1_c[1], color="black", linewidth=2.0,
                     label="Classic", zorder=5)
    if f1_m is not None:
        axes[2].plot(f1_m[0], f1_m[1], color="blue", linewidth=2.0,
                     linestyle="--", label="Modern", zorder=5)

    # References on overlay
    if np.any(si_mask):
        axes[2].plot(si_curve.supercooling[si_mask], si_curve.rate_mm_day[si_mask],
                     color="green", linewidth=1.0, marker="D", markersize=4,
                     alpha=0.6, zorder=3, label=si_curve.name)
    si1_mask = si1_curve.supercooling > 0
    if np.any(si1_mask):
        axes[2].plot(si1_curve.supercooling[si1_mask], si1_curve.rate_mm_day[si1_mask],
                     color="red", linewidth=1.0, marker="D", markersize=4,
                     alpha=0.6, zorder=3, label=si1_curve.name)

    axes[2].set_xlabel("dT (C)", fontsize=10)
    axes[2].set_ylabel("R (mm/day)", fontsize=10)
    axes[2].grid(True, color="green", alpha=0.3, linewidth=0.5)
    axes[2].set_xlim(0, x_lim)
    axes[2].set_ylim(0, y_lim)
    axes[2].set_title("Overlay", fontsize=10, fontweight="bold")
    axes[2].legend(loc="lower right", fontsize=7, framealpha=0.9)

    short_name = result_classic.filename.replace("__", "").replace(".prn", "")
    fig.suptitle(f"{short_name}   te={te:.2f}  tn={tn:.2f}  Td={result_classic.Td:.2f}",
                 fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_simple_overlay(result_classic, result_modern, ax, title=""):
    """Один простой график: Classic vs Modern vs эталоны на одних осях."""
    te = result_classic.te
    tn = result_classic.tn
    si_curve, si1_curve = get_mathcad_references(te, tn, result_classic.Salt, result_classic.Acid)

    f1_c = get_f1_curve(result_classic)
    f1_m = get_f1_curve(result_modern)

    all_dT_max = 0
    all_R_max = 0
    if f1_c is not None:
        all_dT_max = max(all_dT_max, f1_c[0].max())
        all_R_max = max(all_R_max, f1_c[1].max())
    if f1_m is not None:
        all_dT_max = max(all_dT_max, f1_m[0].max())
        all_R_max = max(all_R_max, f1_m[1].max())

    si_mask = si_curve.supercooling > 0
    if np.any(si_mask):
        all_dT_max = max(all_dT_max, si_curve.supercooling[si_mask].max())
        all_R_max = max(all_R_max, si_curve.rate_mm_day[si_mask].max())

    # Эталоны
    if np.any(si_mask):
        ax.plot(si_curve.supercooling[si_mask], si_curve.rate_mm_day[si_mask],
                color="green", linewidth=1.5, marker="D", markersize=5,
                alpha=0.7, zorder=3, label="Si (Cfe=0)")
    si1_mask = si1_curve.supercooling > 0
    if np.any(si1_mask):
        ax.plot(si1_curve.supercooling[si1_mask], si1_curve.rate_mm_day[si1_mask],
                color="red", linewidth=1.5, marker="D", markersize=5,
                alpha=0.7, zorder=3, label="Si1 (Cfe=16ppm)")

    # Classic
    if f1_c is not None:
        ax.plot(f1_c[0], f1_c[1], color="black", linewidth=2.5,
                zorder=5, label="Classic")

    # Modern (Hilbert dual-ch)
    if f1_m is not None:
        ax.plot(f1_m[0], f1_m[1], color="blue", linewidth=2.0,
                linestyle="--", zorder=5, label="Hilbert")

    # PLL
    Td_val = result_modern.Td if hasattr(result_modern, 'Td') else 0.0
    if result_modern.dense_rate_pll is not None:
        dT_m = result_modern.dense_supercooling
        f1_pll = get_f1_curve_from_arrays(dT_m, result_modern.dense_rate_pll, Td_val)
        if f1_pll is not None:
            ax.plot(f1_pll[0], f1_pll[1], color="purple", linewidth=2.0,
                    linestyle="-.", zorder=5, label="PLL")

    # CWT Ridge
    if result_modern.dense_rate_cwt is not None:
        f1_cwt = get_f1_curve_from_arrays(dT_m, result_modern.dense_rate_cwt, Td_val)
        if f1_cwt is not None:
            ax.plot(f1_cwt[0], f1_cwt[1], color="orange", linewidth=2.0,
                    linestyle=":", zorder=5, label="CWT")

    # STFT
    if hasattr(result_modern, 'dense_rate_stft') and result_modern.dense_rate_stft is not None:
        f1_stft = get_f1_curve_from_arrays(dT_m, result_modern.dense_rate_stft, Td_val)
        if f1_stft is not None:
            ax.plot(f1_stft[0], f1_stft[1], color="cyan", linewidth=2.5,
                    linestyle="-", zorder=6, label="STFT")

    x_lim = min(all_dT_max * 1.1, 8.0) if all_dT_max > 0 else 5.0
    y_lim = min(all_R_max * 1.15, 5.0) if all_R_max > 0 else 3.0
    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, y_lim)
    ax.set_xlabel("dT (°C)")
    ax.set_ylabel("R (mm/day)")
    ax.grid(True, color="gray", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title(title, fontsize=11, fontweight="bold")


def plot_pair(result_classic, result_modern, output_path, short_name):
    """Два графика рядом: Classic | Modern. Чистое сравнение."""
    te = result_classic.te
    tn = result_classic.tn
    si_curve, si1_curve = get_mathcad_references(te, tn, result_classic.Salt, result_classic.Acid)

    f1_c = get_f1_curve(result_classic)
    f1_m = get_f1_curve(result_modern)

    # Общие оси (включая отрицательные dT и R)
    all_dT_max, all_R_max = 0, 0
    all_dT_min, all_R_min = 0, 0
    for f1 in [f1_c, f1_m]:
        if f1 is not None:
            all_dT_max = max(all_dT_max, f1[0].max())
            all_dT_min = min(all_dT_min, f1[0].min())
            all_R_max = max(all_R_max, f1[1].max())
            all_R_min = min(all_R_min, f1[1].min())
    si_mask = si_curve.supercooling > 0
    if np.any(si_mask):
        all_dT_max = max(all_dT_max, si_curve.supercooling[si_mask].max())
        all_R_max = max(all_R_max, si_curve.rate_mm_day[si_mask].max())
    x_lim = min(all_dT_max * 1.1, 8.0) if all_dT_max > 0 else 5.0
    y_lim = min(all_R_max * 1.15, 5.0) if all_R_max > 0 else 3.0
    dT_min = all_dT_min * 1.1 if all_dT_min < 0 else -0.5
    y_min = min(all_R_min * 1.15, -1.0) if all_R_min < 0 else -0.3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, f1, res, title, color in [(ax1, f1_c, result_classic, "Classic", "black"),
                                       (ax2, f1_m, result_modern, "Modern (Hilbert)", "blue")]:
        # Эталоны
        if np.any(si_mask):
            ax.plot(si_curve.supercooling[si_mask], si_curve.rate_mm_day[si_mask],
                    color="green", linewidth=1.5, marker="D", markersize=5,
                    alpha=0.7, zorder=3, label="Si (Cfe=0)")
        si1_mask = si1_curve.supercooling > 0
        if np.any(si1_mask):
            ax.plot(si1_curve.supercooling[si1_mask], si1_curve.rate_mm_day[si1_mask],
                    color="red", linewidth=1.5, marker="D", markersize=5,
                    alpha=0.7, zorder=3, label="Si1 (Cfe=16ppm)")
        # LOESS кривая
        if f1 is not None:
            ax.plot(f1[0], f1[1], color=color, linewidth=2.5, zorder=5, label="R(dT)")

        # Фитированная кривая роста: R = s0 * (sigma - s1)^w
        if res.s0 > 0 and res.fit_result is not None:
            from propis_app.core.solubility import supersaturation_percent, supercooling as sc_func
            sol = __import__('propis_app.core.solubility', fromlist=['get_solubility_set']).get_solubility_set(res.Salt, res.Acid)
            dT_fit = np.linspace(0, x_lim, 300)
            T_fit = res.tn - dT_fit
            sigma_fit = supersaturation_percent(T_fit, res.tn, sol)
            R_fit = power_law_model(sigma_fit, res.s0, res.s1, 1.0)
            ax.plot(dT_fit, R_fit, color="magenta", linewidth=1.5, linestyle="--",
                    zorder=6, label=f"fit s0={res.s0:.2f}")

        # Фитированная кривая растворения: R = -s0_d * (|sigma| - s1_d)^w
        if hasattr(res, 's0_d') and res.s0_d > 0:
            from propis_app.core.solubility import supersaturation_percent
            sol = __import__('propis_app.core.solubility', fromlist=['get_solubility_set']).get_solubility_set(res.Salt, res.Acid)
            dT_diss = np.linspace(dT_min, 0, 200)
            T_diss = res.tn - dT_diss
            sigma_diss = supersaturation_percent(T_diss, res.tn, sol)
            # sigma_diss < 0 in dissolution region
            R_diss = -power_law_model(np.abs(sigma_diss), res.s0_d, res.s1_d, 1.0)
            ax.plot(dT_diss, R_diss, color="orange", linewidth=1.5, linestyle="--",
                    zorder=6, label=f"diss s0_d={res.s0_d:.2f}")

        # Info box
        info = (f"s0={res.s0:.3f}  s1={res.s1:.3f}\n"
                f"Td={res.Td:.2f}  Sigm={res.Sigm:.3f}")
        if hasattr(res, 's0_d') and res.s0_d > 0:
            info += f"\ns0_d={res.s0_d:.3f}  s1_d={res.s1_d:.3f}"
        ax.text(0.03, 0.97, info, transform=ax.transAxes, fontsize=7,
                va="top", ha="left", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

        ax.set_xlim(dT_min, x_lim)
        ax.set_ylim(y_min, y_lim)
        ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="-")
        ax.axvline(x=0, color="gray", linewidth=0.8, linestyle="-")
        ax.set_xlabel("dT (°C)", fontsize=12)
        ax.grid(True, color="gray", alpha=0.2)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(title, fontsize=13, fontweight="bold")

    ax1.set_ylabel("R (mm/day)", fontsize=12)

    fig.suptitle(f"{short_name}   te={te:.2f}  tn={tn:.2f}  Td={result_classic.Td:.2f}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def main():
    base = project_root
    plt.rcParams["font.size"] = 11

    # 3 представительных случая: типичный, проблемный, другой файл
    SELECTED = [
        MCD_PARAMS[0],   # 020326_2_1 — типичный
        MCD_PARAMS[9],   # 030326_6_3 — ранее проблемный
        MCD_PARAMS[10],  # 040326_2_2 — другой день
    ]

    print(f"Output: {OUTPUT_DIR}\n")

    for prn_rel, mcd_name, n1, n2, im, isat, im1, isat1, tn in SELECTED:
        prn_path = base / prn_rel
        if not prn_path.exists():
            print(f"SKIP: {prn_path}")
            continue

        prn = read_prn(prn_path)
        params = CycleParams(n1=n1, n2=n2, im=im, isat=isat, im1=im1, isat1=isat1)
        short = mcd_name.replace("__", "")
        print(f"Processing {short}...")

        result_classic = run_classic(prn, params, salt=1, acid=0, face=0,
                                      channel=1, tn_manual=tn)
        result_modern = run_modern(prn, params, salt=1, acid=0, face=0,
                                    channel=1, tn_manual=tn)

        out_file = OUTPUT_DIR / f"{short}.png"
        plot_pair(result_classic, result_modern, out_file, short)

    print(f"\nDone! Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
