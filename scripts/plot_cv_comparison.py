#!/usr/bin/env python3
"""
3 comparison images for CV model hardening (Tasks 4, 5, 2).

Image 1: Mathcad exact + CV free | Mathcad exact + CV(fixed σ_dead)
Image 2: Mathcad exact + CV free | Mathcad exact + CV(weighted + fixed)
Image 3: Mathcad exact + CV free | Mathcad exact + CV(weighted + fixed + T-dep)

Baseline 1 = mathcad_exact.py run_mathcad() ONLY.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.mathcad_exact import run_mathcad
from propis_app.core.kinetics.bcf_model import (
    bcf_model, bcf_model_T, fit_bcf, fit_bcf_fixed_sigma_d,
    _bcf_T_wrapper,
)
from scipy.optimize import curve_fit

PRN = str(project_root / "020326 KDP нейтр XXXIII п.15" / "__020326_2.prn")
SIGMA_D_DFDT = 0.30
OUTPUT_DIR = project_root


def get_data():
    """Run mathcad_exact and extract everything needed."""
    res = run_mathcad(PRN,
                      n1=10, n2=8000, im=5963, isat=6692,
                      im1=5120, isat1=7580,
                      Salt=1, Acid=0, Face=0, d=3.3, ww=1,
                      k1=0.15, span1=0.2, l=0.01,
                      tn_manual=48.60, method='classic')
    return res


def mathcad_f1(res, sigma_grid):
    """Baseline 1: F1 from mathcad_exact (мкм/мин → мм/день)."""
    f1 = res['F1'](sigma_grid) * 1.441
    return np.maximum(f1, 0)


def plot_refs(ax, res):
    """Plot Si/Si1 reference curves."""
    Si, Si1 = res['Si'], res['Si1']
    m0 = Si[:, 0] > 0
    if np.any(m0):
        ax.plot(Si[m0, 0], Si[m0, 1] * 1.441,
                color='green', lw=1.5, marker='D', ms=4, alpha=0.5,
                zorder=3, label='Si (Cfe=0)')
    m1 = Si1[:, 0] > 0
    if np.any(m1):
        ax.plot(Si1[m1, 0], Si1[m1, 1] * 1.441,
                color='red', lw=1.5, marker='D', ms=4, alpha=0.5,
                zorder=3, label='Si1 (Cfe=16)')


def setup_ax(ax, title, sigma_max, rate_max):
    ax.set_xlabel('σ (%)', fontsize=11)
    ax.set_ylabel('R (мм/день)', fontsize=11)
    ax.set_xlim(0, sigma_max)
    ax.set_ylim(0, rate_max)
    ax.grid(True, color='gray', alpha=0.2)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(loc='upper left', fontsize=7, framealpha=0.9)


def main():
    res = get_data()

    VX = res['VX']  # σ% sorted
    VY = res['VY'] * 1.441  # R мм/день
    mask = (VX > 0) & (VY > 0)
    sigma_g = VX[mask]
    rate_g = VY[mask]

    sigma_grid = np.linspace(0, sigma_g.max(), 300)
    f1 = mathcad_f1(res, sigma_grid)
    sigma_max = min(sigma_g.max() * 1.05, 6)
    rate_max = min(f1.max() * 1.15, 4)

    # All CV fits
    bcf_free = fit_bcf(sigma_g, rate_g, auto_weight=False)
    bcf_fixed = fit_bcf_fixed_sigma_d(sigma_g, rate_g, SIGMA_D_DFDT, auto_weight=False)
    bcf_w = fit_bcf(sigma_g, rate_g, auto_weight=True)
    bcf_fixed_w = fit_bcf_fixed_sigma_d(sigma_g, rate_g, SIGMA_D_DFDT, auto_weight=True)

    # T-dependent (force fit for visualization)
    z = res['z']
    T_celsius = z[:, 2]  # temperature
    sigma_z = z[:, 3]    # σ%
    rate_z = z[:, 1] * 1.441  # R мм/день
    mz = (sigma_z > 0) & (rate_z > 0)
    T_K = T_celsius[mz] + 273.15
    T_mean_K = np.mean(T_K)

    bcf_t_forced = None
    sigma_0_init = max(bcf_w.sigma_d / np.exp(4210 / T_mean_K), 1e-6) if bcf_w.sigma_d > 0 else 1e-4
    try:
        xdata = np.vstack([sigma_z[mz], T_K])
        w = 1.0 / (1.0 + rate_z[mz]**2)
        popt, _ = curve_fit(
            _bcf_T_wrapper, xdata, rate_z[mz],
            p0=[bcf_w.beta, sigma_0_init, 4210, bcf_w.sigma_1],
            sigma=1.0/np.sqrt(w), absolute_sigma=False,
            bounds=([0, 1e-10, 0, 0.01], [np.inf, 10, 15000, 100]),
            maxfev=20000)
        bcf_t_forced = {
            'beta': popt[0], 'sigma_0': popt[1],
            'E_half': popt[2], 'sigma_1': popt[3],
            'DH': 2 * 8.314 * popt[2] / 1000,
            'sigma_d_mean': popt[1] * np.exp(popt[2] / T_mean_K),
        }
    except Exception:
        pass

    # ================================================================
    # IMAGE 1: Mathcad + CV free | Mathcad + CV(fixed σ_dead)
    # ================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Left
    ax1.plot(sigma_grid, f1, color='black', lw=2.5, zorder=10, label='Mathcad exact')
    r_cv_free = bcf_model(sigma_grid, bcf_free.beta, bcf_free.sigma_d, bcf_free.sigma_1)
    ax1.plot(sigma_grid, r_cv_free, color='blue', lw=2, ls='--', zorder=9,
             label=f'CV free (σ_d={bcf_free.sigma_d:.2f}%)')
    plot_refs(ax1, res)
    setup_ax(ax1, 'Mathcad exact + CV free fit', sigma_max, rate_max)
    ax1.text(0.97, 0.03, f'corr(β,σ₁)={bcf_free.correlation[0,2]:.3f}' if bcf_free.correlation is not None else '',
             transform=ax1.transAxes, fontsize=7, va='bottom', ha='right', color='blue')

    # Right
    ax2.plot(sigma_grid, f1, color='black', lw=2.5, zorder=10, label='Mathcad exact')
    r_cv_fixed = bcf_model(sigma_grid, bcf_fixed.beta, SIGMA_D_DFDT, bcf_fixed.sigma_1)
    ax2.plot(sigma_grid, r_cv_fixed, color='red', lw=2, ls='--', zorder=9,
             label=f'CV fixed σ_d={SIGMA_D_DFDT}% (df/dt)')
    plot_refs(ax2, res)
    setup_ax(ax2, 'Mathcad exact + CV fixed σ_dead (Task 4)', sigma_max, rate_max)
    if bcf_fixed.se is not None:
        ax2.text(0.97, 0.03, f'β={bcf_fixed.beta:.3f}±{bcf_fixed.se[0]:.3f}  σ₁={bcf_fixed.sigma_1:.3f}',
                 transform=ax2.transAxes, fontsize=7, va='bottom', ha='right', color='red')

    fig.suptitle('020326_2_1 — Task 4: фиксация σ_dead из df/dt', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'cv_task4_comparison.jpg', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved: cv_task4_comparison.jpg")

    # ================================================================
    # IMAGE 2: Mathcad + CV free | Mathcad + CV(weighted + fixed)
    # ================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Left (same as image 1 left)
    ax1.plot(sigma_grid, f1, color='black', lw=2.5, zorder=10, label='Mathcad exact')
    ax1.plot(sigma_grid, r_cv_free, color='blue', lw=2, ls='--', zorder=9,
             label=f'CV free (σ_d={bcf_free.sigma_d:.2f}%)')
    plot_refs(ax1, res)
    setup_ax(ax1, 'Mathcad exact + CV free fit', sigma_max, rate_max)

    # Right: weighted + fixed
    ax2.plot(sigma_grid, f1, color='black', lw=2.5, zorder=10, label='Mathcad exact')
    r_cv_fw = bcf_model(sigma_grid, bcf_fixed_w.beta, SIGMA_D_DFDT, bcf_fixed_w.sigma_1)
    ax2.plot(sigma_grid, r_cv_fw, color='red', lw=2, ls='--', zorder=9,
             label=f'CV wt+fixed σ_d={SIGMA_D_DFDT}%')
    r_cv_w = bcf_model(sigma_grid, bcf_w.beta, bcf_w.sigma_d, bcf_w.sigma_1)
    ax2.plot(sigma_grid, r_cv_w, color='darkorange', lw=1.5, ls=':', zorder=8,
             label=f'CV weighted (σ_d={bcf_w.sigma_d:.2f}%)')
    plot_refs(ax2, res)
    setup_ax(ax2, 'Mathcad exact + CV weighted+fixed (Tasks 4+5)', sigma_max, rate_max)
    ax2.text(0.97, 0.03, f'w=1/(1+R²), Breusch-Pagan p≈0',
             transform=ax2.transAxes, fontsize=7, va='bottom', ha='right', color='gray')

    fig.suptitle('020326_2_1 — Task 5: взвешенный фит + фиксация σ_dead', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'cv_task45_comparison.jpg', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved: cv_task45_comparison.jpg")

    # ================================================================
    # IMAGE 3: Mathcad + CV free | Mathcad + CV full pipeline
    # ================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Left (same)
    ax1.plot(sigma_grid, f1, color='black', lw=2.5, zorder=10, label='Mathcad exact')
    ax1.plot(sigma_grid, r_cv_free, color='blue', lw=2, ls='--', zorder=9,
             label=f'CV free (σ_d={bcf_free.sigma_d:.2f}%)')
    plot_refs(ax1, res)
    setup_ax(ax1, 'Mathcad exact + CV free fit', sigma_max, rate_max)

    # Right: full pipeline
    ax2.plot(sigma_grid, f1, color='black', lw=2.5, zorder=10, label='Mathcad exact')
    ax2.plot(sigma_grid, r_cv_fw, color='red', lw=2, ls='--', zorder=9,
             label=f'CV wt+fixed σ_d={SIGMA_D_DFDT}%')

    if bcf_t_forced is not None:
        T_grid_K = np.full_like(sigma_grid, T_mean_K)
        r_cv_t = bcf_model_T(sigma_grid, T_grid_K,
                              bcf_t_forced['beta'], bcf_t_forced['sigma_0'],
                              bcf_t_forced['E_half'], bcf_t_forced['sigma_1'])
        ax2.plot(sigma_grid, r_cv_t, color='purple', lw=2, ls=':', zorder=8,
                 label=f'CV T-dep (ΔH={bcf_t_forced["DH"]:.0f} kJ/mol)')

    plot_refs(ax2, res)
    setup_ax(ax2, 'Mathcad exact + CV полный (Tasks 4+5+2)', sigma_max, rate_max)

    info_lines = [f'wt+fixed: β={bcf_fixed_w.beta:.3f} σ₁={bcf_fixed_w.sigma_1:.3f}']
    if bcf_t_forced:
        info_lines.append(f'T-dep: ΔH={bcf_t_forced["DH"]:.1f} kJ/mol (лит: 63-75)')
    ax2.text(0.97, 0.03, '\n'.join(info_lines),
             transform=ax2.transAxes, fontsize=7, va='bottom', ha='right', color='gray')

    fig.suptitle('020326_2_1 — Tasks 4+5+2: полное сравнение', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'cv_task452_comparison.jpg', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved: cv_task452_comparison.jpg")

    # ================================================================
    # IMAGE 4: ZOOM dead zone (σ = 0–2%, R = 0–0.5)
    # Left: full range | Right: zoom
    # ================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: full range — все кривые
    ax1.plot(sigma_grid, f1, color='black', lw=2.5, zorder=10, label='Mathcad exact')
    ax1.plot(sigma_grid, r_cv_free, color='blue', lw=2, ls='--', zorder=8,
             label=f'CV free (σ_d={bcf_free.sigma_d:.2f}%)')
    ax1.plot(sigma_grid, r_cv_fw, color='red', lw=2, ls='--', zorder=9,
             label=f'CV wt+fixed σ_d={SIGMA_D_DFDT}%')
    if bcf_t_forced is not None:
        ax1.plot(sigma_grid, r_cv_t, color='purple', lw=2, ls=':', zorder=8,
                 label=f'CV T-dep (ΔH={bcf_t_forced["DH"]:.0f} kJ/mol)')
    plot_refs(ax1, res)
    # Zoom region rectangle
    from matplotlib.patches import Rectangle
    rect = Rectangle((0, 0), 2, 0.5, linewidth=1.5, edgecolor='gray',
                      facecolor='yellow', alpha=0.15, zorder=1)
    ax1.add_patch(rect)
    setup_ax(ax1, 'Полный диапазон', sigma_max, rate_max)

    # Right: ZOOM σ = 0–2%, R = 0–0.5
    sigma_zoom = np.linspace(0, 2, 300)
    f1_zoom = mathcad_f1(res, sigma_zoom)
    r_free_zoom = bcf_model(sigma_zoom, bcf_free.beta, bcf_free.sigma_d, bcf_free.sigma_1)
    r_fw_zoom = bcf_model(sigma_zoom, bcf_fixed_w.beta, SIGMA_D_DFDT, bcf_fixed_w.sigma_1)

    ax2.plot(sigma_zoom, f1_zoom, color='black', lw=2.5, zorder=10, label='Mathcad exact')
    ax2.plot(sigma_zoom, r_free_zoom, color='blue', lw=2, ls='--', zorder=8,
             label=f'CV free (σ_d={bcf_free.sigma_d:.2f}%)')
    ax2.plot(sigma_zoom, r_fw_zoom, color='red', lw=2, ls='--', zorder=9,
             label=f'CV wt+fixed σ_d={SIGMA_D_DFDT}%')

    if bcf_t_forced is not None:
        T_zoom_K = np.full_like(sigma_zoom, T_mean_K)
        r_t_zoom = bcf_model_T(sigma_zoom, T_zoom_K,
                                bcf_t_forced['beta'], bcf_t_forced['sigma_0'],
                                bcf_t_forced['E_half'], bcf_t_forced['sigma_1'])
        ax2.plot(sigma_zoom, r_t_zoom, color='purple', lw=2, ls=':', zorder=8,
                 label=f'CV T-dep')

    # σ_dead markers
    ax2.axvline(x=SIGMA_D_DFDT, color='red', ls=':', lw=1, alpha=0.7)
    ax2.text(SIGMA_D_DFDT + 0.02, 0.47, f'σ_dead={SIGMA_D_DFDT}%\n(df/dt)',
             fontsize=8, color='red', va='top')
    ax2.axvline(x=res['Sigm'], color='gray', ls=':', lw=1, alpha=0.7)
    ax2.text(res['Sigm'] + 0.02, 0.42, f'Sigm={res["Sigm"]:.2f}%\n(ручная)',
             fontsize=8, color='gray', va='top')

    plot_refs(ax2, res)
    ax2.set_xlabel('σ (%)', fontsize=11)
    ax2.set_ylabel('R (мм/день)', fontsize=11)
    ax2.set_xlim(0, 2)
    ax2.set_ylim(0, 0.5)
    ax2.grid(True, color='gray', alpha=0.2)
    ax2.set_title('ZOOM: мёртвая зона (σ = 0–2%)', fontsize=10, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=7, framealpha=0.9)

    fig.suptitle('020326_2_1 — Dead zone zoom: все модели', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'cv_deadzone_zoom.jpg', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved: cv_deadzone_zoom.jpg")


if __name__ == "__main__":
    main()
