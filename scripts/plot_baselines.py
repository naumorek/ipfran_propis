#!/usr/bin/env python3
"""
Baseline 1 (mathcad_exact.py, method='classic') vs Baseline 2 (CV model).
Один график R(σ), две кривые + эталоны.
Файл: 020326_2_1.
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
from propis_app.core.kinetics.bcf_model import bcf_model, fit_bcf_fixed_sigma_d

PRN = str(project_root / "020326 KDP нейтр XXXIII п.15" / "__020326_2.prn")
SIGMA_D_DFDT = 0.30


def main():
    # Baseline 1: mathcad_exact.py (точная копия Mathcad)
    res = run_mathcad(PRN,
                      n1=10, n2=8000, im=5963, isat=6692,
                      im1=5120, isat1=7580,
                      Salt=1, Acid=0, Face=0, d=3.3, ww=1,
                      k1=0.15, span1=0.2, l=0.01,
                      tn_manual=48.60, method='classic')

    VX = res['VX']      # σ% (sorted)
    VY = res['VY']      # R мкм/мин (sorted)
    F1 = res['F1']      # LOESS interp function
    Si = res['Si']       # reference Si
    Si1 = res['Si1']     # reference Si1

    # F1 на сетке (мкм/мин → мм/день: ×1.441)
    sigma_grid = np.linspace(0, VX.max(), 300)
    f1_values = F1(sigma_grid) * 1.441  # мкм/мин → мм/день
    f1_values = np.maximum(f1_values, 0)

    # Baseline 2: CV model (fixed σ_dead из df/dt, weighted)
    # Фитим на тех же данных VX, VY (конвертируем VY в мм/день)
    VY_mmday = VY * 1.441
    mask = (VX > 0) & (VY_mmday > 0)
    bcf = fit_bcf_fixed_sigma_d(VX[mask], VY_mmday[mask],
                                SIGMA_D_DFDT, auto_weight=True)
    cv_values = bcf_model(sigma_grid, bcf.beta, SIGMA_D_DFDT, bcf.sigma_1)

    # Эталоны (Si[j,0]=σ%, Si[j,1]=R мкм/мин → мм/день)
    si_sigma = Si[:, 0]
    si_rate = Si[:, 1] * 1.441
    si1_sigma = Si1[:, 0]
    si1_rate = Si1[:, 1] * 1.441

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 6))

    # Эталоны
    m0 = si_sigma > 0
    if np.any(m0):
        ax.plot(si_sigma[m0], si_rate[m0],
                color='green', lw=1.5, marker='D', ms=5, alpha=0.6,
                zorder=3, label='Si (Cfe=0)')
    m1 = si1_sigma > 0
    if np.any(m1):
        ax.plot(si1_sigma[m1], si1_rate[m1],
                color='red', lw=1.5, marker='D', ms=5, alpha=0.6,
                zorder=3, label='Si1 (Cfe=16 ppm)')

    # Baseline 1: mathcad_exact.py
    ax.plot(sigma_grid, f1_values, color='black', lw=2.5, zorder=10,
            label='Baseline 1: Mathcad exact (2×LOESS)')

    # Baseline 2: CV
    ax.plot(sigma_grid, cv_values, color='blue', lw=2.5, ls='--', zorder=9,
            label=f'Baseline 2: CV (σ_d={SIGMA_D_DFDT}%, β={bcf.beta:.3f}, σ₁={bcf.sigma_1:.3f})')

    # σ_dead
    ax.axvline(x=SIGMA_D_DFDT, color='gray', ls=':', lw=1, alpha=0.5)
    ax.text(SIGMA_D_DFDT + 0.05, f1_values.max() * 0.95,
            f'σ_dead={SIGMA_D_DFDT}%', fontsize=8, color='gray', va='top')

    # Info box
    info = (f"te={res['te']:.2f}  tn={res['tn']:.2f}  Td={res['Td']:.2f}\n"
            f"Sigm={res['Sigm']:.3f}  s2={res['s2']:.3f}  Sig035={res['Sig035']:.3f}")
    ax.text(0.03, 0.97, info, transform=ax.transAxes, fontsize=8,
            va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    ax.set_xlabel('σ (%)', fontsize=13)
    ax.set_ylabel('R (мм/день)', fontsize=13)
    ax.set_xlim(0, min(VX.max() * 1.05, 6))
    ax.set_ylim(0, min(f1_values.max() * 1.15, 4))
    ax.grid(True, color='gray', alpha=0.2)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.set_title('020326_2_1: Baseline 1 (Mathcad exact) vs Baseline 2 (CV)',
                 fontsize=13, fontweight='bold')

    fig.tight_layout()
    out = project_root / 'baselines_comparison.jpg'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
