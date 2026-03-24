#!/usr/bin/env python3
"""
Validate BCF model uncertainty quantification (Tasks 2, 4, 5).

Uses CLASSIC pipeline data (sparse extrema, ~17 points) for BCF fitting,
because Hilbert dense data has drift artifacts in dead zone.

Diagnostics:
  - Correlation matrix (Task 4: degeneracy β vs σ₁)
  - Profile likelihood (Task 4: identifiability)
  - Weighted vs unweighted fit (Task 5: asymmetry)
  - Breusch-Pagan test (Task 5: heteroscedasticity)
  - BCa Bootstrap CI (Task 5)
  - Fixed σ_dead from df/dt (Task 4.4)
  - T-dependent fit (Task 2)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from propis_app.core.prn_reader import read_prn
from propis_app.core.pipeline import run_classic, CycleParams
from propis_app.core.kinetics.bcf_model import (
    fit_bcf, fit_bcf_fixed_sigma_d, fit_bcf_T,
    profile_likelihood, breusch_pagan_test, bootstrap_bcf,
)

# 3 representative test files
TEST_CASES = [
    {
        'name': '020326_2_1',
        'prn': "020326 KDP нейтр XXXIII п.15/__020326_2.prn",
        'n1': 10, 'n2': 8000, 'im': 5963, 'isat': 6692,
        'im1': 5120, 'isat1': 7580, 'tn': 48.60,
        'sigma_d_dfdt': 0.30,  # from df/dt auto-detection
    },
    {
        'name': '020326_5_1',
        'prn': "020326 KDP нейтр XXXIII п.17/__020326_5.prn",
        'n1': 0, 'n2': 9000, 'im': 7227, 'isat': 7576,
        'im1': 6350, 'isat1': 8090, 'tn': 48.60,
        'sigma_d_dfdt': 0.42,
    },
    {
        'name': '030326_2_2',
        'prn': "030326 KDP нейтр XXXIII п.18/__030326_2.prn",
        'n1': 13500, 'n2': 21000, 'im': 4596, 'isat': 5082,
        'im1': 3500, 'isat1': 5690, 'tn': None,
        'sigma_d_dfdt': 0.31,
    },
]


def run_diagnostics(case):
    """Run full BCF uncertainty diagnostics for one test case."""
    print(f"\n{'='*60}")
    print(f"  {case['name']}")
    print(f"{'='*60}")

    prn_path = project_root / case['prn']
    if not prn_path.exists():
        print(f"  SKIP: {prn_path} not found")
        return

    prn = read_prn(prn_path)
    params = CycleParams(
        n1=case['n1'], n2=case['n2'],
        im=case['im'], isat=case['isat'],
        im1=case['im1'], isat1=case['isat1'],
    )

    # Use CLASSIC pipeline — sparse extrema data (17 points), no Hilbert drift
    result = run_classic(prn, params, salt=1, acid=0, face=0,
                         channel=1, tn_manual=case['tn'])

    # Extract R(σ) from classic dense d-step data
    if result.dense_rate is None or result.dense_sigma is None:
        print("  SKIP: no dense data")
        return

    sigma_all = result.dense_sigma
    rate_all = result.dense_rate
    temp_all = result.dense_temperature

    # Growth zone: σ > 0, R > 0
    mask = np.isfinite(sigma_all) & np.isfinite(rate_all) & (sigma_all > 0) & (rate_all > 0)
    sigma_g = sigma_all[mask]
    rate_g = rate_all[mask]
    temp_g = temp_all[mask] if temp_all is not None else None

    print(f"  N data points: {len(sigma_g)}")
    print(f"  σ range: [{sigma_g.min():.3f}, {sigma_g.max():.3f}]%")
    print(f"  R range: [{rate_g.min():.4f}, {rate_g.max():.4f}]")
    print(f"  Sigm (manual): {result.Sigm:.3f}%")

    # ── Task 4: Degeneracy diagnostics ──
    print(f"\n  --- Task 4: Degeneracy ---")

    # Unweighted fit
    bcf_uw = fit_bcf(sigma_g, rate_g, auto_weight=False)
    print(f"  Unweighted: β={bcf_uw.beta:.4f}, σ_d={bcf_uw.sigma_d:.4f}%, σ₁={bcf_uw.sigma_1:.4f}%")
    print(f"  χ²_red={bcf_uw.chi2_reduced:.6f}, RMSE={np.sqrt(bcf_uw.chi2_reduced):.4f}")

    if bcf_uw.correlation is not None:
        corr_bs = bcf_uw.correlation[0, 2]  # β vs σ₁
        corr_bd = bcf_uw.correlation[0, 1]  # β vs σ_d
        print(f"  corr(β, σ₁) = {corr_bs:.3f}  {'⚠ DEGENERATE' if abs(corr_bs) > 0.9 else '✓ OK'}")
        print(f"  corr(β, σ_d) = {corr_bd:.3f}")

    if bcf_uw.se is not None:
        print(f"  SE: β±{bcf_uw.se[0]:.4f}, σ_d±{bcf_uw.se[1]:.4f}, σ₁±{bcf_uw.se[2]:.4f}")

    if bcf_uw.sigma_d_ci is not None:
        print(f"  Wald CI(σ_d): [{bcf_uw.sigma_d_ci[0]:.3f}, {bcf_uw.sigma_d_ci[1]:.3f}]%")

    # Weighted fit
    bcf_w = fit_bcf(sigma_g, rate_g, auto_weight=True)
    print(f"\n  Weighted (w=1/(1+R²)):")
    print(f"  β={bcf_w.beta:.4f}, σ_d={bcf_w.sigma_d:.4f}%, σ₁={bcf_w.sigma_1:.4f}%")
    print(f"  Δσ_d = {bcf_w.sigma_d - bcf_uw.sigma_d:+.4f}%")

    if bcf_w.correlation is not None:
        corr_bs_w = bcf_w.correlation[0, 2]
        print(f"  corr(β, σ₁) = {corr_bs_w:.3f}")

    # Profile likelihood for σ_dead
    best_bcf = bcf_w  # use weighted
    popt = np.array([best_bcf.beta, best_bcf.sigma_d, best_bcf.sigma_1])
    print(f"\n  --- Profile likelihood (σ_dead) ---")
    prof = profile_likelihood(
        best_bcf.sigma_percent, best_bcf.rate_measured, popt, param_index=1,
        weights=best_bcf.weights, se=best_bcf.se)
    print(f"  Profile CI(σ_d): [{prof['ci_lower']:.3f}, {prof['ci_upper']:.3f}]%")
    print(f"  Identifiable: {prof['identifiable']}")

    # Fixed σ_dead from df/dt
    sigma_d_dfdt = case['sigma_d_dfdt']
    print(f"\n  --- Fixed σ_dead = {sigma_d_dfdt}% (df/dt) ---")
    bcf_fixed = fit_bcf_fixed_sigma_d(
        sigma_g, rate_g, sigma_d_fixed=sigma_d_dfdt, auto_weight=True)
    if bcf_fixed.se is not None:
        print(f"  β={bcf_fixed.beta:.4f} ± {bcf_fixed.se[0]:.4f}")
        print(f"  σ₁={bcf_fixed.sigma_1:.4f} ± {bcf_fixed.se[2]:.4f}")
    else:
        print(f"  β={bcf_fixed.beta:.4f}, σ₁={bcf_fixed.sigma_1:.4f}")
    print(f"  RMSE={np.sqrt(bcf_fixed.chi2_reduced):.4f}")

    # ── Task 5: Asymmetry diagnostics ──
    print(f"\n  --- Task 5: Heteroscedasticity ---")
    resid = bcf_uw.rate_measured - bcf_uw.rate_fitted
    bp_stat, bp_p = breusch_pagan_test(resid, bcf_uw.rate_fitted)
    print(f"  Breusch-Pagan: BP={bp_stat:.2f}, p={bp_p:.4f}"
          f"  {'⚠ HETEROSCEDASTIC' if bp_p < 0.05 else '✓ homoscedastic'}")

    # Data distribution
    n_lo = np.sum(sigma_g < 1.0)
    n_hi = np.sum(sigma_g >= 1.0)
    print(f"  Points σ<1%: {n_lo}, σ≥1%: {n_hi} (ratio {n_hi/max(n_lo,1):.1f}:1)")

    # Bootstrap
    print(f"\n  --- BCa Bootstrap (B=500) ---")
    boot = bootstrap_bcf(sigma_g, rate_g, n_boot=500, auto_weight=True)
    if boot:
        for name in ['beta', 'sigma_d', 'sigma_1']:
            b = boot[name]
            print(f"  {name:8s}: {b['estimate']:.4f}  "
                  f"CI=[{b['ci_lower']:.4f}, {b['ci_upper']:.4f}]  "
                  f"z₀={b['z0']:+.3f}, â={b['a_hat']:+.3f}")
    else:
        print("  Bootstrap failed")

    # ── Task 2: T-dependent ──
    print(f"\n  --- Task 2: T-dependent CV ---")
    if temp_g is not None and len(temp_g) == len(sigma_g) and best_bcf.sigma_d > 0.1:
        bcf_t = fit_bcf_T(sigma_g, rate_g, temp_g,
                          bcf_3param=best_bcf, auto_weight=True)
        if bcf_t is not None:
            print(f"  β={bcf_t.beta:.4f}, σ₀={bcf_t.sigma_0:.6f}")
            print(f"  E_half={bcf_t.E_half:.0f} K → ΔH_ads={bcf_t.Delta_H_ads_kJmol:.1f} kJ/mol"
                  f"  (lit: 63-75)")
            print(f"  σ_d(T_mean)={bcf_t.sigma_d_mean:.3f}%")
            print(f"  BIC: 3p={bcf_t.bic_3param:.1f}, 4p={bcf_t.bic_4param:.1f}, "
                  f"ΔBIC={bcf_t.bic_4param - bcf_t.bic_3param:.1f}")
        else:
            print("  Not justified by BIC (ΔBIC ≥ −2)")
    else:
        reason = "σ_dead < 0.1%" if best_bcf.sigma_d <= 0.1 else "no temperature data"
        print(f"  Not triggered ({reason})")


def main():
    print("BCF Model Uncertainty Diagnostics")
    print("Tasks 2 (K_ads(T)), 4 (degeneracy), 5 (asymmetry)")
    print("Using CLASSIC pipeline data (sparse extrema)")

    for case in TEST_CASES:
        try:
            run_diagnostics(case)
        except Exception as e:
            print(f"\n  ERROR in {case['name']}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
