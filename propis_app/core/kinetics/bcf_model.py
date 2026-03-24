"""
BCF (Burton-Cabrera-Frank) kinetic model with dead zone.

Model: R(σ) = β * (σ - σ_d)^2 / (σ_1 + σ - σ_d)

where:
  σ    — supersaturation (%)
  σ_d  — dead zone (critical supersaturation below which no growth)
  β    — kinetic coefficient
  σ_1  — transition parameter (determines curve shape)

At low σ: R ≈ β*(σ-σ_d)^2/σ_1  (parabolic, spiral growth)
At high σ: R ≈ β*(σ-σ_d)        (linear, rough growth)

Reference: Burton, Cabrera, Frank (1951); Cabrera & Vermilyea (1958) for dead zone.
Sangwal (1996), Prog. Cryst. Growth Charact. Mater. 32, 3-43.
Dam & van Enckevort (1990), J. Cryst. Growth 99, 809.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict

import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.stats import norm, chi2


# ============================================================================
#  Dataclasses
# ============================================================================

@dataclass
class BCFResult:
    """Result of BCF model fitting."""
    beta: float          # kinetic coefficient
    sigma_d: float       # dead zone (%)
    sigma_1: float       # transition parameter (%)
    residual: float      # sum of squared residuals

    sigma_percent: np.ndarray   # input σ (%)
    rate_measured: np.ndarray   # measured R
    rate_fitted: np.ndarray     # fitted R

    # --- Uncertainty (Phase 1: Task 4) ---
    pcov: Optional[np.ndarray] = None       # 3×3 covariance matrix
    correlation: Optional[np.ndarray] = None # 3×3 correlation matrix
    se: Optional[np.ndarray] = None          # standard errors [se_β, se_σd, se_σ1]
    beta_ci: Optional[tuple] = None          # (lower, upper) 95% CI
    sigma_d_ci: Optional[tuple] = None
    sigma_1_ci: Optional[tuple] = None
    n_points: int = 0
    n_params: int = 3
    chi2_reduced: float = 0.0               # χ²/(N−p)

    # --- Weighted fit (Phase 2: Task 5) ---
    weights: Optional[np.ndarray] = None
    weighted_residual: float = 0.0
    heteroscedasticity_pvalue: Optional[float] = None


@dataclass
class BCFResultT:
    """Result of temperature-dependent BCF fit (Phase 3: Task 2)."""
    beta: float
    sigma_0: float          # pre-exponential factor
    E_half: float           # ΔH_ads / (2R) in Kelvin
    sigma_1: float          # BCF transition (%)
    residual: float

    # Derived
    Delta_H_ads_kJmol: float = 0.0   # 2 * R_gas * E_half / 1000
    sigma_d_mean: float = 0.0        # σ_dead at T_mean

    sigma_percent: np.ndarray = field(default_factory=lambda: np.array([]))
    T_kelvin: np.ndarray = field(default_factory=lambda: np.array([]))
    rate_measured: np.ndarray = field(default_factory=lambda: np.array([]))
    rate_fitted: np.ndarray = field(default_factory=lambda: np.array([]))
    sigma_d_at_T: Optional[np.ndarray] = None  # σ_dead(T) for each point

    pcov: Optional[np.ndarray] = None
    bic_3param: float = 0.0
    bic_4param: float = 0.0


# ============================================================================
#  Forward models
# ============================================================================

def bcf_model(sigma_percent: np.ndarray, beta: float,
              sigma_d: float, sigma_1: float) -> np.ndarray:
    """
    BCF growth rate model with dead zone (Cabrera-Vermilyea).

    R(σ) = β · (σ − σ_d)² / (σ₁ + σ − σ_d)   for σ > σ_d
    R(σ) = 0                                     for σ ≤ σ_d
    """
    sigma = np.asarray(sigma_percent, dtype=float)
    x = np.maximum(sigma - sigma_d, 0.0)
    denom = np.maximum(sigma_1 + x, 1e-10)
    return beta * x**2 / denom


def bcf_model_T(sigma_percent: np.ndarray, T_kelvin: np.ndarray,
                beta: float, sigma_0: float, E_half: float,
                sigma_1: float) -> np.ndarray:
    """
    Temperature-dependent CV model (Task 2).

    σ_dead(T) = σ₀ · exp(E_half / T)
    R(σ, T) = β · x² / (σ₁ + x),  x = max(σ − σ_dead(T), 0)

    Physics: K_ads(T) = K₀·exp(ΔH_ads/(R·T)), σ_dead ∝ √K_ads
    → σ_dead(T) = σ₀·exp(ΔH_ads/(2R·T)) = σ₀·exp(E_half/T)

    Parameters
    ----------
    sigma_percent : array — supersaturation (%)
    T_kelvin : array — temperature (K)
    beta : float — kinetic coefficient
    sigma_0 : float — pre-exponential (%)
    E_half : float — ΔH_ads/(2R) in Kelvin
    sigma_1 : float — BCF transition (%)
    """
    sigma = np.asarray(sigma_percent, dtype=float)
    T = np.asarray(T_kelvin, dtype=float)
    sigma_d_T = sigma_0 * np.exp(E_half / T)
    x = np.maximum(sigma - sigma_d_T, 0.0)
    denom = np.maximum(sigma_1 + x, 1e-10)
    return beta * x**2 / denom


def _bcf_T_wrapper(sigma_T_flat, beta, sigma_0, E_half, sigma_1):
    """Wrapper for curve_fit: 2 independent variables stacked as rows."""
    sigma = sigma_T_flat[0]
    T_K = sigma_T_flat[1]
    return bcf_model_T(sigma, T_K, beta, sigma_0, E_half, sigma_1)


def bcf_model_fraction(sigma_frac: np.ndarray, beta: float,
                       sigma_d_frac: float,
                       sigma_1_frac: float) -> np.ndarray:
    """BCF model using fractional supersaturation (not percent)."""
    return bcf_model(sigma_frac * 100, beta, sigma_d_frac * 100,
                     sigma_1_frac * 100)


# ============================================================================
#  Utility functions
# ============================================================================

def bic(n: int, p: int, rss: float) -> float:
    """Bayesian Information Criterion: BIC = n·ln(RSS/n) + p·ln(n)."""
    if rss <= 0 or n <= 0:
        return np.inf
    return n * np.log(rss / n) + p * np.log(n)


def _compute_uncertainty(popt, pcov, n_points, n_params=3):
    """Compute SE, correlation, Wald CI, χ²_red from curve_fit output."""
    result = {}

    # Standard errors
    se = np.sqrt(np.diag(pcov))
    result['se'] = se

    # Correlation matrix
    d = np.sqrt(np.diag(pcov))
    d_safe = np.where(d > 0, d, 1e-20)
    corr = pcov / np.outer(d_safe, d_safe)
    result['correlation'] = corr

    # 95% Wald CI
    z = 1.96
    result['beta_ci'] = (popt[0] - z * se[0], popt[0] + z * se[0])
    result['sigma_d_ci'] = (popt[1] - z * se[1], popt[1] + z * se[1])
    result['sigma_1_ci'] = (popt[2] - z * se[2], popt[2] + z * se[2])

    return result


# ============================================================================
#  Phase 1 (Task 4): Fitting with uncertainty
# ============================================================================

def fit_bcf(sigma_percent: np.ndarray, rate: np.ndarray,
            initial_guess: Optional[dict] = None,
            weights: Optional[np.ndarray] = None,
            auto_weight: bool = False,
            sigma_d_fixed: Optional[float] = None) -> BCFResult:
    """
    Fit BCF model to measured data with full uncertainty quantification.

    Parameters
    ----------
    sigma_percent : array — supersaturation (%)
    rate : array — measured growth rate
    initial_guess : dict, optional — {"beta": ..., "sigma_d": ..., "sigma_1": ...}
    weights : array, optional — per-point weights w_i for WLS
    auto_weight : bool — if True, use w_i = 1/(1 + R_i²) (Task 5)
    sigma_d_fixed : float, optional — fix σ_dead from df/dt (Task 4.4)

    Returns
    -------
    BCFResult with pcov, correlation, CI, chi2_reduced
    """
    # Filter valid data
    valid = np.isfinite(sigma_percent) & np.isfinite(rate) & (rate > 0)
    sigma_v = sigma_percent[valid]
    rate_v = rate[valid]
    n = len(sigma_v)

    if n < 4:
        return BCFResult(
            beta=0, sigma_d=0, sigma_1=1,
            residual=np.inf, n_points=n,
            sigma_percent=sigma_v,
            rate_measured=rate_v,
            rate_fitted=np.zeros_like(rate_v),
        )

    # Weights (Task 5)
    if weights is not None:
        w = weights[valid]
    elif auto_weight:
        w = 1.0 / (1.0 + rate_v**2)
    else:
        w = None

    # sigma parameter for curve_fit: sigma = 1/sqrt(weight)
    sigma_cf = 1.0 / np.sqrt(w) if w is not None else None

    # Initial guess
    if initial_guess is None:
        sigma_min = np.min(sigma_v)
        rate_max = np.max(rate_v)
        sigma_max = np.max(sigma_v)
        initial_guess = {
            "beta": rate_max / (sigma_max - sigma_min + 0.1),
            "sigma_d": max(sigma_min - 0.5, 0),
            "sigma_1": (sigma_max - sigma_min) / 2,
        }

    pcov_out = None

    if sigma_d_fixed is not None:
        # Task 4.4: 2-parameter fit with σ_dead fixed
        n_params = 2

        def _model_fixed(sigma, beta, sigma_1):
            return bcf_model(sigma, beta, sigma_d_fixed, sigma_1)

        p0_2 = [initial_guess["beta"], initial_guess["sigma_1"]]
        try:
            popt2, pcov2 = curve_fit(
                _model_fixed, sigma_v, rate_v,
                p0=p0_2,
                sigma=sigma_cf, absolute_sigma=False,
                bounds=([0, 0.01], [np.inf, 100]),
                maxfev=10000,
            )
            beta, sigma_1 = popt2
            sigma_d = sigma_d_fixed
            rate_fitted = bcf_model(sigma_v, beta, sigma_d, sigma_1)
            residual = float(np.sum((rate_v - rate_fitted)**2))

            # Expand 2×2 pcov to 3×3 (σ_dead row/col = 0)
            pcov_out = np.zeros((3, 3))
            pcov_out[0, 0] = pcov2[0, 0]
            pcov_out[0, 2] = pcov2[0, 1]
            pcov_out[2, 0] = pcov2[1, 0]
            pcov_out[2, 2] = pcov2[1, 1]

        except (RuntimeError, ValueError):
            beta = initial_guess["beta"]
            sigma_d = sigma_d_fixed
            sigma_1 = initial_guess["sigma_1"]
            rate_fitted = bcf_model(sigma_v, beta, sigma_d, sigma_1)
            residual = float(np.sum((rate_v - rate_fitted)**2))
    else:
        # Standard 3-parameter fit
        n_params = 3
        p0 = [initial_guess["beta"], initial_guess["sigma_d"],
              initial_guess["sigma_1"]]

        try:
            popt, pcov_out = curve_fit(
                bcf_model, sigma_v, rate_v,
                p0=p0,
                sigma=sigma_cf, absolute_sigma=False,
                bounds=([0, -5, 0.01], [np.inf, np.max(sigma_v), 100]),
                maxfev=10000,
            )
            beta, sigma_d, sigma_1 = popt
            rate_fitted = bcf_model(sigma_v, beta, sigma_d, sigma_1)
            residual = float(np.sum((rate_v - rate_fitted)**2))
        except (RuntimeError, ValueError):
            def objective(params):
                b, sd, s1 = params
                if b <= 0 or s1 <= 0:
                    return 1e20
                fitted = bcf_model(sigma_v, b, sd, s1)
                return np.sum((rate_v - fitted)**2)

            result = minimize(objective, p0, method="Nelder-Mead",
                              options={"maxiter": 10000})
            beta, sigma_d, sigma_1 = result.x
            rate_fitted = bcf_model(sigma_v, beta, sigma_d, sigma_1)
            residual = float(result.fun)

    # Build result
    popt_full = np.array([beta, sigma_d, sigma_1])
    chi2_red = residual / max(n - n_params, 1)

    res = BCFResult(
        beta=beta, sigma_d=sigma_d, sigma_1=sigma_1,
        residual=residual,
        sigma_percent=sigma_v, rate_measured=rate_v, rate_fitted=rate_fitted,
        pcov=pcov_out, n_points=n, n_params=n_params,
        chi2_reduced=chi2_red,
    )

    # Weighted residual
    if w is not None:
        res.weights = w
        res.weighted_residual = float(np.sum(w * (rate_v - rate_fitted)**2))

    # Uncertainty from pcov
    if pcov_out is not None and np.all(np.isfinite(pcov_out)):
        unc = _compute_uncertainty(popt_full, pcov_out, n, n_params)
        res.se = unc['se']
        res.correlation = unc['correlation']
        res.beta_ci = unc['beta_ci']
        res.sigma_d_ci = unc['sigma_d_ci']
        res.sigma_1_ci = unc['sigma_1_ci']

    return res


def fit_bcf_fixed_sigma_d(sigma_percent: np.ndarray, rate: np.ndarray,
                          sigma_d_fixed: float,
                          weights: Optional[np.ndarray] = None,
                          auto_weight: bool = False) -> BCFResult:
    """
    2-parameter CV fit with σ_dead fixed from df/dt (Task 4.4).

    Reduces degeneracy: with σ_dead known, β and σ₁ are better determined.
    The Fisher information matrix becomes well-conditioned.
    """
    return fit_bcf(sigma_percent, rate, sigma_d_fixed=sigma_d_fixed,
                   weights=weights, auto_weight=auto_weight)


# ============================================================================
#  Phase 1 (Task 4): Profile likelihood
# ============================================================================

def profile_likelihood(sigma_percent: np.ndarray, rate: np.ndarray,
                       popt: np.ndarray, param_index: int,
                       n_grid: int = 50,
                       weights: Optional[np.ndarray] = None,
                       se: Optional[np.ndarray] = None) -> dict:
    """
    Profile likelihood for one CV parameter (Raue et al., 2009).

    Fix θ_k on a grid, optimize remaining parameters, record χ².
    95% CI: {θ_k : χ²_profile(θ_k) < χ²_min + 3.84}

    Parameters
    ----------
    sigma_percent, rate : data arrays
    popt : [β, σ_d, σ₁] — best-fit parameters
    param_index : 0=β, 1=σ_d, 2=σ₁
    n_grid : grid resolution
    weights : optional per-point weights
    se : optional standard errors (for grid range)

    Returns
    -------
    dict with 'grid', 'chi2', 'ci_lower', 'ci_upper', 'identifiable'
    """
    valid = np.isfinite(sigma_percent) & np.isfinite(rate) & (rate > 0)
    sigma_v = sigma_percent[valid]
    rate_v = rate[valid]

    if weights is not None:
        w = weights[valid] if len(weights) == len(sigma_percent) else weights
    else:
        w = np.ones(len(sigma_v))

    # Grid range: ±5 SE or reasonable defaults
    if se is not None and se[param_index] > 0:
        half_range = 5 * se[param_index]
    else:
        half_range = max(abs(popt[param_index]) * 0.5, 0.5)

    lo = max(popt[param_index] - half_range, 0.0 if param_index != 1 else -0.5)
    hi = popt[param_index] + half_range
    grid = np.linspace(lo, hi, n_grid)

    # Reference chi2 at optimum
    rate_opt = bcf_model(sigma_v, *popt)
    chi2_min = float(np.sum(w * (rate_v - rate_opt)**2))

    chi2_values = np.full(n_grid, np.inf)
    other_idx = [i for i in range(3) if i != param_index]

    for ig, val in enumerate(grid):
        def obj(params_free):
            p = np.array(popt, dtype=float)
            p[param_index] = val
            for i, idx in enumerate(other_idx):
                p[idx] = params_free[i]
            if p[0] <= 0 or p[2] <= 0.01:
                return 1e20
            fitted = bcf_model(sigma_v, p[0], p[1], p[2])
            return np.sum(w * (rate_v - fitted)**2)

        p0_free = [popt[idx] for idx in other_idx]
        try:
            res = minimize(obj, p0_free, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-8})
            chi2_values[ig] = res.fun
        except Exception:
            pass

    # 95% CI: Δχ² < 3.84
    threshold = chi2_min + 3.84
    within = grid[chi2_values < threshold]

    if len(within) > 0:
        ci_lower = float(within[0])
        ci_upper = float(within[-1])
        identifiable = True
    else:
        ci_lower = float(grid[0])
        ci_upper = float(grid[-1])
        identifiable = False

    # Flat profile check: if χ² changes < 3.84 across entire grid
    chi2_range = np.max(chi2_values[np.isfinite(chi2_values)]) - np.min(chi2_values[np.isfinite(chi2_values)])
    if chi2_range < 3.84:
        identifiable = False

    return {
        'grid': grid,
        'chi2': chi2_values,
        'chi2_min': chi2_min,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'identifiable': identifiable,
        'param_name': ['beta', 'sigma_d', 'sigma_1'][param_index],
    }


# ============================================================================
#  Phase 1 (Task 4): MCMC via emcee
# ============================================================================

def mcmc_bcf(sigma_percent: np.ndarray, rate: np.ndarray,
             p0: np.ndarray,
             sigma_d_fixed: Optional[float] = None,
             n_walkers: int = 32, n_burn: int = 1000, n_steps: int = 4000,
             weights: Optional[np.ndarray] = None) -> dict:
    """
    MCMC sampling of CV parameter posterior (Foreman-Mackey et al., 2013).

    Log-posterior = log-likelihood + log-prior
    Likelihood: marginalized over σ_ε (Jeffrey's prior)
        ln L = −(N−1)/2 · ln(Σ(R_obs − R_model)²)

    Priors:
        β ∈ Uniform(0, 5) mm/day
        σ_dead ∈ Uniform(−0.5, 3) %
        σ₁ ∈ log-Uniform(0.01, 50) %

    Requires: pip install emcee

    Returns
    -------
    dict with 'chain', 'map', 'median', 'hpd_95', 'acceptance_fraction'
    """
    try:
        import emcee
    except ImportError:
        raise ImportError(
            "emcee is required for MCMC. Install: pip install emcee"
        )

    valid = np.isfinite(sigma_percent) & np.isfinite(rate) & (rate > 0)
    sigma_v = sigma_percent[valid]
    rate_v = rate[valid]
    N = len(sigma_v)

    if weights is not None:
        w = weights[valid] if len(weights) == len(sigma_percent) else weights
    else:
        w = np.ones(N)

    if sigma_d_fixed is not None:
        # 2-parameter: β, σ₁
        ndim = 2

        def log_prior(theta):
            beta, sigma_1 = theta
            if not (0 < beta < 5):
                return -np.inf
            if not (0.01 < sigma_1 < 50):
                return -np.inf
            # log-uniform prior on σ₁
            return -np.log(sigma_1)

        def log_likelihood(theta):
            beta, sigma_1 = theta
            model = bcf_model(sigma_v, beta, sigma_d_fixed, sigma_1)
            rss = np.sum(w * (rate_v - model)**2)
            if rss <= 0:
                return -np.inf
            return -(N - 1) / 2.0 * np.log(rss)

        p0_init = np.array([p0[0], p0[2]])  # β, σ₁
        param_names = ['beta', 'sigma_1']
    else:
        # 3-parameter: β, σ_d, σ₁
        ndim = 3

        def log_prior(theta):
            beta, sigma_d, sigma_1 = theta
            if not (0 < beta < 5):
                return -np.inf
            if not (-0.5 < sigma_d < 3):
                return -np.inf
            if not (0.01 < sigma_1 < 50):
                return -np.inf
            return -np.log(sigma_1)  # log-uniform on σ₁

        def log_likelihood(theta):
            beta, sigma_d, sigma_1 = theta
            model = bcf_model(sigma_v, beta, sigma_d, sigma_1)
            rss = np.sum(w * (rate_v - model)**2)
            if rss <= 0:
                return -np.inf
            return -(N - 1) / 2.0 * np.log(rss)

        p0_init = p0[:3]
        param_names = ['beta', 'sigma_d', 'sigma_1']

    def log_posterior(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    # Initialize walkers: Gaussian ball around p0
    pos = p0_init + 1e-3 * np.abs(p0_init) * np.random.randn(n_walkers, ndim)
    # Ensure positive
    for i in range(n_walkers):
        pos[i] = np.maximum(pos[i], 1e-4)

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior)
    sampler.run_mcmc(pos, n_burn + n_steps, progress=False)

    # Discard burn-in
    chain = sampler.get_chain(discard=n_burn, flat=True)

    # MAP estimate
    log_probs = sampler.get_log_prob(discard=n_burn, flat=True)
    map_idx = np.argmax(log_probs)
    map_params = chain[map_idx]

    # Median and 95% HPD
    median_params = np.median(chain, axis=0)
    hpd_95 = {}
    for i, name in enumerate(param_names):
        lo, hi = np.percentile(chain[:, i], [2.5, 97.5])
        hpd_95[name] = (float(lo), float(hi))

    return {
        'chain': chain,
        'param_names': param_names,
        'map': {name: float(map_params[i]) for i, name in enumerate(param_names)},
        'median': {name: float(median_params[i]) for i, name in enumerate(param_names)},
        'hpd_95': hpd_95,
        'acceptance_fraction': float(np.mean(sampler.acceptance_fraction)),
        'n_samples': len(chain),
    }


# ============================================================================
#  Phase 2 (Task 5): Heteroscedasticity test
# ============================================================================

def breusch_pagan_test(residuals: np.ndarray,
                       fitted_values: np.ndarray) -> tuple:
    """
    Breusch-Pagan test for heteroscedasticity.

    Auxiliary regression: e² = a + b·Ŷ + ε
    Statistic: BP = N·R² ~ χ²(1) under H₀ (homoscedastic)

    Returns
    -------
    (bp_statistic, p_value)
    """
    e2 = residuals**2
    n = len(e2)

    # OLS: e² = a + b·Ŷ
    X = np.column_stack([np.ones(n), fitted_values])
    try:
        beta_hat = np.linalg.lstsq(X, e2, rcond=None)[0]
    except np.linalg.LinAlgError:
        return (0.0, 1.0)

    e2_hat = X @ beta_hat
    ss_res = np.sum((e2 - e2_hat)**2)
    ss_tot = np.sum((e2 - np.mean(e2))**2)

    if ss_tot < 1e-20:
        return (0.0, 1.0)

    r_squared = 1.0 - ss_res / ss_tot
    bp = n * r_squared
    p_value = float(1.0 - chi2.cdf(bp, df=1))

    return (float(bp), p_value)


# ============================================================================
#  Phase 2 (Task 5): BCa Bootstrap
# ============================================================================

def bootstrap_bcf(sigma_percent: np.ndarray, rate: np.ndarray,
                  n_boot: int = 2000, confidence: float = 0.95,
                  weights: Optional[np.ndarray] = None,
                  auto_weight: bool = False,
                  sigma_d_fixed: Optional[float] = None) -> Dict:
    """
    BCa bootstrap for CV model parameters (Efron, 1987).

    1. Fit original → θ̂
    2. B=2000 resamples with replacement → θ̂_b
    3. Bias correction: z₀ = Φ⁻¹(#{θ̂_b < θ̂}/B)
    4. Acceleration from jackknife: â = Σ(θ̄−θ̂₋ⱼ)³ / [6·(Σ(θ̄−θ̂₋ⱼ)²)^{3/2}]
    5. BCa CI

    Returns
    -------
    dict: keys 'beta', 'sigma_d', 'sigma_1', each with
          'estimate', 'ci_lower', 'ci_upper', 'boot_distribution'
    """
    valid = np.isfinite(sigma_percent) & np.isfinite(rate) & (rate > 0)
    sigma_v = sigma_percent[valid]
    rate_v = rate[valid]
    n = len(sigma_v)

    if n < 5:
        return {}

    # Original fit
    orig_fit = fit_bcf(sigma_v, rate_v, weights=weights,
                       auto_weight=auto_weight, sigma_d_fixed=sigma_d_fixed)
    theta_hat = np.array([orig_fit.beta, orig_fit.sigma_d, orig_fit.sigma_1])

    # Bootstrap resamples
    boot_params = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        s_boot = sigma_v[idx]
        r_boot = rate_v[idx]
        w_boot = weights[valid][idx] if weights is not None else None
        try:
            bf = fit_bcf(s_boot, r_boot, weights=w_boot,
                         auto_weight=auto_weight, sigma_d_fixed=sigma_d_fixed)
            if bf.residual < np.inf:
                boot_params.append([bf.beta, bf.sigma_d, bf.sigma_1])
        except Exception:
            continue

    if len(boot_params) < 100:
        return {}

    boot_params = np.array(boot_params)
    B = len(boot_params)

    # Jackknife for acceleration
    jack_params = []
    for j in range(n):
        idx_j = np.concatenate([np.arange(j), np.arange(j + 1, n)])
        s_j = sigma_v[idx_j]
        r_j = rate_v[idx_j]
        w_j = weights[valid][idx_j] if weights is not None else None
        try:
            bf_j = fit_bcf(s_j, r_j, weights=w_j, auto_weight=auto_weight,
                           sigma_d_fixed=sigma_d_fixed)
            if bf_j.residual < np.inf:
                jack_params.append([bf_j.beta, bf_j.sigma_d, bf_j.sigma_1])
            else:
                jack_params.append(theta_hat.tolist())
        except Exception:
            jack_params.append(theta_hat.tolist())

    jack_params = np.array(jack_params)

    alpha = 1.0 - confidence
    param_names = ['beta', 'sigma_d', 'sigma_1']
    results = {}

    for k, name in enumerate(param_names):
        theta_k = theta_hat[k]
        boot_k = boot_params[:, k]

        # Bias correction z₀
        prop_below = np.sum(boot_k < theta_k) / B
        prop_below = np.clip(prop_below, 1e-6, 1 - 1e-6)
        z0 = norm.ppf(prop_below)

        # Acceleration â
        jack_k = jack_params[:, k]
        jack_mean = np.mean(jack_k)
        d_jack = jack_mean - jack_k
        num = np.sum(d_jack**3)
        den = 6.0 * (np.sum(d_jack**2))**1.5
        a_hat = num / den if abs(den) > 1e-20 else 0.0

        # BCa quantiles
        z_lo = norm.ppf(alpha / 2)
        z_hi = norm.ppf(1 - alpha / 2)

        def _bca_alpha(z_alpha):
            numer = z0 + z_alpha
            denom = 1.0 - a_hat * numer
            if abs(denom) < 1e-10:
                return 0.5
            return norm.cdf(z0 + numer / denom)

        alpha_lo = _bca_alpha(z_lo)
        alpha_hi = _bca_alpha(z_hi)

        # Clip to valid range
        alpha_lo = np.clip(alpha_lo, 0.5 / B, 1 - 0.5 / B)
        alpha_hi = np.clip(alpha_hi, 0.5 / B, 1 - 0.5 / B)

        sorted_boot = np.sort(boot_k)
        ci_lo = float(sorted_boot[int(alpha_lo * B)])
        ci_hi = float(sorted_boot[min(int(alpha_hi * B), B - 1)])

        results[name] = {
            'estimate': float(theta_k),
            'ci_lower': ci_lo,
            'ci_upper': ci_hi,
            'boot_distribution': boot_k,
            'z0': float(z0),
            'a_hat': float(a_hat),
        }

    return results


# ============================================================================
#  Phase 3 (Task 2): Temperature-dependent fit
# ============================================================================

R_GAS = 8.314  # J/(mol·K)

def fit_bcf_T(sigma_percent: np.ndarray, rate: np.ndarray,
              T_celsius: np.ndarray,
              bcf_3param: Optional[BCFResult] = None,
              E_half_prior: float = 4210.0,
              weights: Optional[np.ndarray] = None,
              auto_weight: bool = False) -> Optional[BCFResultT]:
    """
    Fit temperature-dependent CV model (Task 2).

    σ_dead(T) = σ₀ · exp(E_half / T_K)
    R(σ, T) = β · x² / (σ₁ + x),  x = max(σ − σ_dead(T), 0)

    4 parameters: β, σ₀, E_half, σ₁

    Only runs if 3-parameter σ_dead > 0.1% and ΔBIC < −2.

    Parameters
    ----------
    sigma_percent, rate : data arrays
    T_celsius : temperature array (°C) — same length as sigma_percent
    bcf_3param : result of 3-parameter fit (for initial guess and BIC comparison)
    E_half_prior : initial E_half (K). Default 4210 = 70 kJ/mol / (2·8.314)
    weights, auto_weight : weighting options

    Returns
    -------
    BCFResultT or None (if conditions not met or BIC not improved)
    """
    valid = np.isfinite(sigma_percent) & np.isfinite(rate) & (rate > 0)
    valid &= np.isfinite(T_celsius)
    sigma_v = sigma_percent[valid]
    rate_v = rate[valid]
    T_v = T_celsius[valid] + 273.15  # → Kelvin
    n = len(sigma_v)

    if n < 6:
        return None

    # Condition 1: σ_dead from 3-param fit must be > 0.1%
    if bcf_3param is not None and bcf_3param.sigma_d < 0.1:
        return None

    # Weights
    if weights is not None:
        w = weights[valid]
    elif auto_weight:
        w = 1.0 / (1.0 + rate_v**2)
    else:
        w = None

    sigma_cf = 1.0 / np.sqrt(w) if w is not None else None

    # Initial guess from 3-param fit
    if bcf_3param is not None:
        T_mean = np.mean(T_v)
        # σ_dead = σ₀ · exp(E_half/T_mean) → σ₀ = σ_dead / exp(E_half/T_mean)
        sigma_0_init = bcf_3param.sigma_d / np.exp(E_half_prior / T_mean)
        sigma_0_init = max(sigma_0_init, 1e-6)
        p0 = [bcf_3param.beta, sigma_0_init, E_half_prior, bcf_3param.sigma_1]
    else:
        p0 = [0.5, 1e-4, E_half_prior, 2.0]

    # Fit
    xdata = np.vstack([sigma_v, T_v])
    try:
        popt, pcov = curve_fit(
            _bcf_T_wrapper, xdata, rate_v,
            p0=p0,
            sigma=sigma_cf, absolute_sigma=False,
            bounds=([0, 1e-10, 0, 0.01], [np.inf, 10, 15000, 100]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return None

    beta, sigma_0, E_half, sigma_1 = popt
    rate_fitted = bcf_model_T(sigma_v, T_v, beta, sigma_0, E_half, sigma_1)
    rss_4 = float(np.sum((rate_v - rate_fitted)**2))

    # BIC comparison
    rss_3 = bcf_3param.residual if bcf_3param is not None else rss_4
    bic_3 = bic(n, 3, rss_3)
    bic_4 = bic(n, 4, rss_4)

    # Condition 3: ΔBIC < −2
    if bic_4 >= bic_3 - 2:
        return None  # 4-param model not justified

    sigma_d_at_T = sigma_0 * np.exp(E_half / T_v)
    T_mean = np.mean(T_v)
    Delta_H = 2 * R_GAS * E_half / 1000  # kJ/mol

    return BCFResultT(
        beta=beta, sigma_0=sigma_0, E_half=E_half, sigma_1=sigma_1,
        residual=rss_4,
        Delta_H_ads_kJmol=Delta_H,
        sigma_d_mean=float(sigma_0 * np.exp(E_half / T_mean)),
        sigma_percent=sigma_v, T_kelvin=T_v,
        rate_measured=rate_v, rate_fitted=rate_fitted,
        sigma_d_at_T=sigma_d_at_T,
        pcov=pcov,
        bic_3param=bic_3, bic_4param=bic_4,
    )


# ============================================================================
#  Comparison utility
# ============================================================================

def bcf_to_power_law_comparison(bcf: BCFResult,
                                sigma_range: Optional[np.ndarray] = None
                                ) -> dict:
    """
    Compare BCF fit with equivalent power law parameters.

    At low σ: R ≈ (β/σ₁)*(σ-σ_d)^2  → power law with w=2
    At high σ: R ≈ β*(σ-σ_d)        → power law with w=1
    """
    if sigma_range is None:
        sigma_range = np.linspace(0, 10, 200)

    bcf_rate = bcf_model(sigma_range, bcf.beta, bcf.sigma_d, bcf.sigma_1)

    x = np.maximum(sigma_range - bcf.sigma_d, 1e-10)
    denom = bcf.sigma_1 + x
    effective_w = 2.0 * bcf.sigma_1 / denom

    return {
        "sigma_range": sigma_range,
        "bcf_rate": bcf_rate,
        "effective_exponent": effective_w,
        "low_sigma_w": 2.0,
        "high_sigma_w": 1.0,
        "transition_sigma": bcf.sigma_d + bcf.sigma_1,
    }
