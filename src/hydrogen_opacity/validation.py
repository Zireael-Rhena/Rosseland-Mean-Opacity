"""
validation.py
=============
Physical and numerical consistency checks.

All check functions return None on success and raise AssertionError on failure
(or ValueError for configuration issues).
"""

from __future__ import annotations

import numpy as np

from .constants import PhysicalConstants
from .config import ModelConfig
from .eos import EOSState, solve_eos
from .opacity import OpacityComponents, monochromatic_opacity
from .rosseland import compute_rosseland_mean, rosseland_mean_from_spectrum
from .grids import build_base_x_grid, refine_x_grid_for_thresholds
from .bound_free_hminus import sigma_bf_Hminus
from .bound_free_h import sigma_bf_hydrogenic_shell


def check_eos_consistency(
    state: EOSState,
    const: PhysicalConstants,
    atol: float = 1e-8,
) -> None:
    """
    Verify EOS conservation equations hold to within absolute tolerance.

    Checks:
      1. n_{H0} + n_p + n_{H⁻} ≈ n_{H,tot}         (nuclei conservation)
      2. n_p ≈ n_e + n_{H⁻}                          (charge neutrality)
      3. Σ_n n_n ≈ n_{H0}                            (level population sum)
      4. All densities ≥ 0.

    Parameters
    ----------
    state : EOSState
    const : PhysicalConstants  (unused; kept for API uniformity)
    atol : float
        Absolute tolerance scaled to n_{H,tot}.

    Raises
    ------
    AssertionError if any check fails.
    """
    scale = state.n_H_tot if state.n_H_tot > 0.0 else 1.0

    # 1. Nuclei conservation
    nuclei_err = abs(state.n_H0 + state.n_p + state.n_Hminus - state.n_H_tot) / scale
    assert nuclei_err <= atol, (
        f"Nuclei conservation failed: relative error = {nuclei_err:.3e}  (tol={atol})"
    )

    # 2. Charge neutrality
    charge_scale = max(state.n_p, 1e-300)
    charge_err = abs(state.n_p - state.n_e - state.n_Hminus) / charge_scale
    assert charge_err <= atol, (
        f"Charge neutrality failed: relative error = {charge_err:.3e}  (tol={atol})"
    )

    # 3. Level population sum
    if len(state.level_populations) > 0 and state.n_H0 > 0.0:
        pop_scale = state.n_H0
        pop_err = abs(state.level_populations.sum() - state.n_H0) / pop_scale
        assert pop_err <= atol, (
            f"Level population sum failed: relative error = {pop_err:.3e}  (tol={atol})"
        )

    # 4. Non-negativity
    assert state.n_e >= 0.0, f"n_e < 0: {state.n_e}"
    assert state.n_p >= 0.0, f"n_p < 0: {state.n_p}"
    assert state.n_H0 >= 0.0, f"n_H0 < 0: {state.n_H0}"
    assert state.n_Hminus >= 0.0, f"n_Hminus < 0: {state.n_Hminus}"
    assert np.all(state.level_populations >= 0.0), "Some n_n < 0"


def check_hminus_approximation(
    state: EOSState,
    const: PhysicalConstants,
) -> dict[str, float]:
    """
    Compare exact K_{H⁻} result against the low-T approximation.

    The approximation n_{H⁻}^approx = (1/4) n_{H0} n_e λ³ exp(χ_{H⁻}/k_BT)
    assumes U_H ≈ 2 (ground-state dominated).
    The ratio K_{H⁻}^exact / K_{H⁻}^approx = 1 / U_H(T), so:
      - near 1.0 at T << 1e4 K  (U_H ≈ 2 → ratio ≈ 1)
      - > 1 at higher T  (U_H > 2 → exact > approx)

    Parameters
    ----------
    state : EOSState
    const : PhysicalConstants

    Returns
    -------
    dict with keys 'n_Hminus_exact', 'n_Hminus_approx', 'ratio'
    """
    from .eos import hminus_abundance_approx
    n_Hm_approx = hminus_abundance_approx(state.T, state.n_e, state.n_H0, const)
    ratio = (state.n_Hminus / n_Hm_approx) if n_Hm_approx > 0.0 else float("inf")
    return {
        "n_Hminus_exact": state.n_Hminus,
        "n_Hminus_approx": n_Hm_approx,
        "ratio": ratio,
    }


def eos_diagnostics(
    state: EOSState,
    const: PhysicalConstants,
) -> dict[str, float | int]:
    """
    Return a dictionary of EOS diagnostic quantities for inspection.

    Quantities
    ----------
    n_cut_effective : int
        Density-dependent level cutoff used in this solve.
    partition_function_H : float
        U_H(T, rho) truncated at n_cut_effective.
    neutral_fraction : float
        n_{H0} / n_{H,tot}  (fraction of hydrogen that is neutral).
    ionized_fraction : float
        n_p / n_{H,tot}  (fraction that is ionised).
    hminus_fraction : float
        n_{H⁻} / n_{H,tot}.
    hminus_over_ne : float
        n_{H⁻} / n_e  (relative H⁻ abundance among electrons).

    Parameters
    ----------
    state : EOSState
    const : PhysicalConstants  (unused; kept for API uniformity)

    Returns
    -------
    dict
    """
    n_tot = max(state.n_H_tot, 1e-300)
    n_e   = max(state.n_e, 1e-300)
    return {
        "n_cut_effective":     state.n_cut_effective,
        "partition_function_H": state.partition_function_H,
        "neutral_fraction":    state.n_H0 / n_tot,
        "ionized_fraction":    state.n_p  / n_tot,
        "hminus_fraction":     state.n_Hminus / n_tot,
        "hminus_over_ne":      state.n_Hminus / n_e,
    }


def check_opacity_nonnegative(comp: OpacityComponents) -> None:
    """
    Assert that all opacity components are non-negative at every grid point.

    Parameters
    ----------
    comp : OpacityComponents

    Raises
    ------
    AssertionError if any component has a negative value.
    """
    assert np.all(comp.kappa_es >= 0.0), "kappa_es has negative values"
    assert np.all(comp.kappa_ff >= 0.0), "kappa_ff has negative values"
    assert np.all(comp.kappa_bf_H >= 0.0), "kappa_bf_H has negative values"
    assert np.all(comp.kappa_bf_Hminus >= 0.0), "kappa_bf_Hminus has negative values"
    assert np.all(comp.kappa_total > 0.0), "kappa_total has non-positive values"


def check_threshold_behavior(
    x: np.ndarray,
    T: float,
    comp: OpacityComponents,
    const: PhysicalConstants,
) -> None:
    """
    Verify threshold behavior of bound-free opacities.

    Checks:
    1. H⁻ bound-free is zero below 0.754 eV and above 10 eV.
    2. Neutral-H bound-free (n=1) is zero below the Lyman limit (13.6 eV).

    Parameters
    ----------
    x : ndarray
        x = hν / k_B T grid.
    T : float   [K]
    comp : OpacityComponents
    const : PhysicalConstants
    """
    kBT_ev = const.k_B * T / const.ev_to_erg

    x_hminus_lo = const.chi_Hminus_ev / kBT_ev
    x_hminus_hi = 10.0 / kBT_ev
    x_lyman = const.chi_H_ev / kBT_ev  # n=1 threshold

    # H⁻ below threshold
    below_hminus = x < x_hminus_lo
    if np.any(below_hminus):
        assert np.all(comp.kappa_bf_Hminus[below_hminus] == 0.0), (
            "kappa_bf_Hminus non-zero below 0.754 eV threshold"
        )

    # H⁻ above upper cutoff
    above_hminus = x > x_hminus_hi
    if np.any(above_hminus):
        assert np.all(comp.kappa_bf_Hminus[above_hminus] == 0.0), (
            "kappa_bf_Hminus non-zero above 10 eV cutoff"
        )

    # Neutral-H bound-free (n=1) below Lyman limit
    nu = x * const.k_B * T / const.h
    nu_lyman = const.chi_H_ev * const.ev_to_erg / const.h
    below_lyman = nu < nu_lyman
    if np.any(below_lyman):
        sigma_n1 = sigma_bf_hydrogenic_shell(nu[below_lyman], T, 1, const)
        assert np.all(sigma_n1 == 0.0), (
            "sigma_bf(n=1) non-zero below Lyman limit"
        )


def convergence_test_xgrid(
    T: float,
    rho: float,
    n_max: int,
    const: PhysicalConstants,
    n_x_values: tuple[int, ...] = (200, 500, 1000, 2000),
    tol_frac: float = 0.01,
) -> dict[str, object]:
    """
    Test convergence of κ_R with respect to x-grid resolution.

    Parameters
    ----------
    T : float   [K]
    rho : float   [g cm⁻³]
    n_max : int
    const : PhysicalConstants
    n_x_values : tuple
        Number of x-grid points to test (coarsest to finest).
    tol_frac : float
        Required fractional change between the two finest grids.

    Returns
    -------
    dict with keys 'n_x_list', 'kappa_R_list', 'converged'
    """
    from .config import ModelConfig
    results = []
    for n_x in n_x_values:
        cfg_tmp = ModelConfig(
            T_min_keV=0.001, T_max_keV=10.0, n_T=20,
            rho_min=1e-10, rho_max=1e2, n_rho=13,
            n_max=n_max, x_min=1e-2, x_max=30.0,
            n_x_base=n_x, root_tol=1e-10, max_root_iter=200,
        )
        x_base = build_base_x_grid(cfg_tmp)
        x = refine_x_grid_for_thresholds(x_base, T, n_max, const)
        kR = compute_rosseland_mean(T, rho, n_max, x, const)
        results.append(kR)

    converged = False
    if len(results) >= 2:
        frac = abs(results[-1] - results[-2]) / max(abs(results[-2]), 1e-300)
        converged = frac < tol_frac

    return {
        "n_x_list": list(n_x_values),
        "kappa_R_list": results,
        "converged": converged,
    }


def convergence_test_nmax(
    T: float,
    rho: float,
    const: PhysicalConstants,
    x: np.ndarray,
    n_max_values: tuple[int, ...] = (4, 6, 8, 10),
    tol_frac: float = 0.01,
) -> dict[str, object]:
    """
    Test convergence of κ_R with respect to n_max (maximum H shell).

    Parameters
    ----------
    T : float   [K]
    rho : float   [g cm⁻³]
    const : PhysicalConstants
    x : ndarray
        Base x-grid (will not be re-refined for each n_max).
    n_max_values : tuple
        Sequence of n_max values to test.
    tol_frac : float

    Returns
    -------
    dict with keys 'n_max_list', 'kappa_R_list', 'converged'
    """
    results = []
    for n_max in n_max_values:
        kR = compute_rosseland_mean(T, rho, n_max, x, const)
        results.append(kR)

    converged = False
    if len(results) >= 2:
        frac = abs(results[-1] - results[-2]) / max(abs(results[-2]), 1e-300)
        converged = frac < tol_frac

    return {
        "n_max_list": list(n_max_values),
        "kappa_R_list": results,
        "converged": converged,
    }
