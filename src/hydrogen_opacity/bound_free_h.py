"""
bound_free_h.py
===============
Neutral-hydrogen bound-free (photoionization) opacity.

Each principal quantum shell n contributes a hydrogenic cross-section
above its ionization threshold  χ_n = 13.6 eV / n².

Formula
-------
For hν ≥ χ_n:

    σ_{n,bf}(ν) = n⁻⁵ · (8π / (3√3)) · (m_e e¹⁰) / (c ℏ³ (hν)³)  · g_bf(ν, T)

Note: ℏ (not h) appears in the denominator, per the project specification.

For hν < χ_n:  σ_{n,bf} = 0.

Net opacity (with stimulated-emission correction):
    κ_bf,H = (1/ρ) Σ_n  n_n σ_{n,bf}(ν)  ·  (1 − e^{−hν/k_BT})
"""

import math
import numpy as np
from .constants import PhysicalConstants
from .gaunt import g_bf


def sigma_bf_hydrogenic_shell(
    nu: float | np.ndarray,
    T: float,
    n: int,
    const: PhysicalConstants,
    n_max_phys: float | None = None,
) -> float | np.ndarray:
    """
    Hydrogenic bound-free cross-section for principal quantum number n.

    σ_{n,bf}(ν) = n⁻⁵ · (8π / 3√3) · (m_e e_cgs¹⁰) / (c ℏ³ (hν)³) · g_bf

    The ionization threshold uses the level-dissolution-lowered energy:
        χ_n_eff = 13.6 (1/n² − 1/n_max_phys²) eV     (if n_max_phys is given)
    If n ≥ n_max_phys the shell is dissolved (chi_n_eff ≤ 0): returns 0.
    If n_max_phys is None the bare hydrogenic threshold 13.6/n² eV is used.

    Parameters
    ----------
    nu       : float or ndarray  [Hz]
    T        : float             [K]  (passed to g_bf; unused in baseline g_bf = 1)
    n        : int               principal quantum number
    const    : PhysicalConstants
    n_max_phys : float or None
        Continuous density-dependent effective n_max for ionization-energy lowering.

    Returns
    -------
    sigma : same shape as nu  [cm²]

    Notes
    -----
    * ℏ³ appears in the denominator (not h³) per explicit project specification.
    * g_bf = 1 in the baseline build (can be upgraded to quantum-mechanical values).
    """
    scalar = np.ndim(nu) == 0
    nu = np.atleast_1d(np.asarray(nu, dtype=float))

    # Ionization threshold with optional level-dissolution lowering
    if n_max_phys is not None:
        chi_n_eff_ev: float = const.chi_H_ev * (1.0 / (n * n) - 1.0 / (n_max_phys * n_max_phys))
    else:
        chi_n_eff_ev = const.chi_H_ev / (n * n)

    chi_n_erg: float = chi_n_eff_ev * const.ev_to_erg
    if chi_n_erg <= 0.0:
        return float(0.0) if scalar else np.zeros_like(nu)
    nu_threshold: float = chi_n_erg / const.h  # threshold frequency

    # Hydrogenic prefactor:
    #   (8π / 3√3) · m_e e^10 / (c ℏ³)
    # Units: [g cm⁻³ s] × ... → [cm² (erg)³] when divided by (hν)³
    prefactor: float = (
        (8.0 * math.pi / (3.0 * math.sqrt(3.0)))
        * const.m_e
        * const.e_cgs ** 10
        / (const.c * const.hbar ** 3)
    )

    h_nu: np.ndarray = const.h * nu
    gaunt: np.ndarray = np.asarray(g_bf(nu, T), dtype=float)

    above_threshold = nu >= nu_threshold
    sigma = np.zeros_like(nu)
    if np.any(above_threshold):
        sigma[above_threshold] = (
            prefactor
            * (n ** (-5))
            / h_nu[above_threshold] ** 3
            * gaunt[above_threshold]
        )

    return float(sigma[0]) if scalar else sigma


def alpha_bf_H_true(
    nu: float | np.ndarray,
    T: float,
    level_populations: np.ndarray,
    const: PhysicalConstants,
    n_max_phys: float | None = None,
) -> float | np.ndarray:
    """
    True (stimulated-emission NOT yet applied) bound-free absorption coefficient
    summed over all H levels n = 1..n_max.

    α_bf,H^true = Σ_n  n_n · σ_{n,bf}(ν)

    Parameters
    ----------
    nu : float or ndarray  [Hz]
    T  : float             [K]
    level_populations : ndarray shape (n_max,)
        n_n for n = 1..n_max  [cm⁻³]
    const : PhysicalConstants
    n_max_phys : float or None
        Continuous density-dependent n_max for ionization-energy lowering.

    Returns
    -------
    alpha : same shape as nu  [cm⁻¹]
    """
    scalar = np.ndim(nu) == 0
    nu = np.atleast_1d(np.asarray(nu, dtype=float))
    n_max = len(level_populations)
    alpha = np.zeros_like(nu)
    for n in range(1, n_max + 1):
        nn = level_populations[n - 1]
        if nn == 0.0:
            continue
        alpha += nn * sigma_bf_hydrogenic_shell(nu, T, n, const, n_max_phys=n_max_phys)
    return float(alpha[0]) if scalar else alpha


def kappa_bf_H_net(
    nu: float | np.ndarray,
    T: float,
    rho: float,
    level_populations: np.ndarray,
    const: PhysicalConstants,
    n_max_phys: float | None = None,
) -> float | np.ndarray:
    """
    Net neutral-H bound-free mass opacity (stimulated emission included).

    κ_bf,H = (α_bf,H^true / ρ)  ·  (1 − e^{−hν/k_BT})

    Parameters
    ----------
    nu    : float or ndarray  [Hz]
    T     : float             [K]
    rho   : float             [g cm⁻³]
    level_populations : ndarray  (n_max,)  [cm⁻³]
    const : PhysicalConstants
    n_max_phys : float or None
        Continuous density-dependent n_max for ionization-energy lowering.

    Returns
    -------
    kappa_bf_H : same shape as nu  [cm² g⁻¹]

    Notes
    -----
    Stimulated-emission correction (1 − e^{−x}) is applied exactly once here,
    after summing the true absorption over all shells.
    """
    scalar = np.ndim(nu) == 0
    nu = np.atleast_1d(np.asarray(nu, dtype=float))

    alpha_true = alpha_bf_H_true(nu, T, level_populations, const, n_max_phys=n_max_phys)
    x = const.h * nu / (const.k_B * T)
    stim = -np.expm1(-x)   # 1 − e^{−x}

    result = alpha_true / rho * stim
    return float(result[0]) if scalar else result
