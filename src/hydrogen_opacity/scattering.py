"""
scattering.py
=============
Electron scattering opacity with Klein–Nishina correction.

For x_KN = hν / (m_e c²) → 0 the formula reduces to Thomson scattering.
"""

import numpy as np
from .constants import PhysicalConstants


def sigma_kn(nu: float | np.ndarray, const: PhysicalConstants) -> float | np.ndarray:
    """
    Klein–Nishina total cross-section per electron.

    σ_KN(x) = (3/4) σ_T × {
        (1+x)/x³ × [2x(1+x)/(1+2x) − ln(1+2x)]
        + ln(1+2x)/(2x)
        − (1+3x)/(1+2x)²
    }

    where x = hν / (m_e c²).

    For x < 0.05 a 5-term Taylor series is used to avoid catastrophic
    cancellation in the bracket [2x(1+x)/(1+2x) − ln(1+2x)]:

        σ_KN/σ_T ≈ 1 − 2x + (26/5)x² − (133/10)x³ + (1144/35)x⁴ − …

    Parameters
    ----------
    nu    : float or ndarray   photon frequency  [Hz]
    const : PhysicalConstants

    Returns
    -------
    sigma : same shape as nu   [cm²]
    """
    scalar = np.ndim(nu) == 0
    nu = np.atleast_1d(np.asarray(nu, dtype=float))

    m_e_c2 = const.m_e * const.c * const.c  # erg
    x = const.h * nu / m_e_c2

    sigma = np.empty_like(x)
    low = x < 0.05
    high = ~low

    # Taylor series (Horner form):
    # σ/σ_T = 1 - 2x + (26/5)x² - (133/10)x³ + (1144/35)x⁴
    if np.any(low):
        xl = x[low]
        sigma[low] = const.sigma_T * (
            1.0 + xl * (-2.0 + xl * (26.0/5.0 + xl * (-133.0/10.0
                                      + xl * (1144.0/35.0))))
        )

    if np.any(high):
        xh = x[high]
        one_p2x = 1.0 + 2.0 * xh
        ln_term = np.log1p(2.0 * xh)  # numerically stable
        bracket = 2.0 * xh * (1.0 + xh) / one_p2x - ln_term
        term1 = (1.0 + xh) / xh**3 * bracket
        term2 = ln_term / (2.0 * xh)
        term3 = (1.0 + 3.0 * xh) / one_p2x**2
        sigma[high] = 0.75 * const.sigma_T * (term1 + term2 - term3)

    return float(sigma[0]) if scalar else sigma


def kappa_es(
    nu: float | np.ndarray,
    n_e: float,
    rho: float,
    const: PhysicalConstants,
) -> float | np.ndarray:
    """
    Electron scattering opacity with Klein–Nishina correction.

    κ_es(ν) = n_e σ_KN(ν) / ρ

    Parameters
    ----------
    nu    : float or ndarray   photon frequency  [Hz]
    n_e   : float              electron number density  [cm⁻³]
    rho   : float              mass density             [g cm⁻³]
    const : PhysicalConstants

    Returns
    -------
    kappa : same shape as nu   [cm² g⁻¹]
    """
    return n_e * sigma_kn(nu, const) / rho


# ---------------------------------------------------------------------------
# Poutanen (2017) Compton Rosseland-mean correction
# ---------------------------------------------------------------------------

def lambda_poutanen2017_nondegenerate(
    T_keV: float | np.ndarray,
) -> float | np.ndarray:
    """
    Poutanen (2017) Compton suppression factor for the Rosseland mean.

    Λ(T) = 1 + (T_keV / 39.4)^0.976

    This is a fitting formula for the ratio κ_T / κ_R^Compton in the
    non-degenerate, hot, fully ionized limit, fitted over 2–40 keV.

    This function computes a Rosseland/flux MEAN correction factor, not a
    monochromatic cross-section ratio.  It must be applied only at the final
    mean-opacity level, not inside the frequency-dependent integrand.

    Parameters
    ----------
    T_keV : float or ndarray
        Temperature [keV].  Intended for T_keV >= 2 in this project.

    Returns
    -------
    Lambda : same type/shape as T_keV  (dimensionless, >= 1)

    Notes
    -----
    Reference: Poutanen, J. 2017, ApJ, 835, 119,
               doi:10.3847/1538-4357/835/2/119  (non-degenerate 2–40 keV fit)
    """
    return 1.0 + (T_keV / 39.4) ** 0.976


def kappa_scattering_poutanen2017(
    T_keV: float,
    rho: float,
    n_e: float,
    const: PhysicalConstants,
) -> float:
    """
    Rosseland-mean electron scattering opacity using the Poutanen (2017)
    Compton correction.

    κ_P17 = κ_T / Λ_P17(T)

    where
        κ_T       = n_e σ_T / ρ     (Thomson scattering opacity using EOS n_e)
        Λ_P17(T)  = 1 + (T_keV / 39.4)^0.976

    This is a Rosseland/flux mean correction for Compton scattering.  It is
    NOT a monochromatic cross-section and must NOT be inserted into the
    frequency-dependent opacity integrand.  Apply only at the final
    mean-opacity level in the high-temperature scattering-dominated branch.

    Valid regime:
        - Hot and fully ionized: T_keV >= 2 (H ionization energy 0.0136 keV << T)
        - Non-degenerate electrons: k_B T >> E_F  (satisfied at low densities)
        - Scattering-dominated: absorption opacity negligible
        - Not valid at cold, partially neutral, or H⁻-dominated conditions

    Parameters
    ----------
    T_keV : float
        Temperature [keV].
    rho : float
        Mass density [g cm⁻³].
    n_e : float
        Electron number density from EOS solve [cm⁻³].
        Use the actual EOS value, not a fully-ionized approximation.
    const : PhysicalConstants

    Returns
    -------
    kappa : float  [cm² g⁻¹]

    Notes
    -----
    Reference: Poutanen, J. 2017, ApJ, 835, 119,
               doi:10.3847/1538-4357/835/2/119  (non-degenerate 2–40 keV fit)
    """
    kappa_T = n_e * const.sigma_T / rho
    return kappa_T / lambda_poutanen2017_nondegenerate(T_keV)
