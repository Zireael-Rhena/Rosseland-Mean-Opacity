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

        σ_KN/σ_T ≈ 1 − 2x + (26/5)x² − (133/10)x³ + (1754/105)x⁴ − …

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
    # σ/σ_T = 1 - 2x + (26/5)x² - (133/10)x³ + (1754/105)x⁴
    # Coefficients: 1, -2, 26/5, -133/10, 1754/105
    if np.any(low):
        xl = x[low]
        sigma[low] = const.sigma_T * (
            1.0 + xl * (-2.0 + xl * (26.0/5.0 + xl * (-133.0/10.0
                                      + xl * (1754.0/105.0))))
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
