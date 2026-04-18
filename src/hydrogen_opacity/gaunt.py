"""
gaunt.py
========
Free-free and bound-free Gaunt factors.

Formulas
--------
g_ff(ν, T, Z)  — Karzas & Latter (1961) approximation as specified.
g_bf(ν, T)     — baseline: returns 1.0 everywhere.
"""

import math
import numpy as np
from .constants import PhysicalConstants


def g_ff(
    nu: float | np.ndarray,
    T: float,
    Z: int = 1,
    const: PhysicalConstants | None = None,  # unused; kept for API consistency
) -> float | np.ndarray:
    """
    Free-free velocity-averaged Gaunt factor.

    Formula (exact per spec):
        ν_9 = ν / 10⁹ Hz
        T_4 = T / 10⁴ K
        arg = ν_9 T_4⁻¹ max(0.25,  Z T_4^{−1/2})
        g_ff = ln[ e + exp( 6 − (√3/π) ln(arg) ) ]

    Parameters
    ----------
    nu : float or ndarray   [Hz]
    T : float   [K]
    Z : int     ionic charge (= 1 for protons)
    const : PhysicalConstants  (unused; kept for API uniformity)

    Returns
    -------
    g_ff : same shape as nu  (dimensionless, ≥ 1)

    Notes
    -----
    This approximation is valid over a wide range of ν and T
    and reduces correctly in the high-ν and low-ν limits.
    Reference: Rybicki & Lightman eq. 5.25 / van Hoof et al. (2014) form.
    """
    nu9 = np.asarray(nu, dtype=float) / 1e9
    T4 = T / 1e4
    arg = nu9 / T4 * max(0.25, Z / math.sqrt(T4))
    # Protect against log(0)
    arg = np.where(arg > 0.0, arg, 1e-300)
    inner = 6.0 - (math.sqrt(3.0) / math.pi) * np.log(arg)
    result = np.log(math.e + np.exp(inner))
    # Return scalar if scalar input
    if np.ndim(nu) == 0:
        return float(result)
    return result


def g_bf(
    nu: float | np.ndarray,
    T: float,
) -> float | np.ndarray:
    """
    Bound-free Gaunt factor.

    Baseline implementation: returns 1.0 everywhere.

    Parameters
    ----------
    nu : float or ndarray   [Hz]
    T : float   [K]  (unused in baseline)

    Returns
    -------
    float or ndarray of 1.0  (dimensionless)

    Notes
    -----
    A quantum-mechanical g_bf(n, ν) can be substituted here for
    improved accuracy relative to LANL/TOPS.
    """
    if np.ndim(nu) == 0:
        return 1.0
    return np.ones_like(np.asarray(nu, dtype=float))
