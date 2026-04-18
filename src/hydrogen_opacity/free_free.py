"""
free_free.py
============
Electron-proton (thermal) free-free (bremsstrahlung) absorption opacity.

The NET absorption coefficient already includes the stimulated-emission
factor (1 − e^{−hν/k_BT}).  Do NOT apply this factor again elsewhere.

Formula
-------
α_ff = (4√π e⁶) / (3√3 m_e² h c)
       · sqrt(2 m_e / k_B T)
       · (1 − e^{−hν/k_BT}) / ν³
       · n_e n_p g_ff(ν, T, Z=1)

κ_ff = α_ff / ρ        [cm² g⁻¹]
"""

import math
import numpy as np
from .constants import PhysicalConstants
from .gaunt import g_ff


def alpha_ff_net(
    nu: float | np.ndarray,
    T: float,
    n_e: float,
    n_p: float,
    const: PhysicalConstants,
) -> float | np.ndarray:
    """
    Net free-free absorption coefficient (stimulated emission included).

    α_ff = C_ff · sqrt(1/T) · (1 − e^{−hν/k_BT}) / ν³ · n_e n_p g_ff

    where

        C_ff = (4√π e⁶) / (3√3 m_e² h c) · sqrt(2 m_e / k_B)
             = (4√π) / (3√3) · e⁶ / (m_e² h c) · sqrt(2 m_e / k_B)

    Parameters
    ----------
    nu : float or ndarray   [Hz]
    T  : float              [K]
    n_e : float             electron number density  [cm⁻³]
    n_p : float             proton number density    [cm⁻³]
    const : PhysicalConstants

    Returns
    -------
    alpha_ff : same shape as nu  [cm⁻¹]

    Notes
    -----
    Stimulated emission is already embedded via (1 − exp(−x)).
    Do NOT multiply by (1 − e^{−hν/k_BT}) again.
    """
    scalar = np.ndim(nu) == 0
    nu = np.atleast_1d(np.asarray(nu, dtype=float))

    # Precomputed prefactor (pure constants)
    # C_ff = (4 sqrt(pi) / (3 sqrt(3))) * e^6 / (m_e^2 h c) * sqrt(2 m_e / k_B)
    C_ff: float = (
        (4.0 * math.sqrt(math.pi) / (3.0 * math.sqrt(3.0)))
        * const.e_cgs ** 6
        / (const.m_e ** 2 * const.h * const.c)
        * math.sqrt(2.0 * const.m_e / const.k_B)
    )

    kBT: float = const.k_B * T
    x: np.ndarray = const.h * nu / kBT  # hν / k_BT

    # Stimulated-emission correction
    # Use numerically stable form for small x: (1 − e^{−x}) ≈ x for x ≪ 1
    stim: np.ndarray = -np.expm1(-x)   # = 1 − e^{−x}, more stable

    gaunt: np.ndarray = np.asarray(g_ff(nu, T, Z=1), dtype=float)

    result: np.ndarray = (
        C_ff
        * (1.0 / math.sqrt(T))
        * stim
        / nu ** 3
        * n_e
        * n_p
        * gaunt
    )

    return float(result[0]) if scalar else result


def kappa_ff_net(
    nu: float | np.ndarray,
    T: float,
    rho: float,
    n_e: float,
    n_p: float,
    const: PhysicalConstants,
) -> float | np.ndarray:
    """
    Net free-free mass opacity coefficient.

    κ_ff = α_ff / ρ

    Parameters
    ----------
    nu  : float or ndarray  [Hz]
    T   : float             [K]
    rho : float             [g cm⁻³]
    n_e : float             [cm⁻³]
    n_p : float             [cm⁻³]
    const : PhysicalConstants

    Returns
    -------
    kappa_ff : same shape as nu  [cm² g⁻¹]
    """
    return alpha_ff_net(nu, T, n_e, n_p, const) / rho
