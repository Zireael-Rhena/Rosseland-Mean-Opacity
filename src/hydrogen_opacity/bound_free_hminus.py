"""
bound_free_hminus.py
====================
H⁻ bound-free absorption opacity.

Empirical cross-section fit (Wishart 1979 / John 1988 form):

    σ_{H⁻,bf}(λ) = 1.53e-16 cm² · λ³ · (1/λ − 1/λ₀)^{3/2}

where λ is in μm and λ₀ = 1.64 μm.

Domain restrictions (must be enforced):
  * hν ≤ 0.754 eV  →  σ = 0  (below ionization threshold)
  * 0.754 eV < hν ≤ 10 eV  →  use the fit
  * hν > 10 eV  →  σ = 0  (outside stated validity range; no extrapolation)

Net opacity with stimulated-emission correction:
    κ_bf,H⁻ = (n_{H⁻} / ρ) · σ_{H⁻,bf}(ν) · (1 − e^{−hν/k_BT})
"""

import numpy as np
from .constants import PhysicalConstants


# Fit normalisation constant [cm²]
_SIGMA0_CM2: float = 1.53e-16


def sigma_bf_Hminus(
    nu: float | np.ndarray,
    const: PhysicalConstants,
) -> float | np.ndarray:
    """
    H⁻ bound-free cross-section (empirical fit).

    σ_{H⁻,bf}(λ) = 1.53e-16 cm² · λ³ · (1/λ − 1/λ₀)^{3/2}

    Domain:
      * hν ≤ 0.754 eV  →  0
      * 0.754 eV < hν ≤ 10 eV  →  fit value
      * hν > 10 eV  →  0  (no extrapolation)

    Parameters
    ----------
    nu : float or ndarray  [Hz]
    const : PhysicalConstants

    Returns
    -------
    sigma : same shape as nu  [cm²]

    Notes
    -----
    λ is converted to μm internally.
    λ₀ = 1.64 μm is the threshold wavelength (stored in const).
    The condition 1/λ < 1/λ₀ (i.e. λ > λ₀) is equivalent to hν < 0.754 eV
    and is already excluded by the lower energy cut.
    """
    scalar = np.ndim(nu) == 0
    nu = np.atleast_1d(np.asarray(nu, dtype=float))

    # Convert ν → hν in eV
    h_nu_ev: np.ndarray = const.h * nu / const.ev_to_erg

    # Convert ν → λ in μm
    lam_micron: np.ndarray = (const.c / nu) * 1e4  # cm → μm

    lam0: float = const.lambda0_Hminus_micron  # 1.64 μm

    # Domain mask: 0.754 eV < hν ≤ 10 eV
    in_range = (h_nu_ev > const.chi_Hminus_ev) & (h_nu_ev <= 10.0)

    sigma = np.zeros_like(nu)
    if np.any(in_range):
        lam_r = lam_micron[in_range]
        inv_diff = 1.0 / lam_r - 1.0 / lam0
        # Safety: clamp to non-negative (should never be negative inside the domain,
        # but guard against floating-point edge cases near the threshold)
        inv_diff = np.maximum(inv_diff, 0.0)
        sigma[in_range] = _SIGMA0_CM2 * lam_r ** 3 * inv_diff ** 1.5

    return float(sigma[0]) if scalar else sigma


def alpha_bf_Hminus_true(
    nu: float | np.ndarray,
    n_Hminus: float,
    const: PhysicalConstants,
) -> float | np.ndarray:
    """
    True H⁻ bound-free absorption coefficient (before stimulated-emission correction).

    α_{H⁻,bf}^true = n_{H⁻} · σ_{H⁻,bf}(ν)

    Parameters
    ----------
    nu : float or ndarray  [Hz]
    n_Hminus : float       H⁻ number density  [cm⁻³]
    const : PhysicalConstants

    Returns
    -------
    alpha : same shape as nu  [cm⁻¹]
    """
    return n_Hminus * sigma_bf_Hminus(nu, const)


def kappa_bf_Hminus_net(
    nu: float | np.ndarray,
    T: float,
    rho: float,
    n_Hminus: float,
    const: PhysicalConstants,
) -> float | np.ndarray:
    """
    Net H⁻ bound-free mass opacity (stimulated emission included).

    κ_{H⁻,bf} = (n_{H⁻} / ρ) · σ_{H⁻,bf}(ν) · (1 − e^{−hν/k_BT})

    Parameters
    ----------
    nu : float or ndarray  [Hz]
    T  : float             [K]
    rho : float            [g cm⁻³]
    n_Hminus : float       [cm⁻³]
    const : PhysicalConstants

    Returns
    -------
    kappa : same shape as nu  [cm² g⁻¹]
    """
    scalar = np.ndim(nu) == 0
    nu = np.atleast_1d(np.asarray(nu, dtype=float))

    alpha_true = alpha_bf_Hminus_true(nu, n_Hminus, const)
    x = const.h * nu / (const.k_B * T)
    stim = -np.expm1(-x)   # 1 − e^{−x}

    result = alpha_true / rho * stim
    return float(result[0]) if scalar else result
