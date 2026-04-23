"""
free_free_hminus.py
===================
H⁻ free-free opacity from John (1988), Table 3.

Reference
---------
John, T. L. 1988, A&A, 193, 189–192.

Formula
-------
Absorption coefficient per neutral-H atom per unit electron pressure:

    k_λ^ff(T) = 1e-29 × Σ_{j=1}^{6} θ^{(j+1)/2} × [A_j λ² + B_j + C_j/λ + D_j/λ² + E_j/λ³ + F_j/λ⁴]

where  θ = 5040 / T  (T in K),  λ in µm.

The fit already incorporates the stimulated-emission factor (1 − e^{−hν/k_BT}).
**Do NOT apply a second stimulated-emission correction.**

Volume opacity (cm⁻¹):
    α_{H⁻,ff} = k_λ^ff × n_{H0} × P_e
where  P_e = n_e k_B T  is the electron pressure [dyne cm⁻²].

Mass opacity (cm² g⁻¹):
    κ_{H⁻,ff} = α_{H⁻,ff} / ρ

Domain
------
* Valid temperature range: 1400 K ≤ T ≤ 10 080 K.
* λ > 0.3645 µm  → Table 3a coefficients.
* 0.1823 µm < λ ≤ 0.3645 µm  → Table 3b coefficients.
* λ ≤ 0.1823 µm  → k_λ^ff = 0 (outside fit range).
* Outside temperature range  → k_λ^ff = 0.
"""

from __future__ import annotations

import numpy as np
from .constants import PhysicalConstants


# ---------------------------------------------------------------------------
# John (1988) Table 3 coefficient arrays
# Indices [j-1] for j = 1 … 6.
# Polynomial per j:  A*λ² + B + C/λ + D/λ² + E/λ³ + F/λ⁴
# ---------------------------------------------------------------------------

# Table 3a: λ > 0.3645 µm
_A_3A = np.array([0.0,        2483.3460, -3449.8890,  2200.0400,  -696.2710,   88.2830])
_B_3A = np.array([0.0,         285.8270, -1158.3820,  2427.7190, -1841.4000,  444.5170])
_C_3A = np.array([0.0,       -2054.2910,  8746.5230, -13651.1050, 8624.9700, -1863.8640])
_D_3A = np.array([0.0,        2827.7760, -11485.6320, 16755.5240,-10051.5300, 2095.2880])
_E_3A = np.array([0.0,       -1341.5370,  5303.6090,  -7510.4940,  4400.0670, -901.7880])
_F_3A = np.array([0.0,         208.9520,  -812.9390,   1132.7380,  -655.0200,  132.9850])

# Table 3b: 0.1823 µm < λ ≤ 0.3645 µm
_A_3B = np.array([ 518.1021,  473.2636, -482.2089,  115.5291, 0.0, 0.0])
_B_3B = np.array([-734.8666, 1443.4137, -737.1616,  169.6374, 0.0, 0.0])
_C_3B = np.array([1021.1775,-1977.3395, 1096.8827, -245.6490, 0.0, 0.0])
_D_3B = np.array([-479.0721,  922.3575, -521.1341,  114.2430, 0.0, 0.0])
_E_3B = np.array([  93.1373, -178.9275,  101.7963,  -21.9972, 0.0, 0.0])
_F_3B = np.array([  -6.4285,   12.3600,   -7.0571,    1.5097, 0.0, 0.0])

_T_MIN_K: float = 1400.0
_T_MAX_K: float = 10080.0
_LAM_MIN_MICRON: float = 0.1823   # lower wavelength boundary of Table 3b
_LAM_MID_MICRON: float = 0.3645   # boundary between Table 3a and Table 3b


def _k_lam_ff_scalar(lam_micron: float, theta: float) -> float:
    """
    k_λ^ff for a single (λ, θ) pair.  Returns 0 outside wavelength domain.
    """
    if lam_micron <= _LAM_MIN_MICRON:
        return 0.0
    if lam_micron > _LAM_MID_MICRON:
        A, B, C, D, E, F = _A_3A, _B_3A, _C_3A, _D_3A, _E_3A, _F_3A
    else:
        A, B, C, D, E, F = _A_3B, _B_3B, _C_3B, _D_3B, _E_3B, _F_3B

    lam2 = lam_micron * lam_micron
    ilam = 1.0 / lam_micron
    ilam2 = ilam * ilam
    ilam3 = ilam2 * ilam
    ilam4 = ilam3 * ilam

    result = 0.0
    for j in range(6):  # j = 0..5 → exponent (j+2)/2
        power = theta ** (0.5 * (j + 2))
        poly = (A[j] * lam2 + B[j] + C[j] * ilam
                + D[j] * ilam2 + E[j] * ilam3 + F[j] * ilam4)
        result += power * poly
    return 1e-29 * result


def kappa_ff_Hminus_net(
    nu: float | np.ndarray,
    T: float,
    rho: float,
    n_H0: float,
    n_e: float,
    const: PhysicalConstants,
) -> float | np.ndarray:
    """
    H⁻ free-free mass opacity (stimulated emission already in fit).

    κ_{H⁻,ff} = k_λ^ff(T) × n_{H0} × P_e / ρ

    where  P_e = n_e k_B T  [dyne cm⁻²]  is the electron pressure.

    Parameters
    ----------
    nu    : float or ndarray   photon frequency  [Hz]
    T     : float              temperature        [K]
    rho   : float              mass density       [g cm⁻³]
    n_H0  : float              neutral-H number density  [cm⁻³]
    n_e   : float              electron number density   [cm⁻³]
    const : PhysicalConstants

    Returns
    -------
    kappa : same shape as nu   [cm² g⁻¹]

    Notes
    -----
    * Returns 0 outside T ∈ [1400, 10 080] K or λ ≤ 0.1823 µm.
    * The John (1988) fit already includes the stimulated-emission factor
      (1 − e^{−hν/k_BT}).  No additional correction should be applied.
    """
    scalar = np.ndim(nu) == 0
    nu = np.atleast_1d(np.asarray(nu, dtype=float))

    if T < _T_MIN_K or T > _T_MAX_K:
        result = np.zeros_like(nu)
        return float(result[0]) if scalar else result

    theta = 5040.0 / T
    P_e = n_e * const.k_B * T  # electron pressure [dyne cm⁻²]
    prefactor = n_H0 * P_e / rho  # [cm⁻³ × dyne/cm² / (g/cm³)] = [cm⁻³ × cm²/g × dyne]
    # = n_H0 * P_e / rho  in units: cm⁻³ × (g cm⁻¹ s⁻²) × cm³/g = cm⁻¹ s⁻² × cm³/g
    # k_lam has units cm⁴ dyne⁻¹ so prefactor × k_lam → cm² g⁻¹  ✓

    lam_micron: np.ndarray = (const.c / nu) * 1e4  # cm → µm

    result = np.zeros_like(nu)
    for i, lam in enumerate(lam_micron):
        k_val = _k_lam_ff_scalar(lam, theta)
        result[i] = prefactor * k_val

    return float(result[0]) if scalar else result
