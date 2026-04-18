"""
scattering.py
=============
Electron (Thomson) scattering opacity.

Formula
-------
κ_es = n_e σ_T / ρ

Units: cm² g⁻¹
"""

from .constants import PhysicalConstants


def kappa_es(
    n_e: float,
    rho: float,
    const: PhysicalConstants,
) -> float:
    """
    Electron scattering opacity (Thomson scattering).

    κ_es = n_e σ_T / ρ

    This is frequency-independent (grey).

    Parameters
    ----------
    n_e : float   electron number density  [cm⁻³]
    rho : float   mass density             [g cm⁻³]
    const : PhysicalConstants

    Returns
    -------
    float   [cm² g⁻¹]

    Notes
    -----
    * Non-relativistic limit; no Klein-Nishina correction.
    * σ_T = 6.6525 × 10⁻²⁵ cm²
    """
    return n_e * const.sigma_T / rho
