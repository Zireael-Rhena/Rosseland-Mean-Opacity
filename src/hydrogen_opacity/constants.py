"""
constants.py
============
CGS physical constants and model-specific atomic data.

All values in CGS units (g, cm, s, erg, K, statC).
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PhysicalConstants:
    """
    Container for CGS physical constants used throughout the opacity calculation.

    Attributes
    ----------
    k_B : float
        Boltzmann constant [erg K⁻¹]
    h : float
        Planck constant [erg s]
    hbar : float
        Reduced Planck constant ℏ = h / (2π) [erg s]
    c : float
        Speed of light [cm s⁻¹]
    m_e : float
        Electron mass [g]
    m_H : float
        Hydrogen atom mass [g]
    e_cgs : float
        Electron charge in Gaussian CGS [statC = g^{1/2} cm^{3/2} s⁻¹]
    sigma_T : float
        Thomson cross-section [cm²]
    ev_to_erg : float
        Conversion factor: 1 eV in erg
    chi_H_ev : float
        Hydrogen ionization energy [eV]
    chi_Hminus_ev : float
        H⁻ electron affinity [eV]
    lambda0_Hminus_micron : float
        H⁻ bound-free fit threshold wavelength [μm]  (= 1.64 μm per fit)
    sigma_SB : float
        Stefan-Boltzmann constant [erg cm⁻² s⁻¹ K⁻⁴]
    """

    k_B: float
    h: float
    hbar: float
    c: float
    m_e: float
    m_H: float
    e_cgs: float
    sigma_T: float
    ev_to_erg: float
    chi_H_ev: float
    chi_Hminus_ev: float
    lambda0_Hminus_micron: float
    sigma_SB: float


def load_constants() -> PhysicalConstants:
    """
    Return a frozen dataclass of CGS physical constants.

    Values from NIST CODATA 2018 in CGS.

    Returns
    -------
    PhysicalConstants

    Notes
    -----
    * 1 eV = 1.602176634e-12 erg  (exact since 2019 SI redefinition)
    * hbar = h / (2 pi)
    * sigma_T = (8 pi / 3)(e^2 / m_e c^2)^2  in Gaussian CGS
    * lambda0_Hminus_micron = 1.64 μm (hard-coded fit parameter per spec)
    """
    h: float = 6.6260755e-27           # erg s
    hbar: float = h / (2.0 * math.pi)
    k_B: float = 1.380658e-16           # erg K⁻¹
    c: float = 2.99792458e10            # cm s⁻¹
    m_e: float = 9.1093897e-28       # g
    m_p: float = 1.6726231e-24      # g
    m_H: float = m_p + m_e
    e_cgs: float = 4.80321e-10       # statC
    sigma_T: float = 6.65246e-25   # cm²
    ev_to_erg: float = 1.602176634e-12  # erg eV⁻¹
    chi_H_ev: float = 13.6
    chi_Hminus_ev: float = 0.754
    lambda0_Hminus_micron: float = 1.64  # μm (fit parameter)
    sigma_SB: float = 5.6705e-5    # erg cm⁻² s⁻¹ K⁻⁴

    return PhysicalConstants(
        k_B=k_B,
        h=h,
        hbar=hbar,
        c=c,
        m_e=m_e,
        m_H=m_H,
        e_cgs=e_cgs,
        sigma_T=sigma_T,
        ev_to_erg=ev_to_erg,
        chi_H_ev=chi_H_ev,
        chi_Hminus_ev=chi_Hminus_ev,
        lambda0_Hminus_micron=lambda0_Hminus_micron,
        sigma_SB=sigma_SB,
    )
