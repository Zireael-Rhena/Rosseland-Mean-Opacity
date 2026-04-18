"""
state.py
========
Helper functions converting between frequency/wavelength representations
and principal-quantum-number atomic data.

All results in CGS unless specified otherwise.
"""

import numpy as np
from .constants import PhysicalConstants


def nu_from_x(
    x: float | np.ndarray,
    T: float,
    const: PhysicalConstants,
) -> float | np.ndarray:
    """
    Convert dimensionless photon energy x = hν / k_B T  to frequency ν.

    Parameters
    ----------
    x : array-like
        Dimensionless photon energy.
    T : float
        Temperature [K]
    const : PhysicalConstants

    Returns
    -------
    nu : same shape as x
        Frequency [Hz]

    Formula
    -------
    ν = x k_B T / h
    """
    return x * const.k_B * T / const.h


def lambda_cm_from_x(
    x: float | np.ndarray,
    T: float,
    const: PhysicalConstants,
) -> float | np.ndarray:
    """
    Convert x = hν / k_B T to wavelength in cm.

    Parameters
    ----------
    x : array-like
    T : float   [K]
    const : PhysicalConstants

    Returns
    -------
    wavelength [cm]

    Formula
    -------
    λ = hc / (x k_B T)
    """
    return const.h * const.c / (x * const.k_B * T)


def lambda_micron_from_x(
    x: float | np.ndarray,
    T: float,
    const: PhysicalConstants,
) -> float | np.ndarray:
    """
    Convert x = hν / k_B T to wavelength in microns.

    Parameters
    ----------
    x : array-like
    T : float   [K]
    const : PhysicalConstants

    Returns
    -------
    wavelength [μm]
    """
    return lambda_cm_from_x(x, T, const) * 1e4  # cm → μm


def chi_n_ev(n: int, const: PhysicalConstants) -> float:
    """
    Ionization energy of principal quantum level n  [eV].

    χ_n = 13.6 eV / n²

    Parameters
    ----------
    n : int
        Principal quantum number (≥ 1)
    const : PhysicalConstants

    Returns
    -------
    float  [eV]
    """
    return const.chi_H_ev / (n * n)


def excitation_energy_n_ev(n: int, const: PhysicalConstants) -> float:
    """
    Excitation energy of level n above ground state  [eV].

    E_n^exc = 13.6 eV (1 - 1/n²)

    Parameters
    ----------
    n : int
        Principal quantum number (≥ 1)
    const : PhysicalConstants

    Returns
    -------
    float  [eV]
    """
    return const.chi_H_ev * (1.0 - 1.0 / (n * n))


def degeneracy_n(n: int) -> int:
    """
    Statistical weight (degeneracy) of hydrogen principal level n.

    g_n = 2 n²

    Parameters
    ----------
    n : int

    Returns
    -------
    int
    """
    return 2 * n * n
