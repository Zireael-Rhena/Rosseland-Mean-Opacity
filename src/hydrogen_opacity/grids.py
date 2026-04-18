"""
grids.py
========
Build temperature, density, and spectral x-grids.

The spectral grid is refined around Rosseland-sensitive regions and
opacity thresholds to ensure accurate numerical integration.
"""

import numpy as np
from .constants import PhysicalConstants
from .config import ModelConfig

# 1 keV → K  (1 keV = 1.16045e7 K)
_KEV_TO_K: float = 1.16045e7


def keV_to_K(T_keV: float, const: PhysicalConstants) -> float:
    """
    Convert temperature from keV to Kelvin.

    1 keV = 1.16045 × 10⁷ K.

    Parameters
    ----------
    T_keV : float   [keV]
    const : PhysicalConstants   (unused; kept for API uniformity)

    Returns
    -------
    float   [K]
    """
    return T_keV * _KEV_TO_K


def build_temperature_grid(
    cfg: ModelConfig,
    const: PhysicalConstants,
) -> np.ndarray:
    """
    Build a log-spaced temperature grid in Kelvin.

    Parameters
    ----------
    cfg : ModelConfig
    const : PhysicalConstants

    Returns
    -------
    T_grid : ndarray shape (n_T,)   [K]
    """
    T_min_K = keV_to_K(cfg.T_min_keV, const)
    T_max_K = keV_to_K(cfg.T_max_keV, const)
    return np.logspace(np.log10(T_min_K), np.log10(T_max_K), cfg.n_T)


def build_density_grid(cfg: ModelConfig) -> np.ndarray:
    """
    Build a log-spaced density grid.

    Parameters
    ----------
    cfg : ModelConfig

    Returns
    -------
    rho_grid : ndarray shape (n_rho,)   [g cm⁻³]
    """
    return np.logspace(np.log10(cfg.rho_min), np.log10(cfg.rho_max), cfg.n_rho)


def build_base_x_grid(cfg: ModelConfig) -> np.ndarray:
    """
    Build the base log-spaced x = hν / k_B T grid.

    Parameters
    ----------
    cfg : ModelConfig

    Returns
    -------
    x_base : ndarray shape (n_x_base,)
    """
    return np.logspace(np.log10(cfg.x_min), np.log10(cfg.x_max), cfg.n_x_base)


def refine_x_grid_for_thresholds(
    x_base: np.ndarray,
    T: float,
    n_max: int,
    const: PhysicalConstants,
) -> np.ndarray:
    """
    Return a refined x-grid with dense points near opacity thresholds.

    Added refinement regions:
      1. Rosseland core  x ∈ [0.5, 15]  — highest weight region
      2. Neutral-H ionization thresholds  x_n = χ_n / (k_B T)  for n=1..n_max
      3. H⁻ bound-free lower threshold   x_{H⁻} = 0.754 eV / (k_B T)
      4. H⁻ bound-free upper cutoff      x_{H⁻,max} = 10 eV / (k_B T)

    Parameters
    ----------
    x_base : ndarray
        Base x-grid (sorted).
    T : float   [K]
    n_max : int
        Maximum principal quantum number for H.
    const : PhysicalConstants

    Returns
    -------
    x_refined : ndarray  (sorted, unique, same domain as x_base)
    """
    x_lo: float = float(x_base[0])
    x_hi: float = float(x_base[-1])
    kBT_ev: float = const.k_B * T / const.ev_to_erg

    extra: list[float] = list(np.linspace(0.5, 15.0, 150))

    def _bracket(x_thresh: float) -> None:
        """Add 5 points just below and above x_thresh if inside domain."""
        if x_lo <= x_thresh <= x_hi:
            dx = x_thresh * 1e-3
            lo_start = max(x_lo, x_thresh - 20.0 * dx)
            hi_end = min(x_hi, x_thresh + 20.0 * dx)
            extra.extend(np.linspace(lo_start, x_thresh - dx, 5).tolist())
            extra.extend(np.linspace(x_thresh + dx, hi_end, 5).tolist())

    # Neutral-H thresholds
    for n in range(1, n_max + 1):
        _bracket(const.chi_H_ev / (n * n) / kBT_ev)

    # H⁻ thresholds
    _bracket(const.chi_Hminus_ev / kBT_ev)
    _bracket(10.0 / kBT_ev)

    combined = np.concatenate([x_base, np.asarray(extra, dtype=float)])
    combined = combined[(combined >= x_lo) & (combined <= x_hi)]
    return np.unique(combined)
