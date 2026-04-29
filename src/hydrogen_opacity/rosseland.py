"""
rosseland.py
============
Rosseland-mean opacity integration.

Definition:
    1/κ_R = ∫ (1/κ_ν^tot) w_R(x) dx  /  ∫ w_R(x) dx

Weight function:
    w_R(x) = x⁴ eˣ / (eˣ − 1)²

Integration variable:  x = hν / k_B T.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from .constants import PhysicalConstants
from .config import ModelConfig, ModelOptions
from .grids import build_base_x_grid, refine_x_grid_for_thresholds
from .eos import solve_eos
from .opacity import monochromatic_opacity, OpacityComponents
from .scattering import kappa_scattering_poutanen2017


def rosseland_weight(x: float | np.ndarray) -> float | np.ndarray:
    """
    Rosseland mean weight function.

    w_R(x) = x⁴ eˣ / (eˣ − 1)²

    Parameters
    ----------
    x : float or ndarray    (x = hν / k_B T > 0)

    Returns
    -------
    w : same shape as x  (dimensionless)

    Notes
    -----
    Numerically stable for large x:  eˣ/(eˣ−1)² → eˣ · e^{−2x} = e^{−x}
    Use the form x⁴ exp(x) / (exp(x) − 1)² directly; for x > ~700 scipy
    will overflow, but the Rosseland integrand is negligible there anyway.
    """
    scalar = np.ndim(x) == 0
    x = np.atleast_1d(np.asarray(x, dtype=float))
    ex = np.exp(np.minimum(x, 700.0))  # cap to avoid overflow
    exm1 = ex - 1.0
    # Protect against (exp(x)−1)→0 at very small x
    exm1 = np.where(exm1 > 0.0, exm1, 1e-300)
    w = x ** 4 * ex / exm1 ** 2
    return float(w[0]) if scalar else w


def rosseland_mean_from_spectrum(
    x: np.ndarray,
    kappa_nu: np.ndarray,
) -> float:
    """
    Compute the Rosseland mean given precomputed κ_ν on an x-grid.

    1/κ_R = ∫ w_R(x) / κ_ν dx  /  ∫ w_R(x) dx

    Parameters
    ----------
    x : ndarray
        x = hν / k_B T grid (sorted, > 0).
    kappa_nu : ndarray
        Total monochromatic opacity on the same x-grid  [cm² g⁻¹].
        Must be strictly positive everywhere (no zeros).

    Returns
    -------
    kappa_R : float   [cm² g⁻¹]

    Notes
    -----
    Integration uses the trapezoidal rule on the (possibly non-uniform) x-grid.
    A small floor is applied to κ_ν to prevent 1/κ_ν from diverging where
    opacity is numerically tiny.
    """
    x = np.asarray(x, dtype=float)
    kappa_nu = np.asarray(kappa_nu, dtype=float)

    # Guard: floor opacity at a physically negligible but finite value
    kappa_floor = 1e-100
    kappa_safe = np.maximum(kappa_nu, kappa_floor)

    w = rosseland_weight(x)
    integrand_num = w / kappa_safe
    integrand_den = w

    numerator = trapezoid(integrand_num, x)
    denominator = trapezoid(integrand_den, x)

    if denominator == 0.0:
        raise ValueError("Rosseland denominator is zero — check x-grid.")

    inv_kR = numerator / denominator
    return 1.0 / inv_kR


def compute_rosseland_mean(
    T: float,
    rho: float,
    n_max: int,
    x: np.ndarray,
    const: PhysicalConstants,
    tol: float = 1e-10,
    opts: ModelOptions | None = None,
) -> float:
    """
    Full pipeline: solve EOS → compute opacity spectrum → integrate Rosseland mean.

    Parameters
    ----------
    T : float     [K]
    rho : float   [g cm⁻³]
    n_max : int
    x : ndarray
        Pre-built (and optionally refined) x-grid.
    const : PhysicalConstants
    tol : float
        EOS root-solver tolerance.
    opts : ModelOptions or None
        Physics toggles.  None → production defaults (all on).

    Returns
    -------
    kappa_R : float   [cm² g⁻¹]
    """
    if opts is None:
        opts = ModelOptions()
    state = solve_eos(T, rho, n_max, const, tol=tol, opts=opts)

    # Poutanen (2017) high-T Compton Rosseland-mean correction.
    # Applied only when: compton_mean_mode=="poutanen2017" AND T_keV >= 2.0
    # AND y_e >= 0.999 (fully ionized).  Falls back to KN spectral otherwise.
    # This is a mean-opacity correction, NOT a monochromatic cross-section.
    if opts.compton_mean_mode == "poutanen2017":
        T_keV = T * const.k_B / (1.0e3 * const.ev_to_erg)
        y_e = state.n_e / state.n_H_tot
        if T_keV >= 2.0 and y_e >= 0.999:
            return kappa_scattering_poutanen2017(T_keV, rho, state.n_e, const)

    comp: OpacityComponents = monochromatic_opacity(x, state, const, opts=opts)
    return rosseland_mean_from_spectrum(x, comp.kappa_total)
