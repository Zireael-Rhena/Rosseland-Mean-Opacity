"""
opacity.py
==========
Assembly of the total monochromatic opacity from individual components.

κ_ν^tot = κ_es + κ_ff + κ_bf,H + κ_bf,H⁻ + κ_ff,H⁻

All values in cm² g⁻¹.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import PhysicalConstants
from .config import ModelOptions
from .eos import EOSState
from .scattering import kappa_es as _kappa_es
from .free_free import kappa_ff_net
from .bound_free_h import kappa_bf_H_net
from .bound_free_hminus import kappa_bf_Hminus_net
from .free_free_hminus import kappa_ff_Hminus_net
from .state import nu_from_x


@dataclass(frozen=True)
class OpacityComponents:
    """
    Per-frequency opacity components evaluated on a common x-grid.

    All arrays have the same shape as the x-grid passed to
    ``monochromatic_opacity``.

    Attributes
    ----------
    kappa_es : ndarray
        Electron scattering opacity (Klein–Nishina corrected) [cm² g⁻¹]
    kappa_ff : ndarray
        Net free-free opacity  [cm² g⁻¹]
    kappa_bf_H : ndarray
        Net neutral-H bound-free opacity  [cm² g⁻¹]
    kappa_bf_Hminus : ndarray
        Net H⁻ bound-free opacity  [cm² g⁻¹]
    kappa_ff_Hminus : ndarray
        H⁻ free-free opacity (stim. em. included in fit)  [cm² g⁻¹]
    kappa_total : ndarray
        Sum of all components  [cm² g⁻¹]
    """

    kappa_es: np.ndarray
    kappa_ff: np.ndarray
    kappa_bf_H: np.ndarray
    kappa_bf_Hminus: np.ndarray
    kappa_ff_Hminus: np.ndarray
    kappa_total: np.ndarray


def monochromatic_opacity(
    x: np.ndarray,
    state: EOSState,
    const: PhysicalConstants,
    opts: ModelOptions | None = None,
) -> OpacityComponents:
    """
    Compute all monochromatic opacity components on a given x-grid.

    x = hν / k_B T  (dimensionless photon energy)

    Parameters
    ----------
    x : ndarray
        Dimensionless photon energy grid (must be > 0).
    state : EOSState
        Solved plasma state at (T, ρ).
    const : PhysicalConstants

    Returns
    -------
    OpacityComponents
        Each component is an ndarray with the same shape as ``x``.

    Notes
    -----
    * Stimulated-emission corrections are embedded in κ_ff, κ_bf, and κ_bf,H⁻ terms.
    * κ_ff,H⁻ uses the John (1988) fit which already includes stimulated emission.
    * κ_es uses the Klein–Nishina cross-section (frequency-dependent).
    """
    if opts is None:
        opts = ModelOptions()
    x = np.asarray(x, dtype=float)
    T = state.T
    rho = state.rho

    nu: np.ndarray = nu_from_x(x, T, const)

    # Electron scattering: Klein–Nishina or Thomson depending on toggle
    if opts.use_kn:
        kappa_es_arr = _kappa_es(nu, state.n_e, rho, const)
    else:
        kappa_es_arr = np.full_like(nu, state.n_e * const.sigma_T / rho)

    # Free-free (e-p) — always included
    kappa_ff_arr = kappa_ff_net(nu, T, rho, state.n_e, state.n_p, const)

    # Neutral-H bound-free: use float n_max_phys for threshold lowering for all non-none modes
    n_max_phys_for_bf = state.n_max_phys_effective if opts.lowering_mode != "none" else None
    kappa_bf_H_arr = kappa_bf_H_net(nu, T, rho, state.level_populations, const,
                                     n_max_phys=n_max_phys_for_bf)

    # H⁻ bound-free — always included
    kappa_bf_Hm_arr = kappa_bf_Hminus_net(nu, T, rho, state.n_Hminus, const)

    # H⁻ free-free (John 1988; stim. em. already in fit)
    if opts.use_ff_hminus:
        kappa_ff_Hm_arr = kappa_ff_Hminus_net(nu, T, rho, state.n_H0, state.n_e, const)
    else:
        kappa_ff_Hm_arr = np.zeros_like(nu)

    kappa_total = (kappa_es_arr + kappa_ff_arr + kappa_bf_H_arr
                   + kappa_bf_Hm_arr + kappa_ff_Hm_arr)

    return OpacityComponents(
        kappa_es=kappa_es_arr,
        kappa_ff=kappa_ff_arr,
        kappa_bf_H=kappa_bf_H_arr,
        kappa_bf_Hminus=kappa_bf_Hm_arr,
        kappa_ff_Hminus=kappa_ff_Hm_arr,
        kappa_total=kappa_total,
    )


def total_opacity_from_components(comp: OpacityComponents) -> np.ndarray:
    """
    Return the total opacity array from an OpacityComponents object.

    Parameters
    ----------
    comp : OpacityComponents

    Returns
    -------
    ndarray   [cm² g⁻¹]
    """
    return comp.kappa_total
