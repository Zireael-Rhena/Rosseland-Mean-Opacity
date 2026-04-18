"""
opacity.py
==========
Assembly of the total monochromatic opacity from individual components.

κ_ν^tot = κ_es + κ_ff + κ_bf,H + κ_bf,H⁻

All values in cm² g⁻¹.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import PhysicalConstants
from .eos import EOSState
from .scattering import kappa_es as _kappa_es
from .free_free import kappa_ff_net
from .bound_free_h import kappa_bf_H_net
from .bound_free_hminus import kappa_bf_Hminus_net
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
        Electron scattering opacity (grey, broadcast to x-grid shape) [cm² g⁻¹]
    kappa_ff : ndarray
        Net free-free opacity  [cm² g⁻¹]
    kappa_bf_H : ndarray
        Net neutral-H bound-free opacity  [cm² g⁻¹]
    kappa_bf_Hminus : ndarray
        Net H⁻ bound-free opacity  [cm² g⁻¹]
    kappa_total : ndarray
        Sum of all components  [cm² g⁻¹]
    """

    kappa_es: np.ndarray
    kappa_ff: np.ndarray
    kappa_bf_H: np.ndarray
    kappa_bf_Hminus: np.ndarray
    kappa_total: np.ndarray


def monochromatic_opacity(
    x: np.ndarray,
    state: EOSState,
    const: PhysicalConstants,
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
    * Stimulated-emission corrections are embedded in κ_ff and κ_bf terms.
    * κ_es is grey (frequency-independent); broadcast to x shape.
    """
    x = np.asarray(x, dtype=float)
    T = state.T
    rho = state.rho

    nu: np.ndarray = nu_from_x(x, T, const)

    # Electron scattering (grey)
    kes_val: float = _kappa_es(state.n_e, rho, const)
    kappa_es_arr = np.full_like(x, kes_val)

    # Free-free
    kappa_ff_arr = kappa_ff_net(nu, T, rho, state.n_e, state.n_p, const)

    # Neutral-H bound-free
    kappa_bf_H_arr = kappa_bf_H_net(nu, T, rho, state.level_populations, const)

    # H⁻ bound-free
    kappa_bf_Hm_arr = kappa_bf_Hminus_net(nu, T, rho, state.n_Hminus, const)

    kappa_total = kappa_es_arr + kappa_ff_arr + kappa_bf_H_arr + kappa_bf_Hm_arr

    return OpacityComponents(
        kappa_es=kappa_es_arr,
        kappa_ff=kappa_ff_arr,
        kappa_bf_H=kappa_bf_H_arr,
        kappa_bf_Hminus=kappa_bf_Hm_arr,
        kappa_total=kappa_total,
    )


def total_opacity_from_components(comp: OpacityComponents) -> np.ndarray:
    """
    Return the total opacity array from an OpacityComponents object.

    Equivalent to ``comp.kappa_total`` but provided as an explicit function.

    Parameters
    ----------
    comp : OpacityComponents

    Returns
    -------
    ndarray   [cm² g⁻¹]
    """
    return comp.kappa_total
