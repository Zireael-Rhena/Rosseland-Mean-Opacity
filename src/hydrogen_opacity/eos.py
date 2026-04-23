"""
eos.py
======
Equation of State for pure hydrogen gas in LTE.

Solves for  n_{H0},  n_p,  n_e,  n_{H⁻}  at given (T, ρ)
via a 1-D root-find in the electron number density n_e.

All quantities in CGS.  Energies in erg internally; eV only in docstrings.
"""

from __future__ import annotations

import math
import numpy as np
from scipy.optimize import brentq
from dataclasses import dataclass

from .constants import PhysicalConstants
from .config import ModelOptions


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EOSState:
    """
    Thermodynamic state of the pure-hydrogen plasma.

    Attributes
    ----------
    T : float
        Temperature [K]
    rho : float
        Mass density [g cm⁻³]
    n_H_tot : float
        Total hydrogen nucleus number density [cm⁻³]
    n_e : float
        Electron number density [cm⁻³]
    n_p : float
        Proton number density [cm⁻³]
    n_H0 : float
        Neutral hydrogen number density [cm⁻³]
    n_Hminus : float
        H⁻ number density [cm⁻³]
    level_populations : ndarray shape (n_max,)
        n_n  for n = 1..n_max   [cm⁻³]
        Entries for n > n_cut_effective are zero.
    partition_function_H : float
        U_H(T, rho)  truncated at n_cut_effective  (dimensionless)
    n_cut_effective : int
        Density-dependent effective level cutoff actually used in this solve.
    n_max_phys_effective : float
        Continuous float version of the density-dependent n_max, used for
        ionization-energy lowering (Saha and H bf thresholds).
    """

    T: float
    rho: float
    n_H_tot: float
    n_e: float
    n_p: float
    n_H0: float
    n_Hminus: float
    level_populations: np.ndarray
    partition_function_H: float
    n_cut_effective: int
    n_max_phys_effective: float


# ---------------------------------------------------------------------------
# Atomic / statistical-mechanics helpers
# ---------------------------------------------------------------------------

def partition_function_H(T: float, n_max: int, const: PhysicalConstants) -> float:
    """
    Neutral-hydrogen partition function.

    U_H(T) = Σ_{n=1}^{n_max}  2n²  exp(−E_n^exc / k_B T)

    where  E_n^exc = 13.6 eV (1 − 1/n²).

    Parameters
    ----------
    T : float   [K]
    n_max : int
    const : PhysicalConstants

    Returns
    -------
    float  (dimensionless, ≥ 1)
    """
    kBT: float = const.k_B * T
    U: float = 0.0
    chi_H_erg: float = const.chi_H_ev * const.ev_to_erg
    for n in range(1, n_max + 1):
        gn = 2 * n * n
        E_exc_erg = chi_H_erg * (1.0 - 1.0 / (n * n))
        U += gn * math.exp(-E_exc_erg / kBT)
    return U


def saha_prefactor_H(
    T: float,
    U_H: float,
    const: PhysicalConstants,
    chi_H_eff_ev: float | None = None,
) -> float:
    """
    Hydrogen Saha ionization coefficient S_H(T).

    n_e n_p / n_{H0} = S_H(T)

    S_H(T) = (2π m_e k_B T / h²)^{3/2}  ·  (2 / U_H)  ·  exp(−χ_H_eff / k_B T)

    Parameters
    ----------
    T : float   [K]
    U_H : float
        Neutral-H partition function.
    const : PhysicalConstants
    chi_H_eff_ev : float or None
        Effective ionization energy [eV] after level-dissolution lowering.
        If None, uses const.chi_H_ev = 13.6 eV (no lowering).

    Returns
    -------
    float   [cm⁻³]
    """
    kBT: float = const.k_B * T
    prefactor: float = (2.0 * math.pi * const.m_e * kBT / (const.h ** 2)) ** 1.5
    chi_ev = chi_H_eff_ev if chi_H_eff_ev is not None else const.chi_H_ev
    chi_erg: float = chi_ev * const.ev_to_erg
    return prefactor * (2.0 / U_H) * math.exp(-chi_erg / kBT)


def thermal_de_broglie_e(T: float, const: PhysicalConstants) -> float:
    """
    Thermal de Broglie wavelength of the electron.

    λ_{th,e} = h / sqrt(2π m_e k_B T)

    Parameters
    ----------
    T : float   [K]
    const : PhysicalConstants

    Returns
    -------
    float   [cm]
    """
    return const.h / math.sqrt(2.0 * math.pi * const.m_e * const.k_B * T)


def equilibrium_constant_Hminus(
    T: float,
    U_H: float,
    const: PhysicalConstants,
) -> float:
    """
    Equilibrium constant for H⁻ formation:  n_{H⁻} = K_{H⁻}(T) n_{H0} n_e.

    K_{H⁻}(T) = (1 / (2 U_H))  ·  λ_{th,e}³  ·  exp(χ_{H⁻} / k_B T)

    Parameters
    ----------
    T : float   [K]
    U_H : float
    const : PhysicalConstants

    Returns
    -------
    float   [cm³]
    """
    lam = thermal_de_broglie_e(T, const)
    chi_Hminus_erg: float = const.chi_Hminus_ev * const.ev_to_erg
    return (1.0 / (2.0 * U_H)) * lam ** 3 * math.exp(chi_Hminus_erg / (const.k_B * T))


def hminus_abundance_approx(
    T: float,
    n_e: float,
    n_H0: float,
    const: PhysicalConstants,
) -> float:
    """
    Low-temperature diagnostic approximation for n_{H⁻}.

    n_{H⁻}^approx = (1/4) n_{H0} n_e λ_{th,e}³ exp(χ_{H⁻} / k_B T)

    This approximates U_H ≈ 2 (ground state only).
    Use only for diagnostic comparison, not as the main solver.

    Parameters
    ----------
    T : float   [K]
    n_e : float   [cm⁻³]
    n_H0 : float  [cm⁻³]
    const : PhysicalConstants

    Returns
    -------
    float   [cm⁻³]
    """
    lam = thermal_de_broglie_e(T, const)
    chi_Hminus_erg = const.chi_Hminus_ev * const.ev_to_erg
    return 0.25 * n_H0 * n_e * lam ** 3 * math.exp(chi_Hminus_erg / (const.k_B * T))


def effective_nmax_float(rho: float, const: PhysicalConstants) -> float:
    """
    Density-dependent effective maximum principal quantum number (float).

    Proxy for pressure ionization / level dissolution (Hummer–Mihalas style):

        n_max^phys ≈ 12 (n_H / 10^15 cm^{-3})^{-2/15}

    where  n_H = rho / m_H.

    Parameters
    ----------
    rho : float   [g cm⁻³]
    const : PhysicalConstants

    Returns
    -------
    float  (may be < 1)
    """
    n_H: float = rho / const.m_H
    return 12.0 * (n_H / 1e15) ** (-2.0 / 15.0)


def effective_ncut(rho: float, n_max_user: int, const: PhysicalConstants) -> int:
    """
    Density-dependent effective level cutoff (integer), clamped to [1, n_max_user].

        n_cut = max(1, min(n_max_user, floor(n_max^eff(rho))))

    Parameters
    ----------
    rho : float       [g cm⁻³]
    n_max_user : int  user-configured maximum principal quantum number
    const : PhysicalConstants

    Returns
    -------
    int  in [1, n_max_user]
    """
    n_eff: float = effective_nmax_float(rho, const)
    return max(1, min(n_max_user, int(math.floor(n_eff))))


def level_populations_H(
    T: float,
    n_H0: float,
    n_max: int,
    const: PhysicalConstants,
    n_cut: int | None = None,
) -> np.ndarray:
    """
    Level populations of neutral hydrogen for n = 1..n_max.

    For n ≤ n_cut:
        n_n = n_{H0} · (2n² exp(−E_n^exc / k_B T)) / U_H(T, n_cut)

    For n > n_cut:
        n_n = 0   (dissolved levels above density cutoff)

    The output array always has shape (n_max,) to keep the downstream API
    unchanged.  U_H is computed using n_cut so the populations are normalised
    correctly, and their sum equals n_{H0}.

    Parameters
    ----------
    T : float   [K]
    n_H0 : float   [cm⁻³]
    n_max : int    length of the returned array
    const : PhysicalConstants
    n_cut : int or None
        Effective level cutoff.  If None, use n_max (backward-compatible).

    Returns
    -------
    pops : ndarray shape (n_max,)
        n_n  for n = 1..n_max  [cm⁻³]  (zero for n > n_cut)
    """
    if n_cut is None:
        n_cut = n_max
    n_cut = max(1, min(n_max, n_cut))  # safety clamp
    U_H: float = partition_function_H(T, n_cut, const)
    kBT: float = const.k_B * T
    chi_H_erg: float = const.chi_H_ev * const.ev_to_erg
    pops = np.zeros(n_max, dtype=float)
    for n in range(1, n_cut + 1):
        gn = 2 * n * n
        E_exc_erg = chi_H_erg * (1.0 - 1.0 / (n * n))
        pops[n - 1] = n_H0 * gn * math.exp(-E_exc_erg / kBT) / U_H
    return pops


# ---------------------------------------------------------------------------
# 1-D root solver
# ---------------------------------------------------------------------------

def solve_eos(
    T: float,
    rho: float,
    n_max: int,
    const: PhysicalConstants,
    tol: float = 1e-10,
    opts: ModelOptions | None = None,
) -> EOSState:
    """
    Solve the LTE equation of state for a pure hydrogen gas.

    Variables solved: n_e (electron number density, [cm⁻³]).
    All other densities follow algebraically.

    Conservation equations
    ----------------------
    (1)  n_{H0} + n_p + n_{H⁻} = n_{H,tot}   (nuclei)
    (2)  n_p = n_e + n_{H⁻}                   (charge)
    (3)  n_e n_p / n_{H0} = S_H(T)            (Saha)
    (4)  n_{H⁻} = K_{H⁻}(T,ρ) n_{H0} n_e     (H⁻ equilibrium)

    From (2): n_p = n_e + n_{H⁻}
    From (3): n_{H0} = n_e n_p / S_H
    Substitute into (1):  residual(n_e) = 0.

    A density-dependent level cutoff n_cut(ρ) is applied self-consistently
    to U_H, S_H, K_{H⁻}, and the level populations (see effective_ncut).

    Parameters
    ----------
    T : float   [K]
    rho : float   [g cm⁻³]
    n_max : int
    const : PhysicalConstants
    tol : float
        Absolute tolerance for Brent root solve.

    Returns
    -------
    EOSState
    """
    if opts is None:
        opts = ModelOptions()
    n_H_tot: float = rho / const.m_H
    # Density-dependent level cutoff — applied self-consistently to U_H,
    # the Saha prefactor, the H⁻ equilibrium constant, and level populations.
    n_cut: int = effective_ncut(rho, n_max, const)          # integer: partition fn / level-pop truncation only
    n_max_phys: float = effective_nmax_float(rho, const)    # float: all ionization-energy lowering
    U_H: float = partition_function_H(T, n_cut, const)
    mode: str = opts.lowering_mode
    if mode == "none":
        chi_H_eff_ev: float = const.chi_H_ev
    elif mode == "full":
        chi_H_eff_ev = const.chi_H_ev * (1.0 - 1.0 / (n_max_phys * n_max_phys))
    elif mode == "capped_1eV":
        delta_chi = const.chi_H_ev / (n_max_phys * n_max_phys)
        chi_H_eff_ev = const.chi_H_ev - min(delta_chi, 1.0)
    elif mode == "capped":
        delta_chi = const.chi_H_ev / (n_max_phys * n_max_phys)
        chi_H_eff_ev = const.chi_H_ev - min(delta_chi, opts.delta_chi_max_ev)
    elif mode == "gated_nmax_gt_4":
        if n_max_phys > 4.0:
            chi_H_eff_ev = const.chi_H_ev * (1.0 - 1.0 / (n_max_phys * n_max_phys))
        else:
            chi_H_eff_ev = const.chi_H_ev
    else:
        raise ValueError(f"Unknown lowering_mode {mode!r}; "
                         "choose 'none', 'full', 'capped', 'capped_1eV', or 'gated_nmax_gt_4'")
    S_H: float = saha_prefactor_H(T, U_H, const, chi_H_eff_ev=chi_H_eff_ev)
    K_Hm: float = equilibrium_constant_Hminus(T, U_H, const)

    def residual(n_e: float) -> float:
        """
        residual = n_{H0} + n_p + n_{H⁻} − n_{H,tot}

        where n_p = n_e + n_{H⁻},  n_{H⁻} = K_Hm n_{H0} n_e,
              n_{H0} = n_e n_p / S_H  (from Saha)

        Derivation step:
          n_{H0}(1 + K_Hm n_e) = n_e n_p / S_H   →  n_p = n_e + K_Hm n_{H0} n_e
        Substituting n_{H0} = n_e n_p / S_H:
          n_p  = n_e + K_Hm · (n_e n_p / S_H) · n_e
          n_p (1 − K_Hm n_e² / S_H) = n_e
          n_p = n_e / (1 − K_Hm n_e² / S_H)    [if denominator > 0]

        Then n_{H0} = n_e n_p / S_H,  n_{H⁻} = K_Hm n_{H0} n_e.
        """
        denom = 1.0 - K_Hm * n_e * n_e / S_H
        if denom <= 0.0:
            # unphysical: H⁻ would dominate; n_p would diverge
            return 1.0
        n_p = n_e / denom
        n_H0 = n_e * n_p / S_H
        n_Hm = K_Hm * n_H0 * n_e
        return n_H0 + n_p + n_Hm - n_H_tot

    # Bracket n_e in (0, n_H_tot]
    # Lower bound: tiny positive (fully neutral limit gives residual < 0)
    n_e_lo: float = tol
    # Upper bound: fully ionized (n_e = n_H_tot, all protons, no H⁻)
    n_e_hi: float = n_H_tot

    # Evaluate at endpoints
    f_lo = residual(n_e_lo)
    f_hi = residual(n_e_hi)

    if f_lo * f_hi > 0.0:
        # Same sign — try to handle nearly-fully-neutral regime
        # In fully neutral limit n_e → 0, residual → -n_H_tot < 0
        # In fully ionized limit n_e = n_H_tot, residual could be > 0
        # If both negative: gas is very weakly ionized; use lo = tol, hi adjusted
        if f_lo < 0.0 and f_hi < 0.0:
            # Very low ionization: the bracket doesn't span a root in [tol, n_H_tot]
            # This shouldn't happen for normal H, but as a fallback return tiny n_e
            n_e_sol = tol
        else:
            n_e_sol = n_e_hi
    else:
        n_e_sol = brentq(residual, n_e_lo, n_e_hi, xtol=tol, rtol=tol, maxiter=200)

    # Reconstruct all densities from n_e_sol
    denom_sol = 1.0 - K_Hm * n_e_sol * n_e_sol / S_H
    denom_sol = max(denom_sol, 1e-300)  # guard against divide-by-zero
    n_p_sol = n_e_sol / denom_sol
    n_H0_sol = n_e_sol * n_p_sol / S_H
    n_Hm_sol = K_Hm * n_H0_sol * n_e_sol

    pops = level_populations_H(T, n_H0_sol, n_max, const, n_cut=n_cut)

    return EOSState(
        T=T,
        rho=rho,
        n_H_tot=n_H_tot,
        n_e=n_e_sol,
        n_p=n_p_sol,
        n_H0=n_H0_sol,
        n_Hminus=n_Hm_sol,
        level_populations=pops,
        partition_function_H=U_H,
        n_cut_effective=n_cut,
        n_max_phys_effective=n_max_phys,
    )
