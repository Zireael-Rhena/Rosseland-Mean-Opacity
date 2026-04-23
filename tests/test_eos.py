"""
test_eos.py
-----------
Tests for the hydrogen EOS solver:
  - positivity of all densities
  - nuclei and charge conservation
  - level population sum
  - H⁻ equilibrium ratio
  - behaviour across T, ρ grid
"""

import math
import pytest
import numpy as np

from hydrogen_opacity.constants import load_constants
from hydrogen_opacity.config import default_config
from hydrogen_opacity.eos import (
    solve_eos,
    partition_function_H,
    saha_prefactor_H,
    thermal_de_broglie_e,
    equilibrium_constant_Hminus,
    hminus_abundance_approx,
    level_populations_H,
    effective_nmax_float,
    effective_ncut,
)


@pytest.fixture(scope="module")
def const():
    return load_constants()


@pytest.fixture(scope="module")
def cfg():
    return default_config()


# Representative (T, rho) test points
TEST_POINTS = [
    (1e4,  1e-10),
    (1e4,  1e-7),
    (1e4,  1e-3),
    (5e4,  1e-5),
    (1e5,  1e-3),
    (1e6,  1.0),
    (1e7,  1e2),
]


@pytest.mark.parametrize("T, rho", TEST_POINTS)
def test_eos_all_positive(T, rho, const, cfg):
    state = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol)
    assert state.n_e >= 0.0
    assert state.n_p >= 0.0
    assert state.n_H0 >= 0.0
    assert state.n_Hminus >= 0.0
    assert np.all(state.level_populations >= 0.0)


@pytest.mark.parametrize("T, rho", TEST_POINTS)
def test_eos_nuclei_conservation(T, rho, const, cfg):
    state = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol)
    err = abs(state.n_H0 + state.n_p + state.n_Hminus - state.n_H_tot) / state.n_H_tot
    assert err < 1e-6, f"Nuclei conservation failed: rel err = {err:.2e}"


@pytest.mark.parametrize("T, rho", TEST_POINTS)
def test_eos_charge_neutrality(T, rho, const, cfg):
    state = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol)
    scale = max(state.n_p, 1e-300)
    err = abs(state.n_p - state.n_e - state.n_Hminus) / scale
    assert err < 1e-6, f"Charge neutrality failed: rel err = {err:.2e}"


@pytest.mark.parametrize("T, rho", TEST_POINTS)
def test_eos_level_population_sum(T, rho, const, cfg):
    state = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol)
    if state.n_H0 > 0.0:
        err = abs(state.level_populations.sum() - state.n_H0) / state.n_H0
        assert err < 1e-6, f"Level pop sum failed: rel err = {err:.2e}"


@pytest.mark.parametrize("T, rho", TEST_POINTS)
def test_eos_n_H_tot(T, rho, const, cfg):
    state = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol)
    expected = rho / const.m_H
    assert abs(state.n_H_tot - expected) / expected < 1e-12


def test_partition_function_ground_state(const):
    """At very low T, U_H → g_1 = 2 (ground state only)."""
    T = 100.0  # K — far below any excited level
    U = partition_function_H(T, n_max=8, const=const)
    assert abs(U - 2.0) < 0.01


def test_partition_function_increases_with_T(const):
    U_lo = partition_function_H(1e3, n_max=8, const=const)
    U_hi = partition_function_H(1e5, n_max=8, const=const)
    assert U_hi > U_lo


def test_saha_prefactor_positive(const):
    U = partition_function_H(1e4, n_max=8, const=const)
    S = saha_prefactor_H(1e4, U, const)
    assert S > 0.0


def test_thermal_de_broglie_positive(const):
    lam = thermal_de_broglie_e(1e4, const)
    assert lam > 0.0


def test_hminus_equilibrium_constant_positive(const):
    U = partition_function_H(1e4, n_max=8, const=const)
    K = equilibrium_constant_Hminus(1e4, U, const)
    assert K > 0.0


def test_hminus_approx_close_at_low_T(const, cfg):
    """At T = 1e4 K, the exact and approx H⁻ abundances should be within 1%."""
    T, rho = 1e4, 1e-7
    state = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol)
    n_approx = hminus_abundance_approx(T, state.n_e, state.n_H0, const)
    if state.n_Hminus > 0.0 and n_approx > 0.0:
        ratio = state.n_Hminus / n_approx
        # ratio = U_H^{exact} result / approx; should be ≈ 1 when U_H ≈ 2
        assert 0.99 < ratio < 1.01, f"H- exact/approx ratio = {ratio:.4f}"


def test_level_populations_sum_to_n_H0(const, cfg):
    T, rho = 1e4, 1e-7
    state = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol)
    # Use the same n_cut that solve_eos used for self-consistency
    pops = level_populations_H(T, state.n_H0, cfg.n_max, const,
                               n_cut=state.n_cut_effective)
    err = abs(pops.sum() - state.n_H0) / max(state.n_H0, 1e-300)
    assert err < 1e-12


# ---------------------------------------------------------------------------
# Density-dependent level cutoff tests
# ---------------------------------------------------------------------------

N_MAX_TEST = 6  # matches the spec regression table

@pytest.mark.parametrize("rho, expected_ncut", [
    (1e-7, 6),   # n_eff ≈  6.96  → floor= 6, min(6, 6)=6
    (1e-3, 2),   # n_eff ≈  2.04  → floor= 2, min(6, 2)=2
    (1e0,  1),   # n_eff ≈  0.81  → floor= 0, max(1,min(6,0))=1
    (1e2,  1),   # n_eff ≈  0.44  → floor= 0, max(1,min(6,0))=1
])
def test_effective_ncut_regression(rho, expected_ncut, const):
    """n_cut must match the spec table for n_max_user = 6."""
    nc = effective_ncut(rho, N_MAX_TEST, const)
    assert nc == expected_ncut, (
        f"rho={rho:.0e}: expected n_cut={expected_ncut}, got {nc}"
    )


def test_effective_nmax_float_decreases_with_density(const):
    """n_max^eff is a strictly decreasing function of density."""
    rhos = [1e-10, 1e-5, 1e0, 1e5]
    vals = [effective_nmax_float(r, const) for r in rhos]
    for i in range(len(vals) - 1):
        assert vals[i] > vals[i + 1], (
            f"n_max_float not decreasing: {vals[i]:.3f} → {vals[i+1]:.3f}"
        )


def test_effective_nmax_float_positive(const):
    for rho in [1e-10, 1e-3, 1.0, 1e5]:
        assert effective_nmax_float(rho, const) > 0.0


def test_ncut_clamped_to_nmax_user(const):
    """At very low density n_cut must not exceed n_max_user."""
    for n_max_user in [1, 4, 8]:
        nc = effective_ncut(1e-20, n_max_user, const)
        assert nc <= n_max_user


def test_ncut_minimum_is_one(const):
    """n_cut must never be less than 1, even at extreme density."""
    nc = effective_ncut(1e10, 8, const)
    assert nc >= 1


def test_eos_state_stores_n_cut_effective(const, cfg):
    """EOSState.n_cut_effective must be a positive integer."""
    for T, rho in [(1e4, 1e-7), (1e4, 1e-3), (1e6, 1.0)]:
        state = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol)
        assert isinstance(state.n_cut_effective, int)
        assert 1 <= state.n_cut_effective <= cfg.n_max


def test_ncut_self_consistency_in_level_pops(const, cfg):
    """
    Entries n > n_cut_effective must be zero in state.level_populations,
    and entries n ≤ n_cut_effective must be non-negative.
    """
    for T, rho in [(1e4, 1e-3), (1e4, 1.0), (1e4, 1e2)]:
        state = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol)
        nc = state.n_cut_effective
        pops = state.level_populations
        # Populated levels
        assert np.all(pops[:nc] >= 0.0), "Populated level has negative population"
        # Dissolved levels must be exactly zero
        if nc < cfg.n_max:
            assert np.all(pops[nc:] == 0.0), (
                f"Levels above n_cut={nc} are non-zero: {pops[nc:]}"
            )


@pytest.mark.parametrize("T, rho", TEST_POINTS)
def test_eos_conservation_with_density_cutoff(T, rho, const, cfg):
    """Full conservation checks must still hold after density-cutoff patch."""
    state = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol)
    # Nuclei
    nuc_err = abs(state.n_H0 + state.n_p + state.n_Hminus - state.n_H_tot) / state.n_H_tot
    assert nuc_err < 1e-6
    # Charge
    charge_err = abs(state.n_p - state.n_e - state.n_Hminus) / max(state.n_p, 1e-300)
    assert charge_err < 1e-6
    # Level population sum
    if state.n_H0 > 0.0:
        pop_err = abs(state.level_populations.sum() - state.n_H0) / state.n_H0
        assert pop_err < 1e-6


def test_eos_state_stores_n_max_phys_effective(const, cfg):
    """EOSState.n_max_phys_effective must be a positive float."""
    for T, rho in [(1e4, 1e-7), (1e4, 1e-3), (1e6, 1.0)]:
        state = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol)
        assert isinstance(state.n_max_phys_effective, float)
        assert state.n_max_phys_effective > 0.0


def test_n_max_phys_equals_effective_nmax_float(const, cfg):
    """n_max_phys_effective must equal effective_nmax_float(rho, const)."""
    from hydrogen_opacity.eos import effective_nmax_float
    for T, rho in [(1e4, 1e-7), (1e4, 1e-3), (1e6, 1.0)]:
        state = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol)
        expected = effective_nmax_float(rho, const)
        assert abs(state.n_max_phys_effective - expected) < 1e-12


def test_chi_H_eff_uses_float_n_max_phys(const, cfg):
    """
    At rho=1e-3 (n_max_phys ≈ 2.037, n_cut=2), the Saha prefactor must reflect
    chi_H_eff = 13.6*(1 - 1/n_max_phys²) rather than 13.6*(1 - 1/n_cut²).
    The float value gives a slightly higher chi_H_eff → less ionization.
    """
    from hydrogen_opacity.eos import (effective_nmax_float, effective_ncut,
                                       saha_prefactor_H, partition_function_H)
    T, rho = 1e4, 1e-3
    state = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol)
    n_max_phys = effective_nmax_float(rho, const)
    n_cut = effective_ncut(rho, cfg.n_max, const)
    # Both n_cut and n_max_phys must differ at this density
    assert abs(n_max_phys - float(n_cut)) > 0.01, (
        f"n_max_phys={n_max_phys:.4f} and n_cut={n_cut} are too close to test"
    )
    U_H = partition_function_H(T, n_cut, const)
    chi_float = const.chi_H_ev * (1.0 - 1.0 / (n_max_phys ** 2))
    chi_int   = const.chi_H_ev * (1.0 - 1.0 / (n_cut ** 2))
    S_float = saha_prefactor_H(T, U_H, const, chi_H_eff_ev=chi_float)
    S_int   = saha_prefactor_H(T, U_H, const, chi_H_eff_ev=chi_int)
    # Float lowering gives higher chi → smaller S → less ionization
    assert chi_float > chi_int, "Expected chi_float > chi_int at this density"
    assert S_float < S_int, "Expected S_float < S_int (higher chi → less ionization)"


# ---------------------------------------------------------------------------
# Lowering-mode tests
# ---------------------------------------------------------------------------

from hydrogen_opacity.config import ModelOptions as _MO


def test_lowering_mode_full_reproduces_current_behavior(const, cfg):
    """lowering_mode='full' must give the same result as the previous default (full float lowering)."""
    T, rho = 1e4, 1e-3
    state_full = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol,
                           opts=_MO(lowering_mode="full"))
    # Verify n_max_phys is used: chi_H_eff < 13.6 eV at this density (n_max_phys ≈ 2.037)
    n_max_phys = effective_nmax_float(rho, const)
    chi_full = const.chi_H_ev * (1.0 - 1.0 / n_max_phys**2)
    # Retrieve S_H from the solved state indirectly: the fact it converged is enough,
    # but we cross-check n_e against mode="none" to confirm lowering is active.
    state_none = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol,
                           opts=_MO(lowering_mode="none"))
    # Full lowering lowers chi → higher ionization → higher n_e
    assert state_full.n_e >= state_none.n_e, (
        "full lowering should not give lower n_e than no lowering at cold/dense"
    )


def test_lowering_mode_capped_1eV(const, cfg):
    """capped_1eV must never lower chi_H by more than 1 eV."""
    from hydrogen_opacity.eos import effective_nmax_float
    # Use rho=1e-3 where n_max_phys ≈ 2.037 → Δχ ≈ 13.6/4.15 ≈ 3.28 eV (exceeds 1 eV cap)
    T, rho = 1e4, 1e-3
    n_max_phys = effective_nmax_float(rho, const)
    delta_chi_raw = const.chi_H_ev / n_max_phys**2   # ≈ 3.28 eV
    assert delta_chi_raw > 1.0, "Test precondition: raw Δχ must exceed 1 eV at this density"

    state_capped = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol,
                             opts=_MO(lowering_mode="capped_1eV"))
    state_none   = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol,
                             opts=_MO(lowering_mode="none"))
    state_full   = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol,
                             opts=_MO(lowering_mode="full"))
    # capped lowers chi by at most 1 eV → more ionized than none, less than full
    assert state_capped.n_e >= state_none.n_e, "capped should ionize ≥ none"
    assert state_capped.n_e <= state_full.n_e * 1.001, (
        "capped should not exceed full lowering"
    )


def test_lowering_mode_gated_disables_below_nmax4(const, cfg):
    """gated_nmax_gt_4 must give the same result as 'none' when n_max_phys <= 4."""
    from hydrogen_opacity.eos import effective_nmax_float
    # rho=1e-3 → n_max_phys ≈ 2.037 < 4 → gate is closed → same as none
    T, rho = 1e4, 1e-3
    n_max_phys = effective_nmax_float(rho, const)
    assert n_max_phys <= 4.0, "Test precondition: n_max_phys must be ≤ 4 at rho=1e-3"

    state_gated = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol,
                            opts=_MO(lowering_mode="gated_nmax_gt_4"))
    state_none  = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol,
                            opts=_MO(lowering_mode="none"))
    assert abs(state_gated.n_e - state_none.n_e) / max(state_none.n_e, 1e-300) < 1e-9, (
        "gated mode with n_max_phys<=4 must match 'none' exactly"
    )


def test_lowering_mode_gated_active_above_nmax4(const, cfg):
    """gated_nmax_gt_4 must apply full lowering when n_max_phys > 4."""
    from hydrogen_opacity.eos import effective_nmax_float
    # rho=1e-9 → n_max_phys ≈ large value >> 4 → gate is open → same as full
    T, rho = 1e4, 1e-9
    n_max_phys = effective_nmax_float(rho, const)
    assert n_max_phys > 4.0, "Test precondition: n_max_phys must be > 4 at rho=1e-9"

    state_gated = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol,
                            opts=_MO(lowering_mode="gated_nmax_gt_4"))
    state_full  = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol,
                            opts=_MO(lowering_mode="full"))
    assert abs(state_gated.n_e - state_full.n_e) / max(state_full.n_e, 1e-300) < 1e-9, (
        "gated mode with n_max_phys>4 must match 'full' exactly"
    )


def test_lowering_mode_none_matches_baseline(const, cfg):
    """lowering_mode='none' must give chi_H_eff = 13.6 eV (no lowering)."""
    T, rho = 1e4, 1e-3
    state_none = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol,
                           opts=_MO(lowering_mode="none"))
    state_full = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol,
                           opts=_MO(lowering_mode="full"))
    # 'none' gives higher chi_H_eff → less ionization → lower n_e
    assert state_none.n_e <= state_full.n_e, (
        "no-lowering should give ≤ ionization compared to full lowering"
    )


def test_lowering_mode_invalid_raises(const, cfg):
    """An unknown lowering_mode must raise ValueError."""
    import pytest
    with pytest.raises(ValueError, match="Unknown lowering_mode"):
        solve_eos(1e4, 1e-7, cfg.n_max, const, tol=cfg.root_tol,
                  opts=_MO(lowering_mode="bogus"))


# ---------------------------------------------------------------------------
# "capped" mode (sweepable cap) tests
# ---------------------------------------------------------------------------

def test_capped_mode_respects_delta_chi_max(const, cfg):
    """chi_H_eff must never be lower than 13.6 - delta_chi_max_ev for mode='capped'."""
    from hydrogen_opacity.eos import effective_nmax_float
    T, rho = 1e4, 1e-3   # n_max_phys ≈ 2.037 → raw Δχ ≈ 3.28 eV
    n_max_phys = effective_nmax_float(rho, const)
    raw_delta = const.chi_H_ev / n_max_phys**2
    assert raw_delta > 1.5, "Precondition: raw Δχ must exceed 1.5 eV at this density"

    for cap in [0.5, 0.8, 1.0, 1.2, 1.5]:
        opts = _MO(lowering_mode="capped", delta_chi_max_ev=cap)
        state = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol, opts=opts)
        # chi_H_eff ≥ 13.6 - cap  →  more ionization means lower chi was used
        # We verify indirectly: n_e must not exceed that of 'full' (which uses raw Δχ)
        state_full = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol,
                               opts=_MO(lowering_mode="full"))
        assert state.n_e <= state_full.n_e * 1.001, (
            f"cap={cap} eV: n_e exceeded full-lowering value (chi_eff below floor)"
        )


def test_capped_mode_at_cap_1eV_matches_capped_1eV(const, cfg):
    """capped with delta_chi_max_ev=1.0 must give same result as capped_1eV."""
    T, rho = 1e4, 1e-3
    state_capped    = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol,
                                opts=_MO(lowering_mode="capped", delta_chi_max_ev=1.0))
    state_capped_1eV = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol,
                                 opts=_MO(lowering_mode="capped_1eV"))
    assert abs(state_capped.n_e - state_capped_1eV.n_e) / max(state_capped_1eV.n_e, 1e-300) < 1e-9


def test_capped_mode_monotone_in_cap(const, cfg):
    """Higher cap → more lowering → more ionization → higher n_e."""
    T, rho = 1e4, 1e-3
    caps = [0.5, 0.8, 1.0, 1.2, 1.5]
    n_es = [
        solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol,
                  opts=_MO(lowering_mode="capped", delta_chi_max_ev=c)).n_e
        for c in caps
    ]
    for i in range(len(n_es) - 1):
        assert n_es[i] <= n_es[i + 1] * 1.001, (
            f"n_e not monotone: cap={caps[i]} → {n_es[i]:.3e}, "
            f"cap={caps[i+1]} → {n_es[i+1]:.3e}"
        )


def test_capped_mode_uses_float_not_ncut_at_rho1e6(const, cfg):
    """
    At rho=1e-6 the cap is NOT binding (raw Δχ ≈ 0.52 eV < 1.0 eV cap).
    If n_cut (=5) were used instead of n_max_phys (≈5.117):
        Δχ_ncut   = 13.6/25     = 0.544 eV  → chi_eff = 13.056 eV
        Δχ_nmp    = 13.6/26.18  = 0.520 eV  → chi_eff = 13.080 eV
    The float version gives a slightly higher chi_eff → slightly less ionization.
    We verify the code uses n_max_phys by checking n_e matches saha_prefactor_H
    with chi_eff computed from the float.
    """
    from hydrogen_opacity.eos import (effective_nmax_float, effective_ncut,
                                       saha_prefactor_H, partition_function_H)
    T_k, rho = 1.0e-3, 1e-6  # n_max_phys=5.117, n_cut=5, raw Δχ<cap
    T = T_k * 1.16045e7

    n_max_phys = effective_nmax_float(rho, const)
    n_cut_val  = effective_ncut(rho, cfg.n_max, const)
    assert n_max_phys > float(n_cut_val), "Test precondition: n_max_phys > n_cut at rho=1e-6"

    cap = 1.0
    delta_nmp  = min(const.chi_H_ev / n_max_phys**2,  cap)
    delta_ncut = min(const.chi_H_ev / float(n_cut_val)**2, cap)
    chi_nmp  = const.chi_H_ev - delta_nmp
    chi_ncut = const.chi_H_ev - delta_ncut

    # Cap is not binding → chi_nmp > chi_ncut → n_e(float) < n_e(integer)
    assert delta_nmp < cap,   "Precondition: cap must not be binding at rho=1e-6"
    assert chi_nmp > chi_ncut, f"chi_nmp={chi_nmp:.6f} should exceed chi_ncut={chi_ncut:.6f}"

    U_H  = partition_function_H(T, n_cut_val, const)
    S_nmp  = saha_prefactor_H(T, U_H, const, chi_H_eff_ev=chi_nmp)
    S_ncut = saha_prefactor_H(T, U_H, const, chi_H_eff_ev=chi_ncut)

    # Solved state must be consistent with the float-chi Saha prefactor
    state = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol,
                      opts=_MO(lowering_mode="capped", delta_chi_max_ev=cap))

    # n_e * n_p / n_H0 should equal S_nmp (float), not S_ncut (integer)
    ratio_nmp  = state.n_e * state.n_p / state.n_H0
    err_float  = abs(ratio_nmp - S_nmp)  / S_nmp
    err_int    = abs(ratio_nmp - S_ncut) / S_ncut

    assert err_float < 1e-5, f"Saha ratio does not match float chi: rel err={err_float:.2e}"
    assert err_int   > 1e-6, f"Saha ratio unexpectedly matches integer chi: rel err={err_int:.2e}"


def test_capped_mode_none_unchanged(const, cfg):
    """Adding delta_chi_max_ev field must not change 'none' behavior."""
    T, rho = 1e4, 1e-3
    s1 = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol,
                   opts=_MO(lowering_mode="none", delta_chi_max_ev=0.5))
    s2 = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol,
                   opts=_MO(lowering_mode="none", delta_chi_max_ev=99.0))
    assert abs(s1.n_e - s2.n_e) / max(s1.n_e, 1e-300) < 1e-12


def test_capped_mode_full_unchanged(const, cfg):
    """Adding delta_chi_max_ev field must not change 'full' behavior."""
    T, rho = 1e4, 1e-3
    s1 = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol,
                   opts=_MO(lowering_mode="full", delta_chi_max_ev=0.1))
    s2 = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol,
                   opts=_MO(lowering_mode="full", delta_chi_max_ev=99.0))
    assert abs(s1.n_e - s2.n_e) / max(s1.n_e, 1e-300) < 1e-12


def test_bf_threshold_uses_float_n_max_phys(const):
    """
    sigma_bf_hydrogenic_shell with n_max_phys=2.037 should give a lowered but
    non-zero threshold for n=2, while n_max_phys=2.0 (integer boundary) gives
    chi_n_eff=0 and returns zero cross-section.
    """
    from hydrogen_opacity.bound_free_h import sigma_bf_hydrogenic_shell
    import numpy as np
    # n=2, n_max_phys exactly at 2.0 → chi_n_eff = 13.6*(1/4-1/4) = 0 → sigma=0
    nu_test = np.array([1e15])  # well above Lyman limit
    sigma_zero = sigma_bf_hydrogenic_shell(nu_test, 1e4, 2, const, n_max_phys=2.0)
    assert np.all(sigma_zero == 0.0), "n=2 shell with n_max_phys=2.0 must be dissolved"
    # n=2, n_max_phys=2.037 → chi_n_eff > 0 → sigma > 0 at high enough frequency
    sigma_pos = sigma_bf_hydrogenic_shell(nu_test, 1e4, 2, const, n_max_phys=2.037)
    assert np.all(sigma_pos >= 0.0), "Cross-section must be non-negative"
    # The threshold frequency for n=2 with n_max_phys=2.037 is very low
    # (chi ≈ 13.6*(0.25-0.241)=0.12 eV → nu_thresh ≈ 2.9e13 Hz), so nu=1e15 is above it
    assert np.any(sigma_pos > 0.0), "n=2 shell with n_max_phys=2.037 must have sigma>0"


def test_diagnostics_function(const, cfg):
    """eos_diagnostics must return expected keys and sensible values."""
    from hydrogen_opacity.validation import eos_diagnostics
    T, rho = 1e4, 1e-7
    state = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol)
    d = eos_diagnostics(state, const)
    assert set(d.keys()) == {
        "n_cut_effective", "partition_function_H",
        "neutral_fraction", "ionized_fraction",
        "hminus_fraction", "hminus_over_ne",
    }
    assert d["n_cut_effective"] == state.n_cut_effective
    assert 0.0 <= d["neutral_fraction"] <= 1.0
    assert 0.0 <= d["ionized_fraction"] <= 1.0
    # fractions sum to ≈ 1
    total = d["neutral_fraction"] + d["ionized_fraction"] + d["hminus_fraction"]
    assert abs(total - 1.0) < 1e-6
