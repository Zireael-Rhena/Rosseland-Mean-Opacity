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
    (1e-7, 6),   # n_eff ≈ 15.99  → floor=15, min(6,15)=6
    (1e-3, 3),   # n_eff ≈  3.45  → floor= 3, min(6, 3)=3
    (1e0,  1),   # n_eff ≈  1.09  → floor= 1, min(6, 1)=1
    (1e2,  1),   # n_eff ≈  0.51  → floor= 0, max(1,min(6,0))=1
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
