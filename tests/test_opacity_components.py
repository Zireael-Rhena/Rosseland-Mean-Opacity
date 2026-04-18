"""
test_opacity_components.py
--------------------------
Tests for individual opacity components:
  - Gaunt factors positive
  - electron scattering non-negative and scaling
  - free-free sign and Kramers scaling
  - neutral-H bound-free threshold behavior
  - H⁻ bound-free domain restriction
  - all components non-negative
"""

import pytest
import numpy as np

from hydrogen_opacity.constants import load_constants
from hydrogen_opacity.config import default_config
from hydrogen_opacity.eos import solve_eos
from hydrogen_opacity.gaunt import g_ff, g_bf
from hydrogen_opacity.scattering import kappa_es
from hydrogen_opacity.free_free import alpha_ff_net, kappa_ff_net
from hydrogen_opacity.bound_free_h import sigma_bf_hydrogenic_shell, kappa_bf_H_net, alpha_bf_H_true
from hydrogen_opacity.bound_free_hminus import sigma_bf_Hminus, kappa_bf_Hminus_net
from hydrogen_opacity.grids import build_base_x_grid, refine_x_grid_for_thresholds
from hydrogen_opacity.opacity import monochromatic_opacity
from hydrogen_opacity.validation import check_opacity_nonnegative, check_threshold_behavior
from hydrogen_opacity.state import nu_from_x


@pytest.fixture(scope="module")
def const():
    return load_constants()


@pytest.fixture(scope="module")
def cfg():
    return default_config()


@pytest.fixture(scope="module")
def state_1e4(const, cfg):
    return solve_eos(1e4, 1e-7, cfg.n_max, const, tol=cfg.root_tol)


@pytest.fixture(scope="module")
def state_1e5(const, cfg):
    return solve_eos(1e5, 1e-3, cfg.n_max, const, tol=cfg.root_tol)


class TestGauntFactors:
    def test_g_ff_positive_scalar(self, const):
        val = g_ff(1e14, 1e4, Z=1)
        assert val > 0.0

    def test_g_ff_positive_array(self, const):
        nu = np.logspace(12, 17, 50)
        vals = g_ff(nu, 1e4, Z=1)
        assert np.all(vals > 0.0)

    def test_g_ff_near_1_at_high_nu(self, const):
        # At very high frequency, g_ff → ~1
        val = g_ff(1e17, 1e4, Z=1)
        assert 0.8 < val < 2.0

    def test_g_bf_returns_1(self, const):
        nu = np.array([1e14, 1e15])
        vals = g_bf(nu, 1e4)
        assert np.all(vals == 1.0)

    def test_g_bf_scalar(self):
        assert g_bf(1e14, 1e4) == 1.0


class TestElectronScattering:
    def test_kappa_es_positive(self, state_1e4, const):
        val = kappa_es(state_1e4.n_e, state_1e4.rho, const)
        assert val > 0.0

    def test_kappa_es_scales_with_ne(self, const):
        val1 = kappa_es(1e15, 1e-7, const)
        val2 = kappa_es(2e15, 1e-7, const)
        assert abs(val2 / val1 - 2.0) < 1e-10

    def test_kappa_es_fully_ionized_limit(self, const):
        # For fully ionized H: n_e = rho/m_H
        rho = 1e-3
        n_e = rho / const.m_H
        val = kappa_es(n_e, rho, const)
        expected = const.sigma_T / const.m_H
        assert abs(val - expected) / expected < 1e-10


class TestFreeFree:
    def test_alpha_ff_positive(self, state_1e4, const):
        nu = np.logspace(13, 16, 10)
        alpha = alpha_ff_net(nu, 1e4, state_1e4.n_e, state_1e4.n_p, const)
        assert np.all(alpha >= 0.0)

    def test_kappa_ff_positive(self, state_1e4, const):
        nu = np.logspace(13, 16, 10)
        kff = kappa_ff_net(nu, 1e4, state_1e4.rho, state_1e4.n_e, state_1e4.n_p, const)
        assert np.all(kff >= 0.0)

    def test_kappa_ff_decreases_with_nu(self, state_1e4, const):
        # At high enough ν, ff should be decreasing  (Kramers ∝ ν⁻³ modulo stim)
        nu_lo = 1e14
        nu_hi = 1e15
        kff_lo = kappa_ff_net(nu_lo, 1e4, state_1e4.rho, state_1e4.n_e, state_1e4.n_p, const)
        kff_hi = kappa_ff_net(nu_hi, 1e4, state_1e4.rho, state_1e4.n_e, state_1e4.n_p, const)
        assert kff_lo > kff_hi

    def test_ff_small_at_xray_relative(self, state_1e4, const):
        # At ν = 1e17 Hz, ff should be much smaller than at ν = 1e14 Hz
        # (Kramers ∝ ν^{-3}: ratio ~ (1e17/1e14)^3 = 1e9)
        kff_lo = kappa_ff_net(1e14, 1e4, state_1e4.rho, state_1e4.n_e, state_1e4.n_p, const)
        kff_hi = kappa_ff_net(1e17, 1e4, state_1e4.rho, state_1e4.n_e, state_1e4.n_p, const)
        assert kff_hi < kff_lo * 1e-3


class TestBoundFreeH:
    def test_sigma_zero_below_n1_threshold(self, const):
        nu_thresh = const.chi_H_ev * const.ev_to_erg / (const.h * 1.0)  # n=1
        nu_below = np.array([0.5 * nu_thresh, 0.9 * nu_thresh])
        sigma = sigma_bf_hydrogenic_shell(nu_below, 1e4, 1, const)
        assert np.all(sigma == 0.0)

    def test_sigma_positive_above_n1_threshold(self, const):
        nu_thresh = const.chi_H_ev * const.ev_to_erg / const.h
        nu_above = np.array([1.01 * nu_thresh, 2.0 * nu_thresh])
        sigma = sigma_bf_hydrogenic_shell(nu_above, 1e4, 1, const)
        assert np.all(sigma > 0.0)

    def test_sigma_zero_below_n2_threshold(self, const):
        # n=2 threshold = 13.6/4 = 3.4 eV
        nu_thresh_2 = (const.chi_H_ev / 4.0) * const.ev_to_erg / const.h
        nu_below = np.array([0.5 * nu_thresh_2])
        sigma = sigma_bf_hydrogenic_shell(nu_below, 1e4, 2, const)
        assert float(sigma[0]) == 0.0

    def test_n3_threshold_lower_than_n1(self, const):
        nu_1 = const.chi_H_ev * const.ev_to_erg / const.h
        nu_3 = (const.chi_H_ev / 9.0) * const.ev_to_erg / const.h
        assert nu_3 < nu_1

    def test_hbar_in_denominator_gives_correct_magnitude(self, const):
        """
        Verify the hydrogenic cross-section uses hbar^3, not h^3.
        At nu just above n=1 threshold, sigma should be ~few × 10^{-18} cm².
        """
        nu_thresh = const.chi_H_ev * const.ev_to_erg / const.h
        nu = 1.01 * nu_thresh
        sigma = float(sigma_bf_hydrogenic_shell(nu, 1e4, 1, const))
        # Rough physical order of magnitude: ~6e-18 cm² at threshold for n=1
        assert 1e-20 < sigma < 1e-15, f"sigma_bf(n=1) at threshold = {sigma:.3e}"

    def test_kappa_bf_H_nonnegative(self, state_1e4, const):
        nu = np.logspace(14, 17, 30)
        kbf = kappa_bf_H_net(nu, 1e4, state_1e4.rho, state_1e4.level_populations, const)
        assert np.all(kbf >= 0.0)

    @pytest.mark.parametrize("T, rho", [(1e4, 1e-7), (1e5, 1e-3)])
    def test_kappa_bf_H_threshold_behavior(self, T, rho, const, cfg):
        state = solve_eos(T, rho, cfg.n_max, const)
        x_base = build_base_x_grid(cfg)
        x = refine_x_grid_for_thresholds(x_base, T, cfg.n_max, const)
        comp = monochromatic_opacity(x, state, const)
        check_threshold_behavior(x, T, comp, const)


class TestBoundFreeHminus:
    def test_sigma_zero_below_threshold(self, const):
        # Below 0.754 eV
        nu_lo = np.array([0.5 * const.chi_Hminus_ev * const.ev_to_erg / const.h])
        sigma = sigma_bf_Hminus(nu_lo, const)
        assert float(sigma[0]) == 0.0

    def test_sigma_zero_above_cutoff(self, const):
        # Above 10 eV
        nu_hi = np.array([11.0 * const.ev_to_erg / (const.h)])  # 11 eV
        sigma = sigma_bf_Hminus(nu_hi, const)
        assert float(sigma[0]) == 0.0

    def test_sigma_positive_in_valid_range(self, const):
        # 2 eV — well within [0.754, 10] eV  AND  above the fit zero at lambda0
        nu_mid = np.array([2.0 * const.ev_to_erg / const.h])
        sigma = sigma_bf_Hminus(nu_mid, const)
        assert float(sigma[0]) > 0.0

    def test_sigma_zero_in_gap_below_fit_zero(self, const):
        """
        For 0.754 eV < hν < hc/λ₀, the physical threshold is passed but the
        fit formula gives (1/λ − 1/λ₀) < 0 so σ = 0 (clamped).
        This is correct: the fit rises from 0 at λ = λ₀ = 1.64 μm.
        """
        # hν = 0.754 eV → λ = hc/0.754 eV ≈ 1.646 μm > λ₀ = 1.64 μm
        # So just above chi_Hminus threshold, λ > λ₀  →  (1/λ - 1/λ₀) < 0  → 0
        nu_just_above_chi = const.chi_Hminus_ev * const.ev_to_erg / const.h * 1.001
        sig = float(sigma_bf_Hminus(nu_just_above_chi, const))
        # This should be 0 because λ > λ₀ in this region
        assert sig == 0.0

    def test_sigma_positive_above_fit_zero(self, const):
        """
        Above λ₀ = 1.64 μm frequency (hν = hc/λ₀), sigma becomes positive.
        """
        # nu corresponding to λ₀ - epsilon (slightly shorter wavelength)
        nu_fit_zero = const.c / (const.lambda0_Hminus_micron * 1e-4)
        nu_above_fit = nu_fit_zero * 1.002  # slightly above λ₀ frequency
        sig = float(sigma_bf_Hminus(nu_above_fit, const))
        assert sig > 0.0

    def test_sigma_at_1_micron(self, const):
        """λ = 1 μm → ν = c/λ.  Should give a positive cross-section ~ few × 10⁻¹⁷ cm²."""
        nu = const.c / (1e-4)  # 1 μm in cm
        # 1 μm → hν = hc/λ = (6.626e-27 * 3e10) / 1e-4 ≈ 1.99e-12 erg ≈ 1.24 eV  (within [0.754, 10])
        sigma = float(sigma_bf_Hminus(nu, const))
        assert 1e-19 < sigma < 1e-14, f"sigma at 1 μm = {sigma:.3e}"

    def test_kappa_bf_hminus_nonnegative(self, state_1e4, const):
        nu = np.logspace(13, 16, 50)
        kbf = kappa_bf_Hminus_net(nu, 1e4, state_1e4.rho, state_1e4.n_Hminus, const)
        assert np.all(kbf >= 0.0)

    def test_domain_restriction_at_chi_Hminus(self, const):
        """
        At hν = 0.754 eV (chi_Hminus), σ = 0.
        This is at the domain lower bound where the in_range mask is False.
        """
        nu_thresh = const.chi_Hminus_ev * const.ev_to_erg / const.h
        sigma = float(sigma_bf_Hminus(nu_thresh, const))
        assert sigma == pytest.approx(0.0, abs=1e-50)

    def test_domain_restriction_at_fit_zero_lambda0(self, const):
        """
        At λ = λ₀ = 1.64 μm, the fit formula gives (1/λ - 1/λ₀)^{3/2} = 0, so σ = 0.
        This is the natural zero-crossing of the empirical cross-section.
        """
        nu_at_lambda0 = const.c / (const.lambda0_Hminus_micron * 1e-4)
        sigma = float(sigma_bf_Hminus(nu_at_lambda0, const))
        assert sigma == pytest.approx(0.0, abs=1e-50)

    def test_no_extrapolation_above_10ev(self, const):
        nu_above = np.linspace(
            10.1 * const.ev_to_erg / const.h,
            20.0 * const.ev_to_erg / const.h,
            20,
        )
        sigma = sigma_bf_Hminus(nu_above, const)
        assert np.all(sigma == 0.0)


class TestTotalOpacityAssembly:
    @pytest.mark.parametrize("T, rho", [(1e4, 1e-7), (1e5, 1e-3), (1e6, 1.0)])
    def test_total_opacity_nonnegative(self, T, rho, const, cfg):
        state = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol)
        x_base = build_base_x_grid(cfg)
        x = refine_x_grid_for_thresholds(x_base, T, cfg.n_max, const)
        comp = monochromatic_opacity(x, state, const)
        check_opacity_nonnegative(comp)

    @pytest.mark.parametrize("T, rho", [(1e4, 1e-7), (1e5, 1e-3)])
    def test_total_opacity_equals_sum(self, T, rho, const, cfg):
        state = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol)
        x_base = build_base_x_grid(cfg)
        x = refine_x_grid_for_thresholds(x_base, T, cfg.n_max, const)
        comp = monochromatic_opacity(x, state, const)
        expected = comp.kappa_es + comp.kappa_ff + comp.kappa_bf_H + comp.kappa_bf_Hminus
        np.testing.assert_allclose(comp.kappa_total, expected, rtol=1e-12)
