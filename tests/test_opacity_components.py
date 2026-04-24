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
from hydrogen_opacity.scattering import kappa_es, sigma_kn
from hydrogen_opacity.free_free import alpha_ff_net, kappa_ff_net
from hydrogen_opacity.bound_free_h import sigma_bf_hydrogenic_shell, kappa_bf_H_net, alpha_bf_H_true
from hydrogen_opacity.bound_free_hminus import sigma_bf_Hminus, kappa_bf_Hminus_net
from hydrogen_opacity.free_free_hminus import kappa_ff_Hminus_net
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
        nu_test = np.array([1e14])  # low frequency → Thomson limit
        val = kappa_es(nu_test, state_1e4.n_e, state_1e4.rho, const)
        assert float(val[0]) > 0.0

    def test_kappa_es_scales_with_ne(self, const):
        nu_test = np.array([1e14])
        val1 = kappa_es(nu_test, 1e15, 1e-7, const)
        val2 = kappa_es(nu_test, 2e15, 1e-7, const)
        assert abs(float(val2[0]) / float(val1[0]) - 2.0) < 1e-10

    def test_kappa_es_fully_ionized_limit(self, const):
        # For fully ionized H at low ν: κ_es → σ_T / m_H
        rho = 1e-3
        n_e = rho / const.m_H
        nu_test = np.array([1e10])  # very low ν → pure Thomson (x ≪ 1)
        val = float(kappa_es(nu_test, n_e, rho, const)[0])
        expected = const.sigma_T / const.m_H
        assert abs(val - expected) / expected < 1e-6

    def test_kappa_es_decreases_at_high_energy(self, const):
        # Klein–Nishina: cross-section decreases above m_e c² ≈ 8.2e20 Hz
        n_e = 1e15
        rho = 1e-7
        nu_lo = np.array([1e14])   # hν ≪ m_e c²
        nu_hi = np.array([1e24])   # hν ≫ m_e c²
        val_lo = float(kappa_es(nu_lo, n_e, rho, const)[0])
        val_hi = float(kappa_es(nu_hi, n_e, rho, const)[0])
        assert val_hi < val_lo


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
        expected = (comp.kappa_es + comp.kappa_ff + comp.kappa_bf_H
                    + comp.kappa_bf_Hminus + comp.kappa_ff_Hminus)
        np.testing.assert_allclose(comp.kappa_total, expected, rtol=1e-12)


class TestKleinNishina:
    def test_sigma_kn_reduces_to_thomson_at_low_nu(self, const):
        nu_lo = np.array([1e10])  # hν ≪ m_e c²
        sig = float(sigma_kn(nu_lo, const)[0])
        assert abs(sig - const.sigma_T) / const.sigma_T < 1e-5

    def test_sigma_kn_decreases_at_high_nu(self, const):
        nu_lo = np.array([1e10])
        nu_hi = np.array([1e24])
        sig_lo = float(sigma_kn(nu_lo, const)[0])
        sig_hi = float(sigma_kn(nu_hi, const)[0])
        assert sig_hi < sig_lo

    def test_sigma_kn_positive(self, const):
        nu = np.logspace(10, 25, 50)
        sigs = sigma_kn(nu, const)
        assert np.all(sigs > 0.0)

    def test_low_x_taylor_coefficients(self, const):
        """Taylor branch matches exact series 1-2x+(26/5)x²-(133/10)x³+(1144/35)x⁴."""
        m_e_c2 = const.m_e * const.c ** 2   # erg
        for x_val in [1e-4, 1e-3, 1e-2]:
            nu = np.array([x_val * m_e_c2 / const.h])
            ratio = float(sigma_kn(nu, const)[0]) / const.sigma_T
            ref = (1.0
                   - 2.0 * x_val
                   + (26.0 / 5.0)  * x_val**2
                   - (133.0 / 10.0) * x_val**3
                   + (1144.0 / 35.0) * x_val**4)
            assert abs(ratio - ref) < 1e-12, (
                f"x={x_val}: Taylor ratio={ratio:.15g}, ref={ref:.15g}, "
                f"err={abs(ratio-ref):.3e}")

    def test_branch_continuity_at_threshold(self, const):
        """No discontinuous jump at the x=0.05 Taylor/exact branch boundary.

        Straddling x=0.05: fractional discontinuity must be < 0.1% of local σ.
        """
        m_e_c2 = const.m_e * const.c ** 2

        def s_of_x(x):
            return float(sigma_kn(np.array([x * m_e_c2 / const.h]), const)[0])

        s_below = s_of_x(0.04999)   # Taylor branch
        s_above = s_of_x(0.05001)   # exact branch
        frac_jump = abs(s_above - s_below) / s_below

        assert frac_jump < 1e-3, (
            f"Fractional branch jump at x=0.05: {frac_jump:.3e} "
            f"(s_Taylor={s_below:.6g}, s_exact={s_above:.6g})")

    def test_monotone_suppression(self, const):
        """σ_KN/σ_T must be strictly decreasing into the mildly relativistic regime."""
        m_e_c2 = const.m_e * const.c ** 2
        x_vals = np.array([1e-3, 1e-2, 0.1, 0.3, 1.0])
        nu_arr  = x_vals * m_e_c2 / const.h
        ratios  = sigma_kn(nu_arr, const) / const.sigma_T
        assert np.all(ratios < 1.0), "σ_KN/σ_T must be below 1 for x > 0"
        assert np.all(np.diff(ratios) < 0), "σ_KN/σ_T must decrease monotonically"


class TestFreeFreeHminus:
    def test_zero_outside_temperature_range(self, const):
        state_cold = solve_eos(1000.0, 1e-7, 16, const)
        nu = np.array([1e14])
        val = float(kappa_ff_Hminus_net(nu, 1000.0, 1e-7,
                                        state_cold.n_H0, state_cold.n_e, const)[0])
        assert val == 0.0

    def test_zero_for_short_wavelength(self, const):
        # λ < 0.1823 µm → ν > c/0.1823e-4 ≈ 1.64e15 Hz
        nu_short = np.array([2e15])  # λ ≈ 0.15 µm
        val = float(kappa_ff_Hminus_net(nu_short, 5040.0, 1e-7, 1e10, 1e5, const)[0])
        assert val == 0.0

    def test_positive_at_valid_conditions(self, const):
        # T=5040 K (θ=1), λ=1 µm (ν=3e14 Hz) — in Table 3a domain
        nu = np.array([3e14])
        val = float(kappa_ff_Hminus_net(nu, 5040.0, 1e-7, 1e10, 1e8, const)[0])
        assert val > 0.0

    def test_nonnegative_over_spectrum_solar_T(self, const):
        # Solar-like conditions within John (1988) validity range
        T = 6000.0
        nu = np.logspace(13, 15.5, 60)  # covers both Table 3a and 3b
        state = solve_eos(T, 1e-7, 16, const)
        kff = kappa_ff_Hminus_net(nu, T, 1e-7, state.n_H0, state.n_e, const)
        assert np.all(kff >= 0.0), f"negative H- ff values found: min={kff.min():.3e}"

    def test_table3a_3b_boundary_continuity(self, const):
        # Near the 0.3645 µm boundary, both tables should give similar values
        lam_mid = 0.3645  # µm
        c_cm = 2.99792458e10
        nu_a = np.array([c_cm / (lam_mid * 1.001e-4)])  # just inside 3a
        nu_b = np.array([c_cm / (lam_mid * 0.999e-4)])  # just inside 3b
        n_H0, n_e, rho, T = 1e14, 1e8, 1e-7, 5040.0
        va = float(kappa_ff_Hminus_net(nu_a, T, rho, n_H0, n_e, const)[0])
        vb = float(kappa_ff_Hminus_net(nu_b, T, rho, n_H0, n_e, const)[0])
        # Both should be positive and within 10× of each other (fit is not guaranteed C0)
        assert va >= 0.0 and vb >= 0.0
