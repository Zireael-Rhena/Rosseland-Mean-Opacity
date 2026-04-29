"""
test_poutanen2017.py
--------------------
Tests for the Poutanen (2017) Compton Rosseland-mean correction.

Physics context:
  Poutanen (2017), ApJ, 835, 119, doi:10.3847/1538-4357/835/2/119
  Non-degenerate, 2–40 keV fit:
      Λ_P17(T) = 1 + (T_keV / 39.4)^0.976
      κ_P17    = κ_T / Λ_P17   where  κ_T = n_e σ_T / ρ

Applicability:
  - Hot (T_keV >= 2), fully ionized (y_e >= 0.999)
  - Non-degenerate electrons
  - Scattering-dominated regime
  NOT valid at cold, partially neutral, or line-opacity-dominated conditions.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from hydrogen_opacity.constants import load_constants
from hydrogen_opacity.config import default_config, ModelConfig, ModelOptions
from hydrogen_opacity.grids import build_base_x_grid, refine_x_grid_for_thresholds
from hydrogen_opacity.eos import solve_eos
from hydrogen_opacity.scattering import (
    lambda_poutanen2017_nondegenerate,
    kappa_scattering_poutanen2017,
)
from hydrogen_opacity.rosseland import compute_rosseland_mean


# Boltzmann constant in keV/K (derived consistently from constants module)
def _T_keV(T_K, const):
    return T_K * const.k_B / (1.0e3 * const.ev_to_erg)


@pytest.fixture(scope="module")
def const():
    return load_constants()


@pytest.fixture(scope="module")
def cfg():
    return default_config()


# ---------------------------------------------------------------------------
# 1. Lambda function properties
# ---------------------------------------------------------------------------

class TestLambdaPoutanen2017:
    """Test the Λ_P17(T) suppression factor."""

    def test_lambda_greater_than_one_for_positive_T(self):
        T_vals = np.array([0.1, 1.0, 2.0, 4.0, 8.0, 10.0, 40.0])
        lam = lambda_poutanen2017_nondegenerate(T_vals)
        assert np.all(lam > 1.0), f"Lambda must be > 1 for all T > 0; got {lam}"

    @pytest.mark.parametrize("T_keV, expected", [
        (2.0,  1.0 + (2.0  / 39.4) ** 0.976),
        (4.0,  1.0 + (4.0  / 39.4) ** 0.976),
        (8.0,  1.0 + (8.0  / 39.4) ** 0.976),
        (10.0, 1.0 + (10.0 / 39.4) ** 0.976),
    ])
    def test_lambda_representative_values(self, T_keV, expected):
        lam = float(lambda_poutanen2017_nondegenerate(T_keV))
        assert abs(lam - expected) / expected < 1e-12, (
            f"Lambda({T_keV} keV) = {lam:.10g}, expected {expected:.10g}"
        )

    def test_lambda_monotone_increasing(self):
        T_vals = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        lam = lambda_poutanen2017_nondegenerate(T_vals)
        assert np.all(np.diff(lam) > 0.0), "Λ_P17 must increase monotonically with T"

    def test_lambda_scalar_input(self):
        lam = lambda_poutanen2017_nondegenerate(5.0)
        assert isinstance(lam, float)
        assert lam > 1.0

    def test_lambda_array_input(self):
        T = np.linspace(2.0, 40.0, 50)
        lam = lambda_poutanen2017_nondegenerate(T)
        assert lam.shape == T.shape
        assert np.all(lam > 1.0)


# ---------------------------------------------------------------------------
# 2. P17 scattering opacity less than Thomson
# ---------------------------------------------------------------------------

class TestP17ScatteringLessThanThomson:
    """κ_P17 < κ_T for T >= 2 keV (Compton recoil reduces mean cross-section)."""

    @pytest.mark.parametrize("T_keV", [2.0, 4.0, 8.0, 10.0])
    def test_p17_less_than_thomson(self, T_keV, const):
        rho = 1.0e-9
        # Fully ionized pure H: n_e = rho / m_H
        n_e = rho / const.m_H
        kappa_T = n_e * const.sigma_T / rho   # = sigma_T / m_H
        kappa_P17 = kappa_scattering_poutanen2017(T_keV, rho, n_e, const)
        assert kappa_P17 < kappa_T, (
            f"At T={T_keV} keV: κ_P17={kappa_P17:.6f} must be < κ_T={kappa_T:.6f}"
        )

    def test_suppression_fraction_at_8keV(self, const):
        """At T=8 keV the correction factor is known; check it is in expected range."""
        T_keV = 8.0
        rho = 1.0e-9
        n_e = rho / const.m_H
        kappa_T  = n_e * const.sigma_T / rho
        kappa_P17 = kappa_scattering_poutanen2017(T_keV, rho, n_e, const)
        suppression = 1.0 - kappa_P17 / kappa_T   # fraction suppressed by Compton
        # Λ ≈ 1.205 at T=8 keV → suppression ≈ 1 - 1/1.205 ≈ 17%
        assert 0.10 < suppression < 0.35, (
            f"Suppression at T=8 keV = {suppression:.4f}; expected ~17%"
        )

    def test_p17_positive(self, const):
        rho, n_e = 1.0e-9, 1.0e-9 / const.m_H
        kappa = kappa_scattering_poutanen2017(4.0, rho, n_e, const)
        assert kappa > 0.0


# ---------------------------------------------------------------------------
# 3. P17 not applied below temperature threshold
# ---------------------------------------------------------------------------

class TestP17NotAppliedBelowThreshold:
    """For T_keV < 2, production calculation falls back to KN spectral."""

    @pytest.mark.parametrize("T_keV_low", [0.5, 1.0, 1.9])
    def test_p17_matches_kn_below_threshold(self, T_keV_low, const, cfg):
        """
        compton_mean_mode='poutanen2017' at T < 2 keV must give the same result
        as compton_mean_mode='kn_spectral', because the P17 branch is not triggered.
        """
        KEV_TO_K = 1.0e3 * const.ev_to_erg / const.k_B
        T = T_keV_low * KEV_TO_K
        rho = 1.0e-9

        x_base = build_base_x_grid(cfg)
        x = refine_x_grid_for_thresholds(x_base, T, cfg.n_max, const)

        opts_p17 = ModelOptions(
            use_kn=True, use_ff_hminus=True,
            lowering_mode="capped", delta_chi_max_ev=1.0,
            compton_mean_mode="poutanen2017",
        )
        opts_kn = ModelOptions(
            use_kn=True, use_ff_hminus=True,
            lowering_mode="capped", delta_chi_max_ev=1.0,
            compton_mean_mode="kn_spectral",
        )

        kR_p17 = compute_rosseland_mean(T, rho, cfg.n_max, x, const,
                                        tol=cfg.root_tol, opts=opts_p17)
        kR_kn  = compute_rosseland_mean(T, rho, cfg.n_max, x, const,
                                        tol=cfg.root_tol, opts=opts_kn)

        assert abs(kR_p17 - kR_kn) / kR_kn < 1.0e-12, (
            f"At T={T_keV_low} keV (below P17 threshold), "
            f"kR_p17={kR_p17:.6e} != kR_kn={kR_kn:.6e}"
        )


# ---------------------------------------------------------------------------
# 4. P17 applied in high-T fully-ionized regime
# ---------------------------------------------------------------------------

class TestP17AppliedHighTFullyIonized:
    """For T >= 2 keV and fully-ionized pure H, P17 branch is taken."""

    @pytest.mark.parametrize("T_keV_hi, rho", [
        (2.0, 1.0e-12),
        (4.0, 1.0e-9),
        (8.0, 1.0e-12),
        (10.0, 1.0e-9),
    ])
    def test_p17_applied_gives_expected_value(self, T_keV_hi, rho, const, cfg):
        """
        The P17 model result must equal κ_T / Λ_P17 computed from EOS n_e.
        """
        KEV_TO_K = 1.0e3 * const.ev_to_erg / const.k_B
        T = T_keV_hi * KEV_TO_K

        x_base = build_base_x_grid(cfg)
        x = refine_x_grid_for_thresholds(x_base, T, cfg.n_max, const)

        opts_p17 = ModelOptions(
            use_kn=True, use_ff_hminus=True,
            lowering_mode="capped", delta_chi_max_ev=1.0,
            compton_mean_mode="poutanen2017",
        )
        kR_p17 = compute_rosseland_mean(T, rho, cfg.n_max, x, const,
                                        tol=cfg.root_tol, opts=opts_p17)

        # Reference: compute expected P17 value from EOS state directly
        state = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol, opts=opts_p17)
        y_e = state.n_e / state.n_H_tot
        assert y_e >= 0.999, f"Expected fully ionized at T={T_keV_hi} keV, got y_e={y_e:.6f}"
        kR_expected = kappa_scattering_poutanen2017(T_keV_hi, rho, state.n_e, const)

        assert abs(kR_p17 - kR_expected) / kR_expected < 1.0e-12, (
            f"P17 model: kR={kR_p17:.8e}, expected={kR_expected:.8e}"
        )

    def test_p17_different_from_kn_at_highT(self, const, cfg):
        """P17 gives lower opacity than full-KN spectral at T >= 2 keV."""
        KEV_TO_K = 1.0e3 * const.ev_to_erg / const.k_B
        T = 8.0 * KEV_TO_K
        rho = 1.0e-12

        x_base = build_base_x_grid(cfg)
        x = refine_x_grid_for_thresholds(x_base, T, cfg.n_max, const)

        opts_p17 = ModelOptions(use_kn=True, use_ff_hminus=True,
                                lowering_mode="capped", delta_chi_max_ev=1.0,
                                compton_mean_mode="poutanen2017")
        opts_kn  = ModelOptions(use_kn=True, use_ff_hminus=True,
                                lowering_mode="capped", delta_chi_max_ev=1.0,
                                compton_mean_mode="kn_spectral")

        kR_p17 = compute_rosseland_mean(T, rho, cfg.n_max, x, const,
                                        tol=cfg.root_tol, opts=opts_p17)
        kR_kn  = compute_rosseland_mean(T, rho, cfg.n_max, x, const,
                                        tol=cfg.root_tol, opts=opts_kn)

        # P17 corrects downward: KN spectral overestimates at high T
        assert kR_p17 < kR_kn, (
            f"At T=8 keV, P17 ({kR_p17:.5f}) must be less than KN ({kR_kn:.5f})"
        )

        # The fractional difference should match the expected Lambda correction
        lam = float(lambda_poutanen2017_nondegenerate(8.0))
        state = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol)
        kappa_T = state.n_e * const.sigma_T / rho
        expected_ratio = 1.0 / lam     # κ_P17 / κ_T
        actual_ratio   = kR_p17 / kappa_T
        assert abs(actual_ratio - expected_ratio) / expected_ratio < 1.0e-10


# ---------------------------------------------------------------------------
# 5. Regression: P17 improves high-T residual vs TOPS
# ---------------------------------------------------------------------------

class TestP17Regression:
    """
    Compare P17 vs KN-spectral residuals against TOPS at high T.

    Loads TOPS reference data from data/tops_parsed.npz.
    Skipped if file is not found (optional data dependency).
    """

    _TOPS_PATH = "data/tops_parsed.npz"

    @pytest.fixture(scope="class")
    def tops_data(self):
        if not os.path.exists(self._TOPS_PATH):
            pytest.skip(f"TOPS reference not found at {self._TOPS_PATH}")
        d = np.load(self._TOPS_PATH)
        return d["T_grid"], d["rho_grid"], d["kR_tops"]

    def test_p17_reduces_highT_residual_qualitatively(self, tops_data, const, cfg):
        """
        At T >= 2 keV, the P17 model must give smaller |residual| vs TOPS than
        the KN-spectral model, in the mean across the density grid.
        """
        T_keV_tops, rho_grid, kR_tops = tops_data
        KEV_TO_K = 1.0e3 * const.ev_to_erg / const.k_B

        # Select a few high-T test points
        test_T_keV = np.array([2.0, 4.0, 8.0, 10.0])
        test_rho   = [1.0e-12, 1.0e-9]   # two dilute densities (scattering dominated)

        x_base = build_base_x_grid(cfg)

        opts_p17 = ModelOptions(use_kn=True, use_ff_hminus=True,
                                lowering_mode="capped", delta_chi_max_ev=1.0,
                                compton_mean_mode="poutanen2017")
        opts_kn  = ModelOptions(use_kn=True, use_ff_hminus=True,
                                lowering_mode="capped", delta_chi_max_ev=1.0,
                                compton_mean_mode="kn_spectral")

        rms_p17 = []
        rms_kn  = []

        for T_keV in test_T_keV:
            # Find closest TOPS temperature
            iT = int(np.argmin(np.abs(T_keV_tops - T_keV)))
            T = T_keV_tops[iT] * KEV_TO_K
            x = refine_x_grid_for_thresholds(x_base, T, cfg.n_max, const)

            for rho in test_rho:
                irho = int(np.argmin(np.abs(rho_grid - rho)))
                kR_ref = kR_tops[iT, irho]

                kR_p17 = compute_rosseland_mean(T, rho, cfg.n_max, x, const,
                                                tol=cfg.root_tol, opts=opts_p17)
                kR_kn  = compute_rosseland_mean(T, rho, cfg.n_max, x, const,
                                                tol=cfg.root_tol, opts=opts_kn)

                rms_p17.append(abs(kR_p17 - kR_ref) / kR_ref)
                rms_kn.append(abs(kR_kn  - kR_ref) / kR_ref)

        mean_p17 = float(np.mean(rms_p17))
        mean_kn  = float(np.mean(rms_kn))
        assert mean_p17 < mean_kn, (
            f"P17 mean |residual| ({mean_p17:.4f}) is not smaller than "
            f"KN spectral ({mean_kn:.4f}) at high T — P17 should improve agreement"
        )

    def test_p17_highT_residual_below_2pct(self, tops_data, const, cfg):
        """
        At T >= 2 keV (dilute, scattering-dominated), P17 residual vs TOPS < 2%.
        """
        T_keV_tops, rho_grid, kR_tops = tops_data
        KEV_TO_K = 1.0e3 * const.ev_to_erg / const.k_B

        test_T_keV = np.array([2.0, 4.0, 8.0, 10.0])
        x_base = build_base_x_grid(cfg)

        opts_p17 = ModelOptions(use_kn=True, use_ff_hminus=True,
                                lowering_mode="capped", delta_chi_max_ev=1.0,
                                compton_mean_mode="poutanen2017")

        # Use rho=1e-12 (most scattering-dominated, strongest P17 improvement)
        rho = 1.0e-12
        irho = int(np.argmin(np.abs(rho_grid - rho)))

        for T_keV in test_T_keV:
            iT = int(np.argmin(np.abs(T_keV_tops - T_keV)))
            T = T_keV_tops[iT] * KEV_TO_K
            kR_ref = kR_tops[iT, irho]
            x = refine_x_grid_for_thresholds(x_base, T, cfg.n_max, const)

            kR_p17 = compute_rosseland_mean(T, rho, cfg.n_max, x, const,
                                            tol=cfg.root_tol, opts=opts_p17)
            rd = abs(kR_p17 - kR_ref) / kR_ref
            assert rd < 0.02, (
                f"P17 residual at T={T_keV} keV, rho=1e-12: {100*rd:.2f}% > 2%  "
                f"(kR_P17={kR_p17:.5f}, kR_TOPS={kR_ref:.5f})"
            )
