"""
test_rosseland.py
-----------------
Tests for Rosseland mean integration:
  - weight function properties
  - positivity and finiteness of κ_R
  - grey opacity case recovers the input
  - convergence with x-grid size
"""

import pytest
import numpy as np

from hydrogen_opacity.constants import load_constants
from hydrogen_opacity.config import default_config, ModelConfig
from hydrogen_opacity.grids import build_base_x_grid, refine_x_grid_for_thresholds
from hydrogen_opacity.rosseland import (
    rosseland_weight,
    rosseland_mean_from_spectrum,
    compute_rosseland_mean,
)


@pytest.fixture(scope="module")
def const():
    return load_constants()


@pytest.fixture(scope="module")
def cfg():
    return default_config()


class TestRosselandWeight:
    def test_weight_positive(self):
        x = np.linspace(0.01, 20.0, 200)
        w = rosseland_weight(x)
        assert np.all(w >= 0.0)

    def test_weight_peak_near_3p83(self):
        x = np.linspace(0.1, 15.0, 1000)
        w = rosseland_weight(x)
        x_peak = x[np.argmax(w)]
        assert 3.5 < x_peak < 4.1

    def test_weight_decays_large_x(self):
        w_small = float(rosseland_weight(3.83))
        w_large = float(rosseland_weight(20.0))
        assert w_large < w_small

    def test_weight_scalar_input(self):
        w = rosseland_weight(3.83)
        assert isinstance(w, float)
        assert w > 0.0


class TestRosselandMeanFromSpectrum:
    def test_grey_opacity_recovers_input(self):
        """For a constant κ, κ_R must equal that constant."""
        x = np.linspace(0.01, 30.0, 2000)
        kappa_const = 42.0
        kappa_nu = np.full_like(x, kappa_const)
        kR = rosseland_mean_from_spectrum(x, kappa_nu)
        assert abs(kR - kappa_const) / kappa_const < 1e-4

    def test_returns_positive_float(self):
        x = np.linspace(0.01, 20.0, 500)
        kappa_nu = np.ones_like(x) * 0.5
        kR = rosseland_mean_from_spectrum(x, kappa_nu)
        assert isinstance(kR, float)
        assert kR > 0.0

    def test_larger_opacity_gives_smaller_rosseland(self):
        x = np.linspace(0.01, 20.0, 500)
        kR_lo = rosseland_mean_from_spectrum(x, np.ones_like(x))
        kR_hi = rosseland_mean_from_spectrum(x, np.full_like(x, 100.0))
        assert kR_hi > kR_lo

    def test_harmonic_mean_effect(self):
        """
        If kappa_nu is low in the Rosseland-peak region, κ_R should be low.
        """
        x = np.linspace(0.01, 20.0, 2000)
        w = rosseland_weight(x)
        kappa_nu = np.where(x < 5.0, 1.0, 1000.0)
        kR = rosseland_mean_from_spectrum(x, kappa_nu)
        # κ_R should be dominated by the low-opacity region near x ~ 3.83
        assert kR < 100.0


class TestComputeRosselandMean:
    @pytest.mark.parametrize("T, rho", [
        (1e4, 1e-7),
        (5e4, 1e-5),
        (1e5, 1e-3),
        (1e6, 1.0),
    ])
    def test_positive_and_finite(self, T, rho, const, cfg):
        x_base = build_base_x_grid(cfg)
        x = refine_x_grid_for_thresholds(x_base, T, cfg.n_max, const)
        kR = compute_rosseland_mean(T, rho, cfg.n_max, x, const, tol=cfg.root_tol)
        assert np.isfinite(kR)
        assert kR > 0.0

    def test_rosseland_convergence_with_xgrid(self, const, cfg):
        """κ_R should stabilise as n_x increases."""
        T, rho = 1e4, 1e-7
        kR_values = []
        for n_x in (200, 500, 1000):
            cfg_tmp = ModelConfig(
                T_min_keV=0.001, T_max_keV=10.0, n_T=20,
                rho_min=1e-10, rho_max=1e2, n_rho=13,
                n_max=cfg.n_max, x_min=1e-2, x_max=30.0,
                n_x_base=n_x, root_tol=1e-10, max_root_iter=200,
            )
            x_base = build_base_x_grid(cfg_tmp)
            x = refine_x_grid_for_thresholds(x_base, T, cfg.n_max, const)
            kR = compute_rosseland_mean(T, rho, cfg.n_max, x, const)
            kR_values.append(kR)
        # Fractional change between last two should be < 1%
        frac = abs(kR_values[-1] - kR_values[-2]) / kR_values[-2]
        assert frac < 0.01, f"kappa_R not converged: {kR_values}"
