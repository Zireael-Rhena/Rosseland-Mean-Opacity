"""
test_regression.py
------------------
Regression tests: κ_R at selected (T, ρ) points must reproduce stored
reference values to within 1%.

Reference values were generated with:
  - n_max  = 16
  - x_base = 500 pts, x ∈ [0.01, 30]
  - full threshold refinement
  - EOS tol = 1e-10

If the physics formulas change, update the reference values with
    python -c "
        import sys; sys.path.insert(0,'src')
        from hydrogen_opacity.constants import load_constants
        from hydrogen_opacity.config import default_config
        from hydrogen_opacity.grids import build_base_x_grid, refine_x_grid_for_thresholds
        from hydrogen_opacity.rosseland import compute_rosseland_mean
        const = load_constants(); cfg = default_config()
        x_base = build_base_x_grid(cfg)
        for T, rho in [(1e4, 1e-7), (1e5, 1e-3), (1e6, 1.0), (5e4, 1e-5)]:
            x = refine_x_grid_for_thresholds(x_base, T, cfg.n_max, const)
            kR = compute_rosseland_mean(T, rho, cfg.n_max, x, const)
            print(f'({T:.1e}, {rho:.1e}): {kR:.8e}')
    "
"""

import pytest
import numpy as np

from hydrogen_opacity.constants import load_constants
from hydrogen_opacity.config import default_config
from hydrogen_opacity.grids import build_base_x_grid, refine_x_grid_for_thresholds
from hydrogen_opacity.rosseland import compute_rosseland_mean


@pytest.fixture(scope="module")
def const():
    return load_constants()


@pytest.fixture(scope="module")
def cfg():
    return default_config()


# Reference values (cm² g⁻¹) generated 2026-04-21
# Physics: KN scattering, H⁻ ff (John 1988), new n_max formula,
#          ionization lowering via float n_max_phys (not integer n_cut)
_REGRESSION_CASES = [
    (1.0e4, 1.0e-7, 6.57843451e+01),   # n_cut=15; KN negligible; H- ff dominant
    (1.0e5, 1.0e-3, 2.52158104e+03),   # n_cut=2, n_max_phys≈2.04; KN ~20% at high ν
    (1.0e6, 1.0e0,  5.83780307e+01),   # n_cut=1; KN dominant correction ~21%
    (5.0e4, 1.0e-5, 1.97259640e+03),   # n_cut=7; mixed correction
]


@pytest.mark.parametrize("T, rho, kappa_R_ref", _REGRESSION_CASES)
def test_rosseland_regression(T, rho, kappa_R_ref, const, cfg):
    """
    κ_R at (T, ρ) must match reference to within 1%.
    """
    x_base = build_base_x_grid(cfg)
    x = refine_x_grid_for_thresholds(x_base, T, cfg.n_max, const)
    kR = compute_rosseland_mean(T, rho, cfg.n_max, x, const, tol=cfg.root_tol)
    frac = abs(kR - kappa_R_ref) / kappa_R_ref
    assert frac < 0.01, (
        f"Regression failed for T={T:.1e} K, rho={rho:.1e} g/cc: "
        f"kR={kR:.6e}, ref={kappa_R_ref:.6e}, frac={frac:.4f}"
    )


@pytest.mark.parametrize("T, rho, _", _REGRESSION_CASES)
def test_rosseland_positive_and_finite(T, rho, _, const, cfg):
    x_base = build_base_x_grid(cfg)
    x = refine_x_grid_for_thresholds(x_base, T, cfg.n_max, const)
    kR = compute_rosseland_mean(T, rho, cfg.n_max, x, const, tol=cfg.root_tol)
    assert kR > 0.0
    assert np.isfinite(kR)
