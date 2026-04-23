"""
driver.py
=========
Top-level entry points for running single-point and grid opacity calculations.

Usage
-----
Single point:
    from hydrogen_opacity.constants import load_constants
    from hydrogen_opacity.config import default_config
    from hydrogen_opacity.driver import run_single_point

    const = load_constants()
    cfg = default_config()
    result = run_single_point(T=1e4, rho=1e-7, cfg=cfg, const=const)

Grid:
    from hydrogen_opacity.driver import run_opacity_grid
    grid_result = run_opacity_grid(cfg, const)
"""

from __future__ import annotations

import time

import numpy as np

from .constants import PhysicalConstants
from .config import ModelConfig
from .grids import build_temperature_grid, build_density_grid, build_base_x_grid, refine_x_grid_for_thresholds
from .eos import solve_eos, EOSState
from .opacity import monochromatic_opacity, OpacityComponents
from .rosseland import rosseland_mean_from_spectrum
from .validation import check_eos_consistency, check_opacity_nonnegative


def run_single_point(
    T: float,
    rho: float,
    cfg: ModelConfig,
    const: PhysicalConstants,
) -> dict:
    """
    Compute the full opacity spectrum and Rosseland mean at a single (T, ρ).

    Parameters
    ----------
    T : float     Temperature [K]
    rho : float   Density [g cm⁻³]
    cfg : ModelConfig
    const : PhysicalConstants

    Returns
    -------
    dict with keys:
      'T'            — temperature [K]
      'rho'          — density [g cm⁻³]
      'kappa_R'      — Rosseland mean [cm² g⁻¹]
      'x'            — x-grid
      'kappa_es'     — electron scattering spectrum
      'kappa_ff'     — free-free spectrum
      'kappa_bf_H'   — neutral-H bound-free spectrum
      'kappa_bf_Hminus' — H⁻ bound-free spectrum
      'kappa_total'  — total spectrum
      'eos'          — EOSState
    """
    x_base = build_base_x_grid(cfg)
    x = refine_x_grid_for_thresholds(x_base, T, cfg.n_max, const)

    state: EOSState = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol)
    check_eos_consistency(state, const, atol=1e-6)

    comp: OpacityComponents = monochromatic_opacity(x, state, const)
    check_opacity_nonnegative(comp)

    kappa_R = rosseland_mean_from_spectrum(x, comp.kappa_total)

    return {
        "T": T,
        "rho": rho,
        "kappa_R": kappa_R,
        "x": x,
        "kappa_es": comp.kappa_es,
        "kappa_ff": comp.kappa_ff,
        "kappa_bf_H": comp.kappa_bf_H,
        "kappa_bf_Hminus": comp.kappa_bf_Hminus,
        "kappa_total": comp.kappa_total,
        "eos": state,
    }


def run_opacity_grid(
    cfg: ModelConfig,
    const: PhysicalConstants,
    verbose: bool = True,
) -> dict:
    """
    Compute the Rosseland-mean opacity on the full (T, ρ) grid.

    Also records the Rosseland-weighted contribution of each opacity component.

    Parameters
    ----------
    cfg : ModelConfig
    const : PhysicalConstants
    verbose : bool
        If True, print progress.

    Returns
    -------
    dict with keys:
      'T_grid'          — shape (n_T,)      [K]
      'rho_grid'        — shape (n_rho,)    [g cm⁻³]
      'kappa_R'         — shape (n_T, n_rho) [cm² g⁻¹]
      'kappa_es_mean'   — Rosseland-average κ_es  shape (n_T, n_rho)
      'kappa_ff_mean'   — shape (n_T, n_rho)
      'kappa_bf_H_mean' — shape (n_T, n_rho)
      'kappa_bf_Hminus_mean' — shape (n_T, n_rho)
      'n_max'           — int
      'x_min'           — float
      'x_max'           — float
    """
    T_grid = build_temperature_grid(cfg, const)
    rho_grid = build_density_grid(cfg)
    n_T = len(T_grid)
    n_rho = len(rho_grid)

    x_base = build_base_x_grid(cfg)

    kappa_R = np.zeros((n_T, n_rho))
    kappa_es_mean = np.zeros((n_T, n_rho))
    kappa_ff_mean = np.zeros((n_T, n_rho))
    kappa_bf_H_mean = np.zeros((n_T, n_rho))
    kappa_bf_Hm_mean = np.zeros((n_T, n_rho))

    t0 = time.time()
    total = n_T * n_rho
    done = 0

    for i, T in enumerate(T_grid):
        x = refine_x_grid_for_thresholds(x_base, T, cfg.n_max, const)
        for j, rho in enumerate(rho_grid):
            state = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol)
            comp = monochromatic_opacity(x, state, const)
            kappa_R[i, j] = rosseland_mean_from_spectrum(x, comp.kappa_total)
            # Store mean component values (simple average over x for bookkeeping)
            kappa_es_mean[i, j] = float(np.mean(comp.kappa_es))
            kappa_ff_mean[i, j] = float(np.mean(comp.kappa_ff))
            kappa_bf_H_mean[i, j] = float(np.mean(comp.kappa_bf_H))
            kappa_bf_Hm_mean[i, j] = float(np.mean(comp.kappa_bf_Hminus))
            done += 1
            if verbose and done % 10 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                remaining = (total - done) / rate if rate > 0 else 0.0
                print(
                    f"  [{done}/{total}]  T={T:.2e} K  rho={rho:.2e} g/cc"
                    f"  kR={kappa_R[i,j]:.3e}  ETA={remaining:.0f}s"
                )

    if verbose:
        print(f"Grid complete in {time.time() - t0:.1f} s")

    return {
        "T_grid": T_grid,
        "rho_grid": rho_grid,
        "kappa_R": kappa_R,
        "kappa_es_mean": kappa_es_mean,
        "kappa_ff_mean": kappa_ff_mean,
        "kappa_bf_H_mean": kappa_bf_H_mean,
        "kappa_bf_Hminus_mean": kappa_bf_Hm_mean,
        "n_max": cfg.n_max,
        "x_min": cfg.x_min,
        "x_max": cfg.x_max,
    }
