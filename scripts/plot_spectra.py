#!/usr/bin/env python3
"""
plot_spectra.py
===============
Plot monochromatic opacity spectra and the Rosseland-mean grid.

Usage
-----
    cd hydrogen_opacity
    python scripts/plot_spectra.py [--save] [--output-dir OUTDIR]

Produces
--------
  spectrum_T1e4_rho1e-7.png  — κ(x) components at cool conditions
  spectrum_T1e5_rho1e-3.png  — κ(x) components at warm conditions
  kappa_R_contour.png        — log κ_R contour map over (T, ρ)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not found — skipping plots.")


def plot_spectrum(T: float, rho: float, cfg, const, save: bool, outdir: str) -> None:
    from hydrogen_opacity.grids import build_base_x_grid, refine_x_grid_for_thresholds
    from hydrogen_opacity.eos import solve_eos
    from hydrogen_opacity.opacity import monochromatic_opacity

    x_base = build_base_x_grid(cfg)
    x = refine_x_grid_for_thresholds(x_base, T, cfg.n_max, const)
    state = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol)
    comp = monochromatic_opacity(x, state, const)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(x, comp.kappa_es, label=r"$\kappa_{\rm es}$", ls="--", lw=1.5)
    ax.semilogy(x, np.maximum(comp.kappa_ff, 1e-40), label=r"$\kappa_{\rm ff}$", lw=1.5)
    ax.semilogy(x, np.maximum(comp.kappa_bf_H, 1e-40), label=r"$\kappa_{\rm bf,H}$", lw=1.5)
    ax.semilogy(x, np.maximum(comp.kappa_bf_Hminus, 1e-40), label=r"$\kappa_{\rm bf,H^-}$", lw=1.5)
    ax.semilogy(x, comp.kappa_total, label=r"$\kappa_{\rm tot}$", color="k", lw=2)
    ax.set_xlabel(r"$x = h\nu / k_B T$")
    ax.set_ylabel(r"$\kappa\; [\mathrm{cm}^2\,\mathrm{g}^{-1}]$")
    ax.set_title(f"Monochromatic opacity:  T = {T:.1e} K,  ρ = {rho:.1e} g/cm³")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(x[0], x[-1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = f"spectrum_T{T:.0e}_rho{rho:.0e}.png".replace("+", "").replace("e0", "e")
    if save:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, fname)
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    else:
        plt.show()
    plt.close(fig)


def plot_rosseland_contour(cfg, const, save: bool, outdir: str) -> None:
    from hydrogen_opacity.driver import run_opacity_grid

    grid = run_opacity_grid(cfg, const, verbose=True)
    T_grid = grid["T_grid"]
    rho_grid = grid["rho_grid"]
    kappa_R = grid["kappa_R"]

    fig, ax = plt.subplots(figsize=(8, 6))
    T_mesh, rho_mesh = np.meshgrid(T_grid, rho_grid, indexing="ij")
    levels = np.linspace(np.log10(kappa_R.min() + 1e-100),
                         np.log10(kappa_R.max()), 20)
    cf = ax.contourf(
        np.log10(T_mesh), np.log10(rho_mesh),
        np.log10(kappa_R), levels=levels, cmap="plasma"
    )
    plt.colorbar(cf, ax=ax, label=r"$\log_{10} \kappa_R\; [\mathrm{cm}^2\,\mathrm{g}^{-1}]$")
    ax.set_xlabel(r"$\log_{10} T\; [\mathrm{K}]$")
    ax.set_ylabel(r"$\log_{10} \rho\; [\mathrm{g\,cm}^{-3}]$")
    ax.set_title("Rosseland-Mean Opacity — Pure Hydrogen LTE")
    fig.tight_layout()

    fname = "kappa_R_contour.png"
    if save:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, fname)
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot opacity spectra and Rosseland grid.")
    parser.add_argument("--save", action="store_true", help="Save figures instead of displaying.")
    parser.add_argument("--output-dir", default="figures", help="Directory for saved figures.")
    args = parser.parse_args()

    if not HAS_MPL:
        print("Install matplotlib to use this script.")
        return

    from hydrogen_opacity.constants import load_constants
    from hydrogen_opacity.config import default_config

    const = load_constants()
    cfg = default_config()

    # Spectral plots
    for T, rho in [(1e4, 1e-7), (1e5, 1e-3)]:
        plot_spectrum(T, rho, cfg, const, save=args.save, outdir=args.output_dir)

    # Rosseland contour (uses a reduced grid for speed)
    from hydrogen_opacity.config import ModelConfig
    small_cfg = ModelConfig(
        T_min_keV=0.001, T_max_keV=10.0, n_T=12,
        rho_min=1e-10, rho_max=1e-3, n_rho=10,
        n_max=cfg.n_max, x_min=cfg.x_min, x_max=cfg.x_max,
        n_x_base=cfg.n_x_base, root_tol=cfg.root_tol, max_root_iter=cfg.max_root_iter,
    )
    plot_rosseland_contour(small_cfg, const, save=args.save, outdir=args.output_dir)


if __name__ == "__main__":
    main()
