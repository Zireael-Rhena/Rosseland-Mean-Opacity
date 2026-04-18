#!/usr/bin/env python3
"""
compare_lanl.py
===============
Local comparison utility for LANL/TOPS opacity data.

This script expects pre-downloaded LANL/TOPS data.
It does NOT fetch any data from the internet.

Usage
-----
    python scripts/compare_lanl.py --lanl-file TOPS_data.npz --our-file kappa_R_grid.npz

Expected LANL data file format
-------------------------------
The TOPS data file must be a .npz archive containing:
  'T_grid'   — temperature grid [K]      shape (n_T,)
  'rho_grid' — density grid [g/cc]       shape (n_rho,)
  'kappa_R'  — Rosseland mean [cm²/g]    shape (n_T, n_rho)

Data acquisition (not performed here)
--------------------------------------
LANL/TOPS opacity tables can be downloaded from:
  https://aphysics2.lanl.gov/apps/opacity
  (requires registration)

Interpolation
-------------
The TOPS table is interpolated to the (T, ρ) grid of the local calculation
using bilinear interpolation in log-log space.

Differences expected
---------------------
See README.md for a table of excluded physics.  In brief:
  - H⁻ free-free (excluded here) → our κ_R < LANL in cool/dense regime
  - Bound-bound lines (excluded here) → large underestimate at T < 2e4 K
  - Pressure ionization (excluded here) → differences at ρ > 1 g/cc
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np


def load_tops_data(path: str) -> dict:
    """
    Load a pre-downloaded TOPS/LANL opacity file.

    Parameters
    ----------
    path : str
        Path to .npz file with keys T_grid, rho_grid, kappa_R.

    Returns
    -------
    dict
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"LANL data file not found: {path}\n"
            "Download TOPS data from https://aphysics2.lanl.gov/apps/opacity"
            " and save in the expected format (see script docstring)."
        )
    data = np.load(path, allow_pickle=False)
    required = {"T_grid", "rho_grid", "kappa_R"}
    missing = required - set(data.keys())
    if missing:
        raise KeyError(f"Missing keys in LANL file: {missing}")
    return dict(data)


def interpolate_tops_to_grid(
    tops: dict,
    T_grid: np.ndarray,
    rho_grid: np.ndarray,
) -> np.ndarray:
    """
    Bilinear interpolation of TOPS κ_R onto our (T, ρ) grid in log-log space.

    Parameters
    ----------
    tops : dict
        TOPS data with T_grid, rho_grid, kappa_R.
    T_grid : ndarray   [K]
    rho_grid : ndarray   [g/cc]

    Returns
    -------
    kappa_R_interp : ndarray shape (n_T, n_rho)
    """
    from scipy.interpolate import RegularGridInterpolator

    T_tops = np.asarray(tops["T_grid"])
    rho_tops = np.asarray(tops["rho_grid"])
    kR_tops = np.asarray(tops["kappa_R"])

    # Log-log interpolation
    log_T_tops = np.log10(T_tops)
    log_rho_tops = np.log10(rho_tops)
    log_kR_tops = np.log10(np.maximum(kR_tops, 1e-100))

    interp = RegularGridInterpolator(
        (log_T_tops, log_rho_tops),
        log_kR_tops,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    log_T_our = np.log10(T_grid)
    log_rho_our = np.log10(rho_grid)
    T_mesh, rho_mesh = np.meshgrid(log_T_our, log_rho_our, indexing="ij")
    pts = np.column_stack([T_mesh.ravel(), rho_mesh.ravel()])
    log_kR_interp = interp(pts).reshape(len(T_grid), len(rho_grid))
    return 10.0 ** log_kR_interp


def compare(our_file: str, lanl_file: str, output_dir: str, save_figs: bool) -> None:
    """
    Load both datasets, interpolate TOPS to our grid, and report differences.
    """
    from hydrogen_opacity.io_utils import load_grid_from_npz

    our = load_grid_from_npz(our_file)
    tops = load_tops_data(lanl_file)

    T_grid = our["T_grid"]
    rho_grid = our["rho_grid"]
    kR_our = our["kappa_R"]

    kR_lanl = interpolate_tops_to_grid(tops, T_grid, rho_grid)

    ratio = kR_our / np.maximum(kR_lanl, 1e-100)
    frac_diff = (kR_our - kR_lanl) / np.maximum(kR_lanl, 1e-100)

    print("Comparison statistics (our / LANL):")
    print(f"  Median ratio : {np.nanmedian(ratio):.3f}")
    print(f"  Mean   ratio : {np.nanmean(ratio):.3f}")
    print(f"  Max    ratio : {np.nanmax(ratio):.3f}")
    print(f"  Min    ratio : {np.nanmin(ratio):.3f}")
    print(f"  Frac diff > 10% : {(np.abs(frac_diff) > 0.1).sum()} / {ratio.size}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        T_mesh, rho_mesh = np.meshgrid(T_grid, rho_grid, indexing="ij")

        for ax, data, title in [
            (axes[0], np.log10(np.maximum(kR_our, 1e-100)), "Our code  log₁₀ κ_R"),
            (axes[1], np.log10(ratio), "Ratio  log₁₀(ours / LANL)"),
        ]:
            cf = ax.contourf(np.log10(T_mesh), np.log10(rho_mesh), data, 20, cmap="plasma")
            plt.colorbar(cf, ax=ax)
            ax.set_xlabel("log₁₀ T [K]")
            ax.set_ylabel("log₁₀ ρ [g/cc]")
            ax.set_title(title)

        fig.tight_layout()
        if save_figs:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "lanl_comparison.png")
            fig.savefig(path, dpi=150)
            print(f"Saved: {path}")
        else:
            plt.show()
        plt.close(fig)
    except ImportError:
        print("matplotlib not available — skipping plot.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare local κ_R against pre-downloaded LANL/TOPS data."
    )
    parser.add_argument(
        "--our-file", default="kappa_R_grid.npz",
        help="Path to our opacity grid .npz file."
    )
    parser.add_argument(
        "--lanl-file", required=True,
        help="Path to pre-downloaded LANL/TOPS .npz file."
    )
    parser.add_argument("--save", action="store_true", help="Save figures.")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    compare(args.our_file, args.lanl_file, args.output_dir, args.save)


if __name__ == "__main__":
    main()
