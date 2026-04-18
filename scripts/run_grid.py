#!/usr/bin/env python3
"""
run_grid.py
===========
Compute the Rosseland-mean opacity grid and save to disk.

Usage
-----
    cd hydrogen_opacity
    python scripts/run_grid.py [--output OUTPUT] [--csv]

Output
------
    kappa_R_grid.npz   (default)  — NumPy archive
    kappa_R_grid.csv   (optional) — flat CSV
"""

import argparse
import os
import sys

# Allow running from the hydrogen_opacity/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hydrogen_opacity.constants import load_constants
from hydrogen_opacity.config import default_config
from hydrogen_opacity.driver import run_opacity_grid
from hydrogen_opacity.io_utils import save_grid_to_npz, save_grid_to_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Rosseland-mean opacity grid.")
    parser.add_argument(
        "--output", default="kappa_R_grid",
        help="Output file base name (without extension).  Default: kappa_R_grid",
    )
    parser.add_argument(
        "--csv", action="store_true",
        help="Also write a CSV file in addition to the .npz archive.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    args = parser.parse_args()

    const = load_constants()
    cfg = default_config()

    print(f"Running opacity grid:")
    print(f"  T range  : {cfg.T_min_keV:.3f} – {cfg.T_max_keV:.1f} keV  ({cfg.n_T} points)")
    print(f"  rho range: {cfg.rho_min:.1e} – {cfg.rho_max:.1e} g/cc  ({cfg.n_rho} points)")
    print(f"  n_max    : {cfg.n_max}")
    print(f"  x grid   : {cfg.x_min} – {cfg.x_max}  (base {cfg.n_x_base} pts)")

    grid = run_opacity_grid(cfg, const, verbose=not args.quiet)

    npz_path = args.output if args.output.endswith(".npz") else args.output + ".npz"
    save_grid_to_npz(args.output, grid)
    print(f"Saved: {npz_path}")

    if args.csv:
        csv_path = args.output.rstrip(".npz") + ".csv"
        save_grid_to_csv(csv_path, grid)
        print(f"Saved: {csv_path}")

    # Print a summary
    import numpy as np
    kR = grid["kappa_R"]
    print(f"\nRosseland mean opacity summary:")
    print(f"  min κ_R = {kR.min():.3e} cm²/g")
    print(f"  max κ_R = {kR.max():.3e} cm²/g")
    print(f"  shape   = {kR.shape}  (n_T × n_rho)")


if __name__ == "__main__":
    main()
