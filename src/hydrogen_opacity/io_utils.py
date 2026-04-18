"""
io_utils.py
===========
Save and load opacity grid results.

Supported formats:
  * NumPy .npz archive (recommended for large grids)
  * CSV (for inspection / portability)
"""

from __future__ import annotations

import csv
import os

import numpy as np


def save_grid_to_npz(path: str, result: dict) -> None:
    """
    Save opacity grid results to a NumPy .npz archive.

    Parameters
    ----------
    path : str
        Output file path (will be created or overwritten).
        The .npz extension will be appended if not present.
    result : dict
        Dictionary of arrays (and scalars) to save.
        Expected keys (all optional except T_grid, rho_grid, kappa_R):
          'T_grid'         — temperature grid [K]        shape (n_T,)
          'rho_grid'       — density grid [g/cc]         shape (n_rho,)
          'kappa_R'        — Rosseland mean [cm2/g]      shape (n_T, n_rho)
          'kappa_es'       — electron scattering         shape (n_T, n_rho)  [optional]
          'kappa_ff'       — free-free                   shape (n_T, n_rho)  [optional]
          'kappa_bf_H'     — neutral-H bound-free        shape (n_T, n_rho)  [optional]
          'kappa_bf_Hminus'— H- bound-free               shape (n_T, n_rho)  [optional]

    Notes
    -----
    Scalar metadata (e.g. n_max) should be stored as 0-d arrays.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) if os.path.dirname(path) else ".", exist_ok=True)
    np.savez(path, **result)


def save_grid_to_csv(path: str, result: dict) -> None:
    """
    Save the Rosseland-mean opacity grid to a flat CSV file.

    Columns: T_K, rho_gcc, kappa_R, and any additional component arrays.

    Parameters
    ----------
    path : str
        Output file path.
    result : dict
        Must contain 'T_grid', 'rho_grid', 'kappa_R'.
        Optional component arrays are written as additional columns if present.
    """
    T_grid = np.asarray(result["T_grid"])
    rho_grid = np.asarray(result["rho_grid"])
    kappa_R = np.asarray(result["kappa_R"])

    optional_keys = ["kappa_es", "kappa_ff", "kappa_bf_H", "kappa_bf_Hminus"]
    optional_arrays = {k: np.asarray(result[k]) for k in optional_keys if k in result}

    header = ["T_K", "rho_gcc", "kappa_R"] + list(optional_arrays.keys())

    os.makedirs(os.path.dirname(os.path.abspath(path)) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, T in enumerate(T_grid):
            for j, rho in enumerate(rho_grid):
                row = [T, rho, kappa_R[i, j]]
                for arr in optional_arrays.values():
                    row.append(arr[i, j])
                writer.writerow(row)


def load_grid_from_npz(path: str) -> dict:
    """
    Load an opacity grid from a .npz archive.

    Parameters
    ----------
    path : str
        Path to the .npz file.

    Returns
    -------
    dict
        Keys and arrays as stored by ``save_grid_to_npz``.
    """
    data = np.load(path, allow_pickle=False)
    return dict(data)
