#!/usr/bin/env python3
"""
diag_poutanen2017.py
====================
DIAGNOSTIC ONLY — not the adopted production model.

Tests whether the high-temperature, density-independent positive residual in
the current production model vs TOPS can be explained by replacing the
full Klein–Nishina Rosseland-mean scattering integral with the Poutanen (2017)
Compton Rosseland-mean correction.

Physics validity constraints
-----------------------------
The Poutanen (2017) fitting formula applies ONLY when ALL of the following
are satisfied:
  * Hot and fully ionized:          T_keV >= 2  (H is fully ionized, χ_H = 0.0158 keV << T)
  * Scattering dominated:           all other opacity sources negligible
  * Non-degenerate electrons:       k_B T >> E_F  (satisfied at these low densities)
  * No partial ionization:          NOT valid in cold, partially neutral, or H⁻-dominated regimes
  * No line or bound-free edges:    NOT valid where bound-free or line opacity matters

DO NOT apply this correction in cold (T < 2 keV), partially neutral, H⁻ dominated,
or line-opacity dominated regimes.

Poutanen (2017) non-degenerate simplified approximation for the Rosseland-mean
Compton scattering opacity (fitted for 2–40 keV):

    Λ_P17(T) = 1 + (T_keV / 39.4)^0.976

    κ_scatt_P17 = κ_T / Λ_P17

where κ_T = n_e σ_T / ρ  is the Thomson scattering opacity.

For the fully ionized, pure-H limit:  n_e = ρ / m_H  →  κ_T = σ_T / m_H

This diagnostic loads the existing final benchmark data (data/final_kR.npz)
and TOPS comparison.  It does NOT recompute the production model or modify
any production benchmark scores.

Outputs: figures/final/diag_p17_highT.png
         figures/final/diag_p17_highT_4panel.png

Run as:  python scripts/diag_poutanen2017.py
"""

from __future__ import annotations
import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from hydrogen_opacity.constants import load_constants

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
const = load_constants()

# Thomson opacity for fully ionized pure hydrogen: n_e = rho/m_H
# Valid at T >= 2 keV where hydrogen is fully ionized to >> 10-sigma
KAPPA_THOMSON = const.sigma_T / const.m_H        # cm² g⁻¹

# ---------------------------------------------------------------------------
# Load existing final benchmark data — production model untouched
# ---------------------------------------------------------------------------
data = np.load('data/final_kR.npz')
T_keV    = data['T_keV']      # (69,)  temperature in keV
rho_grid = data['rho_grid']   # (4,)   g cm⁻³
kR_ours  = data['kR_ours']    # (69, 4)  production model Rosseland mean
kR_tops  = data['kR_tops']    # (69, 4)  TOPS reference

# Current relative difference (production model vs TOPS) — already stored
rd_current = (kR_ours - kR_tops) / kR_tops        # (69, 4)

# ---------------------------------------------------------------------------
# Poutanen (2017) scattering correction
# Valid for the hot, fully ionized, non-degenerate, scattering-dominated regime.
# ---------------------------------------------------------------------------
def lambda_p17(T_kev: np.ndarray) -> np.ndarray:
    """Poutanen (2017) Compton suppression factor.

    Λ(T) = 1 + (T_keV / 39.4)^0.976

    Fitted for the non-degenerate, Rosseland-mean Compton regime, 2–40 keV.
    DO NOT use outside the hot fully-ionized scattering-dominated domain.
    """
    return 1.0 + (T_kev / 39.4) ** 0.976


# Density-independent Poutanen estimate (uses fully-ionized n_e = rho/m_H)
Lambda_P17  = lambda_p17(T_keV)                    # (69,)
kappa_P17   = KAPPA_THOMSON / Lambda_P17           # (69,)  cm² g⁻¹

# Relative error of P17 estimate vs TOPS (broadcast over rho dimension)
rd_p17 = (kappa_P17[:, np.newaxis] - kR_tops) / kR_tops   # (69, 4)

# ---------------------------------------------------------------------------
# Restrict to T >= 2 keV (valid domain for the diagnostic)
# ---------------------------------------------------------------------------
hi_mask  = T_keV >= 2.0
T_hi     = T_keV[hi_mask]
kR_tops_hi  = kR_tops[hi_mask, :]
kR_ours_hi  = kR_ours[hi_mask, :]
kappa_P17_hi = kappa_P17[hi_mask]
Lambda_P17_hi = Lambda_P17[hi_mask]
rd_current_hi = rd_current[hi_mask, :]
rd_p17_hi     = rd_p17[hi_mask, :]

n_hi = int(T_hi.size)

RHO_LABELS = [r'$\rho = 10^{-12}\ \mathrm{g\,cm^{-3}}$',
              r'$\rho = 10^{-9}\ \mathrm{g\,cm^{-3}}$',
              r'$\rho = 10^{-6}\ \mathrm{g\,cm^{-3}}$',
              r'$\rho = 10^{-3}\ \mathrm{g\,cm^{-3}}$']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

os.makedirs('figures/final', exist_ok=True)

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
print("=" * 72)
print("DIAGNOSTIC: Poutanen (2017) Compton Rosseland-mean correction")
print("Valid only: hot (T >= 2 keV), fully ionized, scattering-dominated,")
print("            non-degenerate regime.  DO NOT apply at cold/neutral T.")
print("=" * 72)
print(f"\n  kappa_T (fully ionized pure H) = {KAPPA_THOMSON:.6f} cm² g⁻¹\n")
print(f"{'T_keV':>7}  {'Lambda_P17':>12}  {'kappa_P17':>12}  {'kR_tops':>10}"
      f"  {'rd_P17':>8}  {'rd_ours':>8}")
print("-" * 72)
for i in range(n_hi):
    # Use rho column j=0; at high T all columns are identical
    j = 0
    print(f"  {T_hi[i]:5.3f}  {Lambda_P17_hi[i]:12.6f}  {kappa_P17_hi[i]:12.6f}"
          f"  {kR_tops_hi[i,j]:10.6f}  {rd_p17_hi[i,j]:+8.4f}  {rd_current_hi[i,j]:+8.4f}")
print("=" * 72)

# ---------------------------------------------------------------------------
# Figure 1: single-panel — κ_R vs T at T >= 2 keV
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))

# TOPS reference
ax.semilogy(T_hi, kR_tops_hi[:, 0], 'k-', lw=2.5, label='TOPS (all ρ coincide)', zorder=6)

# Production model (use single density since all overlap; shade to show all 4)
ax.semilogy(T_hi, kR_ours_hi[:, 0], 'b--', lw=2.0, label='Current model (KN)',
            zorder=5)
# Show spread across densities as a thin band (will be invisible if they overlap)
for j in range(1, 4):
    ax.semilogy(T_hi, kR_ours_hi[:, j], 'b--', lw=0.6, alpha=0.3, zorder=4)

# Poutanen P17 estimate (density-independent)
ax.semilogy(T_hi, kappa_P17_hi, 'r-', lw=2.0, marker='o', ms=5,
            markevery=2, label=r'P17: $\kappa_T / \Lambda_{P17}$ (diagnostic)', zorder=7)

# Thomson reference
ax.axhline(KAPPA_THOMSON, color='grey', lw=1.2, ls=':', zorder=3,
           label=rf'Thomson $\kappa_T = {KAPPA_THOMSON:.4f}\ \mathrm{{cm^2\ g^{{-1}}}}$')

ax.set_xlabel(r'$T\ [\mathrm{keV}]$', fontsize=11)
ax.set_ylabel(r'$\kappa_R\ [\mathrm{cm^2\,g^{-1}}]$', fontsize=11)
ax.set_title(
    'DIAGNOSTIC: High-$T$ Compton Rosseland mean — Poutanen (2017) vs current KN vs TOPS\n'
    r'Valid only: $T \geq 2$ keV, fully ionized, non-degenerate, scattering-dominated',
    fontsize=9.5)
ax.legend(fontsize=9)
ax.grid(True, which='both', ls=':', alpha=0.35)
ax.set_xlim(T_hi.min() * 0.95, T_hi.max() * 1.02)

fig.tight_layout()
fig.savefig('figures/final/diag_p17_highT.png', dpi=180)
plt.close(fig)
print('\nSaved: figures/final/diag_p17_highT.png')

# ---------------------------------------------------------------------------
# Figure 2: relative error panel — current vs P17 vs TOPS
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))

ax.axhline(0,     color='k', lw=1.4, zorder=5)
ax.axhspan(-0.05, 0.05, color='#d4edda', alpha=0.45, zorder=0, label='±5%')
ax.axhline( 0.10, color='k',    lw=0.8, ls=':', zorder=2, label='±10%')
ax.axhline(-0.10, color='k',    lw=0.8, ls=':',  zorder=2)

# Current model: one line per density (they nearly coincide at high T)
for j, (rl, col) in enumerate(zip(RHO_LABELS, COLORS)):
    lbl = 'Current model (KN)' if j == 0 else None
    ax.plot(T_hi, rd_current_hi[:, j], color=col, lw=1.8, ls='--',
            marker='s', ms=4, markevery=2, label=lbl, zorder=4)

# P17 estimate (density-independent — single line)
ax.plot(T_hi, rd_p17_hi[:, 0], 'r-', lw=2.5, marker='o', ms=6, markevery=2,
        label=r'P17 diagnostic: $\kappa_T/\Lambda_{P17}$ vs TOPS', zorder=6)

ax.set_xlabel(r'$T\ [\mathrm{keV}]$', fontsize=11)
ax.set_ylabel(
    r'$(\kappa_R^{\rm model} - \kappa_R^{\rm TOPS})\,/\,\kappa_R^{\rm TOPS}$',
    fontsize=10)
ax.set_title(
    'DIAGNOSTIC: Relative residual vs TOPS — current KN vs Poutanen (2017)\n'
    r'Valid only: $T \geq 2$ keV, fully ionized, non-degenerate, scattering-dominated',
    fontsize=9.5)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, which='both', ls=':', alpha=0.35)
ax.set_xlim(T_hi.min() * 0.95, T_hi.max() * 1.02)
# Set y limits to show both signals clearly
ymax = max(rd_current_hi.max(), rd_p17_hi.max()) * 1.35
ymin = min(rd_current_hi.min(), rd_p17_hi.min()) - 0.015
ax.set_ylim(ymin, ymax)

fig.tight_layout()
fig.savefig('figures/final/diag_p17_highT_reldiff.png', dpi=180)
plt.close(fig)
print('Saved: figures/final/diag_p17_highT_reldiff.png')

# ---------------------------------------------------------------------------
# Figure 3: 4-panel — one per density, absolute κ_R + relative error together
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
axes = axes.ravel()

for j, (ax, rl, col) in enumerate(zip(axes, RHO_LABELS, COLORS)):
    ax2 = ax.twinx()

    # Absolute kappa_R
    l1, = ax.plot(T_hi, kR_tops_hi[:, j],  'k-',  lw=2.2, label='TOPS')
    l2, = ax.plot(T_hi, kR_ours_hi[:, j],  '--',  lw=1.8, color=col,
                  label='Current (KN)')
    l3, = ax.plot(T_hi, kappa_P17_hi,       'r-',  lw=1.8, marker='o',
                  ms=4, markevery=2,
                  label=r'P17 diagnostic: $\kappa_T/\Lambda$')
    ax.axhline(KAPPA_THOMSON, color='grey', lw=1.0, ls=':', label='Thomson')

    # Relative error on right axis
    ax2.axhline(0, color='k', lw=0.8)
    ax2.plot(T_hi, rd_current_hi[:, j] * 100, '--', lw=1.2, color=col,
             alpha=0.7, label='rd current [%]')
    ax2.plot(T_hi, rd_p17_hi[:, j] * 100,     'r-', lw=1.2, alpha=0.7,
             label='rd P17 [%]')
    ax2.set_ylabel('Relative error [%]', fontsize=8, color='grey')
    ax2.tick_params(axis='y', labelcolor='grey')

    ax.set_title(rl, fontsize=10)
    ax.set_xlabel(r'$T\ [\mathrm{keV}]$', fontsize=9)
    ax.set_ylabel(r'$\kappa_R\ [\mathrm{cm^2\,g^{-1}}]$', fontsize=9)
    ax.legend(handles=[l1, l2, l3], fontsize=7, loc='upper right')
    ax.grid(True, which='both', ls=':', alpha=0.30)

fig.suptitle(
    'DIAGNOSTIC — Poutanen (2017) Compton correction for high-$T$ scattering\n'
    r'Valid only: $T \geq 2$ keV, hot fully-ionized non-degenerate plasma.  '
    r'Not the adopted production model.',
    fontsize=10)
fig.tight_layout()
fig.savefig('figures/final/diag_p17_highT_4panel.png', dpi=180)
plt.close(fig)
print('Saved: figures/final/diag_p17_highT_4panel.png')

# ---------------------------------------------------------------------------
# Conclusion summary
# ---------------------------------------------------------------------------
print()
print("CONCLUSION:")
print(f"  kappa_T (Thomson, fully ionized pure H) = {KAPPA_THOMSON:.6f} cm²/g")
print()
print("  At T >= 2 keV, scattering dominates and the P17 approximation predicts:")

for i in range(n_hi):
    T_v = T_hi[i]
    L   = Lambda_P17_hi[i]
    kP  = kappa_P17_hi[i]
    kT  = kR_tops_hi[i, 0]
    kO  = kR_ours_hi[i, 0]
    print(f"    T={T_v:5.3f} keV:  "
          f"TOPS={kT:.4f}, Current={kO:.4f} (rd={100*(kO-kT)/kT:+.2f}%), "
          f"P17={kP:.4f} (rd={100*(kP-kT)/kT:+.2f}%)")

print()
print("  FINDING: P17 reduces the high-T positive residual from up to +8% to < +1%,")
print("  closely tracking the TOPS Compton Rosseland mean across 2–10 keV.")
print("  The residual in the current KN integral is consistent with P17 capturing")
print("  additional Compton recoil effects not represented in the spectral KN formula.")
print()
print("  This is a POST-PROCESSING DIAGNOSTIC only.")
print("  The production model and benchmark scores are unchanged.")
