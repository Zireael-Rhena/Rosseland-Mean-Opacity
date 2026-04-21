#!/usr/bin/env python3
"""
compare_lanl.py
===============
Compare our Rosseland-mean opacity against the LANL/TOPS gray-opacity table
for pure hydrogen.
"""

import os, sys
import numpy as np

os.makedirs("figures", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

tops = np.load("data/tops_parsed.npz")
ours = np.load("data/ours_kR.npz")

T_keV     = tops["T_grid"]
rho_grid  = tops["rho_grid"]
kR_tops   = tops["kR_tops"]
kR_ours   = ours["kR_ours"]
warn_T    = tops["warn_T_keV"]

off_bound = np.isin(T_keV, warn_T)
reldiff   = (kR_ours - kR_tops) / kR_tops

rho_labels = [r"$\rho=10^{-12}$ g/cc", r"$\rho=10^{-9}$ g/cc",
              r"$\rho=10^{-6}$ g/cc",  r"$\rho=10^{-3}$ g/cc"]
colors     = ["steelblue", "darkorange", "green", "crimson"]

# ── Numeric summary ────────────────────────────────────────────────────────────
print("=" * 70)
print("Stage 3: Comparison summary")
print("=" * 70)

for j, (rho, label) in enumerate(zip(rho_grid, rho_labels)):
    if j == 0:
        valid = ~off_bound
        print(f"\nrho=1e-12 [valid for T < 0.225 keV only, {valid.sum()} pts]")
    else:
        valid = np.ones(len(T_keV), dtype=bool)
        print(f"\nrho={rho:.0e} ({valid.sum()} pts)")

    rd  = reldiff[:, j][valid]
    kRT = kR_tops[:, j][valid]
    kRO = kR_ours[:, j][valid]
    Tv  = T_keV[valid]

    print(f"  rel-diff range  : [{rd.min():+.3f}, {rd.max():+.3f}]")
    print(f"  |rd| < 10%: {(np.abs(rd)<0.10).sum()}/{len(rd)} pts")
    print(f"  |rd| < 50%: {(np.abs(rd)<0.50).sum()}/{len(rd)} pts")
    worst = np.argsort(np.abs(rd))[-3:][::-1]
    print(f"  Worst 3:")
    for ii in worst:
        print(f"    T={Tv[ii]:.4e} keV  ours={kRO[ii]:.3e}  TOPS={kRT[ii]:.3e}  rd={rd[ii]:+.3f}")

# ── Figure 1: kappa_R(T) ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

for j, (ax, rho, label, color) in enumerate(zip(axes, rho_grid, rho_labels, colors)):
    kRT = kR_tops[:, j]
    kRO = kR_ours[:, j]
    if j == 0:
        valid = ~off_bound
        ax.semilogy(T_keV[valid],  kRO[valid],  "-",  color=color, lw=2,   label="Our code")
        ax.semilogy(T_keV[valid],  kRT[valid],  "--", color="k",   lw=1.5, label="TOPS (exact ρ)")
        ax.semilogy(T_keV[~valid], kRO[~valid], "-",  color=color, lw=2,   alpha=0.3)
        ax.semilogy(T_keV[~valid], kRT[~valid], ":",  color="gray",lw=1.5,
                    label=r"TOPS (subst. ρ, T≥0.225 keV)")
        ax.axvline(warn_T[0], color="gray", ls="--", lw=0.8)
    else:
        ax.semilogy(T_keV, kRO, "-",  color=color, lw=2,   label="Our code")
        ax.semilogy(T_keV, kRT, "--", color="k",   lw=1.5, label="TOPS")
    ax.set_xlabel(r"$T$ [keV]", fontsize=10)
    ax.set_ylabel(r"$\kappa_R$ [cm² g$^{-1}$]", fontsize=10)
    ax.set_title(label, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xscale("log")

fig.suptitle("Rosseland-Mean Opacity: Our Code vs LANL/TOPS — Pure H LTE", fontsize=12)
fig.tight_layout()
fig.savefig("figures/comparison_kR.png", dpi=150)
plt.close(fig)
print("\nSaved: figures/comparison_kR.png")

# ── Figure 2: relative difference ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

for j, (rho, label, color) in enumerate(zip(rho_grid, rho_labels, colors)):
    rd = reldiff[:, j]
    if j == 0:
        valid = ~off_bound
        ax.semilogx(T_keV[valid], rd[valid], "-o", color=color, lw=1.5, ms=3,
                    label=label + " (exact ρ)")
        ax.semilogx(T_keV[~valid], rd[~valid], "x", color=color, ms=5, alpha=0.45,
                    label=label + r" (subst. ρ)")
    else:
        ax.semilogx(T_keV, rd, "-o", color=color, lw=1.5, ms=3, label=label)

ax.axhline(0,    color="k", lw=0.8)
ax.axhline( 0.1, color="k", lw=0.5, ls="--", alpha=0.4)
ax.axhline(-0.1, color="k", lw=0.5, ls="--", alpha=0.4)
ax.axvline(warn_T[0], color="gray", lw=0.8, ls="--", alpha=0.7,
           label="off-bound boundary (T=0.225 keV)")
ax.set_xlabel(r"$T$ [keV]", fontsize=12)
ax.set_ylabel(r"$(\kappa^{\rm ours} - \kappa^{\rm TOPS})\,/\,\kappa^{\rm TOPS}$", fontsize=12)
ax.set_title("Relative Difference: Our Code vs LANL/TOPS — Pure H LTE", fontsize=12)
ax.legend(fontsize=8, ncol=2, loc="lower left")
ax.grid(True, alpha=0.3, which="both")
ax.set_ylim(-1.05, 0.25)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
fig.tight_layout()
fig.savefig("figures/comparison_reldiff.png", dpi=150)
plt.close(fig)
print("Saved: figures/comparison_reldiff.png")

# ── Figure 3: cold-T zoom ─────────────────────────────────────────────────────
cold = T_keV <= 0.05
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for j, (rho, label, color) in enumerate(zip(rho_grid, rho_labels, colors)):
    ax.semilogy(T_keV[cold], kR_ours[cold, j], "-",  color=color, lw=2,   label=f"Ours {label}")
    ax.semilogy(T_keV[cold], kR_tops[cold, j], "--", color=color, lw=1.5, alpha=0.7,
                label=f"TOPS {label}")
ax.set_xlabel(r"$T$ [keV]", fontsize=11)
ax.set_ylabel(r"$\kappa_R$ [cm² g$^{-1}$]", fontsize=11)
ax.set_title("Cold-T regime (T ≤ 0.05 keV)", fontsize=11)
ax.legend(fontsize=6, ncol=2)
ax.grid(True, alpha=0.3, which="both")
ax.set_xscale("log")

ax = axes[1]
for j, (rho, label, color) in enumerate(zip(rho_grid, rho_labels, colors)):
    ax.semilogx(T_keV[cold], reldiff[cold, j], "-o", color=color, lw=1.5, ms=4, label=label)
ax.axhline(0, color="k", lw=0.8)
ax.axhline( 0.1, color="k", lw=0.5, ls="--", alpha=0.4)
ax.axhline(-0.1, color="k", lw=0.5, ls="--", alpha=0.4)
ax.set_xlabel(r"$T$ [keV]", fontsize=11)
ax.set_ylabel("Relative difference", fontsize=11)
ax.set_title("Relative difference, cold regime", fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which="both")
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
fig.suptitle("Cold-T Comparison: Our Code vs LANL/TOPS", fontsize=12)
fig.tight_layout()
fig.savefig("figures/comparison_coldT.png", dpi=150)
plt.close(fig)
print("Saved: figures/comparison_coldT.png")
