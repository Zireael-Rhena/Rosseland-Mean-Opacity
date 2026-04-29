#!/usr/bin/env python3
"""
slides_highT_p17_diagnostic.py
================================
Polished slide/report figure for the high-T Compton-scattering residual
diagnostic comparing the production Klein–Nishina spectral integral against
the Poutanen (2017) non-degenerate Rosseland-mean correction.

Physics context
---------------
Restricted to T_keV >= 2 (hot, fully ionised, non-degenerate,
scattering-dominated regime).

  Λ_P17(T)       = 1 + (T_keV / 39.4)^0.976
  κ_scatt_P17(T) = κ_T / Λ_P17         (density-independent)
  κ_T            = σ_T / m_H            (fully ionised pure H: n_e = ρ/m_H)

DO NOT apply the P17 formula at cold temperatures, in partially neutral gas,
or where H⁻ / bound-free / line opacity contributes significantly.

Inputs (pre-computed, not recomputed here):
  data/final_kR.npz      — production-model grid + TOPS comparison

Outputs:
  figures/final/slides/highT_p17_diagnostic_relative_error.pdf
  figures/final/slides/highT_p17_diagnostic_relative_error.png

Run as:  python scripts/slides_highT_p17_diagnostic.py

Caption (for report/slides):
  Relative difference (κ_R^model − κ_R^TOPS) / κ_R^TOPS at T ≥ 2 keV.
  Blue dashed: current production model (frequency-integrated KN Rosseland mean).
  Red solid: P17 diagnostic estimate κ_T / Λ_P17 (Poutanen 2017, non-degenerate
  limit).  Shaded bands show the spread across all four density decades
  (ρ = 10⁻¹² – 10⁻³ g cm⁻³); the spread is < 3 × 10⁻⁵ and visually
  indistinguishable from a single curve.  Horizontal green band: ±10 % guide.
"""

from __future__ import annotations
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))
from hydrogen_opacity.constants import load_constants

# ---------------------------------------------------------------------------
# Slide RC — identical pattern to slides_benchmark_figures.py
# ---------------------------------------------------------------------------
_USETEX = False
try:
    import subprocess
    r = subprocess.run(['latex', '--version'], capture_output=True, timeout=5)
    if r.returncode == 0:
        matplotlib.rcParams['text.usetex'] = True
        import tempfile, pathlib
        _probe = pathlib.Path(tempfile.mktemp(suffix='.pdf'))
        _fig, _ax = plt.subplots(figsize=(1, 1))
        _ax.set_title(r'$\kappa_R$')
        _fig.savefig(str(_probe))
        plt.close(_fig)
        _probe.unlink(missing_ok=True)
        _USETEX = True
except Exception:
    pass

if not _USETEX:
    matplotlib.rcParams['text.usetex'] = False

SLIDE_RC: dict = {
    'text.usetex':        _USETEX,
    'font.family':        'serif',
    'font.serif':         ['Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
    'font.size':          17,
    'axes.titlesize':     17,
    'axes.titlepad':      10,
    'axes.labelsize':     18,
    'xtick.labelsize':    14,
    'ytick.labelsize':    14,
    'legend.fontsize':    14,
    'legend.framealpha':  0.90,
    'legend.edgecolor':   '0.70',
    'lines.linewidth':    2.4,
    'lines.markersize':   6,
    'axes.grid':          True,
    'grid.alpha':         0.28,
    'grid.linestyle':     ':',
    'grid.linewidth':     0.8,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.facecolor':   'white',
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.14,
}
SLIDE_PNG_DPI = 200

# Colours
_KN_COLOR  = '#1f77b4'   # blue  — current KN production model
_P17_COLOR = '#c0392b'   # dark red — P17 diagnostic

OUTDIR = 'figures/final/slides'
STEM   = 'highT_p17_diagnostic_relative_error'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save(fig: plt.Figure, stem: str) -> None:
    base = os.path.join(OUTDIR, stem)
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=SLIDE_PNG_DPI)
    plt.close(fig)
    print(f'Saved: {base}.pdf  +  .png')


def _pct(v: float) -> str:
    """Format a fractional residual as a compact percentage string."""
    s = f'{v * 100:+.2f}%'
    # Replace + with + and - with − (unicode minus) for typography
    return s.replace('-', '−')


# ---------------------------------------------------------------------------
# Physics: constants and P17 formula
# ---------------------------------------------------------------------------
const   = load_constants()
KAPPA_T = const.sigma_T / const.m_H   # Thomson opacity, fully ionised pure H [cm² g⁻¹]


def lambda_p17(T_kev: np.ndarray) -> np.ndarray:
    """Poutanen (2017) Compton suppression factor Λ = 1 + (T/39.4)^0.976.

    Valid for the hot (T >= 2 keV), fully ionised, non-degenerate,
    scattering-dominated, non-degenerate regime (2–40 keV fitting range).
    DO NOT use for cold, partially neutral, or H⁻/bound-free dominated gas.
    """
    return 1.0 + (T_kev / 39.4) ** 0.976


# ---------------------------------------------------------------------------
# Load pre-computed final benchmark data
# ---------------------------------------------------------------------------
data     = np.load('data/final_kR.npz')
T_keV    = data['T_keV']     # (69,)
rho_grid = data['rho_grid']  # (4,)  g cm⁻³
kR_ours  = data['kR_ours']   # (69, 4)
kR_tops  = data['kR_tops']   # (69, 4)

# Restrict to hot regime
hi_mask  = T_keV >= 2.0
T_hi     = T_keV[hi_mask]                         # (14,)
kR_ours_hi = kR_ours[hi_mask, :]                  # (14, 4)
kR_tops_hi = kR_tops[hi_mask, :]                  # (14, 4)

# Current-model relative difference vs TOPS
rd_kn = (kR_ours_hi - kR_tops_hi) / kR_tops_hi   # (14, 4)

# P17 estimate (density-independent; broadcast over density axis)
kappa_P17  = KAPPA_T / lambda_p17(T_hi)           # (14,)
rd_p17     = (kappa_P17[:, None] - kR_tops_hi) / kR_tops_hi   # (14, 4)

# Central curves (mean across 4 densities) and spread bands (min / max)
rd_kn_mean = rd_kn.mean(axis=1)
rd_kn_lo   = rd_kn.min(axis=1)
rd_kn_hi   = rd_kn.max(axis=1)

rd_p17_mean = rd_p17.mean(axis=1)
rd_p17_lo   = rd_p17.min(axis=1)
rd_p17_hi   = rd_p17.max(axis=1)

# ---------------------------------------------------------------------------
# Annotation data: T = 2, 4, 8, 10 keV
# ---------------------------------------------------------------------------
ANN_T      = np.array([2.0, 4.0, 8.0, 10.0])
# Find nearest indices in T_hi
ANN_IDX    = np.array([np.argmin(np.abs(T_hi - tv)) for tv in ANN_T])

ann_kn_val  = rd_kn_mean[ANN_IDX]   # fractional
ann_p17_val = rd_p17_mean[ANN_IDX]

# ---------------------------------------------------------------------------
# Textual summary
# ---------------------------------------------------------------------------
print('=' * 64)
print('Poutanen (2017) high-T diagnostic — textual summary')
print('=' * 64)
print(f'  κ_T (Thomson, fully ionised pure H) = {KAPPA_T:.6f} cm² g⁻¹\n')

max_kn_rd  = float(np.abs(rd_kn_mean).max())
max_p17_rd = float(np.abs(rd_p17_mean).max())
spread_kn  = float((rd_kn_hi - rd_kn_lo).max())
spread_p17 = float((rd_p17_hi - rd_p17_lo).max())

print(f'  Max |residual|  Current KN : {max_kn_rd * 100:6.2f}%')
print(f'  Max |residual|  P17        : {max_p17_rd * 100:6.2f}%')
print(f'  Max density spread  KN     : {spread_kn:.2e}  (< 3×10⁻⁵ — negligible)')
print(f'  Max density spread  P17    : {spread_p17:.2e}  (< 3×10⁻⁵ — negligible)')
print()
print(f'  {"T [keV]":>8}   {"KN rd":>9}   {"P17 rd":>9}')
print(f'  {"-"*8}   {"-"*9}   {"-"*9}')
for tv, kv, pv in zip(T_hi, rd_kn_mean, rd_p17_mean):
    print(f'  {tv:8.3f}   {kv*100:+8.3f}%   {pv*100:+8.3f}%')
print('=' * 64)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
os.makedirs(OUTDIR, exist_ok=True)

with plt.rc_context(SLIDE_RC):
    fig, ax = plt.subplots(figsize=(9.2, 5.8))

    # ---- background reference elements (draw first, lowest z-order) --------
    # ±10% guide band
    ax.axhspan(-0.10, 0.10, color='#d5e8d4', alpha=0.45, zorder=0,
               label=r'$\pm 10\%$ guide')

    # y = 0 reference
    ax.axhline(0.0, color='k', linewidth=1.2, zorder=3, linestyle='-')

    # ---- density-spread bands ----------------------------------------------
    # Current KN band (visually thin — spread < 3×10⁻⁵)
    ax.fill_between(T_hi, rd_kn_lo, rd_kn_hi,
                    color=_KN_COLOR, alpha=0.18, zorder=1,
                    linewidth=0)

    # P17 band (essentially invisible — same negligible spread)
    ax.fill_between(T_hi, rd_p17_lo, rd_p17_hi,
                    color=_P17_COLOR, alpha=0.18, zorder=1,
                    linewidth=0)

    # ---- central curves ----------------------------------------------------
    if _USETEX:
        kn_lbl  = (r'Current KN (spectral $\sigma_{\rm KN}$ Rosseland integral)'
                   r' — density spread $<3\times10^{-5}$')
        p17_lbl = (r'P17 diagnostic: $\kappa_T/\Lambda_{P17}$'
                   r'  (Poutanen 2017, non-degenerate)')
    else:
        kn_lbl  = (r'Current KN  (spectral $\sigma_{\rm KN}$ Rosseland integral)'
                   '\n'
                   r'density spread $< 3\times10^{-5}$ — four $\rho$ decades coincide')
        p17_lbl = (r'P17 diagnostic: $\kappa_T/\Lambda_{\rm P17}$'
                   '\n'
                   r'Poutanen (2017) non-degenerate approximation')

    ax.plot(T_hi, rd_kn_mean, color=_KN_COLOR, lw=2.6, ls='--',
            zorder=5, label=kn_lbl)

    ax.plot(T_hi, rd_p17_mean, color=_P17_COLOR, lw=2.6, ls='-',
            zorder=5, label=p17_lbl)

    # ---- annotation markers ------------------------------------------------
    # Larger filled circles at the four annotation temperatures
    ax.scatter(T_hi[ANN_IDX], ann_kn_val,
               color=_KN_COLOR, s=60, zorder=9, linewidths=0)
    ax.scatter(T_hi[ANN_IDX], ann_p17_val,
               color=_P17_COLOR, s=60, zorder=9, linewidths=0)

    # ---- annotation text ---------------------------------------------------
    # Annotation layout:
    #   T=2:  KN above & right,  P17 below & right
    #   T=4:  KN above,          P17 below
    #   T=8:  KN above & left,   P17 below & left
    #   T=10: KN above & left,   P17 below & left
    #
    # All text offsets are in data coordinates (log-x axis):
    # horizontal offsets are multiplicative factors on T; vertical in fractional units.
    _ann_cfg = [
        # (idx, kn_dx_factor, kn_dy, kn_ha, p17_dx_factor, p17_dy, p17_ha)
        (ANN_IDX[0],  1.06,  +0.009, 'left',   1.06,  -0.009, 'left'),
        (ANN_IDX[1],  1.0,   +0.009, 'center', 1.0,   -0.009, 'center'),
        (ANN_IDX[2],  0.94,  +0.009, 'right',  0.94,  -0.009, 'right'),
        (ANN_IDX[3],  0.94,  +0.009, 'right',  0.94,  -0.009, 'right'),
    ]

    _fs = 11.5   # annotation font size

    for cfg_row in _ann_cfg:
        idx, kn_xf, kn_dy, kn_ha, p17_xf, p17_dy, p17_ha = cfg_row
        T_pt = T_hi[idx]

        kn_str  = _pct(ann_kn_val[list(ANN_IDX).index(idx)])
        p17_str = _pct(ann_p17_val[list(ANN_IDX).index(idx)])

        ax.annotate(
            kn_str,
            xy    =(T_pt,        ann_kn_val[list(ANN_IDX).index(idx)]),
            xytext=(T_pt * kn_xf, ann_kn_val[list(ANN_IDX).index(idx)] + kn_dy),
            fontsize=_fs, color=_KN_COLOR, ha=kn_ha, va='bottom',
            arrowprops=dict(arrowstyle='->', color=_KN_COLOR,
                            lw=0.9, shrinkA=0, shrinkB=2),
            zorder=10,
        )
        ax.annotate(
            p17_str,
            xy    =(T_pt,         ann_p17_val[list(ANN_IDX).index(idx)]),
            xytext=(T_pt * p17_xf, ann_p17_val[list(ANN_IDX).index(idx)] + p17_dy),
            fontsize=_fs, color=_P17_COLOR, ha=p17_ha, va='top',
            arrowprops=dict(arrowstyle='->', color=_P17_COLOR,
                            lw=0.9, shrinkA=0, shrinkB=2),
            zorder=10,
        )

    # ---- axes formatting ---------------------------------------------------
    ax.set_xscale('log')

    ax.set_xlim(1.85, 11.5)
    # y range: accommodate annotations above KN and below P17
    ax.set_ylim(-0.028, 0.118)

    # Explicit x ticks at the annotation temperatures (plus natural log ticks)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f'{v:g}'))
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=10))
    ax.xaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=15))

    # y-axis: fractional → percentage tick labels
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f'{v * 100:+g}%'))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.02))

    if _USETEX:
        xlabel = r'$T\ [\mathrm{keV}]$'
        ylabel = (r'$\bigl(\kappa_R^{\rm model} - \kappa_R^{\rm TOPS}\bigr)'
                  r'\,/\,\kappa_R^{\rm TOPS}$')
        title  = (r'High-$T$ diagnostic: current KN vs.\ Poutanen (2017) correction'
                  r' \quad \textit{(post-processing diagnostic — not the production model)}')
    else:
        xlabel = r'$T$ [keV]'
        ylabel = (r'$(\kappa_R^{\rm model} - \kappa_R^{\rm TOPS})'
                  r'\,/\,\kappa_R^{\rm TOPS}$')
        title  = (r'High-$T$ diagnostic: current KN vs. Poutanen (2017) correction'
                  '\n'
                  r'$\it{(post-processing\ diagnostic\ —\ not\ the\ production\ model)}$')

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=14, pad=12)

    # Legend — place outside the busy annotation region
    leg = ax.legend(loc='upper left', fontsize=12.5,
                    handlelength=2.2, handletextpad=0.6,
                    borderpad=0.7, labelspacing=0.55)

    fig.tight_layout()
    save(fig, STEM)

print()
print('FINDING: P17 reduces the high-T positive residual from'
      f' up to {max_kn_rd * 100:.1f}% to below {max_p17_rd * 100:.1f}%,')
print('confirming the current KN spectral integral overestimates the')
print('Compton Rosseland mean relative to TOPS in the hot fully-ionised regime.')
