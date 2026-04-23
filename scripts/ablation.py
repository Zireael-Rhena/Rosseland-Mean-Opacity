"""
ablation.py
===========
Ablation study: compute kappa_R and EOS diagnostics for models A-D.

Model A: KN only        (use_kn=T, use_ff_hminus=F, use_lowering=F)
Model B: H- ff only     (use_kn=F, use_ff_hminus=T, use_lowering=F)
Model C: lowering only  (use_kn=F, use_ff_hminus=F, use_lowering=T)
Model D: H- ff + KN     (use_kn=T, use_ff_hminus=T, use_lowering=F)

Run as:  python scripts/ablation.py
Outputs: data/ablation_kR.npz, data/ablation_eos.npz
"""

from __future__ import annotations
import sys, re
import numpy as np

sys.path.insert(0, 'src')

from hydrogen_opacity.constants import load_constants
from hydrogen_opacity.config import default_config, ModelOptions
from hydrogen_opacity.grids import build_base_x_grid, refine_x_grid_for_thresholds
from hydrogen_opacity.rosseland import compute_rosseland_mean
from hydrogen_opacity.eos import solve_eos

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
const = load_constants()
cfg   = default_config()
KEV_TO_K = 1.16045e7

tops = np.load('data/tops_parsed.npz')
T_keV    = tops['T_grid']      # (69,)
rho_grid = tops['rho_grid']    # (4,) = [1e-12, 1e-9, 1e-6, 1e-3]
kR_tops  = tops['kR_tops']     # (69, 4)

# ---------------------------------------------------------------------------
# Parse No. Free from raw TOPS file
# ---------------------------------------------------------------------------
def parse_tops_no_free(path: str, T_keV_ref, rho_ref) -> np.ndarray:
    """Parse No. Free (y_e = n_e/n_H) from TOPS text file."""
    n_T   = len(T_keV_ref)
    n_rho = len(rho_ref)
    no_free = np.full((n_T, n_rho), np.nan)
    header_re = re.compile(r'T=\s*([\d.E+\-]+)')
    data_re   = re.compile(r'^\s*([\d.E+\-]+)\s+([\d.E+\-]+)\s+([\d.E+\-]+)\s+([\d.E+\-]+)')
    with open(path) as f:
        lines = f.readlines()
    i_T = -1
    for line in lines:
        m = header_re.search(line)
        if m:
            T_val = float(m.group(1))
            diffs = np.abs(T_keV_ref - T_val)
            i_T = int(np.argmin(diffs)) if diffs.min() < 1e-7 * T_val else -1
            continue
        if i_T < 0:
            continue
        m = data_re.match(line)
        if m:
            rho_val = float(m.group(1))
            nf_val  = float(m.group(4))
            diffs_r = np.abs(rho_ref - rho_val)
            i_rho = int(np.argmin(diffs_r))
            if diffs_r[i_rho] < 1e-7 * rho_val:
                no_free[i_T, i_rho] = nf_val
    return no_free

no_free_tops = parse_tops_no_free('data/tops_hydrogen_gray.txt', T_keV, rho_grid)

# ---------------------------------------------------------------------------
# Ablation model definitions
# ---------------------------------------------------------------------------
MODELS = {
    'A': ModelOptions(use_kn=True,  use_ff_hminus=False, lowering_mode="none"),
    'B': ModelOptions(use_kn=False, use_ff_hminus=True,  lowering_mode="none"),
    'C': ModelOptions(use_kn=False, use_ff_hminus=False, lowering_mode="full"),
    'D': ModelOptions(use_kn=True,  use_ff_hminus=True,  lowering_mode="none"),
}
MODEL_LABELS = {
    'A': 'A: KN only',
    'B': 'B: H⁻ff only',
    'C': 'C: Lowering only',
    'D': 'D: H⁻ff + KN',
}

# ---------------------------------------------------------------------------
# Key diagnostic points
# ---------------------------------------------------------------------------
KEY_POINTS = [
    # Cold / high density
    (5.0e-4, 1e-3),
    (6.0e-4, 1e-3),
    (8.0e-4, 1e-3),
    # Cool / moderate density
    (1.0e-3, 1e-6),
    (2.0e-3, 1e-6),
    (2.5e-3, 1e-6),
    # Cool / dilute
    (1.25e-3, 1e-9),
    (1.50e-3, 1e-9),
]

# ---------------------------------------------------------------------------
# Stage 2: compute kappa_R grids for all models
# ---------------------------------------------------------------------------
x_base = build_base_x_grid(cfg)

# kR arrays: shape (n_models, n_T, n_rho)
model_keys = list(MODELS.keys())
n_models = len(model_keys)
n_T   = len(T_keV)
n_rho = len(rho_grid)

kR_models = np.zeros((n_models, n_T, n_rho))

for mi, mk in enumerate(model_keys):
    opts = MODELS[mk]
    print(f'Computing model {mk}: {MODEL_LABELS[mk]} ...', flush=True)
    for i, T_k in enumerate(T_keV):
        T = T_k * KEV_TO_K
        x = refine_x_grid_for_thresholds(x_base, T, cfg.n_max, const)
        for j, rho in enumerate(rho_grid):
            kR_models[mi, i, j] = compute_rosseland_mean(
                T, rho, cfg.n_max, x, const, tol=cfg.root_tol, opts=opts
            )
    print(f'  done.', flush=True)

np.savez('data/ablation_kR.npz',
         model_keys=model_keys,
         T_keV=T_keV,
         rho_grid=rho_grid,
         kR_models=kR_models,
         kR_tops=kR_tops)
print('Saved: data/ablation_kR.npz')

# ---------------------------------------------------------------------------
# Stage 2b: EOS diagnostics at key points
# ---------------------------------------------------------------------------
diag_rows = []  # list of dicts, one per (model, T_keV, rho)

for mk, opts in MODELS.items():
    for T_k, rho in KEY_POINTS:
        T = T_k * KEV_TO_K

        # TOPS No. Free at this (T, rho)
        dT  = np.abs(T_keV - T_k)
        dr  = np.abs(rho_grid - rho)
        i_T = int(np.argmin(dT))
        i_r = int(np.argmin(dr))
        tops_nf = no_free_tops[i_T, i_r] if (dT[i_T] < 1e-7*T_k and dr[i_r] < 1e-7*rho) else np.nan
        tops_kR = kR_tops[i_T, i_r]      if (dT[i_T] < 1e-7*T_k and dr[i_r] < 1e-7*rho) else np.nan

        state = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol, opts=opts)
        x     = refine_x_grid_for_thresholds(x_base, T, cfg.n_max, const)
        kR    = compute_rosseland_mean(T, rho, cfg.n_max, x, const, tol=cfg.root_tol, opts=opts)

        from hydrogen_opacity.eos import effective_nmax_float
        n_max_phys = state.n_max_phys_effective
        n_cut      = state.n_cut_effective
        chi_H_eff  = (const.chi_H_ev * (1.0 - 1.0/n_max_phys**2)
                      if opts.use_lowering else const.chi_H_ev)
        y_e        = state.n_e / state.n_H_tot
        hminus_oe  = state.n_Hminus / max(state.n_e, 1e-300)
        rd         = (kR - tops_kR) / tops_kR if not np.isnan(tops_kR) else np.nan

        diag_rows.append(dict(
            model=mk, T_keV=T_k, rho=rho,
            n_max_phys=n_max_phys, n_cut=n_cut, chi_H_eff=chi_H_eff,
            n_e=state.n_e, n_p=state.n_p, n_H0=state.n_H0, n_Hminus=state.n_Hminus,
            y_e=y_e, hminus_over_ne=hminus_oe,
            kR=kR, tops_kR=tops_kR, rd=rd, tops_no_free=tops_nf,
        ))

# Save EOS diagnostics
np.savez('data/ablation_eos.npz',
         no_free_tops=no_free_tops,
         T_keV_diag=np.array([r['T_keV'] for r in diag_rows]),
         rho_diag=np.array([r['rho'] for r in diag_rows]),
         model_diag=np.array([r['model'] for r in diag_rows]),
         y_e=np.array([r['y_e'] for r in diag_rows]),
         kR_diag=np.array([r['kR'] for r in diag_rows]),
         rd_diag=np.array([r['rd'] for r in diag_rows]),
)
print('Saved: data/ablation_eos.npz')

# ---------------------------------------------------------------------------
# Print EOS diagnostic table
# ---------------------------------------------------------------------------
print()
print('=' * 110)
print(f"{'Model':6s} {'T(keV)':9s} {'rho':9s} | {'n_max_phys':10s} {'n_cut':5s} {'chi_eff':8s} | "
      f"{'y_e':10s} {'TOPS_nf':10s} | {'n_Hminus/ne':11s} | {'kR':10s} {'TOPS_kR':10s} {'rd':8s}")
print('-' * 110)
for r in diag_rows:
    print(f"{r['model']:6s} {r['T_keV']:9.4e} {r['rho']:9.2e} | "
          f"{r['n_max_phys']:10.4f} {r['n_cut']:5d} {r['chi_H_eff']:8.4f} | "
          f"{r['y_e']:10.4e} {r['tops_no_free']:10.4e} | "
          f"{r['hminus_over_ne']:11.4e} | "
          f"{r['kR']:10.4e} {r['tops_kR']:10.4e} {r['rd']:+8.4f}")
print('=' * 110)
