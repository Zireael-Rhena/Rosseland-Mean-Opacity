"""
cap_sweep.py
============
Stage 2: compute kappa_R and EOS state for the cap sweep.

Models: none, full, capped_0.5 through capped_1.5 (step 0.1 eV)
Densities: rho = 1e-6, 1e-3 g/cm³
T grid: full TOPS temperature grid (69 points)
Physics: H⁻ ff ON, KN ON

Run as:  python scripts/cap_sweep.py
Output:  data/cap_sweep.npz
"""

from __future__ import annotations
import sys, re
import numpy as np

sys.path.insert(0, 'src')

from hydrogen_opacity.constants import load_constants
from hydrogen_opacity.config import default_config, ModelOptions
from hydrogen_opacity.grids import build_base_x_grid, refine_x_grid_for_thresholds
from hydrogen_opacity.eos import solve_eos, effective_nmax_float, effective_ncut
from hydrogen_opacity.rosseland import compute_rosseland_mean

const = load_constants()
cfg   = default_config()
KEV_TO_K = 1.16045e7

# ---------------------------------------------------------------------------
# Load TOPS reference data
# ---------------------------------------------------------------------------
tops = np.load('data/tops_parsed.npz')
T_keV_all = tops['T_grid']    # (69,)
rho_all   = tops['rho_grid']  # (4,)
kR_tops   = tops['kR_tops']   # (69, 4)

# Focus densities
FOCUS_RHOS = [1e-6, 1e-3]
j_rho = [int(np.argmin(np.abs(rho_all - r))) for r in FOCUS_RHOS]  # column indices in kR_tops
T_keV = T_keV_all  # all 69 points

# Parse TOPS No. Free
def parse_tops_no_free(path, T_keV_ref, rho_ref):
    nf = np.full((len(T_keV_ref), len(rho_ref)), np.nan)
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
        m2 = data_re.match(line)
        if m2:
            rv  = float(m2.group(1))
            nfv = float(m2.group(4))
            dr  = np.abs(rho_ref - rv)
            ir  = int(np.argmin(dr))
            if dr[ir] < 1e-7 * rv:
                nf[i_T, ir] = nfv
    return nf

no_free_tops = parse_tops_no_free('data/tops_hydrogen_gray.txt', T_keV_all, rho_all)

# ---------------------------------------------------------------------------
# Build model list
# ---------------------------------------------------------------------------
cap_values = np.round(np.arange(0.5, 1.55, 0.1), 10)  # 0.5 … 1.5

MODEL_SPECS = []   # list of (label, ModelOptions)
MODEL_SPECS.append(("none", ModelOptions(use_kn=True, use_ff_hminus=True, lowering_mode="none")))
MODEL_SPECS.append(("full", ModelOptions(use_kn=True, use_ff_hminus=True, lowering_mode="full")))
for cap in cap_values:
    label = f"capped_{cap:.1f}eV"
    MODEL_SPECS.append((label, ModelOptions(use_kn=True, use_ff_hminus=True,
                                            lowering_mode="capped", delta_chi_max_ev=float(cap))))

MODEL_LABELS = [m[0] for m in MODEL_SPECS]
n_models = len(MODEL_SPECS)
n_T      = len(T_keV)
n_focus  = len(FOCUS_RHOS)  # 2

print(f'Models: {n_models}  T-points: {n_T}  Focus densities: {FOCUS_RHOS}')

# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------
x_base = build_base_x_grid(cfg)

# Arrays: (n_models, n_T, n_focus)
kR_sw    = np.zeros((n_models, n_T, n_focus))
ye_sw    = np.zeros((n_models, n_T, n_focus))
ne_sw    = np.zeros((n_models, n_T, n_focus))
np_sw    = np.zeros((n_models, n_T, n_focus))
nH0_sw   = np.zeros((n_models, n_T, n_focus))
nHm_sw   = np.zeros((n_models, n_T, n_focus))
hmne_sw  = np.zeros((n_models, n_T, n_focus))
chi_sw   = np.zeros((n_models, n_T, n_focus))
nmp_sw   = np.zeros((n_T, n_focus))   # n_max_phys (independent of model)
ncut_sw  = np.zeros((n_T, n_focus), dtype=int)

for jf, (rho, j_col) in enumerate(zip(FOCUS_RHOS, j_rho)):
    for i, T_k in enumerate(T_keV):
        nmp_sw[i, jf]  = effective_nmax_float(rho, const)
        ncut_sw[i, jf] = effective_ncut(rho, cfg.n_max, const)
    print(f'  rho={rho:.0e}: n_max_phys={nmp_sw[0, jf]:.4f}  n_cut={ncut_sw[0, jf]}', flush=True)

for mi, (label, opts) in enumerate(MODEL_SPECS):
    print(f'Computing model "{label}" ...', flush=True)
    for jf, (rho, j_col) in enumerate(zip(FOCUS_RHOS, j_rho)):
        for i, T_k in enumerate(T_keV):
            T   = T_k * KEV_TO_K
            x   = refine_x_grid_for_thresholds(x_base, T, cfg.n_max, const)
            st  = solve_eos(T, rho, cfg.n_max, const, tol=cfg.root_tol, opts=opts)
            kR  = compute_rosseland_mean(T, rho, cfg.n_max, x, const,
                                         tol=cfg.root_tol, opts=opts)
            n_mp = nmp_sw[i, jf]
            # Compute chi_H_eff used in this model
            mode = opts.lowering_mode
            if mode == "none":
                chi = const.chi_H_ev
            elif mode == "full":
                chi = const.chi_H_ev * (1.0 - 1.0 / n_mp**2)
            elif mode in ("capped", "capped_1eV"):
                cap = opts.delta_chi_max_ev if mode == "capped" else 1.0
                chi = const.chi_H_ev - min(const.chi_H_ev / n_mp**2, cap)
            else:
                chi = float('nan')

            ye   = st.n_e / st.n_H_tot
            hmne = st.n_Hminus / max(st.n_e, 1e-300)

            kR_sw[mi, i, jf]   = kR
            ye_sw[mi, i, jf]   = ye
            ne_sw[mi, i, jf]   = st.n_e
            np_sw[mi, i, jf]   = st.n_p
            nH0_sw[mi, i, jf]  = st.n_H0
            nHm_sw[mi, i, jf]  = st.n_Hminus
            hmne_sw[mi, i, jf] = hmne
            chi_sw[mi, i, jf]  = chi
    print(f'  done.', flush=True)

# TOPS reference arrays aligned to focus densities: (n_T, n_focus)
kR_tops_focus = np.column_stack([kR_tops[:, j] for j in j_rho])
nf_tops_focus = np.column_stack([no_free_tops[:, j] for j in j_rho])

np.savez('data/cap_sweep.npz',
         model_labels=MODEL_LABELS,
         cap_values=np.array([float('nan'), float('nan')] + list(cap_values)),
         T_keV=T_keV,
         focus_rhos=np.array(FOCUS_RHOS),
         n_max_phys=nmp_sw,
         n_cut=ncut_sw,
         chi_H_eff=chi_sw,
         n_e=ne_sw,
         n_p=np_sw,
         n_H0=nH0_sw,
         n_Hminus=nHm_sw,
         hminus_over_ne=hmne_sw,
         y_e=ye_sw,
         kR=kR_sw,
         kR_tops=kR_tops_focus,
         y_e_tops=nf_tops_focus)
print('Saved: data/cap_sweep.npz')
print(f'Array shapes: model_labels={len(MODEL_LABELS)}, T_keV={T_keV.shape}, '
      f'kR={kR_sw.shape}, y_e={ye_sw.shape}')
