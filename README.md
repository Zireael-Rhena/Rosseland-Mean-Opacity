# Rosseland-Mean Opacity of Pure Hydrogen Gas

## Scientific Objective

Compute the **Rosseland-mean opacity** $\kappa_R(T, \rho)$ of a **pure hydrogen gas** in **Local Thermodynamic Equilibrium (LTE)** over a grid of temperatures and densities relevant to stellar interiors and atmospheres.

## Formal Analysis Domain

The **formal analysis domain** for this project is:

| Parameter | Range |
|---|---|
| Temperature | 0.0005 – 10 keV |
| Mass density | 10⁻¹² – 10⁻³ g cm⁻³ |

All physical verification, convergence testing, and LANL/TOPS benchmarking are
restricted to this domain.  Runs at densities above 10⁻³ g cm⁻³ are
**out-of-scope stress tests only** and are not part of the formal benchmark domain.
The `default_config()` is set to this domain.

---

## Course Context

- **Course:** ASTRON C207 — Radiation Processes in Astronomy
- **Institution:** UC Berkeley
- **Semester:** Spring 2026
- **Instructor:** Prof. Wenbin Lu

---

## Included Physics

| Source | Symbol | Notes |
|---|---|---|
| Electron scattering | $\kappa_{\text{es}}$ | Thomson cross-section |
| Free-free (e-p) absorption | $\kappa_{\text{ff}}$ | Stimulated-emission corrected; Kramers-like with quantum Gaunt factor |
| Neutral-H bound-free | $\kappa_{\text{bf,H}}$ | Hydrogenic per shell n; stimulated-emission corrected |
| $H^-$ bound-free | $\kappa_{\text{bf,H}^-}$ | Empirical fit; domain-limited to $0.754–10 \ \text{eV}$ |

## Excluded Physics

The following are **intentionally omitted** as a controlled project approximation:

- **$H^-$ free-free** — excluded; would require a separate empirical fit
- **Bound-bound (line) opacity** — excluded; continuum-only model
- **Non-LTE effects** — excluded; strict LTE throughout
- **Full continuum lowering / pressure ionization** — excluded; a conservative density-dependent level cutoff is applied as a reduced proxy (see below)
- **Relativistic scattering corrections** — excluded; non-relativistic temperatures only

---

## Physics Definitions

### Equation of State

The EOS is solved by a 1D root-find in the electron number density $n_e$ at each $(T, \rho)$.

Total hydrogen nucleus density:
$$
    n_{H,tot} = \rho / m_H
$$

Conservation:
$$
    n_{H_0} + n_p + n_{H^-} = n_{H,tot}
$$

$$
    n_p = n_e + n_{H^-}
$$

**Density-dependent level cutoff (conservative pressure-ionization proxy):**

At high density, excited levels are collisionally dissolved.  As a conservative
proxy the sum over principal quantum shells is truncated at an effective cutoff:

$$
n_{\max}^{\rm eff}(\rho) \simeq 100 \left(\frac{\rho/m_H}{10^{12}\ \text{cm}^{-3}}\right)^{-1/6}
$$

The integer cutoff actually used is

$$
n_{\rm cut} = \max\!\left(1,\;\min\!\left(n_{\max}^{\rm user},\;\lfloor n_{\max}^{\rm eff}\rfloor\right)\right)
$$

The default user shell cap is $n_{\max}^{\rm user} = 16$.  The density-dependent
formula typically limits the active shells well below 16 at densities above
$\sim 10^{-6}\ \mathrm{g\,cm^{-3}}$ (e.g., $n_{\rm cut} = 3$ at $10^{-3}\ \mathrm{g\,cm^{-3}}$),
so the cap only matters at the lowest densities in the formal domain.

This cutoff is applied **self-consistently** in the neutral-H partition function
$U_H(T,\rho)$, the Saha prefactor, the $H^-$ equilibrium constant $K_{H^-}$,
and the level populations.  It is a reduced model, not a full non-ideal EOS
treatment (Hummer–Mihalas occupation probabilities are not included).

**Neutral-H partition function:**
$$
    U_H(T,\rho) = \sum_{n=1}^{n_{\rm cut}(\rho)} 2n^2 \exp(-E_n^{\text{exc}} / k_B T)
$$

where $E_n^{\text{exc}} = 13.6 \text{ eV} (1 − 1/n²)$.

**Hydrogen Saha equation:**
$$
    n_e n_p / n_{H_0} = S_H(T) = (2π m_e k_B T / h²)^{3/2} \cdot (2 / U_H) \cdot exp(−χ_H / k_B T)
$$

**H⁻ equilibrium (exact):**
$$
    n_{H^-} = K_{H^-}(T) \cdot n_{H_0} \cdot n_e
$$

$$
    K_{H^-}(T) = (1 / 2 U_H) \cdot λ_{th,e}³ \cdot exp(χ_{H^-} / k_B T)
$$

$$
    λ_{th,e} = h / \sqrt{2π m_e k_B T}
$$

### Electron Scattering
$$
    \kappa_{\text{es}} = n_e \sigma_T / \rho \ \text{[cm² g⁻¹]}
$$

### Free-Free (e-p) Absorption

Net absorption coefficient (stimulated emission already included):
$$
    \alpha_{\text{ff}} = (4\sqrt{\pi} e^6) / (3\sqrt{3} m_e^2 h c) \cdot \sqrt{2 m_e / k_B T} \cdot (1 - e^{-h\nu/k_B T}) / \nu^3 \cdot n_e n_p g_{\text{ff}}(\nu, T)
$$

Free-free Gaunt factor:
$$
    g_{\text{ff}}(\nu, T, Z) \approx \ln[ e + exp( 6 - (\sqrt{3}/\pi) \ln(\nu_9 T_4^{-1} \max(0.25, Z T_4^{-1/2})) ) ]
$$

where $ν_9 = ν / 10⁹ \ \text{Hz}$, $T_4 = T / 10^4 \ \text{K}$.

### Neutral-H Bound-Free Absorption

For shell $n$, ionization threshold $\chi_n = 13.6 \ eV / n^2$.

Cross-section (zero below threshold, hydrogenic above):
$$
    \sigma_{n,bf}(ν) = n^{−5} · (8\pi / 3\sqrt{3}) · (m_e e^{10}) / (c \hbar^3 (h\nu)³) · g_{bf}(\nu, T)
$$

where $g_{bf} = 1$ in the baseline build.

Net opacity (with stimulated-emission correction):
$$
    \kappa_{\text{bf},H} = (1 / \rho) \sum_{n=1}^{n_{\text{max}}} n_n \sigma_{n,bf}(\nu) · (1 − e^{−h\nu/k_B T})
$$

### H⁻ Bound-Free Absorption

Empirical cross-section $(\lambda \text{ in } \mu\text{m}, \lambda_0 = 1.64 \ \mu\text{m})$:
$$
    \sigma_{\text{H}^-, \text{bf}}(\lambda) = 1.53 × 10^{-16} \ \text{cm}^2 · \lambda^3 · (1/\lambda − 1/\lambda_0)^{3/2}
$$

Domain restriction:
- $h\nu ≤ 0.754 \text{ eV} → \sigma = 0$ (below ionization threshold)
- $0.754 \text{ eV} < h\nu ≤ 10 \text{ eV}$ → use the fit
- $h\nu > 10 \text{ eV} → \sigma = 0$ (outside validity range; do not extrapolate)

Net opacity (with stimulated-emission correction):
$$
    \kappa_{\text{bf},\text{H}^-} = (n_{\text{H}^-} / \rho) \sigma_{\text{H}^-, \text{bf}}(\nu) · (1 − e^{−h\nu/k_B T})
$$

**H⁻ free-free is intentionally excluded** as a controlled approximation for this project.

---

## Numerical Rosseland Mean

The Rosseland mean is defined as:
$$
    1 / \kappa_R = ∫ (1 / \kappa_\nu^{\text{tot}}) w_R(x) dx  /  ∫ w_R(x) dx
$$

where $x = h\nu / k_B T$ and the weight function is:
$$
    w_R(x) = x^4 e^x / (e^x − 1)²
$$

Numerical integration is performed over a refined x-grid (default $x \in [0.01, 30]$) using the trapezoidal rule on a threshold-refined non-uniform grid.

---

## Run Instructions

### Install

```bash
cd hydrogen_opacity
pip install -e ".[dev]"
```

### Run a single (T, ρ) point

```python
from hydrogen_opacity.constants import load_constants
from hydrogen_opacity.config import default_config
from hydrogen_opacity.driver import run_single_point

const = load_constants()
cfg = default_config()
result = run_single_point(T=1e4, rho=1e-7, cfg=cfg, const=const)
print(result)
```

### Run a full grid

```bash
python scripts/run_grid.py
```

### Plot spectra

```bash
python scripts/plot_spectra.py
```

---

## Output Description

- `kappa_R_grid.npz` — Numpy archive with arrays:
  - `T_grid` [K]
  - `rho_grid` [g cm⁻³]
  - `kappa_R` [cm² g⁻¹], shape (n_T, n_rho)
  - `kappa_es`, `kappa_ff`, `kappa_bf_H`, `kappa_bf_Hminus` — component contributions

---

## Validation Tests

Run with:

```bash
pytest tests/
```

Tests verify:

1. **Constants sanity** — CGS values within $1\ \text{ppm}$ of NIST CODATA 2018
2. **EOS positivity and conservation** — $n_{\text{H}0}, n_p, n_e, n_{\text{H}^-} \geq 0$; number and charge conservation
3. **Neutral-H bound-free threshold** — $\sigma = 0$ below $h\nu = \chi_n$; continuous above
4. **H⁻ bound-free domain** — $\sigma = 0$ for $h\nu \leq 0.754 \text{ eV}$ and $h\nu > 10 \text{ eV}$
5. **Opacity non-negativity** — all components $\geq 0$ at all $x$
6. **Rosseland positivity and finiteness** — $\kappa_R > 0$ and finite at representative points
7. **Regression** — $\kappa_R$ at selected $(T, \rho)$ reproduces stored reference values to $1\%$

---

## Verification Status

Domain-internal verification has been completed for the formal domain
($T \in [0.0005, 10]\ \mathrm{keV}$, $\rho \in [10^{-12}, 10^{-3}]\ \mathrm{g\,cm^{-3}}$):

| Check | Result |
|---|---|
| EOS positivity and conservation (10 representative points) | ✓ Pass |
| Opacity component non-negativity across full spectrum | ✓ Pass |
| H⁻ bf domain enforcement (0.754–10 eV) | ✓ Pass |
| Neutral-H bf threshold positions (n = 1..16) | ✓ Pass |
| No spectral oscillations or numerical spikes | ✓ Pass |
| x-grid convergence (base vs 4× refined, threshold-refined) | ✓ Pass (<1% in-domain) |
| n_max sensitivity at T ≥ 0.05 keV | ✓ Negligible (<1 ppm) |
| n_max sensitivity at cold T, ρ ≤ 10⁻⁶ (old caveat) | ✓ Resolved by n_max=16 |

**Previous caveat eliminated:** with the old default `n_max=8`, the code
underestimated $\kappa_R$ by ~5% at $(T \approx 0.0005\ \mathrm{keV},\ \rho \approx 10^{-6}\ \mathrm{g\,cm^{-3}})$
because the density cutoff allowed shells up to $n=10$ but the user cap forced $n_{\rm cut}=8$.
The default `n_max=16` removes this discrepancy; those shells are now fully included.

Minor residual: at the most dilute in-domain conditions ($\rho \lesssim 10^{-9}\ \mathrm{g\,cm^{-3}}$),
$n_{\max}^{\rm eff} \approx 34$, so shells 17–34 remain excluded by the cap. The resulting
sensitivity is < 0.5%, which is negligible relative to expected LANL/TOPS differences.

**The code is ready for LANL/TOPS comparison inside the formal domain.**

---

## Limitations and Expected Differences Relative to LANL/TOPS

| Effect | This code | LANL/TOPS |
|---|---|---|
| $H^-$ free-free | omitted | included |
| Bound-bound (lines) | omitted | included |
| Pressure ionization | conservative $n_{\rm cut}(\rho)$ proxy | full Hummer-Mihalas |
| Relativistic corrections | omitted | included at $T > \text{few} \ \text{keV}$ |
| Non-LTE | omitted | available |
| $g_{bf}(\nu)$ Gaunt factor | = 1 | Quantum mechanical |

Expect $\kappa_R$ from this code to **underestimate** LANL/TOPS by factors of a few to ~10× in cool, dense regions where H⁻ free-free and line opacity dominate.

---

## Units

All internal calculations use CGS. Temperatures are accepted in $\text{K}$; the grid is specified in keV and converted via $1 \text{ keV} = 1.16045 × 10⁷ \text{ K}$.
