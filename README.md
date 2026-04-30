# Rosseland-Mean Opacity of Pure Hydrogen Gas

Computes the **Rosseland-mean opacity** $\kappa_R(T, \rho)$ of a **pure hydrogen gas**
in **Local Thermodynamic Equilibrium (LTE)** and benchmarks it against LANL/TOPS
gray Rosseland-mean opacity tables.

| | |
|---|---|
| **Benchmark score** | 238 / 276 grid points (86.2%) within 10% of TOPS |
| **Tests** | 185 passing |
| **Domain** | $T = 0.0005$–$10\ \text{keV}$; $\rho = 10^{-12}$–$10^{-3}\ \text{g\,cm}^{-3}$ |
| **Final model** | `ModelOptions(use_kn=True, use_ff_hminus=True, lowering_mode="capped", delta_chi_max_ev=1.0, compton_mean_mode="poutanen2017")` |
| **Course** | ASTRON C207 — Radiation Processes in Astronomy, UC Berkeley, Spring 2026 |

---

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

## Installation

Requires **Python ≥ 3.11**.

```bash
# Clone and enter the repository
git clone <repo-url>
cd Rosseland-Mean-Opacity

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install the package (editable) with all dependencies
pip install -e ".[dev]"
```

The `.[dev]` extra adds `pytest` for running the test suite.
The main package installs `numpy`, `scipy`, and `matplotlib` (all required to run
the benchmark scripts and reproduce figures).

Verify the installation:

```bash
python -c "import hydrogen_opacity; print('OK')"
```

---

## Quick Start

### Run the test suite

```bash
python -m pytest
```

Expected output: **185 passed**.

### Compute a single opacity point

```python
from hydrogen_opacity.constants import load_constants
from hydrogen_opacity.config import default_config, production_opts
from hydrogen_opacity.grids import build_base_x_grid, refine_x_grid_for_thresholds
from hydrogen_opacity.rosseland import compute_rosseland_mean

const = load_constants()
cfg   = default_config()
opts  = production_opts()          # final production model with P17 correction

T   = 1e7    # K  (≈ 0.86 keV)
rho = 1e-7   # g cm⁻³

x = refine_x_grid_for_thresholds(build_base_x_grid(cfg), T, cfg.n_max, const)
kappa_R = compute_rosseland_mean(T, rho, cfg.n_max, x, const,
                                 tol=cfg.root_tol, opts=opts)
print(f"κ_R = {kappa_R:.4e} cm² g⁻¹")
```

To also inspect the spectrum and EOS state, use `run_single_point` (uses default
`ModelOptions` internally; does not apply the P17 high-T correction):

```python
from hydrogen_opacity.constants import load_constants
from hydrogen_opacity.config import default_config
from hydrogen_opacity.driver import run_single_point

const = load_constants()
cfg = default_config()
result = run_single_point(T=1e7, rho=1e-7, cfg=cfg, const=const)
print(result["kappa_R"], result["eos"])
```

### Run the full opacity grid

```bash
python scripts/run_grid.py
```

Outputs `kappa_R_grid.npz` in the working directory.  Use `--output data/my_grid`
to specify a path.

---

## Reproduce the Final Benchmark

The following sequence regenerates all final benchmark data and figures from scratch.
Run all commands from the repository root.

**Step 1 — run the test suite**

```bash
python -m pytest
```

**Step 2 — regenerate final benchmark data and figures**

```bash
python scripts/final_benchmark_figures.py
```

This script recomputes the full production grid (P17 model), saves results to
`data/final_kR.npz`, and writes the benchmark figures to `figures/final/`.
Runtime: approximately 2–5 minutes depending on hardware.

**Step 3 — regenerate slide-friendly figures**

```bash
python scripts/slides_benchmark_figures.py
```

Loads `data/final_kR.npz` (no physics recomputation).
Writes PDF and PNG to `figures/final/slides/`.

**Step 4 — run Poutanen (2017) diagnostic scripts (optional)**

```bash
python scripts/diag_poutanen2017.py
python scripts/diag_poutanen2017_extended.py
```

These scripts reproduce the high-T Compton residual analysis.
`diag_poutanen2017.py` confirms the P17 correction reduces the high-T bias from
+8.3% to below 1%.  `diag_poutanen2017_extended.py` diagnoses the remaining
sub-percent slope.

---

## Output Files

| Path | Description |
|---|---|
| `data/final_kR.npz` | Production grid: `T_grid`, `rho_grid`, `kR_ours`, `kR_tops` arrays |
| `data/tops_parsed.npz` | Parsed LANL/TOPS reference opacity table |
| `data/tops_hydrogen_gray.txt` | Raw LANL/TOPS gray Rosseland-mean opacity (pure H) |
| `figures/final/benchmark_kR_vs_T.png` | κ_R vs T comparison, 4 densities |
| `figures/final/benchmark_reldiff_vs_T.png` | Relative difference vs TOPS |
| `figures/final/benchmark_coldT_zoom.png` | Cold-T / partially-neutral zoom |
| `figures/final/benchmark_eos_diag.png` | EOS diagnostic (y_e vs TOPS free electrons) |
| `figures/final/highT_p17_production_comparison.png` | KN spectral vs P17 high-T comparison |
| `figures/final/slides/` | Slide-ready PDF+PNG versions of all four benchmark panels |
| `results/final/benchmark_summary.md` | Tabulated benchmark results and qualitative findings |
| `results/final/benchmark_model_card.md` | Machine-readable model card |
| `results/final/report_language.md` | Report-ready prose blocks summarizing results |

---

## Model Configuration

### Final production model

```python
from hydrogen_opacity.config import ModelOptions, production_opts

# Explicit construction:
opts = ModelOptions(
    use_kn=True,
    use_ff_hminus=True,
    lowering_mode="capped",
    delta_chi_max_ev=1.0,
    compton_mean_mode="poutanen2017",
)

# Convenience factory (identical):
opts = production_opts()
```

### Alternative configurations (for comparison)

```python
# Old KN-spectral mode (no P17 correction; preserved for backward compatibility)
opts_kn = ModelOptions(
    use_kn=True,
    use_ff_hminus=True,
    lowering_mode="capped",
    delta_chi_max_ev=1.0,
    compton_mean_mode="kn_spectral",
)

# Without H⁻ free-free (to assess its importance in the cold/dense corner)
opts_no_hff = ModelOptions(
    use_kn=True,
    use_ff_hminus=False,
    lowering_mode="capped",
    delta_chi_max_ev=1.0,
    compton_mean_mode="poutanen2017",
)

# No ionization-energy lowering (Thomson scattering; baseline)
opts_baseline = ModelOptions(
    use_kn=False,
    use_ff_hminus=False,
    lowering_mode="none",
    compton_mean_mode="kn_spectral",
)
```

`ModelOptions` is a frozen dataclass; all fields are documented in
[src/hydrogen_opacity/config.py](src/hydrogen_opacity/config.py).
The `delta_chi_max_ev` parameter is only read when `lowering_mode="capped"`.

---

## Included Physics

| Source | Symbol | Notes |
|---|---|---|
| Electron scattering | $\kappa_{\text{es}}$ | Klein–Nishina spectral treatment at low/intermediate $T$; Poutanen (2017) Compton Rosseland-mean correction in the hot, fully ionized, scattering-dominated regime ($T \geq 2\ \text{keV}$, $y_e \geq 0.999$) |
| Free-free (e-p) absorption | $\kappa_{\text{ff}}$ | Stimulated-emission corrected; Kramers-like with quantum Gaunt factor |
| Neutral-H bound-free | $\kappa_{\text{bf,H}}$ | Hydrogenic per shell n; stimulated-emission corrected; lowered thresholds |
| $H^-$ bound-free | $\kappa_{\text{bf,H}^-}$ | Empirical fit; domain-limited to $0.754–10 \ \text{eV}$ |
| $H^-$ free-free | $\kappa_{\text{ff,H}^-}$ | John (1988) fit; valid $T \in [1400, 10080]\ \text{K}$, $\lambda > 0.1823\ \mu\text{m}$; stim.\ em.\ already in fit |

## Excluded Physics

The following are **intentionally omitted** as a controlled project approximation:

- **Bound-bound (line) opacity** — excluded; continuum-only model
- **Non-LTE effects** — excluded; strict LTE throughout
- **Full continuum lowering / pressure ionization** — excluded; a step-function level cutoff with ionization-energy lowering is applied as a reduced proxy (see below)
- **Smooth Hummer–Mihalas occupation probabilities** — not included; the sharp n_cut approximation creates discontinuities at cold/dense conditions

---

## Physical Scope and Limitations

This is a **continuum-focused, LTE-only, pure-hydrogen** opacity code.

**What this code is:**
- A benchmark-quality implementation of continuum opacity mechanisms for pure hydrogen
- Useful for studying the relative importance of H⁻ free-free, Klein–Nishina scattering, and Compton corrections at stellar interior conditions
- Agreement with TOPS over most of the formal domain (86% within 10%)

**What this code is not:**
- A complete opacity table replacement (bound-bound opacity is omitted)
- A full reproduction of TOPS (TOPS includes lines, full pressure ionization, non-LTE options)
- A first-principles dense-plasma EOS (a simple 1D root-find with phenomenological level cutoff)
- A full Compton redistribution solver (P17 is a Rosseland-mean fitting formula, not a Kompaneets/ETLA solution)
- Applicable outside pure hydrogen (no helium, no metals)

**High-T Compton correction (P17) caveat:**
The Poutanen (2017) correction is a Rosseland/flux **mean-opacity** correction.
It is applied only at the final mean-opacity level and must not be inserted into
the frequency-dependent opacity integrand.  It is valid for non-degenerate electrons
at $T \in [2, 40]\ \text{keV}$ and is applied only when $y_e \geq 0.999$.

**Remaining discrepancies:**
- Partially neutral regime ($T = 1$–$10\ \text{mkeV}$, $\rho = 10^{-6}\ \text{g\,cm}^{-3}$): systematic 30–50% underestimate attributed to missing hydrogen bound-bound (Lyman/Balmer) opacity.
- Cold/dense corner ($T \lesssim 1\ \text{mkeV}$, $\rho = 10^{-3}\ \text{g\,cm}^{-3}$): agreement within a factor of 1.1–1.5 after adopting the 1 eV cap on ionization-energy lowering; remaining discrepancy attributed to missing bound-bound opacity.

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

**Density-dependent level cutoff (pressure-ionization proxy):**

At high density, excited levels are collisionally dissolved.  As a proxy
the sum over principal quantum shells is truncated at an effective cutoff:

$$ 
n_{\max}^{\rm phys}(\rho) \simeq 12 \left(\frac{n_H}{10^{15}\ \text{cm}^{-3}}\right)^{-2/15}, \quad n_H = \rho/m_H
$$

The integer cutoff actually used is

$$
n_{\rm cut} = \max\!\left(1,\;\min\!\left(n_{\max}^{\rm user},\;\lfloor n_{\max}^{\rm phys}\rfloor\right)\right)
$$

The default user shell cap is $n_{\max}^{\rm user} = 16$.  Example values:
$n_{\rm cut} = 16$ at $\rho = 10^{-12}\ \mathrm{g\,cm^{-3}}$,
$n_{\rm cut} = 12$ at $10^{-9}$,
$n_{\rm cut} = 5$ at $10^{-6}$,
$n_{\rm cut} = 2$ at $10^{-3}\ \mathrm{g\,cm^{-3}}$.

**Ionization-energy lowering (softened capped prescription):**

The effective ionization energy for the Saha equation is computed from the
**float-valued** $n_{\max}^{\rm phys}$, not the integer $n_{\rm cut}$.
This avoids discrete jumps at density thresholds.

Raw lowering energy:

$$
\Delta\chi_{\rm raw} = \frac{13.6\ \text{eV}}{(n_{\max}^{\rm phys})^2}
$$

The production model applies a cap $\Delta\chi_{\max} = 1.0\ \text{eV}$:

$$
\Delta\chi_{\rm eff} = \min\!\left(\Delta\chi_{\rm raw},\ \Delta\chi_{\max}\right)
$$

$$
\chi_{H,\rm eff} = 13.6\ \text{eV} - \Delta\chi_{\rm eff}
$$

This value is used in the Saha prefactor $S_H$.  The H bf photoionization
thresholds use the same float $n_{\max}^{\rm phys}$:

$$
\chi_{n,\rm eff} = 13.6\!\left(\frac{1}{n^2} - \frac{1}{(n_{\max}^{\rm phys})^2}\right)\ \text{eV}
$$

Shells with $\chi_{n,\rm eff} \le 0$ ($n \ge n_{\max}^{\rm phys}$) have zero cross-section.
The integer $n_{\rm cut}$ is used **only** for partition-function truncation
and level-population arrays; it does not appear in any energy formula.

**Rationale for the 1 eV cap:** Full-strength lowering
($\Delta\chi_{\rm eff} = \Delta\chi_{\rm raw}$) over-ionizes the cold/dense corner
by up to 16× relative to TOPS at $(T \approx 0.5\ \text{mkeV},\ \rho = 10^{-3}\ \text{g\,cm}^{-3})$.
No lowering under-ionizes by ~1.5–2×.  The 1 eV cap gives the best overall
benchmark score among tested variants and keeps $y_e$ within a factor of 2 of
the TOPS electron fraction at all cold/dense key points.  It is a phenomenological
engineering choice; a full Hummer–Mihalas treatment is not included.

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

### Electron Scattering (Klein–Nishina)

$$
    \kappa_{\text{es}}(\nu) = n_e \sigma_{\rm KN}(\nu) / \rho \ \text{[cm² g⁻¹]}
$$

$$
    \sigma_{\rm KN}(x) = \frac{3}{4}\sigma_T\left[\frac{1+x}{x^3}\left(\frac{2x(1+x)}{1+2x} - \ln(1+2x)\right) + \frac{\ln(1+2x)}{2x} - \frac{1+3x}{(1+2x)^2}\right]
$$

where $x = h\nu / (m_e c^2)$.  Reduces to $\sigma_T$ for $x \to 0$.
At $x < 0.05$ a 5-term Taylor series is used for numerical stability:

$$
    \sigma_{\rm KN}/\sigma_T \approx 1 - 2x + \tfrac{26}{5}x^2 - \tfrac{133}{10}x^3 + \tfrac{1144}{35}x^4
$$

This is a frequency-dependent **total cross-section** correction only.  It does
not include Compton energy redistribution, thermal-electron relativistic
corrections, Kompaneets/Comptonization physics, or line opacity.

### High-Temperature Compton Rosseland Correction (Poutanen 2017)

In the hot, fully ionized, scattering-dominated regime the Rosseland mean is
dominated by electron scattering, and the effective Compton cross-section is
suppressed below the Thomson value due to relativistic recoil.  This is captured
by replacing the full KN spectral Rosseland integral with the Poutanen (2017)
Compton Rosseland-mean correction:

$$
\Lambda_{P17}(T) = 1 + \left(\frac{T_{\rm keV}}{39.4}\right)^{0.976}
$$

$$
\kappa_{\rm scatt} = \kappa_T \,/\, \Lambda_{P17}(T), \qquad \kappa_T = n_e \sigma_T / \rho
$$

where $n_e$ is taken from the EOS solve (not a fully-ionized approximation) and
$T_{\rm keV} = k_B T / 1\ \text{keV}$.

**Applicability:** This correction is applied only when:
- $T_{\rm keV} \geq 2$ (hydrogen fully ionized; $\chi_H = 0.0136\ \text{keV} \ll T$)
- $y_e = n_e / n_{H,\rm tot} \geq 0.999$ (confirmed fully ionized by EOS)
- Scattering-dominated regime (absorption contribution negligible; verified diagnostically)
- Non-degenerate electrons (satisfied at $\rho \leq 10^{-3}\ \text{g\,cm}^{-3}$, $T \geq 2\ \text{keV}$)

Outside this regime the original KN spectral integral is used.

**Important:** This is a Rosseland/flux **mean-opacity** correction, not a monochromatic
cross-section.  It is applied only at the final mean-opacity level and must not be
inserted into the frequency-dependent opacity integrand.

**Reference:** Poutanen, J. 2017, ApJ, 835, 119, doi:10.3847/1538-4357/835/2/119
(non-degenerate 2–40 keV fitting formula)

The correction reduces the high-T positive residual vs TOPS from +8.3% (KN spectral
at T = 10 keV) to below 1%.  Remaining sub-percent slope is attributed to local
temperature-slope mismatch between the P17 analytic fit and the TOPS-implied effective
Compton correction.

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

For shell $n$, effective ionization threshold (with level-dissolution lowering):
$$
    \chi_{n,\rm eff} = 13.6\!\left(\frac{1}{n^2} - \frac{1}{(n_{\max}^{\rm phys})^2}\right)\ \text{eV}
$$
Shells with $\chi_{n,\rm eff} \le 0$ ($n \ge n_{\max}^{\rm phys}$) have zero cross-section.
The continuous float $n_{\max}^{\rm phys}$ is used here, not the integer $n_{\rm cut}$.

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

### H⁻ Free-Free Absorption

Empirical fit from John (1988), Table 3.  The fit already includes the
stimulated-emission factor $(1 - e^{-h\nu/k_BT})$.

$$
    k_\lambda^{\rm ff}(T) = 10^{-29} \sum_{j=1}^{6} \theta^{(j+1)/2} \left[A_j \lambda^2 + B_j + C_j/\lambda + D_j/\lambda^2 + E_j/\lambda^3 + F_j/\lambda^4\right]
$$

where $\theta = 5040/T$ ($T$ in K), $\lambda$ in $\mu$m, and coefficients $(A_j, \ldots, F_j)$
are tabulated in two wavelength ranges: Table 3a ($\lambda > 0.3645\ \mu\text{m}$) and
Table 3b ($0.1823 < \lambda \le 0.3645\ \mu\text{m}$).  Outside these ranges $k_\lambda^{\rm ff} = 0$.

Volume absorption coefficient:
$$
    \alpha_{\text{ff},\text{H}^-} = k_\lambda^{\rm ff} \cdot n_{H^0} \cdot P_e, \quad P_e = n_e k_B T
$$

Mass opacity (no additional stimulated-emission factor):
$$
    \kappa_{\text{ff},\text{H}^-} = \alpha_{\text{ff},\text{H}^-} / \rho
$$

Valid for $T \in [1400, 10080]\ \text{K}$ and $\lambda > 0.1823\ \mu\text{m}$;
returns zero outside these bounds.

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

## Validation Tests

Run with:

```bash
python -m pytest
```

Tests verify:

1. **Constants sanity** — CGS values within $1\ \text{ppm}$ of NIST CODATA 2018
2. **EOS positivity and conservation** — $n_{\text{H}0}, n_p, n_e, n_{\text{H}^-} \geq 0$; number and charge conservation
3. **Neutral-H bound-free threshold** — $\sigma = 0$ below $h\nu = \chi_{n,\rm eff}$; continuous above
4. **H⁻ bound-free domain** — $\sigma = 0$ for $h\nu \leq 0.754 \text{ eV}$ and $h\nu > 10 \text{ eV}$
5. **H⁻ free-free** — zero outside T validity range and short-wavelength cutoff; positive at solar conditions; wavelength-region boundary checked
6. **Klein–Nishina** — reduces to $\sigma_T$ at low $h\nu$; decreases at high $h\nu$; always positive
7. **Opacity non-negativity** — all 5 components $\geq 0$ at all $x$
8. **Rosseland positivity and finiteness** — $\kappa_R > 0$ and finite at representative points
9. **Regression** — $\kappa_R$ at selected $(T, \rho)$ reproduces stored reference values to $1\%$
10. **Poutanen (2017) correction** — $\Lambda_{P17} > 1$; $\kappa_{P17} < \kappa_T$; not applied below threshold; correctly applied and matches formula at high-T fully-ionized points; residual vs TOPS improved

---

## Verification Status

Domain-internal verification has been completed for the formal domain
($T \in [0.0005, 10]\ \mathrm{keV}$, $\rho \in [10^{-12}, 10^{-3}]\ \mathrm{g\,cm^{-3}}$):

| Check | Result |
|---|---|
| EOS positivity and conservation (10 representative points) | ✓ Pass |
| Opacity component non-negativity across full spectrum (all 5) | ✓ Pass |
| H⁻ bf domain enforcement (0.754–10 eV) | ✓ Pass |
| Neutral-H bf threshold positions (float n_max_phys) | ✓ Pass |
| H⁻ ff: zero outside validity domain; positive at solar T | ✓ Pass |
| Klein–Nishina: Thomson limit; decreasing at high ν; correct Taylor coefficients; branch continuity | ✓ Pass |
| No spectral oscillations or numerical spikes | ✓ Pass |
| x-grid convergence (base vs 4× refined, threshold-refined) | ✓ Pass (<3% at boundary; <1% mid-domain) |
| n_max sensitivity at T ≥ 0.05 keV | ✓ Negligible (<1 ppm) |
| float n_max_phys used in all energy-lowering formulas (not n_cut) | ✓ Pass |
| 185/185 unit tests pass (158 original + 27 P17-specific) | ✓ |

**LANL/TOPS comparison — final production model**
(H⁻ ff + KN + P17 high-T Compton + capped lowering, $\Delta\chi_{\max} = 1.0\ \text{eV}$):

| Density | Within 10% | Within 25% |
|---|---|---|
| $10^{-12}\ \text{g\,cm}^{-3}$ | 69/69 | 69/69 |
| $10^{-9}\ \text{g\,cm}^{-3}$ | 62/69 | 66/69 |
| $10^{-6}\ \text{g\,cm}^{-3}$ | 57/69 | 59/69 |
| $10^{-3}\ \text{g\,cm}^{-3}$ | 50/69 | 62/69 |

**Overall: 238/276 = 86.2% within 10% of TOPS.**

Residual discrepancies: bound-bound (line) opacity (dominant at $T = 1\text{–}10\ \text{mkeV}$,
$\rho = 10^{-6}\ \text{g\,cm}^{-3}$), smooth Hummer–Mihalas occupation probabilities
at cold/dense edge, $g_{bf}$ Gaunt factors.  The high-T positive bias is resolved
by the P17 Compton correction (residual reduced from $+8.3\%$ to $<1\%$ at $T \geq 2\ \text{keV}$).

**Cold/dense corner:** The softened 1 eV cap on ionization-energy lowering
regularizes the over-ionization at $(T \lesssim 1\ \text{mkeV},\ \rho = 10^{-3}\ \text{g\,cm}^{-3})$.
The electron fraction $y_e$ is now within a factor of 1.6 of the TOPS value at the coldest
key points, and $\kappa_R$ is within 15–150% (vs.\ 1280% with full-strength lowering).
Remaining opacity discrepancy there is attributed to missing H bound-bound opacity.

**High-T residual improvement from P17 (all densities):**

| T [keV] | Old KN spectral | P17 production |
|---------|----------------|----------------|
| 2 keV | +1.54% | −0.15% |
| 4 keV | +3.14% | −0.05% |
| 8 keV | +6.60% | +0.55% |
| 10 keV | +8.26% | +0.81% |

---

## Limitations and Expected Differences Relative to LANL/TOPS

| Effect | This code | LANL/TOPS |
|---|---|---|
| $H^-$ free-free | John (1988) fit; valid $T=1400$–$10080$ K | full treatment |
| Bound-bound (lines) | omitted | included |
| Pressure ionization | float $n_{\max}^{\rm phys}(\rho)$ with capped lowering (1 eV cap); $n_{\rm cut}$ for discrete sums only | full Hummer-Mihalas |
| Klein–Nishina scattering | included (all $T$) | included |
| Compton Rosseland correction | Poutanen (2017) fitting formula applied at $T \geq 2\ \text{keV}$, $y_e \geq 0.999$ | full Compton treatment |
| Non-LTE | omitted | available |
| $g_{bf}(\nu)$ Gaunt factor | = 1 | quantum mechanical |

At mid-to-high temperatures (0.01–10 keV), agreement is within 10% across all four densities.
The dominant remaining gap is **bound-bound (line) opacity** (omitted), which dominates at
$T = 1\text{–}10\ \text{mkeV}$ and $\rho = 10^{-6}\ \text{g\,cm}^{-3}$.  At the cold/dense corner
($T \lesssim 1\ \text{mkeV}$, $\rho = 10^{-3}\ \text{g\,cm}^{-3}$), the 1 eV cap on
ionization-energy lowering regularizes the EOS to within a factor of ~2 of TOPS;
the remaining $\kappa_R$ discrepancy is attributed to missing bound-bound opacity.

---

## Reproducibility Note

- All 185 tests pass (`python -m pytest`).
- The final production benchmark is regenerated by `python scripts/final_benchmark_figures.py`
  using the P17 production model (`compton_mean_mode="poutanen2017"`).
- The old KN-spectral mode (`compton_mean_mode="kn_spectral"`) is preserved in the
  codebase for ablation comparison and is documented in the figures.
- See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for step-by-step reproduction instructions
  and expected output paths.

---

## Code Structure

```
src/hydrogen_opacity/
  constants.py       — CGS physical constants (NIST CODATA 2018)
  config.py          — ModelConfig, ModelOptions, default_config(), production_opts()
  eos.py             — EOS root-solver (Saha + H⁻ equilibrium, Brent method)
  grids.py           — Temperature, density, and x-grid construction
  scattering.py      — Klein–Nishina σ_KN; Poutanen (2017) Compton Rosseland correction
  free_free.py       — e-p free-free opacity
  bound_free_h.py    — Neutral-H bound-free photoionization
  bound_free_hminus.py — H⁻ bound-free photodetachment
  free_free_hminus.py — H⁻ free-free absorption (John 1988)
  opacity.py         — Assembly of monochromatic opacity components
  rosseland.py       — Rosseland mean integration; compute_rosseland_mean()
  driver.py          — run_single_point(), run_opacity_grid()
  io_utils.py        — NPZ/CSV grid I/O
  validation.py      — EOS consistency and opacity non-negativity checks

scripts/
  run_grid.py                   — Compute full (T, ρ) grid; save to NPZ
  final_benchmark_figures.py    — Recompute production grid + all final figures
  slides_benchmark_figures.py   — Slide-ready PDF+PNG from cached data/final_kR.npz
  diag_poutanen2017.py          — Historical P17 residual diagnostic
  diag_poutanen2017_extended.py — Extended high-T slope diagnosis
  plot_spectra.py               — Representative opacity spectra at key (T, ρ) points

tests/
  test_constants.py         — NIST CODATA sanity checks
  test_eos.py               — EOS positivity, conservation, and Saha checks
  test_opacity_components.py — Per-component opacity domain and sign checks
  test_poutanen2017.py      — 27 P17-specific tests
  test_regression.py        — Regression: κ_R vs stored reference values
  test_rosseland.py         — Rosseland integral positivity and finiteness
```

**Developer notes:**
- Physics toggles belong in `ModelOptions`; grid parameters belong in `ModelConfig`.
- To add a new opacity channel: implement it in a new module, add it to `monochromatic_opacity` in `opacity.py`, and add a guard toggle in `ModelOptions`.
- The P17 correction is applied only in `compute_rosseland_mean` (in `rosseland.py`), after the EOS solve and before (instead of) the spectral integral.  It must not appear inside `monochromatic_opacity` or in the x-grid integrand.
- `EOS` species: H⁰, p, e⁻, H⁻.  All conservation laws are checked in `validation.py`.

---

## References

- **Poutanen, J. 2017**, ApJ, 835, 119, doi:[10.3847/1538-4357/835/2/119](https://doi.org/10.3847/1538-4357/835/2/119) — Compton Rosseland/flux mean correction formula applied at $T \geq 2$ keV.
- **John, T. L. 1988**, A&A, 193, 189 — H⁻ free-free opacity empirical fit (Table 3); valid $T \in [1400, 10080]$ K.
- **Karzas, W. J. & Latter, R. 1961**, ApJS, 6, 167 — Quantum-mechanical free-free Gaunt factors (structure; $g_{bf} = 1$ is used in the present code).
- **Hummer, D. G. & Mihalas, D. 1988**, ApJ, 331, 794 — Occupation probabilities and pressure ionization (not implemented; cited as the physically correct framework omitted here).

---

## Units

All internal calculations use CGS. Temperatures are accepted in $\text{K}$; the grid is specified in keV and converted via $1 \text{ keV} = 1.16045 × 10⁷ \text{ K}$.
