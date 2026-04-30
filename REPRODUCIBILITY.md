# Reproducibility Guide

This document gives exact step-by-step instructions to reproduce all tests, benchmark
data, and figures for the Rosseland-mean opacity project.

All commands are run from the **repository root** (`Rosseland-Mean-Opacity/`).

---

## Environment Setup

```bash
# Python 3.11 or later required
python --version

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install the package and all dependencies
pip install -e ".[dev]"

# Confirm installation
python -c "import hydrogen_opacity; print('OK')"
```

---

## Step 1 — Run the Test Suite

```bash
python -m pytest
```

**Expected result:** `185 passed` (0 failed, 0 errors).

Test breakdown:
- `test_constants.py` — CGS constants vs NIST CODATA 2018
- `test_eos.py` — EOS positivity, conservation, and Saha checks
- `test_opacity_components.py` — per-component sign/domain checks
- `test_poutanen2017.py` — 27 P17-specific tests
- `test_regression.py` — κ_R regression vs stored reference values
- `test_rosseland.py` — Rosseland integral checks

---

## Step 2 — Regenerate Final Benchmark Figures

```bash
python scripts/final_benchmark_figures.py
```

This script:
1. Loads `data/tops_parsed.npz` (LANL/TOPS reference).
2. Recomputes the full production opacity grid on 69 × 4 = 276 (T, ρ) points.
3. Saves the result to `data/final_kR.npz`.
4. Writes figures to `figures/final/`.

**Production model used:**
```python
ModelOptions(
    use_kn=True,
    use_ff_hminus=True,
    lowering_mode="capped",
    delta_chi_max_ev=1.0,
    compton_mean_mode="poutanen2017",
)
```

**Expected benchmark score:** 238 / 276 = 86.2% within 10% of TOPS.

**Expected output files:**

| File | Description |
|---|---|
| `data/final_kR.npz` | Grid arrays: `T_grid`, `rho_grid`, `kR_ours`, `kR_tops` |
| `figures/final/benchmark_kR_vs_T.png` | κ_R vs T, four densities |
| `figures/final/benchmark_reldiff_vs_T.png` | Relative difference vs TOPS |
| `figures/final/benchmark_coldT_zoom.png` | Cold-T / partially-neutral zoom |
| `figures/final/benchmark_eos_diag.png` | EOS diagnostic (y_e vs TOPS) |
| `figures/final/highT_p17_production_comparison.png` | KN spectral vs P17 high-T comparison |

**Runtime:** approximately 2–5 minutes (single-core; most time in EOS root-solves).

---

## Step 3 — Regenerate Slide-Friendly Figures

```bash
python scripts/slides_benchmark_figures.py
```

Loads `data/final_kR.npz` (no physics recomputation).
Writes PDF and PNG to `figures/final/slides/`:

| File | Description |
|---|---|
| `benchmark_kR_vs_T_slide.{pdf,png}` | Main κ_R comparison panel |
| `benchmark_reldiff_vs_T_slide.{pdf,png}` | Relative difference panel |
| `benchmark_coldT_zoom_slide.{pdf,png}` | Cold-T zoom panel |
| `benchmark_eos_diag_slide.{pdf,png}` | EOS diagnostic panel |

---

## Step 4 — Poutanen (2017) Diagnostic Scripts (Optional)

These scripts are informational; they do not affect the production results.

```bash
python scripts/diag_poutanen2017.py
```

Shows that replacing the KN spectral Rosseland integral with the P17 Compton
correction reduces the high-T positive residual vs TOPS:

| T [keV] | Old KN spectral | P17 production |
|---------|----------------|----------------|
| 2 keV   | +1.54%          | −0.15%         |
| 4 keV   | +3.14%          | −0.05%         |
| 8 keV   | +6.60%          | +0.55%         |
| 10 keV  | +8.26%          | +0.81%         |

```bash
python scripts/diag_poutanen2017_extended.py
```

Six-panel diagnostic investigating why a mild positive slope remains after the P17
correction.  Conclusion: local temperature-slope mismatch between the P17 analytic
fit (T₀ = 39.4 keV, α = 0.976) and the TOPS-implied effective Compton factor
(T₀_eff ≈ 37.2 keV, α_eff ≈ 0.993).

Writes additional figures to `figures/final/slides/`:
- `highT_p17_diagnostic_relative_error.{pdf,png}`
- `highT_p17_fit_comparison.{pdf,png}`
- `highT_lambda_eff_comparison.{pdf,png}`
- `highT_lambda_residual.{pdf,png}`

---

## Optional: Run a Full Grid and Save

```bash
python scripts/run_grid.py --output data/my_grid --csv
```

Computes the Rosseland-mean opacity grid using `default_config()` and the
default `ModelOptions` (KN spectral, no P17).  Saves `data/my_grid.npz`
and `data/my_grid.csv`.

---

## Summary of Expected Results

| Metric | Expected value |
|---|---|
| Test count | 185 passed |
| Overall benchmark score | 238 / 276 within 10% of TOPS (86.2%) |
| ρ = 10⁻¹² g cm⁻³ | 69 / 69 within 10% |
| ρ = 10⁻⁹ g cm⁻³  | 62 / 69 within 10% |
| ρ = 10⁻⁶ g cm⁻³  | 57 / 69 within 10% |
| ρ = 10⁻³ g cm⁻³  | 50 / 69 within 10% |
| High-T residual at T = 10 keV (P17) | < +1% |
| High-T residual at T = 10 keV (KN spectral) | +8.26% |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'hydrogen_opacity'`**
Run `pip install -e .` from the repository root, or set `PYTHONPATH=src`.

**`FileNotFoundError: data/tops_parsed.npz`**
The TOPS reference data file must be present in `data/` and need to be regenerated.

**`FileNotFoundError: data/final_kR.npz` (in slides script)**
Run `python scripts/final_benchmark_figures.py` first to generate the cached grid.

**Figures directory does not exist**
The scripts create `figures/final/` and `figures/final/slides/` automatically.

**Different benchmark score**
The score depends on the TOPS reference file (`data/tops_parsed.npz`) and the
production `ModelOptions`.  Using any other `ModelOptions` will change the score.
Verify that `compton_mean_mode="poutanen2017"` is active for the final benchmark.

---

## Scope Notes

- LTE only; non-LTE is not implemented.
- Pure hydrogen only; helium and metals are not included.
- Continuum opacity only; bound-bound (line) opacity is intentionally omitted.
- The P17 Compton correction is a Rosseland mean correction, not a monochromatic
  cross-section, and applies only at T ≥ 2 keV with y_e ≥ 0.999.
- Runs outside ρ ∈ [10⁻¹², 10⁻³] g cm⁻³ are out-of-scope stress tests and are
  not part of the formal benchmark domain.
