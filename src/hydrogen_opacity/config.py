"""
config.py
=========
Model configuration dataclass and default factory.

All temperature ranges are specified in keV and converted to K by grids.py.
Density ranges are in g cm⁻³.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelOptions:
    """
    Runtime physics toggles for ablation studies.

    All defaults preserve the original KN-spectral behavior for backward
    compatibility.  Use ``production_opts()`` for the final production model.

    Attributes
    ----------
    use_kn : bool
        Use Klein–Nishina electron scattering (True) or Thomson (False).
    use_ff_hminus : bool
        Include H⁻ free-free opacity (John 1988).
    lowering_mode : str
        Ionization-energy lowering prescription for the Saha equation.
        One of:
          "none"            — χ_H,eff = 13.6 eV  (no lowering)
          "full"            — χ_H,eff = 13.6(1 − 1/n_max_phys²)  (default)
          "capped_1eV"      — Δχ = min(13.6/n_max_phys², 1.0) eV; χ_H,eff = 13.6 − Δχ
          "capped"          — Δχ = min(13.6/n_max_phys², delta_chi_max_ev); sweepable cap
          "gated_nmax_gt_4" — full lowering only when n_max_phys > 4, else no lowering
        Bound-free threshold lowering via n_max_phys is active for all modes
        except "none".
    delta_chi_max_ev : float
        Maximum ionization-energy lowering [eV] used only when lowering_mode="capped".
        Ignored for all other modes.
    compton_mean_mode : str
        Treatment of the electron-scattering Rosseland mean at high temperature.
        One of:
          "kn_spectral"    — Full Klein–Nishina spectral Rosseland integral
                             (applied at all temperatures; the original behavior).
          "poutanen2017"   — Poutanen (2017) Compton Rosseland-mean correction,
                             applied only when T_keV >= 2 and the EOS electron
                             fraction y_e >= 0.999 (hot, fully ionized regime).
                             Falls back to "kn_spectral" outside that regime.
                             Reference: Poutanen, J. 2017, ApJ, 835, 119,
                             doi:10.3847/1538-4357/835/2/119
    """
    use_kn: bool = True
    use_ff_hminus: bool = True
    lowering_mode: str = "full"
    delta_chi_max_ev: float = 1.0   # cap on Δχ used only when lowering_mode="capped"
    compton_mean_mode: str = "kn_spectral"


def production_opts() -> ModelOptions:
    """Final production configuration — all physics on, P17 high-T Compton correction."""
    return ModelOptions(
        use_kn=True,
        use_ff_hminus=True,
        lowering_mode="capped",
        delta_chi_max_ev=1.0,
        compton_mean_mode="poutanen2017",
    )


@dataclass(frozen=True)
class ModelConfig:
    """
    Configuration for the hydrogen opacity grid calculation.

    Attributes
    ----------
    T_min_keV : float
        Minimum temperature [keV]
    T_max_keV : float
        Maximum temperature [keV]
    n_T : int
        Number of temperature grid points (log-spaced)
    rho_min : float
        Minimum mass density [g cm⁻³]
    rho_max : float
        Maximum mass density [g cm⁻³]
    n_rho : int
        Number of density grid points (log-spaced)
    n_max : int
        Maximum principal quantum number for H level sum
    x_min : float
        Minimum x = hν / k_B T for spectral integration
    x_max : float
        Maximum x = hν / k_B T for spectral integration
    n_x_base : int
        Base number of x-grid points before threshold refinement
    root_tol : float
        Absolute tolerance for the EOS root solver
    max_root_iter : int
        Maximum iterations for the EOS root solver
    """

    T_min_keV: float
    T_max_keV: float
    n_T: int
    rho_min: float
    rho_max: float
    n_rho: int
    n_max: int
    x_min: float
    x_max: float
    n_x_base: int
    root_tol: float
    max_root_iter: int


def default_config() -> ModelConfig:
    """
    Return the default ModelConfig for the ASTRON C207 course project.

    Grid spans:
      T  ∈ [0.0005, 10] keV    (20 points)
      ρ  ∈ [1e-12, 1e-3] g/cm³  (16 points)

    Returns
    -------
    ModelConfig
    """
    return ModelConfig(
        T_min_keV=0.0005,
        T_max_keV=10.0,
        n_T=20,
        rho_min=1e-12,      # g cm⁻³
        rho_max=1e-3,       # g cm⁻³  (formal domain upper bound)
        n_rho=16,
        n_max=16,           # include n = 1..16 for H levels (density cutoff still applies)
        x_min=1e-2,
        x_max=30.0,
        n_x_base=500,
        root_tol=1e-10,
        max_root_iter=200,
    )
