"""
config.py
=========
Model configuration dataclass and default factory.

All temperature ranges are specified in keV and converted to K by grids.py.
Density ranges are in g cm⁻³.
"""

from dataclasses import dataclass


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
