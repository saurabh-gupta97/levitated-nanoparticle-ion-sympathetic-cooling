"""
===================================
Ion-Nanoparticle Simulation Library
===================================
"""

from .parameters import BaseSystemParams

from .potentials import (
    W_i,
    W_p
)

from .floquet_analysis import (
    run_floquet_sweep,
    floquet_solver,
    inhom_steady_state,
    inhom_trajectory
)

from .dynamical_matrices import (
    get_scaling_factors,
    get_drift_matrix_1,
    get_drift_matrix_2,
    get_drive_vectors
)

from .physics_analysis import (
    calculate_purity,
    calculate_energy,
    compute_energy_trajectory,
    compute_steady_state_metrics
)

from .visualization import (
    plot_steady_state_energy,
    plot_zoomed_energy,
    plot_metrics_vs_damping
)

__all__ = [
    "BaseSystemParams",
    "W_i",
    "W_p",
    "run_floquet_sweep",
    "floquet_solver",
    "inhom_steady_state",
    "inhom_trajectory",
    "get_scaling_factors",  
    "get_drift_matrix_1",   
    "get_drift_matrix_2",   
    "get_drive_vectors",
    "calculate_purity",
    "calculate_energy",
    "compute_energy_trajectory",
    "compute_steady_state_metrics",
    "plot_steady_state_energy",
    "plot_zoomed_energy",
    "plot_metrics_vs_damping",
]