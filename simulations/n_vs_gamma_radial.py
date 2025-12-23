"""
Execution script for the coupled Ion-Nanoparticle system in the RADIAL direction.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import time

# --- 1. Path Setup ---
try:
    base_dir = os.path.dirname(__file__)
except NameError:
    base_dir = os.getcwd()

src_path = os.path.abspath(os.path.join(base_dir, '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

# --- 2. Library Imports ---
from ion_nanoparticle_sim_lib import (
    BaseSystemParams,
    W_p, 
    run_floquet_sweep,
    inhom_steady_state,
    get_drift_matrix_1, 
    get_drift_matrix_2, 
    get_drive_vectors,
    compute_energy_trajectory,
    compute_steady_state_metrics,
    plot_steady_state_energy,
    plot_zoomed_energy,
    plot_metrics_vs_damping
)
from dataclasses import dataclass

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Select direction here: 'x' or 'y'
DIRECTION = 'x' 
# ==============================================================================


# --- 3. Define Radial-Specific Parameters ---
@dataclass
class RadialParams(BaseSystemParams):
    """
    Configuration for Radial Dynamics.
    """
    # Conversion factors for different axes stored as attributes
    Gamma_to_gamma_x: float = 719.84
    Gamma_to_gamma_y: float = 782.99
    
    @property
    def Gamma_to_gamma_r(self):
        """Dynamically returns the conversion factor for the active direction."""
        if DIRECTION == 'x':
            return self.Gamma_to_gamma_x
        elif DIRECTION == 'y':
            return self.Gamma_to_gamma_y
        else:
            raise ValueError(f"Invalid DIRECTION: {DIRECTION}")


def load_benchmark_data(filepath, direction):
    """
    Loads benchmark arrays, dynamically selecting keys based on direction.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Construct keys dynamically based on direction string
    # Assuming JSON keys follow pattern: 'population_x_ss_small', 'population_y_ss_small'
    key_pop_large = f'population_{direction}_ss_secular_large'
    key_pop_small = f'population_{direction}_ss_secular_small'
    key_pur_large = f'purity_{direction}_ss_secular_large'
    key_pur_small = f'purity_{direction}_ss_secular_small'
    
    # Check if keys exist to avoid crashing
    if key_pop_small not in data:
        print(f"Warning: Benchmark key '{key_pop_small}' not found in JSON.")
        return np.array([]), np.array([]), np.array([])

    return (
        np.array(data['gamma_p_vec_secular']),
        np.array(data[key_pop_large]),
        np.array(data[key_pop_small]),
        # We return the small population set for plotting comparison by default
        # You can change this if you prefer the 'large' set
    )


# --- 4. Main Execution Block ---
if __name__ == "__main__":
    print(f"--- Starting Simulation (Direction: {DIRECTION}) ---")
    
    # A. Initialize
    params = RadialParams()
    
    # B. Setup Sweep Vectors
    # Logspace from 10^-7 to 10^3, plus one specific point
    gamma_p_vec = np.insert(2*np.pi * np.logspace(-7, 3, num=21), 0, 2*np.pi * 44.5e-9)
    
    # Use the dynamic property to get the correct conversion factor
    Gamma_ba_vec = params.Gamma_to_gamma_r * gamma_p_vec
    
    # C. Setup Matrices (Using the Factory Pattern)
    print("Initializing Drift Matrices...")
    
    # Pass generic DIRECTION variable
    drift1_factory = lambda t: get_drift_matrix_1(t, params, gamma_p_vec, direction=DIRECTION)
    drift2_factory = lambda t: get_drift_matrix_2(t, params, gamma_p_vec, direction=DIRECTION)
    
    drive1_vec, drive2_vec = get_drive_vectors(params, gamma_p_vec, Gamma_ba_vec, direction=DIRECTION)
    
    # D. Run Floquet Solver
    r = 1
    r1 = 1
    # Integration steps calculation
    n_steps = int((params.T_slow / params.T_fast) * r1 * 10**r)
    
    # Define time evaluation points
    t_eval_points = np.linspace(0, params.T_slow, n_steps + 1)
    
    print(f"Running Floquet Sweep over {len(gamma_p_vec)} damping points...")
    X1_hist, Lambda1_hist, X2_hist, Lambda2_hist = run_floquet_sweep(
        gamma_p_vec=gamma_p_vec,
        T_per=params.T_slow,
        t_eval_points=t_eval_points,
        drift1_mat_stack=drift1_factory,
        drive1_vec_stack=drive1_vec,
        drift2_mat_stack=drift2_factory,
        drive2_vec_stack=drive2_vec
    )
    
    # E. Compute Steady State & Metrics
    print("Computing Steady State...")
    Y1_ss, Y1_ss_hist = inhom_steady_state(4, X1_hist, Lambda1_hist)
    Y2_ss, Y2_ss_hist = inhom_steady_state(10, X2_hist, Lambda2_hist)
    
    # 3. Energy Trajectories
    print("Calculating Energies...")
    pot_J, kin_J = compute_energy_trajectory(
        t_list=t_eval_points,
        y1_history=Y1_ss_hist,
        y2_history=Y2_ss_hist,
        params=params,
        potential_func=W_p, 
        idx_gamma=0,
        direction=DIRECTION # <--- Pass Direction
    )
    
    # 4. Metrics
    purity_ss, population_ss = compute_steady_state_metrics(
        Y1_ss, Y2_ss, 
        params, 
        direction=DIRECTION # <--- Pass Direction
    )
    
    # F. Load Benchmark Data
    json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'secular_benchmark.json'))
    try:
        # Load data specific to the current DIRECTION
        gamma_bench, pop_bench_large, pop_bench_small = load_benchmark_data(json_path, DIRECTION)
        
        if len(pop_bench_small) > 0:
            purity_bench = 1 / (2 * pop_bench_small + 1)
        else:
            purity_bench = np.array([])
            
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Warning: Benchmark data issue ({e}). Skipping comparison.")
        gamma_bench, purity_bench, pop_bench_small = np.array([]), np.array([]), np.array([])

    # G. Visualization
    print("Generating Plots...")
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Add direction to filenames to prevent overwriting
    suffix = f"_{DIRECTION}"
    
    fig1 = plot_steady_state_energy(t_eval_points, pot_J, kin_J, params.T_slow)
    fig1.savefig(os.path.join(results_dir, f'energy_vs_time{suffix}.png'), dpi=300)
    
    fig2 = plot_zoomed_energy(t_eval_points, kin_J, params.T_slow, slice_start=5000, slice_end=5050)
    fig2.savefig(os.path.join(results_dir, f'kinetic_energy_vs_time_zoomed{suffix}.png'), dpi=300)
    
    if len(gamma_bench) > 0:
        fig3 = plot_metrics_vs_damping(
            gamma_p_vec, purity_ss, population_ss,
            gamma_bench, purity_bench, pop_bench_small
        )
        fig3.savefig(os.path.join(results_dir, f'n_vs_gamma{suffix}.png'), dpi=300)
    
    print(f"Simulation Complete. Results saved to data/results/ with suffix '{suffix}'")
    # plt.show()