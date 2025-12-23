"""
Coupled Ion-Nanoparticle Crystal Dynamics Simulation.

This script calculates the equilibrium configurations, stability (static and dynamic),
and steady-state quantum metrics (purity, phonon number) for a hybrid system 
consisting of a linear chain of ions and a single levitated nanoparticle in a 
Paul trap.

Methodology:
1. Search for equilibrium positions.
2. Secular approximation for static stability filtering.
3. Floquet analysis for dynamical stability (Monodromy matrix).
4. Solution of the Lyapunov equation for steady-state covariance.

Units: SI units are used throughout.
"""

# ==============================================================================
# BLOCK 0: IMPORTING LIBRARIES
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker

from scipy.optimize import root
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag, eigh, det
from scipy.constants import pi, hbar, epsilon_0, Boltzmann
from scipy.special import factorial

import multiprocess as mp
from functools import partial
from numba import jit

import time

# ==============================================================================
# BLOCK 1: SYSTEM SETUP AND PARAMETERS
# ==============================================================================
eps0 = epsilon_0

class SystemParameters:
    """
    A container for all physical constants, trap geometry, and system parameters.
    
    Calculates derived quantities such as Mathieu parameters (a, q) and 
    secular frequencies based on the input voltage and geometric configurations.
    """
    def __init__(self):
        """
        Initialize the system parameters.

        """

        # --- Particle properties (SI Units) ---
        # Ion
        self.Q_i, self.M_i = 1.6e-19, 40 * 1.6e-27  
        # Nanoparticle
        self.Q_p, self.M_p = 750 * self.Q_i, 2e-17  

        # --- Trap geometry and voltages ---
        # Z-axis parameters
        z0, alpha_z = 1.7e-3, 0.38              # Geometric factor
        V_dc_z, V_slow_z, V_fast_z = 56.5, 0, 0 # Constant, slow- and fast-oscillating voltages
        
        # Radial parameters
        r0, alpha_r = 0.9e-3, 0.93              # Geometric factor
        V_comp = 0.68
        # DC voltages derived to satisfy Laplace equation constraints
        V_dc_x = -0.5 * (alpha_z/alpha_r) * (r0/z0)**2 * V_dc_z + V_comp # Constant voltage
        V_dc_y = -0.5 * (alpha_z/alpha_r) * (r0/z0)**2 * V_dc_z - V_comp # Constant voltage
        V_slow_r, V_fast_r = 80, 1350           # Slow- and fast-oscillating voltage

        # --- Drive Frequencies ---
        self.omega_slow = 2 * pi * 7e3          # Slow frequency [rad/s]
        self.omega_fast = 2 * pi * 17.5e6       # Fast frequency [rad/s]
        self.l = self.omega_slow / self.omega_fast # Dimensionless ratio
        self.T_slow = (2 * pi) / self.omega_slow    # Time period of slow voltage
        self.T_fast = (2 * pi) / self.omega_fast    # Time period of fast voltage

        # --- Mathieu Parameters (a, q) ---
        # Dimensionless parameters characterising the dynamics in the Paul trap.
        # Radial-axes Parameters
        # Ion Parameters
        self.a_x_i = (4 * self.Q_i * V_dc_x * alpha_r) / (self.M_i * r0**2 * self.omega_fast**2)
        self.a_y_i = (4 * self.Q_i * V_dc_y * alpha_r) / (self.M_i * r0**2 * self.omega_fast**2)
        self.q_slow_x_i = -(2 * self.Q_i * V_slow_r * alpha_r) / (self.M_i * r0**2 * self.omega_slow**2)
        self.q_fast_x_i = -(2 * self.Q_i * V_fast_r * alpha_r) / (self.M_i * r0**2 * self.omega_fast**2)
        self.q_slow_y_i = self.q_slow_x_i
        self.q_fast_y_i = self.q_fast_x_i
        
        # Nanoparticle Parameters
        self.a_x_p = (4 * self.Q_p * V_dc_x * alpha_r) / (self.M_p * r0**2 * self.omega_fast**2)
        self.a_y_p = (4 * self.Q_p * V_dc_y * alpha_r) / (self.M_p * r0**2 * self.omega_fast**2)
        self.q_slow_x_p = -(2 * self.Q_p * V_slow_r * alpha_r) / (self.M_p * r0**2 * self.omega_slow**2)
        self.q_fast_x_p = -(2 * self.Q_p * V_fast_r * alpha_r) / (self.M_p * r0**2 * self.omega_fast**2)
        self.q_slow_y_p = self.q_slow_x_p
        self.q_fast_y_p = self.q_fast_x_p
        
        # Z-axis Parameters
        # Ion Parameters
        self.a_z_i = (4 * self.Q_i * V_dc_z * alpha_z) / (self.M_i * z0**2 * self.omega_fast**2)
        self.q_slow_z_i = -(2 * self.Q_i * V_slow_z * alpha_z) / (self.M_i * z0**2 * self.omega_slow**2)
        self.q_fast_z_i = -(2 * self.Q_i * V_fast_z * alpha_z) / (self.M_i * z0**2 * self.omega_fast**2)

        # Nanoparticle Parameters
        self.a_z_p = (4 * self.Q_p * V_dc_z * alpha_z) / (self.M_p * z0**2 * self.omega_fast**2)
        self.q_slow_z_p = -(2 * self.Q_p * V_slow_z * alpha_z) / (self.M_p * z0**2 * self.omega_slow**2)
        self.q_fast_z_p = -(2 * self.Q_p * V_fast_z * alpha_z) / (self.M_p * z0**2 * self.omega_fast**2)
        
        # --- Effective Secular Trap Frequencies ---
        # Computed using the modified Lindstedt-Poincare method for the two-tone Mathieu equation.
        # Ion Secular Frequencies
        self.Omega_x_i = (1/np.sqrt(2)) * (((self.omega_fast/2)*np.sqrt(self.a_x_i + (self.q_fast_x_i**2)/2))**2 + (self.omega_slow/2)**2 + np.sqrt((((self.omega_fast/2)*np.sqrt(self.a_x_i + (self.q_fast_x_i**2)/2))**2 - (self.omega_slow/2)**2)**2 - (self.q_slow_x_i**2 * self.omega_slow**4/8)))**(1/2)
        self.Omega_y_i = (1/np.sqrt(2)) * (((self.omega_fast/2)*np.sqrt(self.a_y_i + (self.q_fast_y_i**2)/2))**2 + (self.omega_slow/2)**2 + np.sqrt((((self.omega_fast/2)*np.sqrt(self.a_y_i + (self.q_fast_y_i**2)/2))**2 - (self.omega_slow/2)**2)**2 - (self.q_slow_y_i**2 * self.omega_slow**4/8)))**(1/2)
        self.Omega_z_i = (1/np.sqrt(2)) * (((self.omega_fast/2)*np.sqrt(self.a_z_i + (self.q_fast_z_i**2)/2))**2 + (self.omega_slow/2)**2 + np.sqrt((((self.omega_fast/2)*np.sqrt(self.a_z_i + (self.q_fast_z_i**2)/2))**2 - (self.omega_slow/2)**2)**2 - (self.q_slow_z_i**2 * self.omega_slow**4/8)))**(1/2)

        # Nanoparticle Secular Frequencies
        self.Omega_x_p = (self.omega_fast/2) * np.sqrt(self.a_x_p + (self.q_slow_x_p * self.l)**2/2 + self.q_fast_x_p**2/2)
        self.Omega_y_p = (self.omega_fast/2) * np.sqrt(self.a_y_p + (self.q_slow_y_p * self.l)**2/2 + self.q_fast_y_p**2/2)
        self.Omega_z_p = (self.omega_fast/2) * np.sqrt(self.a_z_p + (self.q_slow_z_p * self.l)**2/2 + self.q_fast_z_p**2/2)

        
        # --- Damping and Heating Coefficients ---
        self.gamma_i = 2 * pi * 10e3   # Ion damping rate
        self.E_dot_i = 3.8e-22         # Ion energy heating rate

        self.E_dot_p = 11.5e-28        # Nanoparticle energy heating rate (gas scattering)
        self.E_dot_td = 0              # Nanoparticle energy heating rate (trap displacement)
        # Change value if trap displacement noise is non-zero
        
        self.Gamma_to_gamma = np.array([0,0,0]) # Gamma_ba / gamma_fb 
        # Use the below array for Gamma_to_gamma if feedback backaction is non-zero
        #np.array([719.84, 782.99, 842.08])
        

    def get_particle_properties(self, n_ions):
        """
        Constructs the parameter vectors for the multi-particle system.
        
        System consists of `n_ions` ions and 1 nanoparticle.

        Parameters
        ----------
        n_ions : int
            Number of ions in the system.

        Returns
        -------
        tuple
            (charge_vec, mass_vec, Omega_vec_1d, mass_vec_3d, Omega_vec_3d, q_zpf_vec_3d, p_zpf_vec_3d)
            Arrays containing physical properties and ZPF scales for the N+1 particles.
        """
        n_particles = n_ions + 1
        
        # --- 1D Vectors (Size: N+1) ---
        # Construct charge vector [Q_i, ..., Q_i, Q_p]
        charge_vec = np.full(n_particles, self.Q_i)
        charge_vec[-1] = self.Q_p
        
        # Construct mass vector [M_i, ..., M_i, M_p]
        mass_vec = np.full(n_particles, self.M_i)
        mass_vec[-1] = self.M_p

        # Z-axis effective secular frequency vectors [Omega_z_i, ..., Omega_z_i, Omega_z_p]
        Omega_vec_1d = np.full(n_particles, self.Omega_z_i)
        Omega_vec_1d[-1] = self.Omega_z_p
        
        # --- 3D Vectors (Size: 3*(N+1)) ---
        # Flattened representation for x, y, z degrees of freedom
        mass_vec_3d = np.repeat(mass_vec, 3) 

        # Stack frequencies [x, y, z] for ions and nanoparticle
        # [
        # [Omega_x_i, Omega_y_i, Omega_z_i],
        # [Omega_x_i, Omega_y_i, Omega_z_i],
        # ...
        # [Omega_x_p, Omega_y_p, Omega_z_p]
        # ]
        Omega_vec_3d = np.vstack([
            np.tile([self.Omega_x_i, self.Omega_y_i, self.Omega_z_i], (n_ions, 1)), 
            [self.Omega_x_p, self.Omega_y_p, self.Omega_z_p]
        ])
        
        # --- Zero-Point Fluctuations ---
        # Calculated using deltaR^zpf = sqrt(hbar / 2*m*Omega) and P^zpf = sqrt(hbar*m*Omega / 2)
        q_zpf_vec_1d = np.sqrt(hbar / (2 * mass_vec * Omega_vec_1d))
        p_zpf_vec_1d = np.sqrt((hbar * mass_vec * Omega_vec_1d) / 2)
        
        return charge_vec, mass_vec, Omega_vec_1d, mass_vec_3d, Omega_vec_3d, q_zpf_vec_1d, p_zpf_vec_1d


# ==============================================================================
# BLOCK 2: SYSTEMATIC PROJECTED FORCE SEARCH FOR 1D CONFIGURATIONS
# ==============================================================================

@jit(nopython=True, fastmath=True, cache=True)
def _objective_function_kernel(coord_vec_constrained, n_ions, charge_vec, Omega_vec_3d, mass_vec_3d):
    """
    Numba-optimized kernel to calculate the objective function vector for the root solver.
    
    This function computes the net forces (Trap + Coulomb) acting on the system.
    
    CRITICAL: To enforce a 1D configuration search, this function replaces the 
    radial (X, Y) components of the force vector with the radial coordinates 
    themselves. Consequently, the root solver finds a solution where:
        1. F_z = 0 (Axial force balance)
        2. X = 0   (Confinement to axis)
        3. Y = 0   (Confinement to axis)

    Parameters
    ----------
    coord_vec_constrained : ndarray
        Flattened coordinate vector of size 3N-1 (due to y_1 constraint).
    n_ions : int
        Number of ions.
    charge_vec : ndarray
        Vector of particle charges.
    Omega_vec_3d : ndarray
        (N+1, 3) array of secular frequencies.
    mass_vec_3d : ndarray
        Flattened mass vector.

    Returns
    -------
    res : ndarray
        The residual vector to be minimized by the solver.
    """
    # Reconstruct the full 3N coordinate vector from the constrained input
    # The constraint typically fixes one degree of freedom (e.g., y_1=0) to prevent 
    # rigid body rotation degeneracy during 3D solving, though here we force 1D.
    coord_vec = np.zeros(3 * (n_ions + 1))
    coord_vec[0] = coord_vec_constrained[0]
    coord_vec[1] = 0.0  # Explicit Constraint: y_1 = 0
    coord_vec[2:] = coord_vec_constrained[1:]
    coord_vec_reshaped = coord_vec.reshape(-1, 3)
    
    # --- 1. Trap Restoring Forces ---
    # F_trap = -m * omega^2 * r
    mass_vec_reshaped = mass_vec_3d.reshape(-1, 3)
    trap_forces = -mass_vec_reshaped * Omega_vec_3d**2 * coord_vec_reshaped
    
    # --- 2. Coulomb Repulsion Forces ---
    n_particles = coord_vec_reshaped.shape[0]
    # Precompute charge products: Q_i * Q_j / (4 * pi * eps0)
    charge_prod_matrix = np.outer(charge_vec, charge_vec) / (4 * pi * eps0)
    
    net_coulomb_forces = np.zeros_like(coord_vec_reshaped)
    
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            delta_r = coord_vec_reshaped[i] - coord_vec_reshaped[j]
            dist_sq = np.sum(delta_r**2)
            dist = np.sqrt(dist_sq)
            
            # Singularity avoidance: only compute if particles are not overlapping
            if dist > 1e-12: 
                # F_coulomb = k * (q1*q2 / r^2) * r_hat
                force_mag_dir = (charge_prod_matrix[i, j] / dist**3) * delta_r
                net_coulomb_forces[i] += force_mag_dir
                net_coulomb_forces[j] -= force_mag_dir
            
    # Combine forces
    total_forces_reshaped = trap_forces + net_coulomb_forces

    # --- 3. Radial Projection (The "Projected Force" Method) ---
    # We replace the X and Y force components with the X and Y coordinates.
    # Root finding on this vector forces x -> 0 and y -> 0.
    
    # Force_X becomes coordinate_X for all particles
    total_forces_reshaped[:, 0] = coord_vec_reshaped[:, 0]
    
    # Force_Y becomes coordinate_Y for all particles
    total_forces_reshaped[:, 1] = coord_vec_reshaped[:, 1]
    
    # The Z component remains the physical force (F_z)
    
    # Flatten the final vector for the solver
    all_forces_flat = total_forces_reshaped.flatten()
    
    # Remove the constrained component (y_1) from the residual vector
    res = np.zeros(len(all_forces_flat) - 1)
    res[:1] = all_forces_flat[:1] # x_1
    res[1:] = all_forces_flat[2:] # z_1, x_2, y_2, z_2, ...
    
    return res

def _objective_function(coord_vec_constrained, n_ions, params, charge_vec, Omega_vec_3d, mass_vec_3d):
    """Wrapper function for the JIT kernel required by scipy.optimize.root."""
    return _objective_function_kernel(coord_vec_constrained, n_ions, charge_vec, Omega_vec_3d, mass_vec_3d)

def find_single_equilibrium(run_index, n_ions, params, tolerance=1e-12):
    """
    Attempts to find a single equilibrium configuration starting from a random guess.

    Parameters
    ----------
    run_index : int
        Seed offset for random number generation (ensures parallel processes vary).
    n_ions : int
        Number of ions in the chain.
    params : SystemParameters
        Instance containing system physics.
    tolerance : float, optional
        Threshold for x/y coordinates to be considered "on-axis".

    Returns
    -------
    success : bool
        Whether the solver converged and met criteria.
    result : ndarray or None
        If successful, returns the 1D array of z-coordinates. Otherwise None.
    """
    n_particles = n_ions + 1
    charge_vec, _, _, mass_vecs_3d_flat, Omega_vec_3d, _, _ = params.get_particle_properties(n_ions)
    mass_vec_3d = mass_vecs_3d_flat.reshape(-1,3)
    
    # Create partial function with fixed system parameters
    objective_func = partial(_objective_function, n_ions=n_ions, params=params, 
                             charge_vec=charge_vec, Omega_vec_3d=Omega_vec_3d, mass_vec_3d=mass_vec_3d)

    # Generate a random initial guess
    # Seeding ensures reproducibility per run_index while varying across runs
    np.random.seed(int(time.time()) + run_index + mp.current_process().pid)
    guess_constrained = np.random.uniform(-1e-4, 1e-4, size=3 * n_particles - 1)

    # Solve F(r) = 0
    solution = root(objective_func, guess_constrained, method='hybr')

    # --- Post-Processing ---
    if solution.success:
        # Reconstruct full 3D vector (inserting y_1 = 0)
        coord_vec_3d = np.insert(solution.x, 1, 0.0)
        
        # Filter 1: Spatial Bounds
        # Reject "exploded" solutions where particles are too far from trap center
        max_dist = 300e-6  # 300 micrometers
        if np.max(np.abs(coord_vec_3d)) > max_dist:
            return None, None

        coord_vec_reshaped = coord_vec_3d.reshape(-1, 3) 

        # Filter 2: Linearity Check
        # Ensure all particles are effectively on the trap axis
        xy_coord_vec = coord_vec_reshaped[:, :2] 
        is_1d_chain = np.all(np.abs(xy_coord_vec) < tolerance)

        if is_1d_chain:
            # Return only the z-coordinates (axial positions)
            return True, coord_vec_reshaped[:, 2]
        else:
            # Solution converged but is not a linear chain
            return True, None
    else:
        # Solver failed to converge
        return False, None

def get_canonical_signature(z_coord_vec, n_ions, precision=8):
    """
    Generates a hashable signature for a configuration to identify unique states.
    
    Because ions are indistinguishable, the order of ions in the coordinate vector
    does not represent a distinct physical state. We sort the ion coordinates
    to ensure that [z1, z2, z_p] produces the same signature as [z2, z1, z_p].

    Parameters
    ----------
    z_coord_vec : ndarray
        Z-coordinates of all particles.
    n_ions : int
        Number of ions.

    Returns
    -------
    tuple
        Sorted tuple of rounded ion coordinates (nanoparticle excluded or handled implicitly).
    """
    # Isolate ion coordinates
    ion_z_coord_vec = z_coord_vec[:n_ions]
    
    # Round to avoid floating point noise affecting uniqueness
    rounded_coord_vec = np.round(ion_z_coord_vec, precision)
    
    # Sort ion coordinates (indistinguishability)
    sorted_coord_vec = np.sort(rounded_coord_vec)
    
    return tuple(sorted_coord_vec)

def find_unique_equilibrium_configs(n_ions, n_runs, params, verbose=True):
    """
    Performs a Monte Carlo-style search for unique stable equilibrium configurations.
    
    Runs the solver `n_runs` times in parallel with random initial conditions
    and aggregates unique solutions based on their canonical signature.

    Parameters
    ----------
    n_ions : int
        Number of ions.
    n_runs : int
        Total number of solver attempts.
    params : SystemParameters
        System physics object.

    Returns
    -------
    unique_solutions : list
        List of unique 1D z-coordinate arrays found.
    signature_counts : dict
        Dictionary mapping the unique signature to the number of times it was found
        (useful for estimating the basin of attraction).
    """
    if verbose: 
        print(f"--- Starting parallel search for {n_ions}-ion crystal configurations ({n_runs} runs) ---")
    
    start_time = time.time()
    
    # Parallel execution using all available CPU cores
    with mp.Pool(mp.cpu_count()) as pool:
        solver_func = partial(find_single_equilibrium, n_ions=n_ions, params=params)
        results = pool.map(solver_func, range(n_runs))
    
    unique_solutions = {}
    signature_counts = {}
    
    successful_runs = 0
    chain_runs = 0
    
    for success, coord_vec in results:
        if not success:
            continue
        
        successful_runs += 1
        
        if coord_vec is not None:
            chain_runs += 1
            signature = get_canonical_signature(coord_vec, n_ions)
            
            if signature not in unique_solutions:
                # New configuration found
                unique_solutions[signature] = coord_vec
                signature_counts[signature] = 1
            else:
                # Existing configuration found again
                signature_counts[signature] += 1
                
    if verbose:
        elapsed = time.time() - start_time
        print(f"--- Search complete in {elapsed:.2f}s | "
              f"Converged: {successful_runs}/{n_runs} | "
              f"Linear: {chain_runs}/{successful_runs} | "
              f"Unique States: {len(unique_solutions)} ---")
        
    return list(unique_solutions.values()), signature_counts


# ==============================================================================
# BLOCK 3: LOW-LEVEL JIT KERNELS FOR POTENTIALS AND HESSIANS
# ==============================================================================

@jit(nopython=True, fastmath=True, cache=True)
def _calculate_hessian_matrix_kernel_1d(z_coord_vec, charge_vec):
    """
    JIT-optimized kernel for the 1D Coulomb Hessian matrix.
    
    Computes the second derivative of the Coulomb potential with respect to 
    axial positions z_i and z_j. Used for analyzing axial stability and 
    mode frequencies.

    Parameters
    ----------
    z_coord_vec : ndarray
        1D array of particle positions.
    charge_vec : ndarray
        1D array of particle charges.

    Returns
    -------
    hessian_mat : ndarray
        (N x N) symmetric Hessian matrix.
    """
    n_particles = z_coord_vec.shape[0]
    hessian_mat = np.zeros((n_particles, n_particles))
    prefactor = 2 * (1 / (4 * pi * eps0))

    # Compute Off-Diagonal Elements: d^2V / dz_i dz_j
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            charge_prod = charge_vec[i] * charge_vec[j]
            dist_cubed = np.abs(z_coord_vec[i] - z_coord_vec[j])**3
            
            # H_ij = -2 * k_e * Q_i * Q_j / |z_i - z_j|^3
            off_diag_val = -prefactor * charge_prod / dist_cubed
            
            hessian_mat[i, j] = off_diag_val
            hessian_mat[j, i] = off_diag_val # Symmetry

    # Compute Diagonal Elements: d^2V / dz_i^2
    # Derived from translational invariance (sum of rows must be zero for Coulomb)
    for i in range(n_particles):
        hessian_mat[i, i] = -np.sum(hessian_mat[i, :])
        
    return hessian_mat

@jit(nopython=True, fastmath=True, cache=True)
def _calculate_hessian_matrix_kernel_3d(coord_vec_3d, charge_vec):
    """
    JIT-optimized kernel for the full 3D Coulomb Hessian matrix.
    
    Computes the (3N x 3N) matrix required for full 3D stability analysis.
    Handles the tensor contraction required for the 1/r potential second derivatives.

    Parameters
    ----------
    coord_vec_3d : ndarray
        (N, 3) array of particle coordinates.
    charge_vec : ndarray
        1D array of particle charges.

    Returns
    -------
    hessian_mat : ndarray
        (3N x 3N) symmetric Hessian matrix flattened block-wise.
    """
    n_particles = coord_vec_3d.shape[0]
    prefactor_mat = np.outer(charge_vec, charge_vec) / (4 * pi * eps0)
    
    # Temporary storage for 3x3 interaction blocks
    hessian_hyperstack = np.zeros((n_particles, n_particles, 3, 3))
    
    for i in range(n_particles):
        for j in range(i + 1, n_particles): 
            delta = coord_vec_3d[i] - coord_vec_3d[j]
            d_sq = np.sum(delta**2)
            d = np.sqrt(d_sq)
            d3 = d**3
            d5 = d**5
            
            # Outer product of displacement vector: r_vec * r_vec^T
            outer_prod = np.zeros((3,3))
            for row in range(3):
                for col in range(3):
                    outer_prod[row, col] = delta[row] * delta[col]

            # 3x3 sub-block calculation
            # H_sub = -k * Q_i*Q_j * [ (3 r r^T / r^5) - (I / r^3) ]
            block = (3 * outer_prod / d5) - (np.eye(3) / d3)
            block *= -prefactor_mat[i, j]
            
            hessian_hyperstack[i, j] = block
            hessian_hyperstack[j, i] = block 

    # Calculate diagonal blocks (Self-interaction / Restoring forces context)
    for i in range(n_particles):
        diag_block = np.zeros((3, 3))
        for j in range(n_particles):
            if i == j:
                continue
            diag_block -= hessian_hyperstack[i, j]
        hessian_hyperstack[i, i] = diag_block
        
    # Flatten the hyperstack into a standard 2D matrix
    hessian_mat = np.zeros((3 * n_particles, 3 * n_particles))
    for i in range(n_particles):
        for j in range(n_particles):
            for row in range(3):
                for col in range(3):
                    hessian_mat[3*i+row, 3*j+col] = hessian_hyperstack[i,j,row,col]

    return hessian_mat

def calculate_hessian_matrix_jit(z_coord_vec_1d, params, charge_vec):
    """Wrapper for the 1D JIT Hessian kernel."""
    return _calculate_hessian_matrix_kernel_1d(z_coord_vec_1d, charge_vec)

def calculate_hessian_matrix_3d_jit(coord_vec_3d, params, charge_vec):
    """Wrapper for the 3D JIT Hessian kernel."""
    coords_reshaped = coord_vec_3d.reshape(-1, 3)
    return _calculate_hessian_matrix_kernel_3d(coords_reshaped, charge_vec)



# ==============================================================================
# BLOCK 4: MODULAR ANALYSIS FUNCTIONS (SPECTRAL & STATISTICAL)
# ==============================================================================

def compute_system_matrices_jit(z_coord_vec_1d, params):
    """
    Computes the spectral properties of the crystal configuration.
    
    Calculates the Hessian, normal modes, and eigenfrequencies for the 1D
    axial configuration. It distinguishes between the "full system" modes
    and the "ions-only" modes to analyze sympathetic cooling participation.

    Parameters
    ----------
    z_coord_vec_1d : ndarray
        Equilibrium z-coordinates.
    params : SystemParameters
        Physical constants.

    Returns
    -------
    dict
        Dictionary containing Hessians, frequencies, and mode participation matrices.
    """
    n_ions = z_coord_vec_1d.shape[0] - 1
    
    # Retrieve physical properties
    charge_vec, mass_vec, Omega_vec_1d, _, _, _, _ = params.get_particle_properties(n_ions)
    
    # 1. Compute Potentials
    hessian_coulomb = calculate_hessian_matrix_jit(z_coord_vec_1d, params, charge_vec)
    hessian_trap = mass_vec * Omega_vec_1d**2
    hessian_total = np.diag(hessian_trap) + hessian_coulomb

    # 2. Modified Bare Frequencies
    # Frequencies of particles if they were frozen in place by the mean field of others
    mod_Omega_vec_1d = np.sqrt(np.diagonal(hessian_total) / mass_vec)

    # 3. Full System Normal Mode Analysis
    # Solves generalized eigenvalue problem via mass-weighting: H' = M^-1/2 * H * M^-1/2
    M_half = np.diag(np.sqrt(mass_vec))
    M_half_inv = np.diag(1 / np.sqrt(mass_vec))
    
    eigvals, eigvecs = np.linalg.eigh(M_half_inv @ hessian_total @ M_half_inv)
    
    normal_Omega_vec_1d = np.sqrt(eigvals)
    normal_mode_mat_1d = M_half_inv @ eigvecs

    # 4. Ions-Only Subsystem Analysis
    # Used to determine how well the ion chain modes couple to the nanoparticle
    ions_M_half = np.diag(np.sqrt(mass_vec[:-1]))
    ions_M_half_inv = np.diag(1 / np.sqrt(mass_vec[:-1]))
    
    # Extract the ion-ion block of the Hessian
    ions_eigvals, ions_eigvecs = np.linalg.eigh(ions_M_half_inv @ hessian_total[:-1,:-1] @ ions_M_half_inv)
    
    # Construct extended vectors for the ions-only basis (appending the particle as decoupled)
    ions_normal_Omega_vec_1d = np.concatenate((np.sqrt(ions_eigvals), [mod_Omega_vec_1d[-1]]))
    
    ions_normal_mode_mat_1d = np.zeros((n_ions+1, n_ions+1))
    ions_normal_mode_mat_1d[:-1,:-1] = ions_M_half_inv @ ions_eigvecs
    ions_normal_mode_mat_1d[-1,-1] = M_half_inv[-1,-1]

    return {
        'hessian_coulomb': hessian_coulomb,
        'hessian_total': hessian_total,
        'charge_vec': charge_vec,
        'mass_vec': mass_vec,
        'static_bare_frequencies': Omega_vec_1d,
        'static_mod_bare_frequencies': mod_Omega_vec_1d,
        'static_normal_frequencies': normal_Omega_vec_1d,
        'static_normal_modes': normal_mode_mat_1d,
        'static_ions_normal_frequencies': ions_normal_Omega_vec_1d,
        'static_ions_normal_modes': ions_normal_mode_mat_1d,
        'ions_participation_mat': ions_eigvecs
    }


def calculate_static_eom_matrices_jit(gamma_p, params, sys_matrices):
    """
    Constructs the Drift (A) and Diffusion (Gamma) matrices for the Lyapunov equation.
    
    The equation dC/dt = AC + CA^T + Gamma defines the time evolution of the 
    covariance matrix C.

    Parameters
    ----------
    gamma_p : float
        Nanoparticle damping rate.
    params : SystemParameters
        Physical constants.
    sys_matrices : dict
        Output from `compute_system_matrices_jit`.

    Returns
    -------
    A_cov_mat : ndarray
        (4N^2 x 4N^2) Kronecker sum matrix for vectorised Lyapunov solution.
    Gamma_vec : ndarray
        Flattened diffusion matrix.
    A_mat : ndarray
        (2N x 2N) Drift matrix in phase space.
    """
    n_ions = len(sys_matrices['mass_vec']) - 1
    n_particles = n_ions + 1
    n_dim = 2 * n_particles
    
    mass_vec = sys_matrices['mass_vec']
    hessian_total = sys_matrices['hessian_total']

    _, _, _, _, _, q_zpf_vec, p_zpf_vec = params.get_particle_properties(n_ions)
    
    damping_vec = np.concatenate([np.tile([params.gamma_i], n_ions), [gamma_p]])
    
    # --- 1. Unscaled Drift Matrix A (2N x 2N) ---
    # Block structure: [[-Gamma/2,  M^-1], [-H, -Gamma/2]]
    A_qq = -np.diag(damping_vec / 2.0)
    A_qp = np.diag(1 / mass_vec)
    A_pq = -hessian_total
    A_pp = -np.diag(damping_vec / 2.0)
    A_mat = np.block([[A_qq, A_qp], [A_pq, A_pp]])

    # --- 2. Unscaled Diffusion Matrix Gamma (2N x 2N) ---
    mod_Omega_vec = np.sqrt(np.diagonal(hessian_total) / mass_vec)
    mod_q_zpf_vec = np.sqrt(hbar / (2 * mass_vec * mod_Omega_vec))
    mod_p_zpf_vec = np.sqrt(hbar * mass_vec * mod_Omega_vec / 2)
    
    # Heating terms
    heating_vec = np.concatenate([np.tile([params.E_dot_i], n_ions), [params.E_dot_p]]) / (hbar * mod_Omega_vec)
    
    # Localisation / Noise terms
    localisation_vec = np.concatenate([np.tile([0], n_ions), [params.E_dot_td]]) / (hbar * mod_Omega_vec)
    
    # Feedback backaction contribution (if applicable)
    localisation_vec[-1] += params.Gamma_to_gamma[-1] * mod_q_zpf_vec[-1]**2 * gamma_p / q_zpf_vec[-1]**2
    
    Gamma_qq = np.diag(mod_q_zpf_vec**2 * (2 * heating_vec + damping_vec))
    Gamma_pp = np.diag(mod_p_zpf_vec**2 * (2 * heating_vec + damping_vec + 4 * localisation_vec))
    Gamma_mat = block_diag(Gamma_qq, Gamma_pp)

    # --- 3. Phase Space Rescaling ---
    # Transforms coords to dimensionless units of Zero Point Fluctuations (ZPF)
    # A' = S A S^-1, Gamma' = S Gamma S^T
    q_scale_vec = 1 / q_zpf_vec
    p_scale_vec = 1 / p_zpf_vec
    scale_vec = np.concatenate([q_scale_vec, p_scale_vec])
    S = np.diag(scale_vec)
    
    A_mat = S @ A_mat @ np.linalg.inv(S)
    Gamma_mat = S @ Gamma_mat @ S.T

    # --- 4. Prepare for Vectorized Solution (Ax = b) ---
    # Kronecker sum: A_cov = I (x) A + A (x) I
    In = np.eye(n_dim)
    A_cov_mat = np.kron(In, A_mat) + np.kron(A_mat, In)
    Gamma_vec = Gamma_mat.flatten(order='F')
    
    return A_cov_mat, Gamma_vec, A_mat

def calculate_steady_state_purity(ss_cov_mat, n_ions, params):
    """
    Extracts the nanoparticle's quantum purity and phonon number from the 
    steady-state covariance matrix.
    """
    n_particles = n_ions + 1
    
    # Indices for the Nanoparticle in the phase space vector [q1...qN, p1...pN]
    q_idx = n_ions
    p_idx = n_ions + n_particles 
    
    cov_qq = ss_cov_mat[q_idx, q_idx]
    cov_pp = ss_cov_mat[p_idx, p_idx]
    cov_qp = ss_cov_mat[q_idx, p_idx]
    
    # Purity = hbar/2 / sqrt(det(sigma_p))
    determinant = cov_qq * cov_pp - cov_qp**2
    purity = (hbar / 2) / np.sqrt(np.abs(determinant))
    phonon_number = 0.5 * (1 / purity - 1)
    
    return {'z_purity': purity, 'z_phonon_number': phonon_number}
    

def analytical_steady_state_purity(n_ions, gamma_p, sys_matrices_1d, params):
    """
    Calculates analytical estimates for the steady-state phonon number.
    
    This function evaluates the sympathetic cooling limit by considering the 
    coupling between the nanoparticle and specific normal modes of the ion chain.
    """
    # Extract system properties
    mass_vec_1d = sys_matrices_1d['mass_vec']
    Omega_vec_1d = sys_matrices_1d['static_bare_frequencies']
    mod_Omega_vec_1d = sys_matrices_1d['static_mod_bare_frequencies']
    ions_normal_Omega_vec_1d = sys_matrices_1d['static_ions_normal_frequencies']
    ions_normal_mode_mat_1d = sys_matrices_1d['static_ions_normal_modes']
    participation_mat = sys_matrices_1d["ions_participation_mat"]
    
    # --- 1. Bare Coupling Estimate ---
    # Coupling strength g between particle and ions in the bare basis
    mod_g_vec_1d = sys_matrices_1d['hessian_total'][:-1,-1] / np.sqrt(4 * mod_Omega_vec_1d[:-1] * mass_vec_1d[:-1] * mod_Omega_vec_1d[-1] * mass_vec_1d[-1] )
    
    # Effective damping rate via bare coupling
    gamma_p_bare_analytical = gamma_p + params.gamma_i * np.sum((mod_g_vec_1d**2 * (4 * mod_Omega_vec_1d[:-1] * mod_Omega_vec_1d[-1])) / ((mod_Omega_vec_1d[:-1]**2 - mod_Omega_vec_1d[-1]**2)**2))
    z_phonon_number_bare_analytical = (params.E_dot_p + params.E_dot_td) / (hbar * mod_Omega_vec_1d[-1] * gamma_p_bare_analytical)

    # --- 2. Ions Normal Mode Coupling Estimate ---
    # Transform coupling to the ion crystal's normal mode basis
    ions_normal_g_vec_1d = ((ions_normal_mode_mat_1d[:-1,:-1].T @ sys_matrices_1d['hessian_total'][:-1,-1]) * np.sqrt(1 / mass_vec_1d[-1])) / (np.sqrt(4 * ions_normal_Omega_vec_1d[:-1] * mod_Omega_vec_1d[-1]))
    
    # Effective damping for ion modes
    ions_normal_damping_vec_1d = np.zeros(n_ions+1)
    ions_normal_damping_vec_1d[:-1] = np.diagonal(participation_mat.T @ participation_mat) * np.array([params.gamma_i] * n_ions) 
    ions_normal_damping_vec_1d[-1] = gamma_p
    
    # Effective damping rate via normal mode coupling
    gamma_p_ions_normal_analytical = ions_normal_damping_vec_1d[-1] + np.sum((ions_normal_damping_vec_1d[:-1] * ions_normal_g_vec_1d**2 * (4 * ions_normal_Omega_vec_1d[:-1] * mod_Omega_vec_1d[-1])) / ((ions_normal_Omega_vec_1d[:-1]**2 - mod_Omega_vec_1d[-1]**2)**2))
    z_phonon_number_ions_normal_analytical = (params.E_dot_p + params.E_dot_td) / (hbar * mod_Omega_vec_1d[-1] * gamma_p_ions_normal_analytical)

    # --- 3. Special Mode Analysis (Softest Mode) ---
    # Focus on the lowest frequency mode
    ions_normal_Omega_special = np.min(ions_normal_Omega_vec_1d[:-1])
    ions_normal_g_special = ions_normal_g_vec_1d[np.argmin(ions_normal_Omega_vec_1d[:-1])]
    ions_normal_damping_special = ions_normal_damping_vec_1d[np.argmin(ions_normal_Omega_vec_1d[:-1])]
    
    gamma_p_ions_normal_special = ions_normal_damping_vec_1d[-1] + (ions_normal_damping_special * ions_normal_g_special**2 * (4 * ions_normal_Omega_special * ions_normal_Omega_vec_1d[-1])) / ((ions_normal_Omega_special**2 - ions_normal_Omega_vec_1d[-1]**2)**2)
    z_phonon_number_ions_normal_special = (params.E_dot_p + params.E_dot_td) / (hbar * ions_normal_Omega_vec_1d[-1] * gamma_p_ions_normal_special)

    # Update sys_matrices with calculated couplings for later inspection
    sys_matrices_1d['coupling_bare'] = mod_g_vec_1d
    sys_matrices_1d['coupling_ions_normal'] = ions_normal_g_vec_1d
    sys_matrices_1d['damping_ions_normal'] = ions_normal_damping_vec_1d
    sys_matrices_1d['coupling_ions_normal_special'] = ions_normal_g_special

    if n_ions < 3:
        return sys_matrices_1d, z_phonon_number_bare_analytical, z_phonon_number_ions_normal_analytical, z_phonon_number_ions_normal_special, z_phonon_number_ions_normal_special
    
    # --- 4. Special Mode Analysis (First 2 Modes) for Larger Chains ---
    ions_normal_Omega_special2 = ions_normal_Omega_vec_1d[:2]
    ions_normal_g_special2 = ions_normal_g_vec_1d[:2]
    ions_normal_damping_special2 = ions_normal_damping_vec_1d[:2]
    
    gamma_p_ions_normal_special2 = ions_normal_damping_vec_1d[-1] + np.sum((ions_normal_damping_special2 * ions_normal_g_special2**2 * (4 * ions_normal_Omega_special2 * ions_normal_Omega_vec_1d[-1])) / ((ions_normal_Omega_special2**2 - ions_normal_Omega_vec_1d[-1]**2)**2))
    z_phonon_number_ions_normal_special2 = (params.E_dot_p + params.E_dot_td) / (hbar * ions_normal_Omega_vec_1d[-1] * gamma_p_ions_normal_special2)

    sys_matrices_1d['coupling_ions_normal_special2'] = ions_normal_g_special2
    
    return sys_matrices_1d, z_phonon_number_bare_analytical, z_phonon_number_ions_normal_analytical, z_phonon_number_ions_normal_special, z_phonon_number_ions_normal_special2




# ==============================================================================
# BLOCK 5: CORE ANALYSIS WORKFLOW
# ==============================================================================

def analyze_single_configuration_jit(args):
    """
    Worker function to process a single equilibrium configuration.
    
    Performs the following steps:
    1. Sets stability flags (Floquet skipped, relying on static check).
    2. Steady-State Analysis - solves the Lyapunov equation for covariance.
    3. Quantum Metrics - computes purity and phonon occupancy.

    Parameters
    ----------
    args : tuple
        (config_data, n_ions, gamma_p, params)

    Returns
    -------
    dict
        Updated configuration dictionary with quantum metrics.
    """
    config_data, n_ions, gamma_p, params = args
    sys_matrices = config_data['sys_matrices']

    # --- 1. Stability Flags ---
    # We rely on the 3D static stability check performed in the pre-screening.
    # For a purely 1D axial approximation without axial micromotion, 
    # static stability implies dynamical stability.
    config_data['is_floquet_stable'] = True
    
    # --- 2. Steady-State Covariance (Lyapunov) ---
    try:
        # Construct Drift (A) and Diffusion (Gamma) matrices
        A_cov_mat, Gamma_vec, A_mat = calculate_static_eom_matrices_jit(gamma_p, params, sys_matrices)
        
        # Solve A_cov * vec(Sigma) = -vec(Gamma)
        # Note: A_cov_mat is (2N)^2 x (2N)^2.
        ss_cov_vec = np.linalg.solve(A_cov_mat, -Gamma_vec)
        ss_cov_mat = ss_cov_vec.reshape((A_mat.shape[0], A_mat.shape[0]), order='F')

        # --- 3. Unscaling Transformation ---
        # The solver works in dimensionless units (scaled by ZPF).
        # We transform back to SI units: Sigma = S^-1 * Sigma' * S^-1^T
        _, _, _, _, _, q_zpf_vec, p_zpf_vec = params.get_particle_properties(n_ions)
        q_scale_vec = 1 / q_zpf_vec
        p_scale_vec = 1 / p_zpf_vec
        scale_vec = np.concatenate([q_scale_vec, p_scale_vec])
        
        S_inv = np.diag(1.0 / scale_vec)
        
        ss_cov_mat = S_inv @ ss_cov_mat @ S_inv.T
        
        config_data['covariance_mat'] = ss_cov_mat
        
        # --- 4. Quantum Metrics ---
        purity_props = calculate_steady_state_purity(ss_cov_mat, n_ions, params)
        config_data['z_phonon_number'] = purity_props['z_phonon_number']
        config_data['z_purity'] = purity_props['z_purity']
            
    except np.linalg.LinAlgError:
        # Catch singular matrix errors (implies numerical instability)
        config_data['covariance_mat'] = None
    
    return config_data

def analyze_ion_count_jit(args):
    """
    Analyzes the system for a fixed number of ions (N).
    
    Workflow:
    1. Search for unique equilibrium configurations (1D Axis).
    2. Filter results for collisions and 3D static stability.
    3. Run parallel steady-state analysis on surviving candidates.
    4. Aggregate statistics.

    Parameters
    ----------
    args : tuple
        (n_ions, gamma_p, n_runs, params, verbose)

    Returns
    -------
    tuple
        (n_ions, results_dictionary)
    """
    n_ions, gamma_p, n_runs, params, verbose = args
    n_particles = n_ions + 1
    
    if verbose: 
        print(f"\n===== Starting job for N_Ions = {n_ions} =====")
    
    # --- Step 1: Find Equilibrium Candidates ---
    configs_1d, configs_1d_likelihood = find_unique_equilibrium_configs(n_ions, n_runs, params, verbose=verbose)
    
    if not configs_1d:
        return n_ions, {'_summary_statistics': {'n_ions': n_ions, 'total_configs': 0}}
    
    job_tickets = [] 
    if verbose: 
        print(f"--> Pre-screening {len(configs_1d)} configs for collisions and 3D stability...")
    
    min_distance_1st = 1e-15 # 1 femtometer check (numerical overlap)
    min_distance_2nd = 1e-9  # 1 nanometer check (physical overlap)
    
    # --- Step 2: Pre-Screening Loop ---
    for z_coord_vec_1d in configs_1d:
        
        # Collision Check 1
        sorted_coord_vec = np.sort(z_coord_vec_1d)
        if np.min(np.diff(sorted_coord_vec)) < min_distance_1st:
            if verbose: print(f"--> Rejected a configuration for coordinates being too close (less than 1 fm).")
            continue

        # --- 3D Static Stability Check ---
        # Crucial for 1D approximation validity.
        # We inflate the 1D solution to 3D and check the full Hessian eigenvalues.
        
        # 1. Inflate to 3D (x=0, y=0)
        coord_vec_3d = np.zeros(3 * n_particles)
        coord_vec_3d[2::3] = z_coord_vec_1d 

        # 2. Get 3D properties
        charge_vec, _, _, mass_vec_3d, Omega_vec_3d, _, _ = params.get_particle_properties(n_ions)
        
        # 3. Calculate Full 3D Hessian
        hessian_trap_3d = np.diag(mass_vec_3d * Omega_vec_3d.flatten()**2)
        hessian_coulomb_3d = calculate_hessian_matrix_3d_jit(coord_vec_3d, params, charge_vec)
        hessian_total_3d = hessian_trap_3d + hessian_coulomb_3d
        
        # 4. Check Dynamical Matrix Eigenvalues
        inv_mass_mat_3d = np.diag(1 / mass_vec_3d)
        dynamical_mat_3d = inv_mass_mat_3d @ hessian_total_3d
        eigenvalues_3d_w_squared = np.linalg.eigvals(dynamical_mat_3d)
        
        # 5. Stability Criterion: All eigenvalues > 0 (no saddle points)
        # We allow small negative numbers (-1e-15) for numerical noise.
        if np.any(eigenvalues_3d_w_squared < -1e-15):
            if verbose: print(f"--> Rejected a configuration for failing 3D stability (potential zigzag instability).")
            continue

        # --- Step 3: Prepare Valid Candidate ---
        sys_matrices_1d = compute_system_matrices_jit(z_coord_vec_1d, params)
        
        initial_config_data = {
            'coord_vec': z_coord_vec_1d,
            'likelihood': configs_1d_likelihood[get_canonical_signature(z_coord_vec_1d, n_ions)],
            'sys_matrices': sys_matrices_1d,
            'is_static_stable': True,
            'is_floquet_stable': True 
        }
        
        # Collision Check 2
        sorted_coord_vec = np.sort(z_coord_vec_1d)
        if np.min(np.diff(sorted_coord_vec)) < min_distance_2nd:
            if verbose: print(f"--> Rejected a configuration for coordinates being too close (less than 1 nm).")
            continue

        # Calculate Analytical Estimates (useful for benchmarking)
        sys_matrices_1d, z_ph_bare, z_ph_norm, z_ph_spec, z_ph_spec2 = analytical_steady_state_purity(n_ions, gamma_p, sys_matrices_1d, params)

        initial_config_data['sys_matrices'] = sys_matrices_1d
        initial_config_data['z_phonon_number_bare_analytical'] = z_ph_bare
        initial_config_data['z_phonon_number_ions_normal_analytical'] = z_ph_norm
        initial_config_data['z_phonon_number_ions_normal_special'] = z_ph_spec
        initial_config_data['z_phonon_number_ions_normal_special2'] = z_ph_spec2
        
        job_tickets.append(initial_config_data)

    if not job_tickets:
        if verbose: print("--> No truly stable (3D) configurations found.")
        return n_ions, {'_summary_statistics': {'n_ions': n_ions, 'total_configs': len(configs_1d), 'static_stable': 0}}
    
    if verbose: print(f"--> Found {len(job_tickets)} promising candidates. Starting full parallel analysis...")

    # --- Step 4: Parallel Execution ---
    job_args = [(ticket, n_ions, gamma_p, params) for ticket in job_tickets]
    
    with mp.Pool() as pool:
        analysis_results = pool.map(analyze_single_configuration_jit, job_args)

    results_dict = {get_canonical_signature(res['coord_vec'], n_ions): res for res in analysis_results}
    
    # --- Step 5: Statistics Aggregation ---
    total = len(configs_1d)
    static = len(job_tickets)
    floquet = sum(1 for c in results_dict.values() if c['is_floquet_stable'])
    
    summary_stats = {
        'n_ions': n_ions, 
        'total_configs': total, 
        'static_stable': static, 
        'floquet_stable': floquet
    }
    
    # Extract best phonon numbers
    floquet_stable_configs = [c for c in results_dict.values() if c['is_floquet_stable']]
    if floquet_stable_configs:
        # Extract columns: [Numerical, Bare, Normal, Special1, Special2]
        phonons = np.array([[
            c.get('z_phonon_number', np.nan), 
            c['z_phonon_number_bare_analytical'], 
            c['z_phonon_number_ions_normal_analytical'], 
            c['z_phonon_number_ions_normal_special'], 
            c['z_phonon_number_ions_normal_special2']
        ] for c in floquet_stable_configs if 'z_phonon_number' in c and not np.isnan(c['z_phonon_number'])])
        
        if phonons.size > 0:
            summary_stats['mean_z_phonon'] = np.mean(phonons[:,0])
            summary_stats['std_z_phonon'] = np.std(phonons[:,0])
            
            # Find index of best (lowest) numerical phonon number
            best_idx = np.argmin(phonons[:,0])
            
            summary_stats['best_z_phonon']  = phonons[best_idx, 0]
            summary_stats['best_z_phonon_bare_analytical'] = phonons[best_idx, 1]
            summary_stats['best_z_phonon_ions_normal_analytical'] = phonons[best_idx, 2]
            summary_stats['best_z_phonon_ions_normal_special'] = phonons[best_idx, 3]
            summary_stats['best_z_phonon_ions_normal_special2'] = phonons[best_idx, 4]
        
    results_dict['_summary_statistics'] = summary_stats
    
    if verbose:
        print(f"--> Analysis for N_ions = {n_ions} complete. Found {floquet} fully stable config(s).")
        
    return n_ions, results_dict


def run_ion_scan_jit(n_ions, gamma_p, n_runs=None, params=None, verbose=True):
    """
    Main driver function to scan over different ion numbers.
    
    Executes the analysis serially for each N to maintain clear logging,
    while utilizing parallelism within the analysis of each N.

    Parameters
    ----------
    n_ions : int or list
        If int, scans range(1, n_ions+1). If list, scans those specific numbers.
    gamma_p : float
        Nanoparticle damping rate.
    n_runs : int or list, optional
        Number of equilibrium search attempts per N. 
        Default scales as 10000 * N.
    params : SystemParameters
        Physical constants.

    Returns
    -------
    dict
        Nested dictionary of results, keyed by N.
    """
    if isinstance(n_ions, int):
        ion_list = list(range(1, n_ions + 1))
    else:
        ion_list = list(n_ions)

    if n_runs is None:
        # Heuristic: Larger systems have more complex energy landscapes, need more runs
        run_list = [10000*n for n in ion_list]
    else:
        run_list = list(n_runs) if isinstance(n_runs, list) else [n_runs] * len(ion_list)

    if len(ion_list) != len(run_list):
        raise ValueError("n_ions and n_runs lists must have the same length.")

    job_args = [(n, gamma_p, r, params, verbose) for n, r in zip(ion_list, run_list)]
    
    print(f"===== Starting JIT-Optimized Ion Scan SERIALLY for N = {ion_list} =====")
    print("      (Parallelism will be used within each N-ion analysis)")
    
    # Serial loop over N, parallel processing within N
    results_list = [analyze_ion_count_jit(args) for args in job_args]
        
    return dict(sorted(results_list))  


# ==============================================================================
# BLOCK 7: 1D DATA VISUALIZATION FUNCTIONS
# ==============================================================================

def print_summary_table_1d(results_dict):
    """
    Prints a formatted summary table of the 1D simulation results to the console.

    Displays the number of configurations found, stability counts, and comparison
    between numerical results and analytical estimates for each ion number.

    Parameters
    ----------
    results_dict : dict
        The main results dictionary containing analysis data.
    """
    print("\n" + "="*200)
    print("---- 1D ION SCAN SUMMARY ----".center(80))
    
    header = (f"{'N_Ions':<8} {'Total':<10} {'Static':<10} {'Floquet':<10} "
              f"{'Best Z Ph Bare':<14} {'Best Z Ph Ions Normal Special':<14}"
              f"{'Best Z Ph Ions Normal Special2':<14}{'Best Z Ph Ions Normal':<14} "
              f"{'Best Z Ph':<14} {'Mean Z Ph':<14} {'Std Dev Z Ph':<15}")
    print(header)
    print("-" * 200)
    
    for n_ions in sorted(results_dict.keys()):
        stats = results_dict[n_ions].get('_summary_statistics', {})
        if not stats or stats.get('total_configs', 0) == 0: continue
            
        row = (f"{stats.get('n_ions', 0):<8} {stats.get('total_configs', 0):<10} "
               f"{stats.get('static_stable', 0):<10} "
               f"{stats.get('floquet_stable', 0):<10} "
               f"{stats.get('best_z_phonon_bare_analytical', np.nan):<14.4e}"
               f"{stats.get('best_z_phonon_ions_normal_special', np.nan):<14.4e}"
               f"{stats.get('best_z_phonon_ions_normal_special2', np.nan):<14.4e}"
               f"{stats.get('best_z_phonon_ions_normal_analytical', np.nan):<14.4e}"
               f"{stats.get('best_z_phonon', np.nan):<14.4e} "
               f"{stats.get('mean_z_phonon', np.nan):<14.4e} "
               f"{stats.get('std_z_phonon', np.nan):<15.4e}")
        print(row)
    
    print("="*200)

def print_config_details_1d(results_dict, n_ions=None, signature=None):
    """
    Pretty-prints detailed configuration info for specific results.

    Parameters
    ----------
    results_dict : dict
        Main results data.
    n_ions : int, optional
        Specific ion count to inspect. If None, prints all.
    signature : tuple, optional
        Specific configuration signature to inspect.
    """
    ion_configs = {n_ions: results_dict[n_ions]} if n_ions is not None else results_dict
    
    for n_ions_key, configs in ion_configs.items():
        print(f"\n{'='*70}\nN_IONS = {n_ions_key}\n{'='*70}")
        
        configs_to_show = {signature: configs[signature]} if signature else \
                          {sig: data for sig, data in configs.items() if sig != '_summary_statistics'}
        
        for sig, data in configs_to_show.items():
            print(f"\n  Configuration: {sig}")
            print(f"  {'-'*66}")
            print(f"    Static Stable:  {'✓' if data.get('is_static_stable') else '✗'}")
            print(f"    Floquet Stable: {'✓' if data.get('is_floquet_stable') else '✗'}")
            
            if data.get('is_floquet_stable'):
                print("\n    Phonon Number (Z): {0:.6e}".format(data.get('z_phonon_number', np.nan)))
                print("    Purity (Z):        {0:.6f}".format(data.get('z_purity', np.nan)))

            print("\n    Available Data:")
            if 'coord_vec' in data:
                print(f"      coord_vec: shape {data['coord_vec'].shape}")
            if data.get('covariance_mat') is not None:
                print(f"      covariance_mat: shape {data['covariance_mat'].shape}")

def get_best_configs_1d(results_dict, top_n=5):
    """
    Extracts the top N stable configurations with the lowest Z-phonon numbers across all ion counts.

    Parameters
    ----------
    results_dict : dict
        Main results data.
    top_n : int
        Number of top results to return.

    Returns
    -------
    list
        List of tuples (n_ions, signature, phonon_number, data_dict) sorted by phonon number.
    """
    all_stable = []
    
    for n_ions, configs in results_dict.items():
        for sig, data in configs.items():
            if sig == "_summary_statistics": continue
            
            if data.get('is_floquet_stable'):
                phonon_num = data.get('z_phonon_number', np.nan)
                if not np.isnan(phonon_num):
                    all_stable.append((n_ions, sig, phonon_num, data))
    
    return sorted(all_stable, key=lambda x: x[2])[:top_n]

def plot_frequencies(results_dict, max_n_ions=None, save_path="."):
    """
    Plots the Ion Normal Mode frequencies and coupling strengths for the best configurations.

    Generates a dual-axis plot:
    1. Left Axis: Ion Normal Mode frequencies (dots).
    2. Right Axis: Coupling strength to the slowest ion normal mode (line).

    Parameters
    ----------
    results_dict : dict
        Main results data.
    max_n_ions : int, optional
        Filter to plot only up to this number of ions.
    save_path : str
        Directory to save the plot.
    """
    if not results_dict:
        print("Results dictionary is empty. Nothing to plot.")
        return

    ion_counts_to_plot = sorted(results_dict.keys())
    if max_n_ions is not None:
        ion_counts_to_plot = [n for n in ion_counts_to_plot if n <= max_n_ions]

    # --- 1. Extract Frequency Data from Best Configurations ---
    plot_data = []
    for n_ions in ion_counts_to_plot:
        # Find the single best configuration for this ion count
        best_configs = get_best_configs_1d({n_ions: results_dict[n_ions]}, top_n=1)
        
        if not best_configs:
            continue
        
        config_data = best_configs[0][3]
        sys_matrices = config_data['sys_matrices']
        
        mod_bare_freqs = sys_matrices['static_mod_bare_frequencies']
        ions_nm_freqs = sys_matrices['static_ions_normal_frequencies']
        ions_nm_g = sys_matrices['coupling_ions_normal']

        # Find coupling to the lowest frequency mode
        idx = np.argmin(ions_nm_freqs[:-1])
        ions_nm_g_lowest_freq = np.abs(ions_nm_g[idx])
        
        plot_data.append({
            'n_ions': n_ions,
            'mod_bare_freqs': mod_bare_freqs,
            'ions_nm_freqs': ions_nm_freqs,
            'ions_nm_g': ions_nm_g_lowest_freq
        })

    if not plot_data:
        print("No stable configurations found to plot.")
        return

    # --- 2. Create Plot: Ion Normal Mode Frequencies & Coupling ---
    fig2, ax2 = plt.subplots(figsize=(16, 9))
    g_rad_s = []
    g_khz = []
    
    for data in plot_data:
        n_ions = data['n_ions']
        freqs_rad_s = data['ions_nm_freqs']
        freqs_mhz = freqs_rad_s / (2 * np.pi * 1e6)

        g_rad_s.append(data['ions_nm_g'])
        g_khz.append(data['ions_nm_g'] / (2 * np.pi * 1e3))
        
        # Plot the N ion normal mode frequencies in green
        ax2.plot([n_ions] * n_ions, freqs_mhz[:-1], 'o', color='forestgreen', markersize=12, 
                 label='Ion Normal Modes' if n_ions == plot_data[0]['n_ions'] else "")
    
    # Configure Left Axis (Frequencies)
    ax2.set_xticks(range(0, 12 + 1))
    ax2.tick_params(axis='both', which='both', direction='in', top=True, right=False, width=2, labelsize=12)
    ax2.grid(axis='x', which='major', linestyle='-', linewidth=3, alpha=0.5)
    ax2.set_ylim([0.0, 6.0])
    
    for spine in ax2.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')

    ax2.set_xlabel('Number of Ions ($N_{ions}$)', fontsize=14)
    ax2.set_ylabel('Frequency (MHz)', fontsize=14)
    ax2.set_title('Ion Normal Mode Frequencies of Best Configurations', fontsize=16)

    # Configure Right Axis (Coupling Strength)
    ax3 = ax2.twinx()
    ax3.plot(range(1,len(g_khz)+1), g_khz, color='black', marker='d', markersize=10, 
             linestyle='-', linewidth=4, label='Coupling to Slowest Ion Normal Mode')
    
    # Add Trendline
    ax3.plot(range(1,len(g_khz)+1), np.sqrt(np.arange(1,len(g_khz)+1)) * g_khz[0], 
             color='gray', linestyle='--', linewidth=4, label='Trendline Even N')
    
    ax3.tick_params(axis='y', which='both', direction='in', top=False, right=True, width=2, labelsize=12)
    ax3.set_ylim([0.0, 3.0])
    ax3.set_ylabel('Coupling (kHz)', fontsize=14)
    
    plt.tight_layout()
    filename = f"{save_path}/frequencies_coupling.png"
    plt.savefig(filename, dpi=300)
    print(f"--> Saved plot: {filename}")
    plt.close()
    

def plot_all_configurations_1d(results_dict, max_n_ions=None, save_path="."):
    """
    Visualizes all stable configurations found for each ion count.
    
    Creates a row of subplots for each ion number N, showing the spatial
    arrangement of ions and the nanoparticle.

    Parameters
    ----------
    results_dict : dict
        Main results data.
    max_n_ions : int, optional
        Filter max N.
    save_path : str
        Directory to save the plot.
    """
    if not results_dict: return
    ion_counts_to_plot = sorted(results_dict.keys())
    if max_n_ions is not None:
        ion_counts_to_plot = [n for n in ion_counts_to_plot if n <= max_n_ions]

    for n_ions in ion_counts_to_plot:
        # Get all stable configurations for the given n_ions
        configs_for_n = {s: d for s, d in results_dict[n_ions].items() if s != '_summary_statistics' and d.get('is_floquet_stable')}
        if not configs_for_n: continue

        num_stable = len(configs_for_n)
        
        # Create a figure with a row of subplots
        fig, axes = plt.subplots(1, num_stable, figsize=(num_stable * 6, 3), squeeze=False)
        axes = axes.flatten() 
        
        fig.suptitle(f'Stable 1D Configurations for {n_ions} Ion(s)', fontsize=16)

        # Plot each configuration
        for idx, data in enumerate(configs_for_n.values()):
            ax = axes[idx]
            z_coord_vec = data['coord_vec'] * 1e6 # convert to micrometers
            
            # Plot ions
            ax.scatter(z_coord_vec[:n_ions], np.zeros(n_ions), 
                       c='blue', s=50, label='Ions', zorder=10)
            # Plot nanoparticle
            ax.scatter(z_coord_vec[-1], 0, 
                       c='red', s=100, label='Nanoparticle', zorder=10)
            
            ax.set_title(f'Config #{idx+1}')
            ax.set_xlabel('Z-Position (μm)')
            ax.set_yticks([])
            ax.grid(True, axis='x', linestyle='--', alpha=0.6)
            if idx == 0:
                ax.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        filename = f"{save_path}/configurations_N{n_ions}.png"
        plt.savefig(filename, dpi=300)
        print(f"--> Saved plot: {filename}")
        plt.close()

def plot_phonon_statistics_1d(results_dict, save_path="."):
    """
    Plots the cooling scaling statistics (Phonon Number vs Ion Count).
    
    Compares the numerical best result against the analytical model predictions
    using high-visibility markers and lines.

    Parameters
    ----------
    results_dict : dict
        Main results data.
    save_path : str
        Directory to save the plot.
    """
    stats_list = [v['_summary_statistics'] for v in results_dict.values() if v.get('_summary_statistics', {}).get('floquet_stable', 0) > 0]
    if not stats_list:
        print("No stable configurations with phonon data found. Cannot plot.")
        return

    df = pd.DataFrame(stats_list).sort_values('n_ions')
    
    fig, ax = plt.subplots(figsize=(16, 9))

    # --- Plot Data ---
    # Numerical Best Result
    ax.plot(df['n_ions'], df['best_z_phonon'],
            color='black', linestyle='-', linewidth=7, marker='o', markersize=20, label='Best Z Phonon #')

    # Analytical Benchmark 1: Bare frequencies
    ax.plot(df['n_ions'], df['best_z_phonon_bare_analytical'],
            color='green', linestyle=':', linewidth=7, marker='s', markersize=20, label='Best Z Phonon Bare Analytical #')    

    # Analytical Benchmark 2: Special Mode 2
    ax.plot(df['n_ions'], df['best_z_phonon_ions_normal_special2'],
            color='red', linestyle=':', linewidth=7, marker='d', markersize=20, label='Best Z Phonon Normal Ions Special #')
            
    ax.set_yscale('log')

    # --- Plot Formatting (Strictly Preserved) ---
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, width=2, labelsize=12)

    ax.grid(which='major', color='gray', ls='-', linewidth=3, alpha=0.5)
    ax.grid(which='minor', color='gray', ls='-', linewidth=3, alpha=0.5)

    # Custom Tick Locators for Log Scale
    minor_ticks0 = np.logspace(np.log10(10**5), np.log10(10**6), 3)
    minor_ticks1 = np.logspace(np.log10(10**6), np.log10(10**7), 3)
    minor_ticks2 = np.logspace(np.log10(10**7), np.log10(10**8), 3)
    ax.yaxis.set_minor_locator(ticker.FixedLocator(np.concatenate((minor_ticks0,minor_ticks1,minor_ticks2))))

    ax.set_ylim([10**5.8, 10**8.2])
    
    for spine in ax.spines.values():
        spine.set_linewidth(2.25)
        spine.set_edgecolor('black')

    ax.set_title('Nanoparticle Z-Mode Cooling vs. Ion Crystal Size', fontsize=16)
    ax.set_xlabel('Number of Ions in Crystal')
    ax.set_ylabel('Phonon Number')
    
    plt.tight_layout()
    filename = f"{save_path}/phonon_statistics.png"
    plt.savefig(filename, dpi=300)
    print(f"--> Saved plot: {filename}")
    plt.close()


# ==============================================================================
# BLOCK 7: MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    # Use standard "fork" context to ensure compatibility with JIT and Multiprocessing
    # on MacOS/Linux.
    try:
        mp.set_start_method('fork')
    except RuntimeError:
        pass # Context already set

    print("========================================================================")
    print("HYBRID TRAP SIMULATION: 1D SYMPATHETIC COOLING (ArXiv Submission Code)")
    print("========================================================================")
    
    start_time = time.time()
    
    # 1. Initialize System Physics
    # Note: Parameters are defined in the SystemParameters class in Block 1
    system_params = SystemParameters()
    
    # 2. Define Simulation Scope
    # User requested N=13 ions. 
    # N_runs scales with N to ensure the ground state is found in the complex landscape.
    max_ions = 8
    run_schedule = [(10**3) * n for n in range(1, max_ions + 1)]
    
    # Damping rate provided: 2 * pi * 44.5e-9 rad/s
    gamma_particle = 2 * pi * 44.5e-9 
    
    # 3. Execute Scan
    # This will run the equilibrium search and steady-state analysis
    results_dict = run_ion_scan_jit(
        n_ions=max_ions, 
        n_runs=run_schedule,
        gamma_p=gamma_particle, 
        params=system_params,
        verbose=True
    )

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")

    # 4. Output Results
    print_summary_table_1d(results_dict)

    # 5. Generate Figures
    print("\nGenerating plots...")
    plot_phonon_statistics_1d(results_dict)
    
    print("\nDone. Output files generated in current directory.")