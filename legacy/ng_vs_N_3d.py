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
        # DC voltages derived to satisfy Laplace equation constraints
        V_dc_x = -0.5 * (alpha_z/alpha_r) * (r0/z0)**2 * V_dc_z # Constant voltage
        V_dc_y = -0.5 * (alpha_z/alpha_r) * (r0/z0)**2 * V_dc_z # Constant voltage
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
        #np.array([561.39, 750.09, 842.08])
        

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
        q_zpf_vec_3d = np.sqrt(hbar / (2 * mass_vec_3d * Omega_vec_3d.flatten()))
        p_zpf_vec_3d = np.sqrt((hbar * mass_vec_3d * Omega_vec_3d.flatten()) / 2)
        
        return charge_vec, mass_vec, Omega_vec_1d, mass_vec_3d, Omega_vec_3d, q_zpf_vec_3d, p_zpf_vec_3d




# ==============================================================================
# BLOCK 2: COMPUTING EQUILIBRIUM COORDINATES
# ==============================================================================
# This block contains high-performance (JIT-compiled) kernels for calculating 
# forces and finding the equilibrium positions of the ion-nanoparticle crystal.

# --- Equilibrium Search Kernels ---

@jit(nopython=True, fastmath=True, cache=True)
def _objective_function_kernel(coord_vec, n_ions, charge_vec, Omega_vec_3d, mass_vec_3d):
    """
    Calculates the net forces on all particles (Trap + Coulomb).
    
    This function acts as the objective for the root finder. Equilibrium is found
    when the net force on every particle is zero.
    
    Parameters
    ----------
    coord_vec : ndarray
        Flattened 1D array of coordinates (size 3N).
    n_ions : int
        Number of ions (used for logic, though array sizes dictate loop).
    charge_vec : ndarray
        Vector of charges for all N particles.
    Omega_vec_3d : ndarray
        Reshaped matrix of trap frequencies (secular approximation).
    mass_vec_3d : ndarray
        Reshaped matrix of masses.

    Returns
    -------
    ndarray
        Flattened 1D array of net forces (size 3N).
    """
    # Reshape 1D vector to (n_particles, 3) for 3D vector math
    coords_reshaped = coord_vec.reshape(-1, 3)
    
    # --- 1. Trap Confinement Forces ---
    # Calculated using the secular frequencies: F = -m * Omega^2 * r
    # We reshape mass inputs to match coordinate dimensions for broadcasting
    mass_vec_3d = mass_vec_3d.reshape(-1, 3)
    trap_forces = -mass_vec_3d * Omega_vec_3d**2 * coords_reshaped
    
    # --- 2. Coulomb Interaction Forces ---
    # Pre-compute charge products Q_i * Q_j / (4*pi*eps0)
    charge_prod_matrix = np.outer(charge_vec, charge_vec) / (4 * pi * eps0)
    
    net_coulomb_forces = np.zeros_like(coords_reshaped)

    
    # N-body loop
    n_particles = coords_reshaped.shape[0]
    # Choose a pair of particles that hasn't been accounted for already
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            # Create a (3,1) array containing the difference of the coordinates of the two partciles along each axis
            dist_vec = coords_reshaped[i] - coords_reshaped[j]
            dist_sq = np.sum(dist_vec**2)
            dist = np.sqrt(dist_sq)
            
            # F_coulomb = (k * Q1 * Q2 / r^3) * vec(r)
            force_mag_dir = (charge_prod_matrix[i, j] / dist**3) * dist_vec
            
            # Newton's 3rd Law: Force on i is opposite to force on j
            net_coulomb_forces[i] += force_mag_dir
            net_coulomb_forces[j] -= force_mag_dir
            
    # Combine forces and flatten back to 1D for the solver
    all_forces_flat = (trap_forces + net_coulomb_forces).flatten()
    
    return all_forces_flat

def _objective_function(coord_vec, n_ions, charge_vec, Omega_vec_3d, mass_vec_3d):
    """Wrapper function to expose the JIT kernel to the SciPy solver."""
    return _objective_function_kernel(coord_vec, n_ions, charge_vec, Omega_vec_3d, mass_vec_3d)

def find_single_equilibrium(run_index, n_ions, params):
    """
    Attempts to find a single equilibrium configuration starting from a random guess.
    
    Designed to be run in parallel.

    Parameters
    ----------
    run_index : int
        Unique ID for this run (used for random number generator seeding).
    n_ions : int
        Number of ions in the system.
    params : SystemParameters
        Instance containing physical constants.

    Returns
    -------
    ndarray or None
        Flattened coordinate vector if successful, else None.
    """
    n_particles = n_ions + 1
    
    # Extract property arrays
    charge_vec, _, _, mass_vec_3d_flat, Omega_vec_3d, _, _ = params.get_particle_properties(n_ions)
    mass_vec_3d = mass_vec_3d_flat.reshape(-1, 3)
    
    # Create a partial function with fixed parameters for the solver
    objective_func = partial(
        _objective_function, 
        n_ions=n_ions, 
        charge_vec=charge_vec, 
        Omega_vec_3d=Omega_vec_3d, 
        mass_vec_3d=mass_vec_3d
    )
    
    # --- Random Seeding for Parallelism ---
    # Ensure every process gets a unique seed using time + run ID + process ID
    np.random.seed(int(time.time()) + run_index + mp.current_process().pid)
    
    # --- Seed Guess ---
    # Random uniform distribution [1e-6, 2e-4] meters.
    # The lower bound (1 micron) avoids the Coulomb singularity at r=0.
    guess = np.random.uniform(-1e-4, 1e-4, size=3 * n_particles)
    
    # Solve F(r) = 0
    solution = root(objective_func, guess, method='hybr')
    
    return solution.x if solution.success else None


def get_canonical_signature(coord_vec, n_ions, precision=8):
    """
    Generates a unique signature for a crystal configuration to identify distinct geometries.

    Because ions are identical, swapping indices (e.g., Ion 1 <-> Ion 2) results in 
    the same physical state but a different numerical coordinate vector. This function 
    rounds and sorts coordinates to create a permutation-invariant signature.

    Parameters
    ----------
    coord_vec : ndarray
        The flattened coordinate vector of size 3(N_ions + 1).
    n_ions : int
        The number of ions in the system.
    precision : int, optional
        Number of decimal places to round to for uniqueness checks (default is 8).

    Returns
    -------
    tuple
        A sorted tuple of ion coordinates (ordered by Z, then Y, then X) 
        serving as a hashable signature.
    """
    # Reshape and extract only ion coordinates (nanoparticle is distinguishable)
    ion_coords = coord_vec.reshape(-1, 3)[:n_ions]
    
    # Round to avoid floating point noise
    rounded_coords = np.round(ion_coords, precision)
    
    # Sort coordinates spatially (Z, then Y, then X) to create a canonical order
    return tuple(map(tuple, sorted(rounded_coords.tolist(), key=lambda p: (p[2], p[1], p[0]))))

def find_unique_equilibrium_configs(n_ions, n_runs, params, verbose=True):
    """
    Runs multiple parallel searches to identify all unique stable crystal configurations.

    Executes a search by spawning `n_runs` optimization attempts from random seed conditions in parallel.

    Parameters
    ----------
    n_ions : int
        Number of ions in the crystal.
    n_runs : int
        Number of random seed guesses to attempt.
    params : SystemParameters
        Instance containing physical constants and trap parameters.
    verbose : bool, optional
        If True, prints progress and statistics (default is True).

    Returns
    -------
    list of ndarray
        A list containing the unique flattened coordinate vectors found (3N dimensional).
    """
    if verbose: 
        print(f"--- Starting parallel search for {n_ions}-ion crystal configurations ({n_runs} runs) ---")
    
    start_time = time.time()
    
    # Parallel execution
    with mp.Pool(mp.cpu_count()) as pool:
        solver_func = partial(find_single_equilibrium, n_ions=n_ions, params=params)
        results = pool.map(solver_func, range(n_runs))
    
    # Filter for unique solutions
    unique_solutions = {}
    successful_runs = 0
    
    for coord_vec in results:
        if coord_vec is not None:
            successful_runs += 1
            signature = get_canonical_signature(coord_vec, n_ions)
            
            # Store if we haven't seen this geometry before
            if signature not in unique_solutions:
                unique_solutions[signature] = coord_vec
                
    if verbose:
        print(f"--- Search complete in {time.time() - start_time:.2f}s | "
              f"Success: {successful_runs}/{n_runs} | Unique: {len(unique_solutions)} ---")
              
    return list(unique_solutions.values())



# ==============================================================================
# BLOCK 3: COMPUTING HESSIAN MATRIX
# ==============================================================================
# This block contains the JIT-compiled kernel for calculating the Hessian matrix 
# of the Coulomb interaction potential. This matrix describes the coupling 
# between different degrees of freedom in the crystal.


@jit(nopython=True, fastmath=True, cache=True)
def _calculate_hessian_matrix_kernel(coord_vec, charge_vec):
    """
    Calculates the Hessian matrix (second derivatives) of the Coulomb potential energy.
    
    H_ij = d^2(V_Coulomb) / (dx_i dx_j)
    
    This matrix represents the linearized stiffness of the Coulomb repulsion 
    between particles. It is added to the trap potential curvature to form 
    the total stiffness matrix.

    Parameters
    ----------
    coord_vec : ndarray
        Positions of particles, shape (N, 3).
    charge_vec : ndarray
        Charges of particles, shape (N,).

    Returns
    -------
    ndarray
        The (3N, 3N) Hessian matrix describing Coulomb interactions.
    """
    n_particles = coord_vec.shape[0]
    
    # Precompute charge constants: k_e * Q_i * Q_j
    # Creates a matrix of dimension (N, N) containing the product of charges on each ordered pair of particles.
    prefactor_mat = np.outer(charge_vec, charge_vec) / (4 * pi * eps0)
    
    # Use a hyperstack to store (3,3) blocks corresponding to each ordered pair of particles before flattening. Total N^2 blocks.
    hessian_hyperstack = np.zeros((n_particles, n_particles, 3, 3))
    
    # --- Off-Diagonal Blocks (Interaction Terms) ---
    for i in range(n_particles):
        for j in range(i + 1, n_particles): # Iterate over unique pairs
            d_vec = coord_vec[i] - coord_vec[j]
            d_sq = np.sum(d_vec**2)
            d = np.sqrt(d_sq)
            d3 = d**3
            d5 = d**5
            
            # T_ab = (3 * r_a * r_b / r^5) - (kronecker_delta(a,b) / r^3)
            outer_prod = np.zeros((3,3))
            for row in range(3):
                for col in range(3):
                    outer_prod[row, col] = d_vec[row] * d_vec[col]

            # The identity matrix np.eye(3) is used as the Kronecker delta function over particle indices
            block = (3 * outer_prod / d5) - (np.eye(3) / d3)
            
            # Scale by charge prefactor (negative sign from potential derivative)
            block *= -prefactor_mat[i, j]
            
            # Store symmetric blocks
            hessian_hyperstack[i, j] = block
            hessian_hyperstack[j, i] = block

    # --- Diagonal Blocks (Self-Stiffness) ---
    # The diagonal block for particle 'i' is the negative sum of all off-diagonal blocks involving 'i'.
    for i in range(n_particles):
        diag_block = np.zeros((3, 3))
        for j in range(n_particles):
            if i == j:
                continue
            diag_block -= hessian_hyperstack[i, j]
        hessian_hyperstack[i, i] = diag_block
        
    # Flatten the Hessian hyperstack to a (3N, 3N) matrix with the ordering [a_x, a_y, a_z, ..., b_x, b_y, b_z]
    hessian_mat = np.zeros((3 * n_particles, 3 * n_particles))
    for i in range(n_particles):
        for j in range(n_particles):
            for row in range(3):
                for col in range(3):
                    # Map hyperstack indices to flat matrix indices
                    hessian_mat[3*i+row, 3*j+col] = hessian_hyperstack[i,j,row,col]

    return hessian_mat

def calculate_hessian_matrix_jit(coord_vec, charge_vec, params):
    """
    Wrapper to call the JIT-compiled Hessian kernel.
    
    Parameters
    ----------
    coord_vec : ndarray
        Coordinate vector. Can be flattened (3N,) or shaped (N, 3).
    charge_vec : ndarray
        Vector of particle charges.
    params : SystemParameters
        Instance containing physical constants (unused in kernel but kept for interface consistency).

    Returns
    -------
    ndarray
        The (3N, 3N) Hessian matrix.
    """
    # Ensure input is shaped (N, 3) for the kernel
    coords_reshaped = coord_vec.reshape(-1, 3)
    return _calculate_hessian_matrix_kernel(coords_reshaped, charge_vec)


# ==============================================================================
# BLOCK 4: MODULAR ANALYSIS FUNCTIONS (JIT-OPTIMIZED)
# ==============================================================================

def compute_system_matrices_jit(coord_vec, params):
    """
    Computes all fundamental matrices (Hessians, mass vectors, charge vectors) 
    for a given crystal configuration by calling the optimized JIT kernels.

    This acts as a centralized data preparation step to avoid redundant 
    calculations during the dynamics simulation.

    Parameters
    ----------
    coord_vec : ndarray
        The equilibrium position vector of the crystal (3N).
    params : SystemParameters
        Instance containing physical constants.

    Returns
    -------
    dict
        A dictionary containing:
        - 'hessian_coulomb': (3N, 3N) Coulomb stiffness matrix.
        - 'hessian_trap': (3N, 3N) Trap stiffness matrix (diagonal).
        - 'hessian_total': (3N, 3N) Total stiffness matrix.
        - 'mass_vec_3d': (3N,) Flattened mass vector.
        - 'charge_vec': (N,) Charge vector.
    """
    n_ions = coord_vec.shape[0] - 1
    charge_vec, mass_vec, _, mass_vec_3d, Omega_vec_3d, _, _ = params.get_particle_properties(n_ions)
    
    # Call the fast JIT kernel for the most expensive part (Coulomb interaction)
    hessian_coulomb = calculate_hessian_matrix_jit(coord_vec, charge_vec, params)
    
    # Compute trap stiffness (diagonal matrix K = m * Omega^2)
    # Flattening the mass and Omega vectors ensures we get a 1D vector for diagonal construction
    hessian_trap_diag = mass_vec_3d * Omega_vec_3d.flatten()**2
    
    # Total Hessian (Stiffnes) = Trap Stiffness + Coulomb Stiffness
    hessian_total = np.diag(hessian_trap_diag) + hessian_coulomb
    
    return {
        'hessian_coulomb': hessian_coulomb,
        'hessian_total': hessian_total,
        'hessian_trap': hessian_trap_diag,
        'mass_vec_3d': mass_vec_3d,
        'charge_vec': charge_vec,
        'mass_vec': mass_vec
    }


    
def calculate_static_eom_matrices_jit(gamma_p, params, sys_matrices):
    """
    Constructs the Drift and Diffusion matrices for the equations of motion.

    Parameters
    ----------
    gamma_p : float
        Nanoparticle damping rate [rad/s].
    params : SystemParameters
        System constants.
    sys_matrices : dict
        Pre-computed Hessians and mass vectors from `compute_system_matrices_jit`.

    Returns
    -------
    tuple
        (A_cov_mat, Gamma_vec, A_mat)
        - A_cov_mat: Drift matrix for the second-order moments.
        - Gamma_vec: Flattened diffusion matrix.
        - A_mat: Drift matrix for the first-order moments.
    """
    n_ions = len(sys_matrices['mass_vec']) - 1
    n_particles = n_ions + 1
    n_dim = 6 * n_particles
    
    mass_vec = sys_matrices['mass_vec_3d']
    H_total = sys_matrices['hessian_total']
    _, _, _, _, _, q_zpf_vec_3d, p_zpf_vec_3d = params.get_particle_properties(n_ions)

    # Construct damping vector [gamma_i, ..., gamma_p]
    damping_vec = np.concatenate([np.tile([params.gamma_i], 3*n_ions), np.tile([gamma_p], 3)])
    
    # Calculate renormalised frequencies using the *coupled* stiffness matrix
    new_Omega_vec_3d = np.sqrt(np.abs(np.diagonal(H_total) / mass_vec))

    # --- 1. Drift Matrix ---
    # [[ -Gamma/2,  1/m ], 
    #  [ -H_total, -Gamma/2 ]]
    A_qq = -np.diag(damping_vec / 2.0)
    A_qp = np.diag(1 / mass_vec)
    A_pq = -H_total
    A_pp = -np.diag(damping_vec / 2.0)
    A_mat = np.block([[A_qq, A_qp], [A_pq, A_pp]])

    # --- 2. Diffusion Matrix Gamma ---
    # Calculate ZPF using renormalised frequencies
    new_q_zpf_vec_3d = np.sqrt(hbar / (2 * mass_vec * new_Omega_vec_3d))
    new_p_zpf_vec_3d = np.sqrt(hbar * mass_vec * new_Omega_vec_3d / 2)
    
    # Heating rates (Recoil, Gas Scattering)
    heating_vec = np.concatenate([np.tile([params.E_dot_i], 3*n_ions), np.tile([params.E_dot_p], 3)]) / (hbar * new_Omega_vec_3d)
    
    # Localization rates (Trap displacement, Back-action) 
    localisation_vec = np.concatenate([np.tile([0], 3*n_ions), np.tile([params.E_dot_td], 3)]) / (hbar * new_Omega_vec_3d)
    localisation_vec[-3:] += (params.Gamma_to_gamma * new_q_zpf_vec_3d[-3:]**2 * gamma_p)/(q_zpf_vec_3d[-3:]**2)
    
    # Diagonal blocks 
    Gamma_qq = np.diag(new_q_zpf_vec_3d**2 * (2 * heating_vec + damping_vec))
    Gamma_pp = np.diag(new_p_zpf_vec_3d**2 * (2 * heating_vec + damping_vec + 4 * localisation_vec))

    Gamma_mat = block_diag(Gamma_qq, Gamma_pp)

    # --- 3. Scaling Transformation ---
    # Normalize to dimensionless units (q = deltaR/deltaR_zpf, p = P/P_zpf)
    q_scale_vec_3d = 1 / q_zpf_vec_3d
    p_scale_vec_3d = 1 / p_zpf_vec_3d
    scale_vec = np.concatenate([q_scale_vec_3d, p_scale_vec_3d])
    S = np.diag(scale_vec)
    
    # Transform matrices
    A_mat = S @ A_mat @ np.linalg.inv(S) 
    Gamma_mat = S @ Gamma_mat @ S.T

    # --- 4. Final Forms ---
    In = np.eye(n_dim)
    # Convert A_mat and Gamma_mat to a form that works for the flattened covariance matrix
    A_cov_mat = np.kron(In, A_mat) + np.kron(A_mat, In)
    Gamma_vec = Gamma_mat.flatten(order='F')
    
    return A_cov_mat, Gamma_vec, A_mat
    

def calculate_steady_state_purity(ss_cov_mat, n_ions, params):
    """
    Extracts purity and phonon number for the nanoparticle from the covariance matrix.
    
    Parameters
    ----------
    ss_cov_mat : ndarray
        Steady-state covariance matrix (6N, 6N).
    n_ions : int
        Number of ions (used to identify nanoparticle indices).
    params : SystemParameters
        Instance containing physical constants.

    Returns
    -------
    dict
        Dictionary containing 'purity' and 'phonon_number' for 'x', 'y', and 'z' axes.
    """
    n_dim = 3 * (n_ions + 1)
    q_local_indices = 3 * n_ions + np.arange(3) # The 3 indices at the end of the position block
    p_global_indices = n_dim + q_local_indices  # The 3 indices at the end of the momentum block, which follows the position block
    
    # Extract Variances (Diagonal elements)
    cov_qq_vec = ss_cov_mat.diagonal()[q_local_indices]
    cov_pp_vec = ss_cov_mat.diagonal()[p_global_indices]
    
    # Extract Covariances (Off-diagonal elements)
    # Correctly index (q, p) pairs
    cov_qp_vec = ss_cov_mat[q_local_indices, p_global_indices]
    
    # Calculate Purity and Phonon Number
    determinants = cov_qq_vec * cov_pp_vec - cov_qp_vec**2
    purities = (hbar / 2) / np.sqrt(np.abs(determinants))
    phonon_numbers = 0.5 * (1 / purities - 1)
    
    return {
        'x': {'purity': purities[0], 'phonon_number': phonon_numbers[0]},
        'y': {'purity': purities[1], 'phonon_number': phonon_numbers[1]},
        'z': {'purity': purities[2], 'phonon_number': phonon_numbers[2]}
    }

def calculate_nanoparticle_modes_jit(sys_matrices):
    """
    Computes normal modes using pre-computed matrices of system properties.

    Parameters
    ----------
    sys_matrices : dict
        Dictionary containing 'hessian_total' and 'mass_vec_3d'.

    Returns
    -------
    dict
        Dictionary containing 'frequencies' (ndarray) and 'modes' (ndarray).
    """
    H = sys_matrices['hessian_total']
    M = np.diag(sys_matrices['mass_vec_3d'])
    
    # Generalized eigenvalue problem: H v = w^2 M v
    w2, modes = eigh(H, M) 
    freqs = np.sqrt(np.maximum(w2, 0))
    
    return {'frequencies': freqs, 'modes': modes}


# ==============================================================================
# BLOCK 5: STABILITY ANALYSIS AND FLOQUET COMPONENT DECOMPOSITION
# ==============================================================================
# This block handles the time-dependent aspects of the simulation. It prepares 
# the Fourier components for the Floquet solver and performs the stability check.

def calculate_linear_vector(t, coord_vec, params, charge_vec, mass_vec):
    """
    Calculates the time-dependent linear force vector F(t) for the Floquet analysis.
    
    This vector accounts for the 'micromotion' driving force arising from the 
    time-dependence of the trap potential and the equilibrium Coulomb forces.

    Parameters
    ----------
    t : float
        Current time [s].
    coord_vec : ndarray
        Equilibrium positions (3N).
    params : SystemParameters
        System constants.
    charge_vec : ndarray
        Vector of particle charges.
    mass_vec : ndarray
        Vector of particle masses.

    Returns
    -------
    ndarray
        Flattened force vector (3N).
    """
    # Ensure input is (N, 3) for broadcasting
    coord_vec = coord_vec.reshape(-1, 3)
    
    n_particles = len(mass_vec)
    n_ions = n_particles - 1
    
    # --- 1. Build Parameter Matrices (N, 3) ---
    
    # 'a' parameter matrix
    ion_row_a = [params.a_x_i, params.a_y_i, params.a_z_i]
    particle_row_a = [params.a_x_p, params.a_y_p, params.a_z_p]
    a_mat = np.vstack([np.tile(ion_row_a, (n_ions, 1)), particle_row_a])

    # 'q_slow' parameter matrix
    ion_row_q_slow = [params.q_slow_x_i, params.q_slow_y_i, params.q_slow_z_i]
    particle_row_q_slow = [params.q_slow_x_p, params.q_slow_y_p, params.q_slow_z_p]
    q_slow_mat = np.vstack([np.tile(ion_row_q_slow, (n_ions, 1)), particle_row_q_slow])

    # 'q_fast' parameter matrix
    ion_row_q_fast = [params.q_fast_x_i, params.q_fast_y_i, params.q_fast_z_i]
    particle_row_q_fast = [params.q_fast_x_p, params.q_fast_y_p, params.q_fast_z_p]
    q_fast_mat = np.vstack([np.tile(ion_row_q_fast, (n_ions, 1)), particle_row_q_fast])

    # --- 2. Time-Dependent Terms ---
    cos_slow = np.cos(params.omega_slow * t)
    cos_fast = np.cos(params.omega_fast * t)
    l_sq = params.l**2

    # 3. Curvature matrix W(t)
    # W(t) = (Omega_rf^2 / 4) * [a - 2*q_slow*cos(ws*t) - 2*q_fast*cos(wf*t)]
    W_mat = (params.omega_fast**2 / 4) * (
        a_mat - 2 * q_slow_mat * l_sq * cos_slow - 2 * q_fast_mat * cos_fast
    )

    # 4. Trap Force F = -m * W(t) * r
    trap_forces = -mass_vec[:, np.newaxis] * W_mat * coord_vec

    # --- 3. Static Coulomb Force ---
    charge_prod_mat = np.outer(charge_vec, charge_vec) / (4 * pi * eps0)
    
    delta = coord_vec[:, np.newaxis, :] - coord_vec[np.newaxis, :, :]
    d_mat = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(d_mat, np.inf)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_d3_mat = d_mat**-3
        coulomb_force_tensor = charge_prod_mat[..., np.newaxis] * delta * inv_d3_mat[..., np.newaxis]
    
    net_coulomb_forces = np.sum(np.nan_to_num(coulomb_force_tensor), axis=1)
    
    return -(trap_forces + net_coulomb_forces).flatten()
    

def precompute_floquet_components_vectorized(coord_vec, gamma_p, params, charge_vec, mass_vec, hessian_mat):
    """
    Decomposes the system dynamics into Fourier components for efficient JIT integration.

    Splits the Drift Matrix and Diffusion Vector into:
    - Constant term
    - Slow Cosine term (cos(w_slow * t))
    - Fast Cosine term (cos(w_fast * t))

    Parameters
    ----------
    coord_vec : ndarray
        Equilibrium positions.
    gamma_p : float
        Nanoparticle damping rate.
    params : SystemParameters
        System constants.
    hessian_mat : ndarray
        Coulomb Hessian matrix.

    Returns
    -------
    dict
        Dictionary of scaled matrices (A_const, A_slow, A_fast) and vectors (g_const, g_slow, g_fast).
    """
    n_particles = len(mass_vec)
    n_ions = n_particles - 1
    n_dim = 6 * n_particles
    
    # --- 1. Construct Constant Drift Matrix (A_const) ---
    # Basis: [x1, p1, y1, p1, z1, p1, x2, p2...] (Interleaved)
    
    # Ion Blocks
    block_x_ion = np.array([[-params.gamma_i/2, 1/params.M_i], [-params.M_i*(params.omega_fast**2/4)*params.a_x_i, -params.gamma_i/2]])
    block_y_ion = np.array([[-params.gamma_i/2, 1/params.M_i], [-params.M_i*(params.omega_fast**2/4)*params.a_y_i, -params.gamma_i/2]])
    block_z_ion = np.array([[-params.gamma_i/2, 1/params.M_i], [-params.M_i*params.Omega_z_i**2, -params.gamma_i/2]])
    ion_const_blocks = [block_x_ion, block_y_ion, block_z_ion] * n_ions
    
    # Nanoparticle Blocks
    block_x_p = np.array([[0, 1/params.M_p], [-params.M_p*(params.omega_fast**2/4)*params.a_x_p, 0]])
    block_y_p = np.array([[0, 1/params.M_p], [-params.M_p*(params.omega_fast**2/4)*params.a_y_p, 0]])
    block_z_p = np.array([[0, 1/params.M_p], [-params.M_p*params.Omega_z_p**2, 0]])
    p_const_blocks = [block_x_p, block_y_p, block_z_p]
    
    A_const = block_diag(*(ion_const_blocks + p_const_blocks))
    
    # Add Coulomb coupling
    # Note: In interleaved basis, force/hessian terms go to lower-left of 2x2 blocks
    # Indices: Rows 1, 3, 5... Cols 0, 2, 4...
    A_const[1::2, 0::2] -= hessian_mat 
    
    # Add Nanoparticle Damping
    np_diag_indices = np.arange(n_dim - 6, n_dim)
    A_const[np_diag_indices, np_diag_indices] += -gamma_p / 2.0

    # --- 2. Construct Oscillating Coefficients (A_slow, A_fast) ---
    A_slow_coeff = np.zeros((n_dim, n_dim))
    A_fast_coeff = np.zeros((n_dim, n_dim))
    
    # Pre-calculate amplitude factors: K = m * (Omega_rf^2 / 2) * q
    ion_slow_val_x = params.M_i * (params.omega_fast**2/2) * params.q_slow_x_i * params.l**2
    ion_fast_val_x = params.M_i * (params.omega_fast**2/2) * params.q_fast_x_i
    ion_slow_val_y = params.M_i * (params.omega_fast**2/2) * params.q_slow_y_i * params.l**2
    ion_fast_val_y = params.M_i * (params.omega_fast**2/2) * params.q_fast_y_i
    
    p_slow_val_x = params.M_p * (params.omega_fast**2/2) * params.q_slow_x_p * params.l**2
    p_fast_val_x = params.M_p * (params.omega_fast**2/2) * params.q_fast_x_p
    p_slow_val_y = params.M_p * (params.omega_fast**2/2) * params.q_slow_y_p * params.l**2
    p_fast_val_y = params.M_p * (params.omega_fast**2/2) * params.q_fast_y_p

    # Fill Matrix (Ions)
    for i in range(n_ions):
        idx_px, idx_qx = 6*i + 1, 6*i + 0
        A_slow_coeff[idx_px, idx_qx] = ion_slow_val_x
        A_fast_coeff[idx_px, idx_qx] = ion_fast_val_x
        
        idx_py, idx_qy = 6*i + 3, 6*i + 2
        A_slow_coeff[idx_py, idx_qy] = ion_slow_val_y
        A_fast_coeff[idx_py, idx_qy] = ion_fast_val_y

    # Fill Matrix (Nanoparticle)
    idx_px_p, idx_qx_p = 6*n_ions + 1, 6*n_ions + 0
    A_slow_coeff[idx_px_p, idx_qx_p] = p_slow_val_x
    A_fast_coeff[idx_px_p, idx_qx_p] = p_fast_val_x
    
    idx_py_p, idx_qy_p = 6*n_ions + 3, 6*n_ions + 2
    A_slow_coeff[idx_py_p, idx_qy_p] = p_slow_val_y
    A_fast_coeff[idx_py_p, idx_qy_p] = p_fast_val_y

    # --- 3. Deconstruct Linear Drive Vector g(t) ---
    K_zero_t  = calculate_linear_vector(0, coord_vec, params, charge_vec, mass_vec)
    K_pi_slow = calculate_linear_vector(pi/params.omega_slow, coord_vec, params, charge_vec, mass_vec)
    K_pi_fast = calculate_linear_vector(pi/params.omega_fast, coord_vec, params, charge_vec, mass_vec)
    
    K_const = 0.5 * (K_pi_slow + K_pi_fast)
    K_slow_coeff = K_zero_t - K_pi_fast
    K_fast_coeff = K_pi_slow - K_zero_t
    
    g_const, g_slow_coeff, g_fast_coeff = (np.zeros(n_dim) for _ in range(3))
    g_const[1::2] = -K_const
    g_slow_coeff[1::2] = -K_slow_coeff
    g_fast_coeff[1::2] = -K_fast_coeff

    # --- 4. Scaling (Normalization) ---
    _, _, _, _, _, q_zpf_vec_3d, p_zpf_vec_3d = params.get_particle_properties(n_ions)
    
    # A_const is [x1, p1, y1, p1...]
    scale_vec_interleaved = np.zeros(n_dim)
    # Position indices: 0, 2, 4...
    scale_vec_interleaved[0::2] = 1.0 / q_zpf_vec_3d
    # Momentum indices: 1, 3, 5...
    scale_vec_interleaved[1::2] = 1.0 / p_zpf_vec_3d
    
    S = np.diag(scale_vec_interleaved)
    S_inv = np.diag(1.0 / scale_vec_interleaved)
    
    return {
        "A_const": S @ A_const @ S_inv,
        "A_slow_coeff": S @ A_slow_coeff @ S_inv,
        "A_fast_coeff": S @ A_fast_coeff @ S_inv,
        "g_const": S @ g_const,
        "g_slow_coeff": S @ g_slow_coeff,
        "g_fast_coeff": S @ g_fast_coeff
    }

@jit(nopython=True, fastmath=True, cache=True)
def _dY_aug_dt_kernel(t, Y_flat, A_const, A_slow_coeff, A_fast_coeff, g_const, g_slow_coeff, g_fast_coeff, omega_slow, omega_fast):
    """JIT-compiled derivative for the Augmented State Matrix."""
    cos_slow = np.cos(omega_slow * t)
    cos_fast = np.cos(omega_fast * t)
    
    A_t = A_const + cos_slow * A_slow_coeff + cos_fast * A_fast_coeff
    g_t = g_const + cos_slow * g_slow_coeff + cos_fast * g_fast_coeff
    
    n_dim = A_t.shape[0]

    # We want to convert to a homogeneous system of equations.
    # Constructs an augmented drift matrix by appending the driving vector as the last column and a row of zeroes as the last row.
    # The corresponding vector is augmented by appending 1 at the end.
    A_aug_t = np.zeros((n_dim + 1, n_dim + 1))
    A_aug_t[:-1, :-1] = A_t
    A_aug_t[:-1, -1] = g_t
    
    Y_aug_mat = Y_flat.reshape(n_dim + 1, n_dim + 1)
    return (A_aug_t @ Y_aug_mat).flatten()

def dY_aug_dt_optimized(t, Y_flat, components, omega_slow, omega_fast):
    """Wrapper for the JIT kernel compatible with scipy.integrate."""
    return _dY_aug_dt_kernel(t, Y_flat,
                             components["A_const"], components["A_slow_coeff"], 
                             components["A_fast_coeff"], components["g_const"], 
                             components["g_slow_coeff"], components["g_fast_coeff"],
                             omega_slow, omega_fast)

def analyze_floquet_stability_jit(coord_vec, gamma_p, params, hessian_coulomb, charge_vec, mass_vec):
    """
    Computes the Monodromy Matrix to determine stability.
    
    Parameters
    ----------
    coord_vec : ndarray
        Equilibrium positions.
    gamma_p : float
        Nanoparticle damping.
    hessian_coulomb : ndarray
        Coulomb stiffness matrix.

    Returns
    -------
    ndarray
        The (N+1, N+1) Monodromy Matrix.
    """
    components = precompute_floquet_components_vectorized(coord_vec, gamma_p, params, charge_vec, mass_vec, hessian_coulomb)
    
    n_dim = components["A_const"].shape[0]
    y0_aug_flat = np.eye(n_dim + 1).flatten()
    
    fun = partial(dY_aug_dt_optimized,
                  components=components, 
                  omega_slow=params.omega_slow, 
                  omega_fast=params.omega_fast)
    
    solution = solve_ivp(fun=fun, 
                         t_span=[0, params.T_slow], 
                         y0=y0_aug_flat,
                         method="DOP853", 
                         rtol=1e-6, 
                         atol=1e-6)
    
    if solution.success:
        return solution.y[:, -1].reshape(n_dim + 1, n_dim + 1)
    
    return np.full((n_dim + 1, n_dim + 1), np.nan)





# ==============================================================================
# BLOCK 6: CORE ANALYSIS FUNCTIONS
# ==============================================================================
# This block orchestrates the high-level analysis pipeline:
# 1. Finding unique equilibria.
# 2. Filtering for geometric stability (Secular/Static).
# 3. Computing Floquet stability (Dynamical).
# 4. Calculating steady-state quantum metrics (Purity, Phonon Number).

def analyze_single_configuration_jit(args):
    """
    Worker function to perform the computationally expensive Floquet and 
    steady-state analysis on a single, pre-screened crystal configuration.

    Designed to be run in parallel via multiprocessing.

    Parameters
    ----------
    args : tuple
        (config_data, n_ions, gamma_p, params)
        - config_data: dict with 'coord_vec' and pre-computed 'sys_matrices'.
        - n_ions: Number of ions.
        - gamma_p: Nanoparticle damping.
        - params: SystemParameters instance.

    Returns
    -------
    dict
        The updated config_data dictionary populated with stability flags, 
        covariance matrices, and quantum metrics.
    """
    config_data, n_ions, gamma_p, params = args

    # Extract pre-computed data
    coord_vec = config_data['coord_vec']
    sys_matrices = config_data['sys_matrices']
    
    # --- 1. Floquet Stability Analysis ---
    # Integrate the equations of motion over one period to find the Monodromy Matrix
    monodromy_mat = analyze_floquet_stability_jit(
        coord_vec, 
        gamma_p, 
        params, 
        sys_matrices['hessian_coulomb'],
        sys_matrices['charge_vec'],
        sys_matrices['mass_vec']
    )
    config_data['monodromy_mat'] = monodromy_mat

    # Check Stability: All eigenvalues must have modulus <= 1
    try:
        # We strip the last row/col (Augmented 1) to check the physical sub-block
        eigenvalues = np.linalg.eigvals(monodromy_mat[:-1, :-1])
        max_magnitude = np.max(np.abs(eigenvalues))
        
        # Threshold slightly > 1.0 to account for numerical precision errors
        stability_threshold = 1.0001
        
        if max_magnitude > stability_threshold:
            config_data['is_floquet_stable'] = False
            return config_data # Early exit for unstable crystals
            
    except np.linalg.LinAlgError:
        # Catch cases where eigenvalue calculation fails
        config_data['is_floquet_stable'] = False
        return config_data

    config_data['is_floquet_stable'] = True
    
    # --- 2. Steady-State Quantum Analysis ---
    # Solve the Lyapunov equation for the Covariance Matrix V
    try:
        # Construct Drift (A) and Diffusion (Gamma) matrices
        A_cov_mat, Gamma_vec, A_mat = calculate_static_eom_matrices_jit(gamma_p, params, sys_matrices)
        
        # Solve A_cov * vec(V) = -Gamma_vec
        ss_cov_vec = np.linalg.solve(A_cov_mat, -Gamma_vec)
        ss_cov_mat = ss_cov_vec.reshape((A_mat.shape[0], A_mat.shape[0]), order='F')

        # --- 3. Un-scale the Covariance Matrix ---
        # Transform back from dimensionless units to physical units (position/momentum)
        _, _, _, _, _, q_zpf_vec_3d, p_zpf_vec_3d = params.get_particle_properties(n_ions)
        
        scale_vec = np.concatenate([1 / q_zpf_vec_3d, 1 / p_zpf_vec_3d])
        
        S = np.diag(scale_vec)
        S_inv = np.diag(1.0 / scale_vec) # Optimization: Invert diagonal directly
        
        # V_physical = S^-1 * V_dimensionless * S^-1
        ss_cov_mat = S_inv @ ss_cov_mat @ S_inv
        
        config_data['covariance_mat'] = ss_cov_mat
        
        # --- 4. Calculate Metrics ---
        purity_props = calculate_steady_state_purity(ss_cov_mat, n_ions, params)
        for axis in ['x', 'y', 'z']:
            config_data[f'{axis}_phonon_number'] = purity_props[axis]['phonon_number']
            config_data[f'{axis}_purity'] = purity_props[axis]['purity']
            
    except np.linalg.LinAlgError:
        # Handle rare solver failures (singular matrices)
        config_data['covariance_mat'] = None
        config_data['is_floquet_stable'] = False 
        
    # --- 5. Normal Mode Analysis ---
    config_data['np_normal_modes'] = calculate_nanoparticle_modes_jit(sys_matrices)
    
    return config_data


def analyze_ion_count_jit(args):
    """
    Orchestrator for analyzing a specific number of ions (N).

    1. Finds all unique equilibrium geometries.
    2. Filters them for static stability (Secular approximation).
    3. Dispatches stable candidates to parallel workers for full Floquet analysis.

    Parameters
    ----------
    args : tuple
        (n_ions, gamma_p, n_runs, params, verbose)

    Returns
    -------
    tuple
        (n_ions, results_dict)
        results_dict contains configuration data keyed by geometric signature,
        plus a summary statistics entry.
    """
    n_ions, gamma_p, n_runs, params, verbose = args
    if verbose: print(f"\n===== Starting job for N_Ions = {n_ions} =====")
    
    # === STEP 1: Find all unique equilibrium configurations ===
    configs = find_unique_equilibrium_configs(n_ions, n_runs, params, verbose=verbose)
    
    if not configs:
        return n_ions, {'_summary_statistics': {'n_ions': n_ions, 'total_configs': 0}}
    
    # === STEP 2: Pre-filter for Static Stability (Serial) ===
    job_tickets = [] 
    if verbose: print(f"--> Pre-screening {len(configs)} unique configs for static stability...")

    for config_flat in configs:
        coord_vec = config_flat.reshape(-1, 3)
        
        # Compute Hessians and Matrices
        sys_matrices = compute_system_matrices_jit(coord_vec, params)
        
        M_mat = np.diag(sys_matrices['mass_vec_3d'])
        H_mat = sys_matrices['hessian_total']

        # Solve generalized eigenvalue problem: H v = w^2 M v
        # We only need eigenvalues to check for negative stiffness (instability)
        eigenvalues_w_squared = eigh(H_mat, M_mat, eigvals_only=True)
        
        # Criterion: All squared frequencies must be positive (Real frequencies)
        if not np.any(eigenvalues_w_squared <= 0):
            initial_config_data = {
                'coord_vec': coord_vec,
                'sys_matrices': sys_matrices,
                'static_eigenvalues': eigenvalues_w_squared,
                'is_static_stable': True,
                'is_floquet_stable': False # Default until proven otherwise
            }
            job_tickets.append(initial_config_data)

    if not job_tickets:
        if verbose: print("--> No statically stable configurations found.")
        return n_ions, {'_summary_statistics': {'n_ions': n_ions, 'total_configs': len(configs), 'static_stable': 0}}
    
    if verbose: 
        print(f"--> Found {len(job_tickets)} promising candidates. Starting full parallel analysis...")

    # === STEP 3: Run full analysis on stable candidates (Parallel) ===
    # Prepare arguments for the worker function
    job_args = [(ticket, n_ions, gamma_p, params) for ticket in job_tickets]
    
    with mp.Pool() as pool:
        analysis_results = pool.map(analyze_single_configuration_jit, job_args)

    # === STEP 4: Aggregate Results ===
    # Store results in a dictionary keyed by the canonical signature of the geometry
    results_dict = {get_canonical_signature(res['coord_vec'].flatten(), n_ions): res for res in analysis_results}
    
    # Compute Summary Statistics
    total = len(configs)
    static = len(job_tickets)
    floquet_stable_configs = [c for c in results_dict.values() if c['is_floquet_stable']]
    floquet = len(floquet_stable_configs)
    
    summary_stats = {
        'n_ions': n_ions, 
        'total_configs': total, 
        'static_stable': static, 
        'floquet_stable': floquet
    }
    
    # Calculate statistics for stable configurations
    if floquet_stable_configs:
        for axis in ['x', 'y', 'z']:
            # Filter out NaNs to ensure robust statistics
            phonons = [c[f'{axis}_phonon_number'] for c in floquet_stable_configs if not np.isnan(c[f'{axis}_phonon_number'])]
            purities = [c[f'{axis}_purity'] for c in floquet_stable_configs if not np.isnan(c[f'{axis}_purity'])]
            
            summary_stats[f'mean_{axis}_phonon'] = np.mean(phonons) if phonons else np.nan
            summary_stats[f'std_{axis}_phonon'] = np.std(phonons) if phonons else np.nan
            summary_stats[f'best_{axis}_phonon'] = np.min(phonons) if phonons else np.nan
            
            summary_stats[f'mean_{axis}_purity'] = np.mean(purities) if purities else np.nan
            summary_stats[f'std_{axis}_purity'] = np.std(purities) if purities else np.nan
    
    results_dict['_summary_statistics'] = summary_stats
    
    if verbose:
        print(f"--> Analysis for N_ions = {n_ions} complete. Found {floquet} fully stable config(s).")
        
    return n_ions, results_dict

def run_ion_scan_jit(n_ions, gamma_p, n_runs=None, params=None, verbose=True):
    """
    Top-level function to scan over a range of ion numbers (N).

    Executes the analysis serially for each N to avoid nested multiprocessing issues,
    while utilizing parallelism within the analysis of each N.

    Parameters
    ----------
    n_ions : int or list of int
        If int, scans from 1 to n_ions. If list, scans specific counts provided.
    gamma_p : float
        Nanoparticle damping rate [rad/s].
    n_runs : int or list of int, optional
        Number of Monte Carlo attempts for finding equilibria. 
        If None, scales automatically with factorial(N).
    params : SystemParameters, optional
        System constants.
    verbose : bool, optional
        Print progress to console.

    Returns
    -------
    dict
        Nested dictionary of results, keyed by ion count N.
    """
    # Normalize inputs
    if isinstance(n_ions, int):
        ion_list = list(range(1, n_ions + 1))
    else:
        ion_list = list(n_ions)

    if n_runs is None:
        # Heuristic: Factorial scaling for configuration search space
        run_list = [5000 + int(factorial(n)) for n in ion_list]
    else:
        run_list = list(n_runs) if isinstance(n_runs, list) else [n_runs] * len(ion_list)

    if len(ion_list) != len(run_list):
        raise ValueError("n_ions and n_runs lists must have the same length.")

    # Prepare job arguments
    job_args = [(n, gamma_p, r, params, verbose) for n, r in zip(ion_list, run_list)]
    
    print(f"===== Starting JIT-Optimized Ion Scan SERIALLY for N = {ion_list} =====")
    print("      (Parallelism will be used within each N-ion analysis)")
    
    # Execute serially
    results_list = [analyze_ion_count_jit(args) for args in job_args]
        
    return dict(sorted(results_list))


# ==============================================================================
# BLOCK 7: DATA VISUALIZATION AND REPORTING FUNCTIONS
# ==============================================================================

def print_config_details(results_dict, n_ions=None, signature=None):
    """
    Prints detailed information about specific configurations in the results dictionary.

    Parameters
    ----------
    results_dict : dict
        The nested results dictionary returned by run_ion_scan_jit.
    n_ions : int, optional
        Filter to show only configurations for this number of ions.
    signature : tuple, optional
        Filter to show only the configuration with this specific geometric signature.
    """
    ion_configs = {n_ions: results_dict[n_ions]} if n_ions is not None else results_dict
    
    for n_ions_key, configs in ion_configs.items():
        print(f"\n{'='*70}")
        print(f"N_IONS = {n_ions_key}")
        print('='*70)
        
        configs_to_show = {signature: configs[signature]} if signature else configs
        
        for sig, data in configs_to_show.items():
            # Skip summary statistics entries
            if sig == '_summary_statistics':
                continue

            print(f"\n  Configuration: {sig}")
            print(f"  {'-'*66}")
            print(f"    Static Stable:  {'✓' if data['is_static_stable'] else '✗'}")
            print(f"    Floquet Stable: {'✓' if data['is_floquet_stable'] else '✗'}")
            
            if data['is_floquet_stable']:
                print(f"\n    Phonon Numbers:")
                for axis in ['x', 'y', 'z']:
                    val = data.get(f'{axis}_phonon_number', np.nan)
                    print(f"      {axis.upper()}: {val:.6e}")
                
                print(f"\n    Purities:")
                for axis in ['x', 'y', 'z']:
                    val = data.get(f'{axis}_purity', np.nan)
                    print(f"      {axis.upper()}: {val:.6f}")
            
            print(f"\n    Available Data Objects:")
            if 'coord_vec' in data:
                print(f"      coord_vec: shape {data['coord_vec'].shape}")
            if 'monodromy_mat' in data:
                print(f"      monodromy_mat: shape {data['monodromy_mat'].shape}")
            if 'covariance_mat' in data and data['covariance_mat'] is not None:
                print(f"      covariance_mat: shape {data['covariance_mat'].shape}")


def get_best_configs(results_dict, axis='z', top_n=5):
    """
    Extracts the top N stable configurations with the lowest phonon numbers.

    Parameters
    ----------
    results_dict : dict
        Nested results dictionary.
    axis : str, optional
        Axis to optimize for ('x', 'y', or 'z'). Default is 'z'.
    top_n : int, optional
        Number of top results to return.

    Returns
    -------
    list of tuple
        [(n_ions, signature, phonon_number, config_data), ...]
    """
    all_stable = []
    
    for n_ions, configs in results_dict.items():
        for sig, data in configs.items():
            if sig == "_summary_statistics":
                continue
            
            if data.get('is_floquet_stable'):
                phonon_num = data.get(f'{axis}_phonon_number', np.nan)
                if not np.isnan(phonon_num):
                    all_stable.append((n_ions, sig, phonon_num, data))
    
    # Sort by phonon number (ascending) and take top N
    return sorted(all_stable, key=lambda x: x[2])[:top_n]


def print_summary_table(results_dict):
    """
    Prints a formatted summary table of statistics for all ion counts scanned.
    
    Displays total configurations found, stability counts, and phonon number statistics.
    """
    print("\n" + "="*160)
    print("---- ION SCAN SUMMARY ----".center(160))
    
    header = (f"{'N_Ions':<8} {'Total':<10} {'Static':<10} {'Floquet':<10} "
              f"{'Best X Ph':<14} {'Mean X Ph':<14} {'Std Dev X Ph':<15} "
              f"{'Best Y Ph':<14} {'Mean Y Ph':<14} {'Std Dev Y Ph':<15} "
              f"{'Best Z Ph':<14} {'Mean Z Ph':<14} {'Std Dev Z Ph':<15}")
    print(header)
    print("-" * 200)
    
    for n_ions in sorted(results_dict.keys()):
        stats = results_dict[n_ions].get('_summary_statistics', {})
        
        # Skip if no data or empty stats
        if not stats or stats.get('total_configs', 0) == 0:
            continue
            
        row = (f"{stats['n_ions']:<8} {stats['total_configs']:<10} "
               f"{stats['static_stable']:<10} {stats['floquet_stable']:<10} "
               f"{stats.get('best_x_phonon', np.nan):<14.4e} {stats.get('mean_x_phonon', np.nan):<14.4e} {stats.get('std_x_phonon', np.nan):<15.4e} "
               f"{stats.get('best_y_phonon', np.nan):<14.4e} {stats.get('mean_y_phonon', np.nan):<14.4e} {stats.get('std_y_phonon', np.nan):<15.4e} "
               f"{stats.get('best_z_phonon', np.nan):<14.4e} {stats.get('mean_z_phonon', np.nan):<14.4e} {stats.get('std_z_phonon', np.nan):<15.4e}")
        print(row)
    
    print("="*200)


def plot_all_configurations(results_dict, max_n_ions=None, save_plots=True):
    """
    Generates 3D scatter plots of all Floquet-stable crystal configurations.

    Parameters
    ----------
    results_dict : dict
        Results dictionary.
    max_n_ions : int, optional
        Limit plotting to crystals with <= max_n_ions.
    save_plots : bool, optional
        If True, saves figures to disk.
    """
    if not results_dict: return

    ion_counts_to_plot = sorted(results_dict.keys())
    if max_n_ions is not None:
        ion_counts_to_plot = [n for n in ion_counts_to_plot if n <= max_n_ions]

    for n_ions in ion_counts_to_plot:
        # Filter for stable configs
        configs_for_n = {s: d for s, d in results_dict[n_ions].items() 
                         if s != '_summary_statistics' and d.get('is_floquet_stable')}
        
        if not configs_for_n: 
            print(f"No stable configurations to plot for N={n_ions}")
            continue

        num_stable = len(configs_for_n)
        # Dynamic figure sizing based on number of subplots
        fig, axes = plt.subplots(1, num_stable, figsize=(num_stable * 5, 5), 
                                 subplot_kw={'projection': '3d'})
        
        # Ensure axes is iterable even if only 1 plot
        if num_stable == 1: 
            axes = [axes]
            
        fig.suptitle(f'Floquet Stable Configurations for {n_ions} Ion(s)', fontsize=16)

        for idx, data in enumerate(configs_for_n.values()):
            ax = axes[idx]
            # Reshape coordinate vector to (N+1, 3) and convert to micrometers
            coord_vec = data['coord_vec'].reshape(-1, 3) * 1e6 
            
            ion_coords = coord_vec[:n_ions]
            p_coords = coord_vec[-1]
            
            ax.scatter(ion_coords[:, 0], ion_coords[:, 1], ion_coords[:, 2], 
                       color='blue', s=50, label='Ions')
            ax.scatter(p_coords[0], p_coords[1], p_coords[2], 
                       color='red', s=100, label='Nanoparticle')
            
            ax.set_title(f'Config #{idx+1}')
            ax.set_xlabel('X (μm)')
            ax.set_ylabel('Y (μm)')
            ax.set_zlabel('Z (μm)')
            
            # Add legend only to the first subplot to reduce clutter
            if idx == 0: 
                ax.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_plots:
            filename = f'crystal_configs_N{n_ions}.png'
            plt.savefig(filename, dpi=300)
            print(f"Saved configuration plot: {filename}")
        
        # plt.show()


def plot_phonon_statistics(results_dict, axis='z', save_plots=True):
    """
    Plots the scaling of Phonon Number (Mean and Best) vs. Crystal Size.

    Parameters
    ----------
    results_dict : dict
        Results dictionary.
    axis : str, optional
        Axis to plot ('x', 'y', or 'z').
    save_plots : bool, optional
        If True, saves the figure.
    """
    # Extract valid statistics
    stats_list = []
    for v in results_dict.values():
        summary = v.get('_summary_statistics', {})
        if summary.get('floquet_stable', 0) > 0:
            stats_list.append(summary)
            
    if not stats_list:
        print("No stable configurations with phonon data found. Cannot plot statistics.")
        return

    df = pd.DataFrame(stats_list).sort_values('n_ions')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Mean with Error Bars (Standard Deviation)
    ax.errorbar(df['n_ions'], df[f'mean_{axis}_phonon'], yerr=df[f'std_{axis}_phonon'],
                fmt='-o', capsize=5, color='black', label=f'Mean {axis.upper()} Phonon #')
    
    # Plot Best (Minimum) Phonon Number
    ax.plot(df['n_ions'], df[f'best_{axis}_phonon'],
            linestyle='--', marker='s', color='red', label=f'Best {axis.upper()} Phonon #')
    
    ax.set_yscale('log')
    ax.set_title(f'Nanoparticle {axis.upper()}-Mode Cooling vs. Ion Crystal Size', fontsize=16)
    ax.set_xlabel('Number of Ions in Crystal')
    ax.set_ylabel('Phonon Number')
    
    # Ensure x-axis only shows integer ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.7)
    
    plt.tight_layout()
    
    if save_plots:
        filename = f'phonon_statistics_axis_{axis}.png'
        plt.savefig(filename, dpi=300)
        print(f"Saved statistics plot: {filename}")
        
    # plt.show()


# ==============================================================================
# BLOCK 8: MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    # Ensure multiprocessing works correctly on Windows/macOS (spawn method)
    # Note: 'multiprocess' library usually handles this, but good practice to be safe.
    mp.freeze_support()

    print("--- Initializing Crystal Simulation ---")
    start_time = time.time()
    
    # 1. Initialize System Parameters
    # Create instance of the parameters class defined in Block 1
    system_params = SystemParameters()
    
    # 2. Define Scan Parameters
    # Example: Scan a single case of N=13 ions
    # For a full sweep, change to n_ions=[1, 2, 3, ...]
    target_ions = 6 
    runs_per_ion = 10000  # Reduced for demo; use higher values (e.g. 1M) for production
    damping_rate = 2 * pi * 44.55e-9
    
    # 3. Run the Simulation
    results_dict = run_ion_scan_jit(
        n_ions=target_ions, 
        n_runs=runs_per_ion,
        gamma_p=damping_rate, 
        params=system_params,
        verbose=True
    )

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")

    # 4. Reporting
    print_summary_table(results_dict)
    
    # 5. Visualization (Saved to disk)
    print("\n--- Generating Plots ---")
    plot_all_configurations(results_dict)
    plot_phonon_statistics(results_dict, axis='z')
    
    print("\n--- Done. ---")
    # plt.show() # Uncomment to display plots interactively at the end