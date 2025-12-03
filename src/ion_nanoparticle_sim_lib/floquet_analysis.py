import numpy as np
from numpy.linalg import matrix_power, inv
from scipy.integrate import solve_ivp

import time

def coupled_dZ_dt(t, Z_flat, dim, drift_func, drive_vec):
    """
    Computes the time derivative of the combined state vector Z(t) = [X(t), Lambda(t)].

    Solves the coupled matrix differential equations for the first and second moments:
    1. Homogeneous evolution (Monodromy matrix): dX/dt = drift_mat(t) * X(t)
    2. Inhomogeneous evolution: dLambda/dt = X(t)^(-1) * drive_vec
    
    Parameters
    ----------
    t : float
        Current time [s].
    Z_flat : ndarray
        Flattened state vector containing components of X (dim*dim) and Lambda (dim).
    dim : int
        Dimension of the phase space (4 for first moments, 10 for second moments).
    drift_func : callable
        Function returning the drift matrix of shape (dim, dim).
    drive_vec : ndarray
        The diffusion/drive vector of shape (dim,).

    Returns
    -------
    ndarray
        Flattened derivative vector [dX/dt, dLambda/dt].
    """
    # Reconstruct Monodromy matrix X (dim, dim) and Lambda (dim)
    X_mat = Z_flat[:dim*dim].reshape(dim, dim) 
    Lambda_vec = Z_flat[dim*dim:] 

    # Compute derivatives
    # drift_mat(t) @ X
    dX_dt = drift_func(t) @ X_mat
    # X^(-1) @ drive_vec
    dLambda_dt = inv(X_mat) @ drive_vec

    return np.concatenate([dX_dt.flatten(), dLambda_dt])


def floquet_solver(dim, gamma_p, t_eval_points, T_per, drift_mat_factory, drive_vec_factory):

    # Initial conditions for the Floquet simulation
    # Initial Monodromy matrix X(0) = Identity matrix
    # Initial Lambda(0) = Zero vector
    X_init_mat = np.eye(dim)
    Lambda_init_vec = np.zeros(dim)
    Z_init_flat = np.concatenate([X_init_mat.flatten(), Lambda_init_vec])

    # First Moments (Dimension 4)
    solution = solve_ivp(
        fun=lambda t, z: coupled_dZ_dt(t, z, dim, drift_mat_factory, drive_vec_factory),
        t_span=[0, T_per],
        y0=Z_init_flat,
        method="DOP853", 
        rtol=1e-12, atol=1e-12,
        t_eval=t_eval_points
    )

    return solution.y.T


def run_floquet_sweep(gamma_p_vec, t_eval_points, T_per, drift1_mat_stack, drive1_vec_stack, drift2_mat_stack, drive2_vec_stack):
    n_gamma = len(gamma_p_vec)

    all_X1_histories = []
    all_Lambda1_histories = []

    all_X2_histories = []
    all_Lambda2_histories = []

    loop_start_time = time.time()
    print(f"Starting parameter sweep over {n_gamma} gamma_p values...")

    for i, gamma_p in enumerate(gamma_p_vec):
        # Extract system-specific parameters for the i-th damping rate
        drift1_mat_single = lambda t: drift1_mat_stack(t)[i]
        drive1_vec_single = drive1_vec_stack[i]

        drift2_mat_single = lambda t: drift2_mat_stack(t)[i]
        drive2_vec_single = drive2_vec_stack[i]

        start_time = time.time()
        print(f"  Processing gamma_p index {i+1}/{n_gamma} ({gamma_p:.2e})...", end='')
        
        Z1_flat_history = floquet_solver(4, gamma_p, t_eval_points, T_per, drift1_mat_single, drive1_vec_single)
        Z2_flat_history = floquet_solver(10, gamma_p, t_eval_points, T_per, drift2_mat_single, drive2_vec_single)

        all_X1_histories.append(Z1_flat_history[:, :16].reshape(len(t_eval_points), 4, 4))
        all_Lambda1_histories.append(Z1_flat_history[:, 16:])
    
        all_X2_histories.append(Z2_flat_history[:, :100].reshape(len(t_eval_points), 10, 10))
        all_Lambda2_histories.append(Z2_flat_history[:, 100:])

        end_time = time.time()
        print(f" Done ({end_time - start_time:.2f}s)")

    loop_end_time = time.time()
    print(f"Sweep complete in {loop_end_time - loop_start_time:.4f} seconds.")

    # Data Aggregation
    X1_mat_history = np.stack(all_X1_histories, axis=1)
    Lambda1_vec_history = np.stack(all_Lambda1_histories, axis=1)
    X2_mat_history = np.stack(all_X2_histories, axis=1)
    Lambda2_vec_history = np.stack(all_Lambda2_histories, axis=1)

    return X1_mat_history, Lambda1_vec_history, X2_mat_history, Lambda2_vec_history


def inhom_steady_state(dim, X_mat_history, Lambda_vec_history):
    # Extract Monodromy matrices X(T) and inhomogeneous vectors Lambda(T) at the end of one period
    # Index -1 corresponds to t = T_slow
    X_mat_T = X_mat_history[-1, :, :, :]
    Lambda_vec_T = Lambda_vec_history[-1, :, :]
    I = np.eye(dim)

    n_gamma = X_mat_history.shape[1]
    
    # Calculate the steady-state vector Y_ss at t = 0 (start of period after t = m * T_slow where m-->infinity)
    # Y_ss(0) = (I - X(T))^-1 * X(T) * Lambda(T)
    # We add a new axis to the Lambda_vec_T to convert it to a (n_gamma, dim, 1) array. The matrix multiplication results in a (n_gamma, dim, 1) array which is then reshaped. 
    # Result shape: (n_gamma, dim)
    Y_vec_ss = (inv(I - X_mat_T) @ X_mat_T @ Lambda_vec_T[:, :, np.newaxis]).reshape(n_gamma, dim)
    # Propagate the steady-state solution over one full time period [0, T_per]
    # Formula: Y_ss(t) = X(t) * [Y_ss(0) + Lambda(t)]
    # Numpy broadcasting adds Y_vec_ss (n_gamma, dim) to Lambda history (n_steps, n_gamma, dim) 
    sum_of_vectors = Y_vec_ss + Lambda_vec_history
    # We add a new axis to sum_of_vectors to convert it to a (n_steps, n_gamma, dim, 1) array and then remove the last axis after the matrix multiplication. 
    # Result shape: (n_steps, n_gamma, dim)
    Y_vec_ss_history = (X_mat_history @ sum_of_vectors[..., np.newaxis]).squeeze(axis=-1)

    return Y_vec_ss, Y_vec_ss_history


def inhom_trajectory(m, n, dim, X0_vec, X_mat_history, Lambda_vec_history, Y_vec_ss):
    """
    Generic vectorized calculation for any dimension (First or Second moments).
    
    Parameters
    ----------
    m : int
        Number of full periods elapsed.
    n : int
        Time step index within the current period.
    dim : int
        Dimension of the system (4 or 10).
    X0_vec : ndarray
        Initial condition vector (dim,).
    X_mat_history : ndarray
        History of Monodromy matrices (n_steps, n_gamma, dim, dim).
    Lambda_vec_history : ndarray
        History of Inhomogeneous vectors (n_steps, n_gamma, dim).
    Y_vec_ss : ndarray
        Steady state vector at t=0 (n_gamma, dim).
        
    Returns
    -------
    ndarray
        Batch of state vectors of shape (n_gamma, dim).
    """
    X_mat_T = X_mat_history[-1, :, :, :] # Monodromy matrix at T_slow
    monodromy_pow_m = matrix_power(X_mat_T, m)

    X_mat_n = X_mat_history[n, :, :, :]
    Lambda_vec_n = Lambda_vec_history[n, :, :]
    I = np.eye(dim)
    
    # Homogeneous Part
    # Broadcasts single initial condition X0_vec across the batch
    Y_hom_batch = (X_mat_n @ monodromy_pow_m) @ X0_vec
    
    # Inhomogeneous Part
    # 1. Term: (I - M^m) * Y_ss(0)
    # Reshape Y1_vec_ss to (n_gamma, dim, 1) for matrix multiplication
    term1 = (I - monodromy_pow_m) @ Y_vec_ss[:, :, np.newaxis]
    
    # 2. Add Lambda(t). Squeeze term1 to (n_gamma, dim) before addition.
    term2 = term1.squeeze(axis=-1) + Lambda_vec_n
    
    # 3. Multiply by X(t)
    Y_inhom_batch = (X_mat_n @ term2[:, :, np.newaxis]).squeeze(axis=-1)
    
    return Y_hom_batch + Y_inhom_batch