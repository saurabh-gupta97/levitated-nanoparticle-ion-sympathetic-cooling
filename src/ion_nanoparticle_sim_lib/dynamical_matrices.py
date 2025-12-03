import numpy as np
from .parameters import BaseSystemParams
from .potentials import W_i, W_p

def get_scaling_factors(params: BaseSystemParams, direction: str):
    """
    Computes scaling factors for the system based on the direction.
    
    Parameters
    ----------
    params : BaseSystemParams
        System parameters dataclass.
    direction : str
        'x' or 'y'.
        
    Returns
    -------
    tuple
        (scale1_mat, scale1_vec, scale2_mat, scale2_vec, bi, ci, bp, cp)
    """
    if direction == 'x' or direction == 0:
        r_zpf_i, pr_zpf_i = params.x_zpf_i, params.px_zpf_i
        r_zpf_p, pr_zpf_p = params.x_zpf_p, params.px_zpf_p
    elif direction == 'y' or direction == 1:
        r_zpf_i, pr_zpf_i = params.y_zpf_i, params.py_zpf_i
        r_zpf_p, pr_zpf_p = params.y_zpf_p, params.py_zpf_p
    else:
        raise ValueError("Direction must be 'x' or 'y'")

    # Scaling constants
    bi = 1.0 / r_zpf_i
    bp = 1.0 / r_zpf_p
    ci = 1.0 / pr_zpf_i
    cp = 1.0 / pr_zpf_p

    # --- First Moments (Dim 4) ---
    scale1_vec = np.array([bi, ci, bp, cp])
    scale1_mat = np.outer(scale1_vec, 1/scale1_vec)

    # --- Second Moments (Dim 10) ---
    scale2_vec = np.array([
        bi*bi, ci*ci, bi*ci,       # Ion
        bp*bp, cp*cp, bp*cp,       # Particle
        bi*bp, ci*cp, bi*cp, bp*ci # Cross terms
    ])
    scale2_mat = np.outer(scale2_vec, 1/scale2_vec)

    return scale1_mat, scale1_vec, scale2_mat, scale2_vec, bi, ci, bp, cp


def get_drift_matrix_1(t, params: BaseSystemParams, gamma_p_vec, direction: str):
    """
    Constructs the 1st-order drift matrix for a batch of gamma values.
    """
    # 1. Get Scaling
    scale1_mat, _, _, _, _, _, _, _ = get_scaling_factors(params, direction)
    n_gamma = len(gamma_p_vec)

    # 2. Select Direction-Dependent Physics
    if direction == 'x' or direction == 0:
        K_r = params.K_x
    elif direction == 'y' or direction == 1:
        K_r = params.K_y
    else:
        raise ValueError(f"Invalid direction: {direction}")
    W_r_i = W_i(t, params, direction)
    W_r_p = W_p(t, params, direction)
    # 3. Construct Physical Matrix (4x4)
    # Drift = [[-gamma/2, 1/M, 0, 0], [-M(W-K/M), -gamma/2, -K, 0], ...]
    
    drift_phys = np.zeros((n_gamma, 4, 4))
    
    # Ion block (fixed for all gammas)
    drift_phys[:, 0, 0] = -params.gamma_dop / 2
    drift_phys[:, 0, 1] = 1 / params.M_i
    drift_phys[:, 1, 0] = -params.M_i * (W_r_i - K_r / params.M_i)
    drift_phys[:, 1, 1] = -params.gamma_dop / 2
    drift_phys[:, 1, 2] = -K_r

    # Particle block (gamma dependent)
    drift_phys[:, 2, 3] = 1 / params.M_p
    drift_phys[:, 3, 0] = -K_r
    drift_phys[:, 3, 2] = -params.M_p * (W_r_p - K_r / params.M_p)
    
    # Apply Gamma Vector
    # Broadcast gamma_p_vec (shape: n_gamma) into the matrix diagonals
    drift_phys[:, 2, 2] = -gamma_p_vec / 2.0
    drift_phys[:, 3, 3] = -gamma_p_vec / 2.0

    # 4. Apply Scaling
    return scale1_mat * drift_phys


def get_drift_matrix_2(t, params: BaseSystemParams, gamma_p_vec, direction: str):
    """
    Constructs the 2nd-order drift matrix (Covariance) for a batch of gamma values.
    """
    _, _, scale2_mat, _, _, _, _, _ = get_scaling_factors(params, direction)
    n_gamma = len(gamma_p_vec)
    
    if direction == 'x' or direction == 0:
        K_r = params.K_x 
    elif direction == 'y' or direction == 1:
        K_r = params.K_y
    else:
        raise ValueError(f"Invalid direction: {direction}")
    W_r_i = W_i(t, params, direction)
    W_r_p = W_p(t, params, direction)
    # Base Matrix (n_gamma, 10, 10)
    drift_phys = np.zeros((n_gamma, 10, 10))

    # --- Ion Block (Rows 0-2) ---
    drift_phys[:, 0, 0] = -params.gamma_dop
    drift_phys[:, 0, 2] = 2 / params.M_i
    
    drift_phys[:, 1, 1] = -params.gamma_dop
    drift_phys[:, 1, 2] = -2 * params.M_i * (W_r_i - K_r / params.M_i)
    drift_phys[:, 1, 9] = -2 * K_r  # coupling term

    drift_phys[:, 2, 0] = -params.M_i * (W_r_i - K_r / params.M_i)
    drift_phys[:, 2, 1] = 1 / params.M_i
    drift_phys[:, 2, 2] = -params.gamma_dop
    drift_phys[:, 2, 6] = -K_r      # coupling term

    # --- Particle Block (Rows 3-5) ---
    # Gamma dependent terms added later
    drift_phys[:, 3, 5] = 2 / params.M_p
    
    drift_phys[:, 4, 5] = -2 * params.M_p * (W_r_p - K_r / params.M_p)
    drift_phys[:, 4, 8] = -2 * K_r

    drift_phys[:, 5, 3] = -params.M_p * (W_r_p - K_r / params.M_p)
    drift_phys[:, 5, 4] = 1 / params.M_p
    drift_phys[:, 5, 6] = -K_r

    # --- Cross Terms (Rows 6-9) ---
    drift_phys[:, 6, 8] = 1 / params.M_p
    drift_phys[:, 6, 9] = 1 / params.M_i
    
    drift_phys[:, 7, 2] = -K_r
    drift_phys[:, 7, 5] = -K_r
    drift_phys[:, 7, 8] = -params.M_i * (W_r_i - K_r / params.M_i)
    drift_phys[:, 7, 9] = -params.M_p * (W_r_p - K_r / params.M_p)

    drift_phys[:, 8, 0] = -K_r
    drift_phys[:, 8, 6] = -params.M_p * (W_r_p - K_r / params.M_p)
    drift_phys[:, 8, 7] = 1 / params.M_i
    
    drift_phys[:, 9, 3] = -K_r
    drift_phys[:, 9, 6] = -params.M_i * (W_r_i - K_r / params.M_i)
    drift_phys[:, 9, 7] = 1 / params.M_p

    # --- Apply Gamma (Damping) ---
    # Particle diagonal damping
    drift_phys[:, 3, 3] = -gamma_p_vec
    drift_phys[:, 4, 4] = -gamma_p_vec
    drift_phys[:, 5, 5] = -gamma_p_vec
    
    # Cross-term damping (average of ion and particle)
    avg_gamma = -(params.gamma_dop + gamma_p_vec) / 2.0
    drift_phys[:, 6, 6] = avg_gamma
    drift_phys[:, 7, 7] = avg_gamma
    drift_phys[:, 8, 8] = avg_gamma
    drift_phys[:, 9, 9] = avg_gamma

    return scale2_mat * drift_phys



def get_drive_vectors(params: BaseSystemParams, gamma_p_vec, Gamma_ba_vec, direction):
    """
    Constructs the 1st and 2nd order drive vectors.
    Converts Energy Rates (E_dot) into Scattering Rates (Gamma) to ensure correct units.
    """
    # 1. Scaling
    _, scale1_vec, _, scale2_vec, _, _, _, _ = get_scaling_factors(params, direction)
    n_gamma = len(gamma_p_vec)
    
    # 2. Select ZPF and Frequency based on direction
    # We need the secular frequency (Omega) to convert E_dot -> Gamma
    # Gamma_heating = E_dot / (hbar * Omega)
    if direction == 'x' or direction == 0:
        r_zpf_i, pr_zpf_i = params.x_zpf_i, params.px_zpf_i
        r_zpf_p, pr_zpf_p = params.x_zpf_p, params.px_zpf_p
        
        Gamma_gas = params.Gamma_gas_x
        Gamma_td = params.Gamma_td_x
        Gamma_dop = params.Gamma_dop_x
        
    elif direction == 'y' or direction == 1:
        r_zpf_i, pr_zpf_i = params.y_zpf_i, params.py_zpf_i
        r_zpf_p, pr_zpf_p = params.y_zpf_p, params.py_zpf_p
        
        Gamma_gas = params.Gamma_gas_y
        Gamma_td = params.Gamma_td_y
        Gamma_dop = params.Gamma_dop_y
    else:
        raise ValueError(f"Invalid direction: {direction}")

    # --- First Order (Dim 4) ---
    drive1 = np.zeros((n_gamma, 4))
    drive1_scaled = scale1_vec * drive1

    # --- Second Order (Dim 10) ---
    # Ion Noise (Constant)
    # <q^2> term: r_zpf^2 * (2*Gamma_h + gamma)
    term_i_pos = r_zpf_i**2 * (2 * Gamma_dop + params.gamma_dop)
    # <p^2> term
    term_i_mom = pr_zpf_i**2 * (2 * Gamma_dop + params.gamma_dop)

    # Particle Noise (Gamma Dependent)
    # <q^2> term
    term_p_pos = r_zpf_p**2 * (2 * Gamma_gas + gamma_p_vec) # E_dot_gas needs to be in params!
    
    # <p^2> term (includes Feedback and Trap Displacement)
    # Note: Gamma_td and Gamma_fb need to be handled. 
    # Assuming params.E_dot_td is defined. 
    # Gamma_ba_vec is passed as argument since it changes with gamma_p
    term_p_mom = pr_zpf_p**2 * (2 * Gamma_gas + gamma_p_vec + 4 * Gamma_td + 4 * Gamma_ba_vec)

    drive2 = np.zeros((n_gamma, 10))
    drive2[:, 0] = term_i_pos
    drive2[:, 1] = term_i_mom
    drive2[:, 3] = term_p_pos
    drive2[:, 4] = term_p_mom

    drive2_scaled = scale2_vec * drive2

    return drive1_scaled, drive2_scaled