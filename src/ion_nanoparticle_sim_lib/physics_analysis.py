import numpy as np
from .parameters import BaseSystemParams

def calculate_purity(y1_vec, y2_vec, idx_gamma, params: BaseSystemParams, direction = 'x'):
    """
    Calculates the purity of the nanoparticle COM motion state from the covariance matrix.
    
    Purity = 1 / sqrt(det(V_dimensionless))
    """
    # 1. Select Direction-Dependent ZPFs
    if direction == 'x' or direction == 0:
        r_zpf_p = params.x_zpf_p
        pr_zpf_p = params.px_zpf_p
    elif direction == 'y' or direction == 1:
        r_zpf_p = params.y_zpf_p
        pr_zpf_p = params.py_zpf_p
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'x' or 'y'.")

    # 2. Recover Scaling Factors from params
    bp = 1.0 / r_zpf_p 
    cp = 1.0 / pr_zpf_p 

    # 3. Extract moments (indices 2,3 for first moments, 3,4,5 for second moments of NP)
    # First moments (Means) <q> and <p>
    mean_q_p = y1_vec[idx_gamma, 2] / (bp * r_zpf_p)
    mean_p_p = y1_vec[idx_gamma, 3] / (cp * pr_zpf_p)
    
    # Second moments <q^2>, <p^2>, and <{q,p}/2>
    mean_q2_p = y2_vec[idx_gamma, 3] / ((bp * r_zpf_p)**2)
    mean_p2_p = y2_vec[idx_gamma, 4] / ((cp * pr_zpf_p)**2)
    mean_qp_p = y2_vec[idx_gamma, 5] / ((bp * r_zpf_p) * (cp * pr_zpf_p))
    
    # 4. Calculate Variances
    var_q_p = mean_q2_p - (mean_q_p**2)
    var_p_p = mean_p2_p - (mean_p_p**2)
    cov_qp_p = mean_qp_p - (mean_q_p * mean_p_p)
    
    # 5. Determinant
    det_V_dimless = (var_q_p * var_p_p) - (cov_qp_p**2)
    
    return 1.0 / np.sqrt(det_V_dimless)


def calculate_energy(y1_vec, y2_vec, t, idx_gamma, params: BaseSystemParams, potential_func, direction = 'x'):
    """
    Calculates the instantaneous mean mechanical energies in SI units [Joules].
    
    Parameters
    ----------
    potential_func : callable
        A function f(t, params, direction) that returns W(t) [s^-2].
    direction : str
        'x' or 'y'
    """
    # 1. Select Direction-Dependent ZPFs
    if direction == 'x' or direction == 0:
        r_zpf_p = params.x_zpf_p
        pr_zpf_p = params.px_zpf_p
    elif direction == 'y' or direction == 1:
        r_zpf_p = params.y_zpf_p
        pr_zpf_p = params.py_zpf_p
    else:
        raise ValueError(f"Invalid direction: {direction}")

    # 2. Recover Scaling Factors
    bp = 1.0 / r_zpf_p
    cp = 1.0 / pr_zpf_p

    # 3. Recover physical moments [m^2] and [kg^2 m^2/s^2]
    mean_X2_phys = y2_vec[idx_gamma, 3] / (bp**2)
    mean_P2_phys = y2_vec[idx_gamma, 4] / (cp**2)
    
    # 4. Kinetic Energy: E_k = <P^2> / 2m
    E_kin = mean_P2_phys / (2 * params.M_p)
    
    # 5. Potential Energy: E_p = 1/2 * m * W(t) * <X^2>
    # We call the potential function passed as an argument, passing the direction
    W_p_t = potential_func(t, params, direction)
    E_pot = (params.M_p * W_p_t * mean_X2_phys) / 2
    
    E_tot = E_kin + E_pot
    
    return np.array([E_kin, E_pot, E_tot])


def compute_energy_trajectory(t_list, y1_history, y2_history, params: BaseSystemParams, potential_func, idx_gamma=0, direction = 'x'):
    """
    Computes energy trajectories over time.
    """
    potential_history = []
    kinetic_history = []

    for idx_t, t_val in enumerate(t_list):
        energy_vals = calculate_energy(
            y1_vec=y1_history[idx_t], 
            y2_vec=y2_history[idx_t], 
            t=t_val, 
            idx_gamma=idx_gamma, 
            params=params, 
            potential_func=potential_func,
            direction=direction
        )
        
        kinetic_history.append(energy_vals[0])
        potential_history.append(energy_vals[1])

    return np.array(potential_history), np.array(kinetic_history)


def compute_steady_state_metrics(y1_ss_vec, y2_ss_vec, params: BaseSystemParams, direction = 'x'):
    """
    Computes purity and phonon number for all damping rates.
    """
    n_gamma = y1_ss_vec.shape[0]
    
    purity_list = []
    for i in range(n_gamma):
        p = calculate_purity(y1_ss_vec, y2_ss_vec, i, params, direction)
        purity_list.append(p)
        
    purity_ss_micro = np.array(purity_list)

    # Phonon Number: n = 1/2 * (1/Purity - 1)
    population_ss_micro = (np.divide(1.0, purity_ss_micro) - 1.0) * 0.5
    
    return purity_ss_micro, population_ss_micro