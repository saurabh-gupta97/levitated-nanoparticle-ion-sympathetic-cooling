import numpy as np
from .parameters import BaseSystemParams

def time_dep_potential_general(t, omega_slow, omega_fast, a, q_slow, q_fast):
    """Calculates W(t) based on Mathieu parameters."""
    l = omega_slow / omega_fast
    return (omega_fast**2 / 4) * (
        a
        - 2 * q_slow * l**2 * np.cos(omega_slow * t) 
        - 2 * q_fast * np.cos(omega_fast * t)
    )

# --- Ion Wrappers ---
def W_x_i(t, params: BaseSystemParams):
    return time_dep_potential_general(
        t, params.omega_slow, params.omega_fast, 
        params.a_x_i, params.q_slow_x_i, params.q_fast_x_i
    )

def W_y_i(t, params: BaseSystemParams):
    return time_dep_potential_general(
        t, params.omega_slow, params.omega_fast, 
        params.a_y_i, params.q_slow_y_i, params.q_fast_y_i
    )

# --- Nanoparticle Wrappers ---
def W_x_p(t, params: BaseSystemParams):
    return time_dep_potential_general(
        t, params.omega_slow, params.omega_fast, 
        params.a_x_p, params.q_slow_x_p, params.q_fast_x_p
    )

def W_y_p(t, params: BaseSystemParams):
    return time_dep_potential_general(
        t, params.omega_slow, params.omega_fast, 
        params.a_y_p, params.q_slow_y_p, params.q_fast_y_p
    )

# --- Direction Dispatchers ---
def W_i(t, params: BaseSystemParams, direction: str):
    if direction == 'x' or direction == 0:
        return W_x_i(t, params)
    elif direction == 'y' or direction == 1:
        return W_y_i(t, params)
    else:
        raise ValueError(f"Invalid direction: {direction}")

def W_p(t, params: BaseSystemParams, direction: str):
    if direction == 'x' or direction == 0:
        return W_x_p(t, params)
    elif direction == 'y' or direction == 1:
        return W_y_p(t, params)
    else:
        raise ValueError(f"Invalid direction: {direction}")