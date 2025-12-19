"""
Coupled Ion-Nanoparticle Dynamics Simulation in the RADIAL direction.

This script calculates the steady-state dynamics of a system consisting 
of a single ion and a nanoparticle levitated in a linear Paul trap. 
It sets up the equations of motion in the Floquet formalism to solve 
for the first and second moments (covariance matrix) of the system.

SI units are used throughout unless specified otherwise.
"""
# ==============================================================================
# BLOCK 0: Import Libraries
# ==============================================================================
import numpy as np
from numpy.linalg import inv, matrix_power
from scipy.integrate import solve_ivp
import time
import matplotlib.pyplot as plt



# ==============================================================================
# BLOCK 1: Physical Constants and System Parameters
# =============================================================================
# Physical constants
hbar = 1.05 * 10**(-34)
kB   = 1.38 * 10**(-23)
eps0 = 8.85 * 10**(-12)


# Ion and nanoparticle parameters
Q_i, M_i = 1.6 * 10**(-19), 40 * 1.6 * 10**(-27)
Q_p, M_p = 750 * Q_i, 2 * 10**(-17)



# Paul trap parameters
dz = 1.7 * 10**(-3)
alpha_z = 0.38
V_0_z, V_slow_z, V_fast_z = 56.5, 0, 0

dx = 0.9 * 10**(-3)
alpha_x = 0.93
V_0_x = -0.5 * (alpha_z/alpha_x) * (dx/dz)**2 * V_0_z
V_slow_x, V_fast_x = 80, 1350

dy = 0.9 * 10**(-3)
alpha_y = 0.93
V_0_y =  -0.5 * (alpha_z/alpha_y) * (dx/dz)**2 * V_0_z
V_slow_y, V_fast_y = -80, -1350

omega_slow, omega_fast = 2 * np.pi * 7 * 10**3, 2 * np.pi * 17.5 * 10**6
l = omega_slow / omega_fast



# Mathieu parameters defined using system parameters
a_x_i      = +(4 * Q_i * V_0_x * alpha_x) / (M_i * dx**2 * omega_fast**2)
q_slow_x_i = -(2 * Q_i * V_slow_x * alpha_x) / (M_i * dx**2 * omega_slow**2)
q_fast_x_i = -(2 * Q_i * V_fast_x * alpha_x) / (M_i * dx**2 * omega_fast**2)

a_y_i      = +(4 * Q_i * V_0_y * alpha_y) / (M_i * dy**2 * omega_fast**2)
q_slow_y_i = -(2 * Q_i * V_slow_y * alpha_y) / (M_i * dy**2 * omega_slow**2)
q_fast_y_i = -(2 * Q_i * V_fast_y * alpha_y) / (M_i * dy**2 * omega_fast**2)

a_z_i      = +(4 * Q_i * V_0_z * alpha_z) / (M_i * dz**2 * omega_fast**2)
q_slow_z_i = -(2 * Q_i * V_slow_z * alpha_z) / (M_i * dz**2 * omega_slow**2)
q_fast_z_i = -(2 * Q_i * V_fast_z * alpha_z) / (M_i * dz**2 * omega_fast**2)

a_x_p      = +(4 * Q_p * V_0_x * alpha_x) / (M_p * dx**2 * omega_fast**2)
q_slow_x_p = -(2 * Q_p * V_slow_x * alpha_x) / (M_p * dx**2 * omega_slow**2)
q_fast_x_p = -(2 * Q_p * V_fast_x * alpha_x) / (M_p * dx**2 * omega_fast**2)

a_y_p      = +(4 * Q_p * V_0_y * alpha_y) / (M_p * dy**2 * omega_fast**2)
q_slow_y_p = -(2 * Q_p * V_slow_y * alpha_y) / (M_p * dy**2 * omega_slow**2)
q_fast_y_p = -(2 * Q_p * V_fast_y * alpha_y) / (M_p * dy**2 * omega_fast**2)

a_z_p      = +(4 * Q_p * V_0_z * alpha_z) / (M_p * dz**2 * omega_fast**2)
q_slow_z_p = -(2 * Q_p * V_slow_z * alpha_z) / (M_p * dz**2 * omega_slow**2)
q_fast_z_p = -(2 * Q_p * V_fast_z * alpha_z) / (M_p * dz**2 * omega_fast**2)



# Values of Gamma_ba / gamma_fb for different axes
Gamma_to_gamma_x = 561.39
Gamma_to_gamma_y = 750.09
Gamma_to_gamma_z = 842.08



# Choose the RADIAL (x or y) axis along which computation is to be performed
a_r_i = a_x_i
q_slow_r_i = q_slow_x_i
q_fast_r_i = q_fast_x_i

a_r_p = a_x_p
q_slow_r_p = q_slow_x_p
q_fast_r_p = q_fast_x_p

Gamma_to_gamma_r = Gamma_to_gamma_x



# Time-dependent trap frequency^2
def W_i(t):
    """
    Calculates the time-dependent confining potential curvature for the Ion.
    
    The potential is of the form V(t) ~ M * W(t) * x^2.
    
    Parameters
    ----------
    t : float
        Time in seconds.
        
    Returns
    -------
    float
        W(t) in [s^-2].
    """
    return (omega_fast**2 / 4) * (
        a_r_i
        - 2 * q_slow_r_i * l**2 * np.cos(omega_slow * t)
        - 2 * q_fast_r_i * np.cos(omega_fast * t)
    )

def W_p(t):
    """
    Calculates the time-dependent confining potential curvature for the Nanoparticle.
    
    The potential is of the form V(t) ~ M * W(t) * x^2.
    
    Parameters
    ----------
    t : float
        Time in seconds.
        
    Returns
    -------
    float
        W(t) in [s^-2].
    """
    return (omega_fast**2 / 4) * (
        a_r_p
        - 2 * q_slow_r_p * l**2 * np.cos(omega_slow * t)
        - 2 * q_fast_r_p * np.cos(omega_fast * t)
    )



# Effective secular trap frequencies of the ion and the nanoparticle
Omega_r_i = (1/np.sqrt(2)) * np.sqrt(((omega_fast/2)*np.sqrt(a_r_i + (q_fast_r_i**2)/2))**2 + (omega_slow/2)**2 + np.sqrt((((omega_fast/2)*np.sqrt(a_r_i + (q_fast_r_i**2)/2))**2 - (omega_slow/2)**2)**2 - (q_slow_r_i**2 * omega_slow**4)/8))
Omega_z_i = (1/np.sqrt(2)) * np.sqrt(((omega_fast/2)*np.sqrt(a_z_i + (q_fast_z_i**2)/2))**2 + (omega_slow/2)**2 + np.sqrt((((omega_fast/2)*np.sqrt(a_z_i + (q_fast_z_i**2)/2))**2 - (omega_slow/2)**2)**2 - (q_slow_z_i**2 * omega_slow**4)/8))

Omega_r_p = (omega_fast/2) * np.sqrt(a_r_p + (q_slow_r_p * l)**2/2 + q_fast_r_p**2/2)
Omega_z_p = (omega_fast/2) * np.sqrt(a_z_p + (q_slow_z_p * l)**2/2 + q_fast_z_p**2/2)



# Distance between the ion and the nanoparticle trap centres along the trap axis 
dist = np.cbrt(((Q_i * Q_p)/(4 * np.pi * eps0)) * ((1/(M_i * Omega_z_i**2)) + (1/(M_p * Omega_z_p**2))))



# Renormalised effective secular trap frequencies of the ion and the nanoparticle in the radial direction
new_Omega_r_i = np.sqrt(
    Omega_r_i**2 - ((Q_i * Q_p)/(4 * np.pi * eps0 * M_i * dist**3))
)
new_Omega_r_p = np.sqrt(
    Omega_r_p**2 - ((Q_i * Q_p)/(4 * np.pi * eps0 * M_p * dist**3))
)


# Zero-point fluctuations in the position and momentum of the ion and the nanoparticle computed from the renormalised frequencies
r_zpf_i = np.sqrt(hbar / (2 * M_i * new_Omega_r_i))
p_zpf_i = np.sqrt(hbar * M_i * new_Omega_r_i / 2)

r_zpf_p = np.sqrt(hbar / (2 * M_p * new_Omega_r_p))
p_zpf_p = np.sqrt(hbar * M_p * new_Omega_r_p / 2)



# Coupling rate between ion and nanoparticle in the radial direction
g_ip = (1/hbar) * ((Q_i * Q_p)/(4 * np.pi * eps0)) * ((r_zpf_i * r_zpf_p)/dist**3)
# hbar * g / (r_zpf_i * r_zpf_p). This is the coefficient of the coupling term K_ip * deltaR_i * deltaR_p
K_ip = (Q_i * Q_p)/(4 * np.pi * eps0 * dist**3)



# Dissipation rates of the ion and nanoparticle
# These are the damping and heating rates from Doppler cooling of the ion 
gamma_i = 2 * np.pi * 10 * 10**3
Gamma_h_i = 3.8 * 10**(-22)/(hbar * new_Omega_r_i)

# This is an array of the damping rates of the nanoparticle over which we wish to compute the occupation of nanoparticle COM motion
gamma_p_vec = 2*np.pi * np.insert(np.logspace(-7, 3, num=21), 0, 44.5e-9)

# This is the heating rate of the nanoparticle from background gas scattering
Gamma_h_p = 11.5 * 10**(-28)/(hbar * new_Omega_r_p)
# This is the trap displacement noise heating rate of the nanoparticle. The commented value is the typical value observed in experiments.
Gamma_td_p = 0 #2.8 * 10**(-26)/(hbar * new_Omega_r_p)
# This is the back action noise heating rate of the nanoparticle
Gamma_fb_p_vec = Gamma_to_gamma_r * gamma_p_vec


n_gamma = len(gamma_p_vec)


# ==============================================================================
# BLOCK 2: Values of occupation number computed without micromotion
# ==============================================================================
# This is an array of the damping rates of the nanoparticle over which the following occupation values were computed
gamma_p_vec_secular = np.insert(2*np.pi * 10**np.arange(-7, 3 + (1/60), 1/30.), 0, 2*np.pi * 44.5e-9)



# These are the arrays of nanoparticle COM motion occupation in different directions computed in the absence of micromotion. These values can be obtained from the expression of the steady state occupation number, or numerically, for instance, using the n_vs_gamma.nb Mathematica notebook.
# These are the occupation numbers in the z-axis with typical trap displacement heating rates in experiments
population_z_ss_secular_large = np.array([4.6411e8, 4.62302e8, 4.62044e8, 4.61766e8, 4.61467e8, 
4.61143e8, 4.60795e8, 4.6042e8, 4.60015e8, 4.59578e8, 
4.59108e8, 4.58602e8, 4.58056e8, 4.57468e8, 4.56835e8, 
4.56153e8, 4.5542e8, 4.5463e8, 4.53781e8, 4.52867e8, 
4.51884e8, 4.50828e8, 4.49693e8, 4.48474e8, 4.47166e8, 
4.45761e8, 4.44254e8, 4.42639e8, 4.40907e8, 4.39053e8, 
4.37068e8, 4.34945e8, 4.32675e8, 4.30251e8, 4.27664e8, 
4.24905e8, 4.21966e8, 4.18838e8, 4.15511e8, 4.11979e8, 
4.08231e8, 4.0426e8, 4.00058e8, 3.95618e8, 3.90933e8, 
3.85997e8, 3.80806e8, 3.75355e8, 3.69642e8, 3.63665e8, 
3.57424e8, 3.50922e8, 3.44162e8, 3.37149e8, 3.2989e8, 
3.22395e8, 3.14676e8, 3.06745e8, 2.98619e8, 2.90315e8, 
2.81851e8, 2.7325e8, 2.64533e8, 2.55724e8, 2.46848e8, 
2.37932e8, 2.29e8, 2.20079e8, 2.11195e8, 2.02375e8, 
1.93642e8, 1.85021e8, 1.76535e8, 1.68205e8, 1.6005e8, 
1.52088e8, 1.44335e8, 1.36805e8, 1.29509e8, 1.22458e8, 
1.15658e8, 1.09116e8, 1.02835e8, 9.68176e7, 9.10638e7, 
8.55726e7, 8.03415e7, 7.53667e7, 7.06435e7, 6.6166e7, 
6.19279e7, 5.79219e7, 5.41402e7, 5.05749e7, 4.72174e7, 
4.40591e7, 4.10913e7, 3.83052e7, 3.56922e7, 3.32436e7, 
3.09509e7, 2.88057e7, 2.68001e7, 2.49261e7, 2.31763e7, 
2.15433e7, 2.00201e7, 1.86002e7, 1.7277e7, 1.60446e7, 
1.48972e7, 1.38293e7, 1.28358e7, 1.19118e7, 1.10527e7, 
1.02542e7, 9.51213e6, 8.82274e6, 8.18242e6, 7.58781e6, 
7.03574e6, 6.52328e6, 6.04765e6, 5.60628e6, 5.19677e6, 
4.81686e6, 4.46446e6, 4.13761e6, 3.8345e6, 3.55342e6, 
3.29281e6, 3.05119e6, 2.82719e6, 2.61955e6, 2.42709e6, 
2.2487e6, 2.08337e6, 1.93015e6, 1.78815e6, 1.65657e6, 
1.53465e6, 1.42168e6, 1.317e6, 1.22001e6, 1.13016e6, 
1.04691e6, 969787., 898339., 832149., 770832., 714032., 661415., 
612675., 567528., 525707., 486970., 451089., 417854., 387070., 
358557., 332148., 307687., 285031., 264046., 244611., 226609., 
209937., 194495., 180194., 166948., 154680., 143317., 132794.,
123048., 114022., 105662., 97919.1, 90748.4, 84107.2, 77956.6, 
72260.2, 66984.5, 62098.5, 57573.4, 53382.6, 49501.3, 45906.7, 
42577.6, 39494.4, 36639., 33994.6, 31545.4, 29277.2, 27176.6, 
25231.1, 23429.3, 21760.7, 20215.3, 18784.1, 17458.6, 16231.1, 
15094.2, 14041.3, 13066.2, 12163.2, 11326.8, 10552.3, 9834.95, 
9170.61, 8555.35, 7985.55, 7457.84, 6969.12, 6516.51, 6097.34,
5709.13, 5349.61, 5016.64, 4708.28, 4422.7, 4158.21, 3913.27, 
3686.42, 3476.33, 3281.77, 3101.58, 2934.7, 2780.15, 2637.02, 
2504.46, 2381.7, 2268., 2162.71, 2065.2, 1974.89, 1891.25, 1813.79, 
1742.05, 1675.62, 1614.09, 1557.11, 1504.34, 1455.46, 1410.2, 
1368.28, 1329.46, 1293.51, 1260.21, 1229.37, 1200.82, 1174.37, 
1149.87, 1127.19, 1106.18, 1086.72, 1068.7, 1052.01, 1036.56, 
1022.25, 1008.99, 996.714, 985.344, 974.815, 965.063, 956.032, 
947.668, 939.922, 932.748, 926.105, 919.952, 914.253, 908.976, 
904.088, 899.562, 895.369, 891.487, 887.891, 884.561, 881.476, 
878.62, 875.974, 873.523, 871.254, 869.151, 867.204, 865.4, 863.729, 
862.181, 860.746, 859.417, 858.185, 857.044, 855.986, 855.004, 
854.094, 853.25, 852.466, 851.738, 851.062, 850.434, 849.849, 849.305])

# These are the occupation numbers in the z-axis with no trap displacement heating
population_z_ss_secular_small = np.array([1.83137e7, 1.82423e7, 1.82322e7, 1.82212e7, 1.82094e7, 
1.81966e7, 1.81829e7, 1.81681e7, 1.81521e7, 1.81349e7, 
1.81163e7, 1.80963e7, 1.80748e7, 1.80516e7, 1.80266e7, 
1.79997e7, 1.79708e7, 1.79396e7, 1.79061e7, 1.787e7, 
1.78313e7, 1.77896e7, 1.77448e7, 1.76967e7, 1.76451e7, 
1.75897e7, 1.75302e7, 1.74665e7, 1.73981e7, 1.7325e7, 
1.72467e7, 1.71629e7, 1.70733e7, 1.69777e7, 1.68756e7, 
1.67667e7, 1.66508e7, 1.65273e7, 1.63961e7, 1.62567e7, 
1.61088e7, 1.59521e7, 1.57863e7, 1.56111e7, 1.54263e7, 
1.52315e7, 1.50267e7, 1.48116e7, 1.45861e7, 1.43503e7, 
1.41041e7, 1.38475e7, 1.35808e7, 1.3304e7, 1.30176e7, 
1.27219e7, 1.24173e7, 1.21044e7, 1.17837e7, 1.14561e7, 
1.11221e7, 1.07827e7, 1.04388e7, 1.00912e7, 9.74097e6, 
9.38912e6, 9.03669e6, 8.68469e6, 8.33416e6, 7.98611e6, 
7.64154e6, 7.30138e6, 6.96653e6, 6.63783e6, 6.31606e6, 
6.0019e6, 5.69599e6, 5.39887e6, 5.111e6, 4.83276e6, 
4.56446e6, 4.30632e6, 4.05849e6, 3.82104e6, 3.59401e6, 
3.37734e6, 3.17093e6, 2.97463e6, 2.78826e6, 2.61159e6, 
2.44436e6, 2.28629e6, 2.13708e6, 1.9964e6, 1.86391e6, 
1.73929e6, 1.62219e6, 1.51226e6, 1.40916e6, 1.31254e6, 
1.22207e6, 1.13743e6, 1.05829e6, 984345., 915300., 850865., 
790765., 734736., 682527., 633899., 588624., 546488., 507287., 
470827., 436929., 405420., 376140., 348938., 323672., 300210., 
278426., 258205., 239438., 222022., 205864., 190873., 176968., 
164071., 152111., 141021., 130737., 121203., 112365., 104172., 
96577.7, 89538.8, 83015.2, 76969.3, 71366.6, 66174.7, 61363.8, 
56906.1, 52775.8, 48949., 45403.4, 42118.6, 39075.5, 36256.3, 
33644.5, 31225.1, 28983.9, 26907.7, 24984.5, 23203.1, 21553., 
20024.5, 18608.7, 17297.3, 16082.6, 14957.5, 13915.5, 12950.3, 
12056.3, 11228.3, 10461.4, 9751.14, 9093.28, 8483.97, 7919.65, 7397., 
6912.93, 6464.6, 6049.37, 5664.81, 5308.64, 4978.77, 4673.27, 
4390.32, 4128.28, 3885.58, 3660.82, 3452.65, 3259.86, 3081.31, 
2915.94, 2762.79, 2620.96, 2489.6, 2367.94, 2255.27, 2150.93, 
2054.29, 1964.79, 1881.9, 1805.14, 1734.05, 1668.2, 1607.23, 1550.75, 
1498.45, 1450.02, 1405.16, 1363.61, 1325.14, 1289.51, 1256.5, 
1225.94, 1197.64, 1171.42, 1147.15, 1124.66, 1103.84, 1084.56, 
1066.7, 1050.16, 1034.84, 1020.66, 1007.52, 995.35, 984.081, 973.645, 
963.98, 955.029, 946.74, 939.062, 931.952, 925.368, 919.269, 913.622, 
908.391, 903.547, 899.061, 894.907, 891.059, 887.495, 884.195, 
881.139, 878.308, 875.687, 873.259, 871.011, 868.928, 867., 865.214, 
863.56, 862.028, 860.61, 859.296, 858.079, 856.952, 855.908, 854.942, 
854.047, 853.218, 852.45, 851.739, 851.081, 850.471, 849.906, 
849.383, 848.898, 848.45, 848.034, 847.649, 847.293, 846.963, 
846.657, 846.374, 846.112, 845.869, 845.644, 845.435, 845.242, 
845.063, 844.897, 844.744, 844.601, 844.469, 844.347, 844.233, 
844.128, 844.03, 843.94, 843.855, 843.777, 843.704, 843.636, 843.572, 
843.513, 843.457, 843.404, 843.355, 843.308, 843.263, 843.221, 
843.18, 843.14, 843.102, 843.064, 843.026, 842.989, 842.952])

# These are the occupation numbers in the radial direction with typical trap displacement heating rates in experiments
population_r_ss_secular_large = np.array([5.89211e10, 3.37432e10, 3.17988e10, 2.99363e10, 
2.81555e10, 2.64563e10, 2.48376e10, 2.32985e10, 
2.18373e10, 2.04524e10, 1.91415e10, 1.79025e10, 
1.6733e10, 1.56305e10, 1.45923e10, 1.36158e10, 
1.26983e10, 1.1837e10, 1.10292e10, 1.02723e10, 
9.5636e9, 8.90056e9, 8.28066e9, 7.70148e9, 7.16069e9, 
6.65602e9, 6.18531e9, 5.74651e9, 5.33764e9, 4.95681e9, 
4.60226e9, 4.2723e9, 3.96532e9, 3.67981e9, 3.41437e9, 
3.16764e9, 2.93837e9, 2.72538e9, 2.52754e9, 2.34384e9, 
2.17327e9, 2.01495e9, 1.86801e9, 1.73165e9, 1.60514e9, 
1.48777e9, 1.3789e9, 1.27792e9, 1.18428e9, 1.09745e9, 
1.01694e9, 9.42299e8, 8.73101e8, 8.08957e8, 7.495e8, 
6.94392e8, 6.43318e8, 5.95985e8, 5.52121e8, 5.11474e8, 
4.7381e8, 4.3891e8, 4.06574e8, 3.76615e8, 3.48857e8, 
3.23141e8, 2.99316e8, 2.77245e8, 2.56798e8, 2.37857e8, 
2.2031e8, 2.04057e8, 1.89e8, 1.75054e8, 1.62135e8, 
1.50169e8, 1.39085e8, 1.28818e8, 1.19309e8, 1.10501e8, 
1.02343e8, 9.47867e7, 8.77881e7, 8.13059e7, 7.5302e7, 
6.97414e7, 6.45911e7, 5.98211e7, 5.54031e7, 5.13114e7, 
4.75217e7, 4.40119e7, 4.07612e7, 3.77505e7, 3.49622e7, 
3.23797e7, 2.9988e7, 2.77729e7, 2.57214e7, 2.38215e7, 
2.20618e7, 2.04321e7, 1.89228e7, 1.7525e7, 1.62304e7, 
1.50315e7, 1.39211e7, 1.28927e7, 1.19403e7, 1.10582e7, 
1.02413e7, 9.48478e6, 8.78411e6, 8.13521e6, 7.53424e6, 
6.97767e6, 6.46222e6, 5.98484e6, 5.54273e6, 5.13328e6, 
4.75409e6, 4.4029e6, 4.07766e6, 3.77645e6, 3.49748e6, 
3.23913e6, 2.99987e6, 2.77828e6, 2.57306e6, 2.38301e6, 
2.20699e6, 2.04398e6, 1.89301e6, 1.7532e6, 1.62371e6, 
1.50379e6, 1.39273e6, 1.28988e6, 1.19462e6, 1.1064e6, 
1.0247e6, 949039., 878965., 814068., 753965., 698303., 646753., 
599012., 554798., 513850., 475928., 440807., 408281., 378159., 
350261., 324425., 300498., 278338., 257816., 238809., 221207., 
204906., 189809., 175827., 162878., 150886., 139780., 129494., 
119968., 111146., 102976., 95409.7, 88402.2, 81912.4, 75902.1, 
70335.8, 65180.8, 60406.7, 55985.2, 51890.4, 48098.2, 44586.1, 
41333.5, 38321.2, 35531.5, 32947.8, 30555.1, 28339.1, 26286.9, 
24386.2, 22626., 20995.9, 19486.1, 18088., 16793.1, 15593.8, 14483.2, 
13454.7, 12502.1, 11619.9, 10802.9, 10046.2, 9345.48, 8696.5, 
8095.47, 7538.84, 7023.34, 6545.92, 6103.78, 5694.3, 5315.08, 
4963.87, 4638.61, 4337.38, 4058.4, 3800.04, 3560.76, 3339.17, 
3133.94, 2943.88, 2767.86, 2604.84, 2453.87, 2314.05, 2184.56, 
2064.64, 1953.58, 1850.72, 1755.47, 1667.25, 1585.54, 1509.88, 
1439.8, 1374.91, 1314.8, 1259.14, 1207.59, 1159.85, 1115.63, 1074.69, 
1036.76, 1001.64, 969.116, 938.993, 911.096, 885.26, 861.332, 
839.172, 818.65, 799.643, 782.041, 765.74, 750.643, 736.661, 723.712, 
711.72, 700.613, 690.328, 680.802, 671.98, 663.81, 656.243, 649.236, 
642.746, 636.736, 631.17, 626.015, 621.24, 616.819, 612.724, 608.932, 
605.42, 602.167, 599.155, 596.365, 593.782, 591.389, 589.173, 
587.121, 585.22, 583.46, 581.83, 580.32, 578.922, 577.627, 576.428, 
575.317, 574.288, 573.336, 572.454, 571.637, 570.88, 570.179, 569.53, 
568.929, 568.372, 567.857, 567.379, 566.937, 566.528, 566.148, 565.797])

# These are the occupation numbers in the radial direction with no trap displacement heating
population_r_ss_secular_small = np.array([2.32642e9, 1.3323e9, 1.25553e9, 1.18199e9, 1.11168e9, 
1.04459e9, 9.80679e8, 9.19908e8, 8.62216e8, 8.07532e8, 
7.55774e8, 7.06856e8, 6.6068e8, 6.17149e8, 5.76158e8, 
5.37602e8, 5.01374e8, 4.67367e8, 4.35473e8, 4.05587e8, 
3.77605e8, 3.51426e8, 3.2695e8, 3.04082e8, 2.8273e8, 
2.62804e8, 2.44219e8, 2.26893e8, 2.10749e8, 1.95713e8, 
1.81714e8, 1.68686e8, 1.56565e8, 1.45293e8, 1.34812e8, 
1.2507e8, 1.16018e8, 1.07608e8, 9.9797e7, 9.25435e7, 
8.58091e7, 7.95579e7, 7.37561e7, 6.83723e7, 6.3377e7, 
5.87429e7, 5.44443e7, 5.04575e7, 4.67603e7, 4.33318e7, 
4.0153e7, 3.72058e7, 3.44737e7, 3.1941e7, 2.95935e7, 
2.74176e7, 2.5401e7, 2.35321e7, 2.18002e7, 2.01954e7, 
1.87082e7, 1.73303e7, 1.60535e7, 1.48706e7, 1.37747e7, 
1.27593e7, 1.18186e7, 1.09471e7, 1.01398e7, 9.39197e6, 
8.69918e6, 8.05742e6, 7.46295e6, 6.91228e6, 6.40221e6, 
5.92974e6, 5.49211e6, 5.08675e6, 4.71128e6, 4.36351e6, 
4.0414e6, 3.74306e6, 3.46672e6, 3.21078e6, 2.97373e6, 
2.75418e6, 2.55083e6, 2.36249e6, 2.18805e6, 2.02649e6, 
1.87687e6, 1.73828e6, 1.60994e6, 1.49106e6, 1.38097e6, 
1.27901e6, 1.18457e6, 1.09711e6, 1.01611e6, 941095., 
871618., 807272., 747679., 692488., 641373., 594034., 550192., 
509588., 471984., 437157., 404903., 375032., 347367., 321746., 
298018., 276042., 255691., 236842., 219386., 203220., 188247., 
174381., 161540., 149647., 138632., 128432., 118985., 110236., 
102133., 94628.8, 87679., 81242.8, 75282., 69761.6, 64649.1, 59914.2, 
55529.2, 51468.2, 47707.1, 44224., 40998.2, 38010.7, 35243.9, 
32681.5, 30308.4, 28110.7, 26075.3, 24190.3, 22444.6, 20827.9, 
19330.5, 17943.9, 16659.6, 15470.3, 14368.8, 13348.7, 12403.9, 
11529., 10718.7, 9968.27, 9273.28, 8629.64, 8033.54, 7481.49, 
6970.23, 6496.74, 6058.23, 5652.11, 5276., 4927.68, 4605.1, 4306.34, 
4029.66, 3773.42, 3536.11, 3316.34, 3112.8, 2924.3, 2749.72, 2588.05, 
2438.32, 2299.65, 2171.22, 2052.29, 1942.14, 1840.13, 1745.65, 
1658.16, 1577.13, 1502.08, 1432.58, 1368.22, 1308.61, 1253.41, 
1202.28, 1154.93, 1111.08, 1070.47, 1032.86, 998.024, 965.766, 
935.89, 908.222, 882.598, 858.867, 836.889, 816.536, 797.686, 
780.228, 764.06, 749.087, 735.22, 722.378, 710.484, 699.469, 689.268, 
679.821, 671.071, 662.968, 655.464, 648.514, 642.078, 636.117, 
630.596, 625.484, 620.749, 616.363, 612.302, 608.541, 605.058, 
601.832, 598.845, 596.078, 593.515, 591.142, 588.945, 586.909, 
585.024, 583.278, 581.662, 580.164, 578.778, 577.493, 576.304, 
575.203, 574.182, 573.238, 572.363, 571.552, 570.802, 570.107, 
569.463, 568.867, 568.315, 567.804, 567.33, 566.892, 566.486, 566.11, 
565.761, 565.439, 565.14, 564.863, 564.607, 564.37, 564.15, 563.947, 
563.758, 563.583, 563.422, 563.272, 563.133, 563.005, 562.886, 
562.776, 562.674, 562.579, 562.492, 562.411, 562.336, 562.266, 
562.202, 562.142, 562.087, 562.036, 561.989, 561.945, 561.904, 
561.866, 561.832, 561.799, 561.769, 561.742, 561.716, 561.692, 
561.67, 561.65, 561.631, 561.614, 561.597, 561.582, 561.568])



# These are the arrays of purity values computed from the above occupation numbers as Purity = 1/(2 n + 1), where n is the occupation number
# The nomenclature for different axes and the presence/absence of trap displacement heating remains the same as above
purity_z_ss_secular_large = 1/(2 * population_z_ss_secular_large + 1)
purity_z_ss_secular_small = 1/(2 * population_z_ss_secular_small + 1)

purity_r_ss_secular_large = 1/(2 * population_r_ss_secular_large + 1)
purity_r_ss_secular_small = 1/(2 * population_r_ss_secular_small + 1)



# =========================================================================================
# BLOCK 3: Setting up the Floquet equations of motion (drift matrices and diffusion vectors
# =========================================================================================

# These are scaling factors which we use to make the range of values in our Floquet system more tractable
bi = 1/r_zpf_i
bp = 1/r_zpf_p
ci = 1/p_zpf_i
cp = 1/p_zpf_p


# Floquet system for first-order moments of the quadrature operators. The ordering is [deltaR_i, P_i, deltaR_p, P_p].
# The vector of scaling factors for each component
scale1_vec = np.array([bi,
                       ci, 
                       bp,
                       cp])

# Hadamard product to element-wise scale the drift matrix of first-order moments. It creates the matrix of ratios (s_i / s_j) and multiplies element-wise.
scale1_mat = np.outer(scale1_vec, 1/scale1_vec)

# These are initial values for the first-order moments scaled by the scaling factors
# These initial conditions can be chosen arbitrarily since we are only interested in the steady state values. We choose a thermal state at 300 K to be the initial state.
X10_vec = scale1_vec * np.array([
    1.0,  # <deltaR_i>
    0.0,  # <P_i>
    1.0,  # <deltaR_p>
    0.0   # <P_p>
])

# Initial temperature and thermal occupation numbers (nbars) of the ion and nanoparticle COM motion
temp0 = 300
nbar_i0 = (kB * temp0) / (hbar * new_Omega_r_i)
nbar_p0 = (kB * temp0) / (hbar * new_Omega_r_p)

# Initial standard deviations in the quadratures for a thermal state at the initial temperature temp0
r_std_i0 = r_zpf_i * np.sqrt(2 * nbar_i0 + 1)
p_std_i0 = p_zpf_i * np.sqrt(2 * nbar_i0 + 1)

r_std_p0 = r_zpf_p * np.sqrt(2 * nbar_p0 + 1)
p_std_p0 = p_zpf_p * np.sqrt(2 * nbar_p0 + 1)


# The (n_gamma, 4, 4) drift matrix and the (n_gamma, 1, 4) driving vector for the first moments
# The first dimension corresponds to the different gamma_p values in the gamma_p_vec
def drift1_mat(t):
    """
    Computes the Drift Matrix for the first-order moments.
    
    The system evolves as d<X>/dt = drift1_mat(t)<X> + drive1_vec.
    Includes vectorized support for multiple nanoparticle damping rates (gamma_p).

    Parameters
    ----------
    t : float
        Current time in simulation [s].

    Returns
    -------
    ndarray
        Shape (n_gamma, 4, 4). The scaled drift matrix for every gamma_p value.
    """
    # drift1_phys_mat is (4, 4) array
    drift1_phys_mat = np.array([
        [-gamma_i/2, 1/M_i, 0, 0],
        [-M_i*(W_i(t) - K_ip/M_i), -gamma_i/2, -K_ip, 0],
        [0, 0, 0, 1/M_p],                                    # gamma_p will be added below
        [-K_ip, 0, -M_p*(W_p(t) - K_ip/M_p), 0]              # gamma_p will be added below
    ])

    # Converts drift1_phys_mat to a (n_gamma, 4, 4) array
    drift1_phys_mat = np.tile(drift1_phys_mat, (n_gamma, 1, 1))
    
    # Add nanoparticle damping (gamma_p) to the relevant diagonal elements
    drift1_phys_mat[:, 2, 2] = drift1_phys_mat[:, 3, 3] = -gamma_p_vec/2.0

    return scale1_mat * drift1_phys_mat

# drive1_vec is a (n_gamma, 4) array. The driving terms for first order moments are all zero. 
drive1_vec = scale1_vec * np.tile(np.array([0,0,0,0]), (n_gamma, 1))


# Floquet system for second-order moments of the quadrature operators. The ordering is [deltaR_i^2, P_i^2, deltaR_i P_i, deltaR_p^2, P_p^2, deltaR_p P_p, deltaR_i deltaR_p, P_i, P_p, deltaR_i P_p, deltaR_p P_i].
# The vector of scaling factors for each component
scale2_vec = np.array([
    bi*bi, ci*ci, bi*ci,       # Ion terms
    bp*bp, cp*cp, bp*cp,       # Particle terms
    bi*bp, ci*cp, bi*cp, bp*ci # Cross terms
])

# Hadamard product to element-wise scale the drift matrix.
scale2_mat = np.outer(scale2_vec, 1/scale2_vec)

# These are initial values for the second-order moments scaled by the scaling factors
# These initial conditions are chosen for a thermal state at 300 K to be the initial state.
X20_vec = scale2_vec * np.array([
    (bi**2) * r_std_i0**2 + X10_vec[0]**2,  # <deltaR_i^2>
    (ci**2) * p_std_i0**2 + X10_vec[1]**2,  # <P_i^2>
    (bi * ci) * 0,                          # <{deltaR_i, P_i}/2> (assuming zero initial correlation)
    (bp**2) * r_std_p0**2 + X10_vec[2]**2,  # <deltaR_p^2>
    (cp**2) * p_std_p0**2 + X10_vec[3]**2,  # <P_p^2>
    (bp * cp) * 0,                          # <{deltaR_p, P_p}/2> (assuming zero initial correlation)
    (bi * bp) * 0,                          # <deltaR_i deltaR_p>
    (ci * cp) * 0,                          # <P_i P_p>
    (bi * cp) * 0,                          # <deltaR_i P_p>
    (bp * ci) * 0                           # <deltaR_p P_i>
])


# The (n_gamma, 10, 10) drift matrix and the (n_gamma, 1, 10) driving vector for the second moments
# The first dimension corresponds to the different gamma_p values in the gamma_p_vec
def drift2_mat(t):
    """
    Computes the Drift Matrix for the second-order moments (Covariance Matrix evolution).
    
    Includes vectorized support for multiple nanoparticle damping rates (gamma_p).

    Parameters
    ----------
    t : float
        Current time [s].

    Returns
    -------
    ndarray
        Shape (n_gamma, 10, 10). The scaled drift matrix for every gamma_p value.
    """
    # drift2_phys_mat is (10, 10) array
    drift2_phys_mat = np.array([
      [-gamma_i, 0, 2/M_i, 0, 0, 0, 0, 0, 0, 0],
      [0, -gamma_i, -2*M_i*(W_i(t) - K_ip/M_i), 0, 0, 0, 0, 0, 0, -2*K_ip],
      [-M_i*(W_i(t) - K_ip/M_i), 1/M_i, -gamma_i, 0, 0, 0, -K_ip, 0, 0, 0],
      [0, 0, 0, 0, 0, 2/M_p, 0, 0, 0, 0],                                                   # gamma_p will be added below
      [0, 0, 0, 0, 0, -2*M_p*(W_p(t) - K_ip/M_p), 0, 0, -2*K_ip, 0],                        # gamma_p will be added below
      [0, 0, 0, -M_p*(W_p(t) - K_ip/M_p), 1/M_p, 0, -K_ip, 0, 0, 0],                        # gamma_p will be added below
      [0, 0, 0, 0, 0, 0, 0, 0, 1/M_p, 1/M_i],                                               # gamma_i, gamma_p will be added below
      [0, 0, -K_ip, 0, 0, -K_ip, 0, 0, -M_i*(W_i(t) - K_ip/M_i), -M_p*(W_p(t) - K_ip/M_p)], # gamma_i, gamma_p will be added below
      [-K_ip, 0, 0, 0, 0, 0, -M_p*(W_p(t) - K_ip/M_p), 1/M_i, 0, 0],                        # gamma_i, gamma_p will be added below
      [0, 0, 0, -K_ip, 0, 0, -M_i*(W_i(t) - K_ip/M_i), 1/M_p, 0, 0]                         # gamma_i, gamma_p will be added below
    ])

    # Converts drift2_phys_mat to a (n_gamma, 10, 10) array
    drift2_phys_mat = np.tile(drift2_phys_mat, (n_gamma, 1, 1))
    
    # Add ion and nanoparticle damping (gamma_p) to the relevant diagonal elements
    drift2_phys_mat[:, 3, 3] = drift2_phys_mat[:, 4, 4] = drift2_phys_mat[:, 5, 5] = - gamma_p_vec
    drift2_phys_mat[:, 6, 6] = drift2_phys_mat[:, 7, 7] = drift2_phys_mat[:, 8, 8] = drift2_phys_mat[:, 9, 9] = - (gamma_i + gamma_p_vec)/2
    
    return scale2_mat * drift2_phys_mat

# drive2_vec is a (10) array
drive2_vec = np.array([
    r_zpf_i**2 * (2 * Gamma_h_i + gamma_i),
    p_zpf_i**2 * (2 * Gamma_h_i + gamma_i),
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
])

# drive2_vec is a (n_gamma, 10) array
drive2_vec = np.tile(drive2_vec, (n_gamma, 1))

drive2_vec[:, 3] = r_zpf_p**2 * (2 * Gamma_h_p + gamma_p_vec)
drive2_vec[:, 4] = p_zpf_p**2 * (2 * Gamma_h_p + gamma_p_vec + 4 * Gamma_td_p + 4 * Gamma_fb_p_vec)

drive2_vec = scale2_vec * drive2_vec



# ==============================================================================
# BLOCK 4: Timestep and Periods for Floquet Analysis
# ==============================================================================
# Resolution multipliers for the integration time steps
r = 1
r1 = 1

# Drive periods [s]
T_slow = 2 * np.pi / omega_slow
T_fast = 2 * np.pi / omega_fast

# Integration steps per slow period (T_slow)
# Ensures sufficient resolution for the fast RF drive (T_fast)
n_steps = int((T_slow / T_fast) * r1 * 10**r)

# Time step size [s] and evaluation points array
dt = T_slow / n_steps
t_eval_points = np.linspace(0, T_slow, n_steps + 1)

# Number of periods to integrate over (Floquet theory requires 1 period)
n_pers = 1



# ==============================================================================
# BLOCK 5: Floquet Solver
# ==============================================================================
def coupled_dZ_dt_single(t, Z_flat, dim, drift_func, drive_vec):
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

# Initial conditions for the Floquet simulation. These are not the physical initial conditions, those were defined in BLOCK 3.
# Initial Monodromy matrix X(0) = Identity matrix
# Initial Lambda(0) = Zero vector
X1_init_mat = np.eye(4)
Lambda1_init_vec = np.zeros(4)
# Flatten and tile for batch processing placeholders
Z1_init_flat = np.concatenate([X1_init_mat.flatten(), Lambda1_init_vec])
Z1_init_flat = np.tile(Z1_init_flat, n_gamma)

X2_init_mat = np.eye(10)
Lambda2_init_vec = np.zeros(10)
Z2_init_flat = np.concatenate([X2_init_mat.flatten(), Lambda2_init_vec])
Z2_init_flat = np.tile(Z2_init_flat, n_gamma)


# Floquet loop
print(f"Starting parameter sweep over {n_gamma} gamma_p values...")
start_time = time.time()

# Storage for trajectory histories
all_X1_histories = []
all_Lambda1_histories = []
all_X2_histories = []
all_Lambda2_histories = []

# Reshape initial conditions for iteration
Z1_init_stack = Z1_init_flat.reshape(n_gamma, -1)
Z2_init_stack = Z2_init_flat.reshape(n_gamma, -1)

for i, gamma_p in enumerate(gamma_p_vec):
    # Extract system-specific parameters for the i-th damping rate
    drift1_mat_single = lambda t: drift1_mat(t)[i]
    drive1_vec_single = drive1_vec[i]
    z1_init_single = Z1_init_stack[i]

    drift2_mat_single = lambda t: drift2_mat(t)[i]
    drive2_vec_single = drive2_vec[i]
    z2_init_single = Z2_init_stack[i]

    # Solve ODEs for the i-th damping rate
    loop_start = time.time()
    print(f"  Processing gamma_p index {i+1}/{n_gamma} ({gamma_p:.2e})...", end='')
    
    # First Moments (Dimension 4)
    solution1 = solve_ivp(
        fun=lambda t, z: coupled_dZ_dt_single(t, z, 4, drift1_mat_single, drive1_vec_single),
        t_span=[0, T_slow],
        y0=z1_init_single,
        method="DOP853", rtol=1e-12, atol=1e-12,
        t_eval=t_eval_points
    )
    
    # Second Moments (Dimension 10)
    solution2 = solve_ivp(
        fun=lambda t, z: coupled_dZ_dt_single(t, z, 10, drift2_mat_single, drive2_vec_single),
        t_span=[0, T_slow],
        y0=z2_init_single,
        method="DOP853", rtol=1e-12, atol=1e-12,
        t_eval=t_eval_points
    )
    
    print(f" Done ({time.time() - loop_start:.2f}s)")
    
    # Separate X(t) and Lambda(t) from flattened results and store
    Z1_flat_history = solution1.y.T
    all_X1_histories.append(Z1_flat_history[:, :16].reshape(len(t_eval_points), 4, 4))
    all_Lambda1_histories.append(Z1_flat_history[:, 16:])
    
    Z2_flat_history = solution2.y.T
    all_X2_histories.append(Z2_flat_history[:, :100].reshape(len(t_eval_points), 10, 10))
    all_Lambda2_histories.append(Z2_flat_history[:, 100:])

print(f"Sweep complete in {time.time() - start_time:.4f} seconds.")



# Data Aggregation
X1_mat_history = np.stack(all_X1_histories, axis=1)
Lambda1_vec_history = np.stack(all_Lambda1_histories, axis=1)
X2_mat_history = np.stack(all_X2_histories, axis=1)
Lambda2_vec_history = np.stack(all_Lambda2_histories, axis=1)



# Extract Monodromy matrices X(T) and inhomogeneous vectors Lambda(T) at the end of one period
# Index -1 corresponds to t = T_slow
X1_mat_T = X1_mat_history[-1, :, :, :]
Lambda1_vec_T = Lambda1_vec_history[-1, :, :]
I4 = np.eye(4)

X2_mat_T = X2_mat_history[-1]
Lambda2_vec_T = Lambda2_vec_history[-1]
I10 = np.eye(10)

# Compute Floquet multipliers (eigenvalues of the Monodromy matrix) to check stability
# For a stable periodic solution, the modulus of all eigenvalues must be <= 1
print("Eigenvalues of first-order Monodromy matrix:\n", 
      np.linalg.eig(X1_mat_T[0, :, :])[0])
print("Eigenvalues of second-order Monodromy matrix:\n", 
      np.linalg.eig(X2_mat_T[0, :, :])[0])

print("Modulus of eigenvalues (First Order):", 
      np.absolute(np.linalg.eig(X1_mat_T[0, :, :])[0]))
print("Modulus of eigenvalues (Second Order):", 
      np.absolute(np.linalg.eig(X2_mat_T[0, :, :])[0]))


# ==============================================================================
# BLOCK 6: Steady State Solutions
# ==============================================================================

# --- First Moments (Mean Values) ---

# Calculate the steady-state vector Y_ss at t = 0 (start of period after t = m * T_slow where m-->infinity)
# Y_ss(0) = (I - X(T))^-1 * X(T) * Lambda(T)
# We add a new axis to the Lambda1_vec_T to convert it to a (n_gamma, 4, 1) array. The matrix multiplication results in a (n_gamma, 4, 1) array which is then reshaped. 
# Result shape: (n_gamma, 4)
Y1_vec_ss = (inv(I4 - X1_mat_T) @ X1_mat_T @ Lambda1_vec_T[:, :, np.newaxis]).reshape(n_gamma, 4)

# Propagate the steady-state solution over one full time period [0, T_slow]
# Formula: Y_ss(t) = X(t) * [Y_ss(0) + Lambda(t)]
# Numpy broadcasting adds Y1_vec_ss (n_gamma, 4) to Lambda history (n_steps, n_gamma, 4) 
sum_of_vectors1 = Y1_vec_ss + Lambda1_vec_history
# We add a new axis to sum_of_vectors to convert it to a (n_steps, n_gamma, 4, 1) array and then remove the last axis after the matrix multiplication. 
# Result shape: (n_steps, n_gamma, 4)
Y1_vec_ss_history = (X1_mat_history @ sum_of_vectors1[..., np.newaxis]).squeeze(axis=-1)


# --- Second Moments (Covariance) ---

# Calculate steady-state covariance vector at t = 0
# Result shape: (n_gamma, 10)
Y2_vec_ss = (inv(I10 - X2_mat_T) @ X2_mat_T @ Lambda2_vec_T[:, :, np.newaxis]).reshape(n_gamma, 10)

# Propagate covariance solution over the full period
sum_of_vectors_2 = Y2_vec_ss + Lambda2_vec_history
Y2_vec_ss_history = (X2_mat_history @ sum_of_vectors_2[..., np.newaxis]).squeeze(axis=-1)



def Y1_vec(m, n):
    """
    Calculates the full solution state vector for the first moments (dimension 4).

    Computes Y(t) = Y_hom(t) + Y_inhom(t) for the m-th period and n-th time step (t = m * T_slow + n * dt).

    Parameters
    ----------
    m : int
        Number of full periods (T_slow) elapsed.
    n : int
        Time step index within the current period [0, n_steps].

    Returns
    -------
    ndarray
        State vector of shape (4,).
    """
    X_mat_n = X1_mat_history[n]
    Lambda_vec_n = Lambda1_vec_history[n]
    monodromy_pow_m = matrix_power(X1_mat_T, m)
    
    # Homogeneous evolution of the initial condition X10_vec
    Y_hom_vec = X_mat_n @ monodromy_pow_m @ X10_vec
    
    # Inhomogeneous evolution driving towards steady state
    # Formula: X(t) * [ (I - M^m) * Y_ss(0) + Lambda(t) ]
    Y_inhom_vec = X_mat_n @ ((I4 - monodromy_pow_m) @ Y1_vec_ss + Lambda_vec_n)
    
    return Y_hom_vec + Y_inhom_vec


def Y2_vec(m, n):
    """
    Calculates the full solution state vector for the second moments (dimension 10).

    Computes Y(t) = Y_hom(t) + Y_inhom(t) for the m-th period and n-th time step (t = m * T_slow + n * dt).

    Parameters
    ----------
    m : int
        Number of full periods (T_slow) elapsed.
    n : int
        Time step index within the current period [0, n_steps].

    Returns
    -------
    ndarray
        State vector of shape (10,).
    """
    X_mat_n = X2_mat_history[n]
    Lambda_vec_n = Lambda2_vec_history[n]
    monodromy_pow_m = matrix_power(X2_mat_T, m)
    
    Y_hom_vec = X_mat_n @ monodromy_pow_m @ X20_vec
    Y_inhom_vec = X_mat_n @ ((I10 - monodromy_pow_m) @ Y2_vec_ss + Lambda_vec_n)
    
    return Y_hom_vec + Y_inhom_vec



# Vectorized Full Solution Functions (Batch Processing)
def Y1_vec_batch(m, n):
    """
    Vectorized calculation of the first moments for all gamma_p values simultaneously.

    Parameters
    ----------
    m : int
        Number of full periods elapsed.
    n : int
        Time step index within the current period.

    Returns
    -------
    ndarray
        Batch of state vectors of shape (n_gamma, 4).
    """
    # Extract batch matrices/vectors for time step 'n'
    X_mat_n = X1_mat_history[n, :, :, :]        # Shape: (n_gamma, 4, 4)
    Lambda_vec_n = Lambda1_vec_history[n, :, :] # Shape: (n_gamma, 4)
    
    # Compute matrix power for the stack of Monodromy matrices
    monodromy_pow_m = matrix_power(X1_mat_T, m) # Shape: (n_gamma, 4, 4)
    
    # --- Homogeneous Part ---
    # Broadcasts single initial condition X10_vec across the batch
    Y_hom_batch = (X_mat_n @ monodromy_pow_m) @ X10_vec
    
    # --- Inhomogeneous Part ---
    # 1. Term: (I - M^m) * Y_ss(0)
    # Reshape Y1_vec_ss to (n_gamma, 4, 1) for matrix multiplication
    term1 = (I4 - monodromy_pow_m) @ Y1_vec_ss[:, :, np.newaxis]
    
    # 2. Add Lambda(t). Squeeze term1 to (n_gamma, 4) before addition.
    term2 = term1.squeeze(axis=-1) + Lambda_vec_n
    
    # 3. Multiply by X(t)
    Y_inhom_batch = (X_mat_n @ term2[:, :, np.newaxis]).squeeze(axis=-1)
    
    return Y_hom_batch + Y_inhom_batch


def Y2_vec_batch(m, n):
    """
    Vectorized calculation of the second moments for all gamma_p values simultaneously.
    
    Parameters
    ----------
    m : int
        Number of full periods elapsed.
    n : int
        Time step index within the current period.
    Returns
    -------
    ndarray
        Batch of state vectors of shape (n_gamma, 10).
    """
    X_mat_n = X2_mat_history[n, :, :, :]          # Shape: (n_gamma, 10, 10)
    Lambda_vec_n = Lambda2_vec_history[n, :, :]   # Shape: (n_gamma, 10)
    
    monodromy_pow_m = matrix_power(X2_mat_T, m)   # Shape: (n_gamma, 10, 10)
    
    # Homogeneous Part
    Y_hom_batch = (X_mat_n @ monodromy_pow_m) @ X20_vec
    
    # Inhomogeneous Part
    term1 = (I10 - monodromy_pow_m) @ Y2_vec_ss[:, :, np.newaxis]
    term2 = term1.squeeze(axis=-1) + Lambda_vec_n
    Y_inhom_batch = (X_mat_n @ term2[:, :, np.newaxis]).squeeze(axis=-1)
    
    return Y_hom_batch + Y_inhom_batch



# =================================================================================
# BLOCK 7: Time-dependent Purity, Energy and Steady State Purity, Population
# =================================================================================

def calculate_purity(y1_vec, y2_vec, idx_gamma):
    """
    Calculates the purity of the nanoparticle COM motion state from the covariance matrix.

    For a Gaussian state, purity is defined as Purity = 1 / sqrt(det(V)), where V is 
    the covariance matrix of the dimensionless quadrature operators (q, p).
    
    Parameters
    ----------
    y1_vec : ndarray
        First-order moments (means) vector.
    y2_vec : ndarray
        Second-order moments (covariance) vector.
    idx_gamma : int
        Index of the current damping rate in the gamma_p parameter sweep.

    Returns
    -------
    float
        The purity of the state (Scalar between 0 and 1, where 1 is a pure state).
    """
    # Extract moments for the nanoparticle (indices 2, 3, 4, 5 in the specific flattening scheme)
    # The simulation vector 'y1_vec' is already scaled by 'bp', 'cp'. We undo this scaling.
    # We normalize explicitly by ZPF to ensure variables refer to dimensionless operators:
    # q = deltaR / r_zpf  and  p = P / p_zpf
    
    # First moments (Means) <q> and <p>
    mean_q_p = y1_vec[idx_gamma, 2] / (bp * r_zpf_p)
    mean_p_p = y1_vec[idx_gamma, 3] / (cp * p_zpf_p)
    
    # Second moments <q^2>, <p^2>, and <{q,p}/2>
    mean_q2_p = y2_vec[idx_gamma, 3] / ((bp * r_zpf_p)**2)
    mean_p2_p = y2_vec[idx_gamma, 4] / ((cp * p_zpf_p)**2)
    mean_qp_p = y2_vec[idx_gamma, 5] / ((bp * r_zpf_p) * (cp * p_zpf_p))
    
    # Calculate Variances and Covariance of the dimensionless variables
    # Var(A) = <A^2> - <A>^2
    var_q_p = mean_q2_p - (mean_q_p**2)
    var_p_p = mean_p2_p - (mean_p_p**2)
    cov_qp_p = mean_qp_p - (mean_q_p * mean_p_p)
    
    # Calculate the determinant of the 2x2 dimensionless covariance matrix
    # Det(V) = Var(q)*Var(p) - Cov(q,p)^2
    det_V_dimless = (var_q_p * var_p_p) - (cov_qp_p**2)
    
    return 1.0 / np.sqrt(det_V_dimless)


def calculate_energy(y1_vec, y2_vec, t, idx_gamma):
    """
    Calculates the instantaneous mean mechanical energies in SI units [Joules].

    Reverses the numerical scaling to recover physical quantities <q^2> and <p^2>.

    Parameters
    ----------
    y1_vec : ndarray
        First-order moments vector (unused in variance calculation but kept for interface consistency).
    y2_vec : ndarray
        Second-order moments vector.
    t : float
        Current time [s] (required for time-dependent potential W_p(t)).
    idx_gamma : int
        Index of the current damping rate.

    Returns
    -------
    ndarray
        Array of shape (3,) containing [Kinetic Energy, Potential Energy, Total Energy].
    """
    # Recover physical moments in SI units [m^2] and [kg^2 m^2/s^2]
    # Scaling factors bi, ci, bp, cp are inverted here (e.g., divided by bp^2)
    mean_X2_phys = y2_vec[idx_gamma, 3] / (bp**2)
    mean_P2_phys = y2_vec[idx_gamma, 4] / (cp**2)
    # mean_XP_phys = y2_vec[idx_gamma, 5] / (bp * cp) # Cross term (unused for energy)
    
    # Calculate Energies
    # Kinetic: E_k = <P^2> / 2m
    E_kin = mean_P2_phys / (2 * M_p)
    
    # Potential: E_p = 1/2 * m * Omega(t)^2 * <deltaR^2> -> using W_p(t) curvature function
    E_pot = (M_p * W_p(t) * mean_X2_phys) / 2
    
    # Total Energy
    E_tot = E_kin + E_pot
    
    return np.array([E_kin, E_pot, E_tot])


def compute_energy_trajectory(t_list, y1_history, y2_history, idx_gamma=0):
    """
    Computes the raw kinetic and potential energy trajectories in Joules.
    
    This function isolates the heavy calculation loop from the plotting logic.

    Parameters
    ----------
    t_list : ndarray
        Array of time evaluation points.
    y1_history : ndarray
        History of first-order moments.
    y2_history : ndarray
        History of second-order moments.
    idx_gamma : int, optional
        Index of the damping rate to analyze.

    Returns
    -------
    tuple (ndarray, ndarray)
        (potential_energy_joules, kinetic_energy_joules)
    """
    potential_history = []
    kinetic_history = []

    # Loop through time steps to reconstruct energy from moments
    for idx_t, t_val in enumerate(t_list):
        # calculate_energy returns [Kinetic, Potential, Total] in Joules
        energy_vals = calculate_energy(y1_history[idx_t], y2_history[idx_t], t_val, idx_gamma)
        
        # Store raw values (Joules)
        kinetic_history.append(energy_vals[0])
        potential_history.append(energy_vals[1])

    return np.array(potential_history), np.array(kinetic_history)



def compute_steady_state_metrics(y1_ss_vec, y2_ss_vec, n_gamma):
    """
    Computes purity and phonon number for all damping rates in the steady state.

    Utilizes the pre-calculated steady-state vectors Y_ss_vec.

    Parameters
    ----------
    y1_ss_vec : ndarray
        Steady-state first moments (n_gamma, 4).
    y2_ss_vec : ndarray
        Steady-state second moments (n_gamma, 10).
    n_gamma : int
        Number of damping points.

    Returns
    -------
    tuple
        (purity_array, population_array)
    """
    # Vectorize the purity calculation function to apply it over the gamma index array
    vectorized_purity_func = np.vectorize(
        lambda i: calculate_purity(y1_ss_vec, y2_ss_vec, i)
    )
    
    idx_gamma_list = np.arange(n_gamma)
    purity_ss_micro = vectorized_purity_func(idx_gamma_list)

    # Compute Phonon Number (Population) from Purity
    # Relation for Gaussian states: n = 1/2 * (1/Purity - 1)
    population_ss_micro = (np.divide(1.0, purity_ss_micro) - 1.0) * 0.5
    
    return purity_ss_micro, population_ss_micro


# ==============================================================================
# BLOCK 8: Plotting and Visualization
# ==============================================================================
def plot_steady_state_energy(t_list, potential_joules, kinetic_joules):
    """
    Plots the steady-state energy using pre-calculated energy vectors.

    Handles unit conversion (J -> eV) and visualization scaling (Kinetic x50).

    Parameters
    ----------
    t_list : ndarray
        Time points.
    potential_joules : ndarray
        Potential energy array in Joules.
    kinetic_joules : ndarray
        Kinetic energy array in Joules.
    """
    # Unit Conversion: Joules -> eV
    pot_ev = potential_joules / 1.6e-19
    kin_ev = kinetic_joules / 1.6e-19
    
    # Normalize time
    t_normalized = t_list / T_slow

    # --- Figure Setup ---
    fig1, ax1 = plt.subplots(figsize=(12, 7))

    # --- Plotting ---
    # Note: Kinetic energy is multiplied by 50 here for visualization contrast
    line1, = ax1.plot(t_normalized, pot_ev, color='gray', label='Potential Energy')
    line2, = ax1.plot(t_normalized, kin_ev * 50, color='black', label='Kinetic Energy (x50)')

    # --- Formatting ---
    ax1.set_title('Steady-State Energy of the Nanoparticle', fontsize=16)
    ax1.set_xlabel(r'$t / T_{\rm slow}$', fontsize=12)
    ax1.set_ylabel('Energy [eV]', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    ax1.set_xlim(-0.01, 1.01)
    ax1.grid(True, linestyle='-', linewidth=2, alpha=0.6)

    # Inward ticks and box style
    ax1.tick_params(axis='both', which='major', direction='in', 
                    top=True, right=True, labelsize=12)

    # Thicken spines
    for spine in ax1.spines.values():
        spine.set_linewidth(2)

    ax1.legend(handles=[line1, line2], loc='upper center', fontsize=12)
    fig1.tight_layout()

    filename = 'energy_vs_time_micromotion.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Full energy plot saved to {filename}")
    # plt.show()


def plot_zoomed_energy(t_list, kinetic_history, slice_start=5000, slice_end=5050):
    """
    Generates a zoomed-in view of the kinetic energy to visualize micromotion.

    Parameters
    ----------
    t_list : ndarray
        Array of time evaluation points.
    kinetic_history : list or ndarray
        History of kinetic energy values (in Joules).
    slice_start : int
        Start index for the zoom window.
    slice_end : int
        End index for the zoom window.
    """
    # Convert slice to eV
    # Note: kinetic_history is assumed to be raw Joules here. 
    # If passing the previously scaled list (x50), adjust divisor accordingly.
    # Here we assume raw input for safety.
    kin_energy_ev = np.array(kinetic_history[slice_start:slice_end]) / 1.6e-19
    t_zoomed = t_list[slice_start:slice_end] / T_slow

    # --- Figure Setup ---
    fig2, ax2 = plt.subplots(figsize=(27, 9))

    # --- Plotting ---
    line_kin, = ax2.plot(t_zoomed, kin_energy_ev, 
                         color='black', linewidth=6, label='Kinetic Energy')

    # --- Formatting ---
    ax2.set_title('Steady-State Energy (Micromotion Zoom)', fontsize=16)
    ax2.set_xlabel(r'$t / T_{\rm slow}$', fontsize=12)
    ax2.set_ylabel('Energy [eV]', fontsize=12, color='blue') # Label kept blue per request
    ax2.tick_params(axis='y', labelcolor='black')

    # --- Aesthetic Customization ---
    # Inward ticks, top and right spines included
    ax2.tick_params(axis='both', which='major', direction='in', 
                    top=True, right=True, length=8, width=4, labelsize=12)

    # Thicken frame spines
    for spine in ax2.spines.values():
        spine.set_linewidth(6)

    fig2.tight_layout()

    # --- Saving ---
    filename = 'energy_vs_time_zoomed.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Zoomed plot saved to {filename}")
    # plt.show()

def plot_metrics_vs_damping(gamma_vec, purity_micro, pop_micro, 
                            gamma_sec_vec, purity_sec, pop_sec):
    """
    Plots Purity and Phonon Number against Damping Rate (Gamma).

    Phonon Number is on the left Y-axis and Purity is on the right Y-axis.
    
    Compares the full Floquet solution (with micromotion) against the 
    Secular approximation. Uses a dual y-axis (TwinX).

    Parameters
    ----------
    gamma_vec : ndarray
        Damping rates for the Floquet simulation [rad/s].
    purity_micro, pop_micro : ndarray
        Results from the Floquet simulation.
    gamma_sec_vec : ndarray
        Damping rates for the Secular approximation [rad/s].
    purity_sec, pop_sec : ndarray
        Pre-computed results from the Secular approximation.
    """
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    ax4 = ax3.twinx() # Create second y-axis sharing the same x-axis

    # Convert gamma to Hz for plotting: Gamma / (2*pi)
    x_freq_micro = gamma_vec / (2 * np.pi)
    x_freq_sec = gamma_sec_vec / (2 * np.pi)

    # --- Plotting (Micromotion) ---
    # Purity on Right Axis (Red)
    l_pur_micro, = ax4.loglog(x_freq_micro, purity_micro, color='red', linestyle=':', 
                              linewidth=3, marker="^", markersize=10, 
                              label='Purity (Floquet)')
    
    # Population on Left Axis (Black)
    l_pop_micro, = ax3.loglog(x_freq_micro, pop_micro, color='black', linestyle=':', 
                              linewidth=3, marker="^", markersize=10, 
                              label='Population (Floquet)')

    # --- Plotting (Secular Approximation) ---
    # Purity on Right Axis (Red solid)
    l_pur_sec, = ax4.loglog(x_freq_sec, purity_sec, color='red', linestyle='-', 
                            linewidth=3, label='Purity (Secular)')
    
    # Population on Left Axis (Black solid)
    l_pop_sec, = ax3.loglog(x_freq_sec, pop_sec, color='black', linestyle='-', 
                            linewidth=3, label='Population (Secular)')


    # --- Formatting ---
    # Right Axis (Purity)
    ax4.set_title('Steady-State Metrics vs. Damping', fontsize=16)
    ax4.set_ylabel('Purity', fontsize=12, color='red')
    ax4.tick_params(axis='y', labelcolor='red')

    # Left Axis (Population)
    ax3.set_xlabel('Nanoparticle Damping $\gamma_p / 2\pi$ [Hz]', fontsize=12)
    ax3.set_ylabel('Phonon number $\\bar{n}$', fontsize=12)

    # Grid and Ticks
    ax3.grid(True, linestyle='-', linewidth=2, alpha=0.6)
    
    # Ax3 Ticks (Left)
    ax3.tick_params(axis='both', which='major', direction='in', 
                    top=True, right=False, width=2, labelsize=12)
    # Ax4 Ticks (Right)
    ax4.tick_params(axis='both', which='major', direction='in', 
                    top=True, right=True, width=2, labelsize=12)

    # Spines
    for spine in ax3.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')

    # --- Unified Legend ---
    # Collect handles and labels from both axes
    handles = [l_pop_micro, l_pop_sec, l_pur_micro, l_pur_sec]
    labels = [h.get_label() for h in handles]
    
    # Place legend (using ax3 coordinates)
    ax3.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=2, fontsize=12)

    fig3.tight_layout()

    # --- Saving ---
    filename = 'purity_population_vs_damping.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {filename}")
    # plt.show()




# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == "__main__":
    print("--- Starting Simulation Analysis ---")

    # ---------------------------------------------------------
    # 1. Data Preparation
    # ---------------------------------------------------------
    print("Calculating energy trajectories (gamma_p index 0)...")
    idx_gamma_plot = 0
    
    # Calculate raw energies ONCE
    pot_J, kin_J = compute_energy_trajectory(
        t_eval_points, 
        Y1_vec_ss_history, 
        Y2_vec_ss_history, 
        idx_gamma=idx_gamma_plot
    )
    
    print("Computing steady-state metrics...")
    purity_ss, population_ss = compute_steady_state_metrics(
        Y1_vec_ss, 
        Y2_vec_ss, 
        n_gamma
    )

    # ---------------------------------------------------------
    # 2. Plotting
    # ---------------------------------------------------------
    
    # Figure 1: Full Energy (Passes raw Joules; function handles x50 scaling)
    print("Generating Figure 1: Full Steady-State Energy...")
    plot_steady_state_energy(t_eval_points, pot_J, kin_J)

    # Figure 2: Zoomed Energy (Passes raw Joules; function plots as-is)
    print("Generating Figure 2: Zoomed Kinetic Energy...")
    plot_zoomed_energy(t_eval_points, kin_J, slice_start=5000, slice_end=5050)

    # Figure 3: Metrics vs Damping
    print("Generating Figure 3: Metrics vs Damping Comparison...")
    plot_metrics_vs_damping(
        gamma_p_vec, purity_ss, population_ss,
        gamma_p_vec_secular, purity_r_ss_secular_small, population_r_ss_secular_small
    )

    print("\n--- Execution Complete ---")
    plt.show()