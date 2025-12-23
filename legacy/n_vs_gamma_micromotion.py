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

V_comp = 0.68

dx = 0.9 * 10**(-3)
alpha_x = 0.93
V_0_x = -0.5 * (alpha_z/alpha_x) * (dx/dz)**2 * V_0_z + V_comp
V_slow_x, V_fast_x = 80, 1350

dy = 0.9 * 10**(-3)
alpha_y = 0.93
V_0_y =  -0.5 * (alpha_z/alpha_y) * (dx/dz)**2 * V_0_z - V_comp
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
Gamma_to_gamma_x = 719.84
Gamma_to_gamma_y = 782.99
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
gamma_p_vec = 2 * np.pi * np.insert(np.logspace(-7, 3, num=21), 0, 44.5e-9)

# This is the heating rate of the nanoparticle from background gas scattering
Gamma_h_p = 11.5 * 10**(-28) / (hbar * new_Omega_r_p)
# This is the trap displacement noise heating rate of the nanoparticle. The commented value is the typical value observed in experiments.
Gamma_td_p = 0 #2.8 * 10**(-26)/(hbar * new_Omega_r_p)
# This is the back action noise heating rate of the nanoparticle
Gamma_fb_p_vec = Gamma_to_gamma_r * gamma_p_vec


n_gamma = len(gamma_p_vec)


# ==============================================================================
# BLOCK 2: Values of occupation number computed without micromotion
# ==============================================================================
# This is an array of the damping rates of the nanoparticle over which the following occupation values were computed
gamma_p_vec_secular = np.insert(2 * np.pi * 10**np.arange(-7, 3 + (1/60), 1/30.), 0, 2 * np.pi * 44.5e-9)



# These are the arrays of nanoparticle COM motion occupation in different directions computed in the absence of micromotion. These values can be obtained from the expression of the steady state occupation number, or numerically, for instance, using the n_vs_gamma.nb Mathematica notebook.
# These are the occupation numbers in the z-axis with typical trap displacement heating rates in experiments
population_z_ss_secular_large = np.array([
4.6411e8, 4.62302e8, 4.62044e8, 4.61766e8, 4.61467e8, 
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
population_z_ss_secular_small = np.array([
1.83137e7, 1.82423e7, 1.82322e7, 1.82212e7, 1.82094e7, 
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
population_r_ss_secular_large = np.array([
7.40399e10, 4.28297e10, 4.03933e10, 3.80558e10, 
3.58177e10, 3.3679e10, 3.16391e10, 2.96969e10, 
2.78509e10, 2.60991e10, 2.44392e10, 2.28688e10, 
2.1385e10, 1.99848e10, 1.86653e10, 1.74231e10, 
1.6255e10, 1.51578e10, 1.4128e10, 1.31624e10, 
1.22579e10, 1.14111e10, 1.0619e10, 9.8786e9, 
9.18694e9, 8.54122e9, 7.93871e9, 7.37683e9, 6.85309e9,
6.36513e9, 5.9107e9, 5.48765e9, 5.09398e9, 4.72776e9,
4.3872e9, 4.07058e9, 3.77631e9, 3.50288e9, 3.24887e9,
3.01296e9, 2.7939e9, 2.59053e9, 2.40175e9, 2.22656e9,
2.06399e9, 1.91316e9, 1.77324e9, 1.64346e9, 
1.52309e9, 1.41147e9, 1.30796e9, 1.212e9, 1.12303e9, 
1.04055e9, 9.64092e8, 8.93226e8, 8.27545e8, 7.66672e8,
7.10258e8, 6.5798e8, 6.09536e8, 5.64648e8, 5.23055e8,
4.84518e8, 4.48813e8, 4.15733e8, 3.85085e8, 
3.56692e8, 3.30389e8, 3.06022e8, 2.83449e8, 2.62539e8,
2.43169e8, 2.25227e8, 2.08607e8, 1.93211e8, 
1.78951e8, 1.65743e8, 1.53508e8, 1.42176e8, 1.3168e8, 
1.21958e8, 1.12953e8, 1.04613e8, 9.68887e7, 8.97341e7,
8.31076e7, 7.69703e7, 7.1286e7, 6.60213e7, 6.11453e7,
5.66293e7, 5.24468e7, 4.85731e7, 4.49854e7, 
4.16627e7, 3.85853e7, 3.57352e7, 3.30956e7, 3.06509e7,
2.83868e7, 2.62899e7, 2.43479e7, 2.25494e7, 
2.08836e7, 1.93409e7, 1.79122e7, 1.6589e7, 1.53636e7, 
1.42286e7, 1.31775e7, 1.22041e7, 1.13025e7, 1.04676e7,
9.69433e6, 8.97819e6, 8.31495e6, 7.70071e6, 
7.13185e6, 6.60501e6, 6.1171e6, 5.66523e6, 5.24674e6, 
4.85917e6, 4.50023e6, 4.16781e6, 3.85994e6, 3.57482e6,
3.31077e6, 3.06622e6, 2.83974e6, 2.63e6, 2.43574e6, 
2.25584e6, 2.08923e6, 1.93493e6, 1.79203e6, 1.65969e6,
1.53712e6, 1.42361e6, 1.31849e6, 1.22113e6, 1.13097e6, 
1.04746e6, 970129., 898508., 832179., 770750., 713859., 661172., 
612377., 567187., 525336., 486577., 450681., 417437., 386650., 
358137., 331730., 307275., 284626., 263651., 244225., 226235., 
209574., 194143., 179853., 166618., 154362., 143010., 132498., 
122762., 113745., 105395., 97661.2, 90499.1, 83866.1, 77723.1, 
72034., 66765.3, 61885.7, 57366.7, 53181.6, 49305.6, 45716.1, 
42391.7, 39312.9, 36461.6, 33821., 31375.4, 29110.5, 27013., 25070.4, 
23271.4, 21605.2, 20062.2, 18633.2, 17309.7, 16084., 14948.9, 
13897.6, 12924.1, 12022.4, 11187.3, 10414., 9697.78, 9034.48, 
8420.18, 7851.27, 7324.39, 6836.44, 6384.54, 5966.03, 5578.43, 
5219.47, 4887.04, 4579.16, 4294.03, 4029.96, 3785.41, 3558.92, 
3349.17, 3154.91, 2975., 2808.39, 2654.09, 2511.18, 2378.84, 2256.27, 
2142.76, 2037.63, 1940.27, 1850.11, 1766.6, 1689.27, 1617.64, 
1551.31, 1489.88, 1432.99, 1380.31, 1331.51, 1286.32, 1244.47, 
1205.71, 1169.81, 1136.57, 1105.78, 1077.27, 1050.86, 1026.41, 
1003.76, 982.783, 963.357, 945.366, 928.705, 913.275, 898.984, 
885.75, 873.493, 862.142, 851.629, 841.893, 832.877, 824.526, 
816.793, 809.63, 802.997, 796.854, 791.165, 785.897, 781.017, 
776.498, 772.313, 768.437, 764.847, 761.523, 758.444, 755.593, 
752.952, 750.507, 748.242, 746.144, 744.202, 742.403, 740.736, 
739.193, 737.764, 736.441, 735.215, 734.08, 733.029, 732.055, 
731.153, 730.318, 729.545, 728.828, 728.165, 727.551, 726.982, 
726.454, 725.966, 725.514])

# These are the occupation numbers in the radial direction with no trap displacement heating
population_r_ss_secular_small = np.array([
2.92334e9, 1.69106e9, 1.59486e9, 1.50257e9, 1.4142e9, 
1.32976e9, 1.24922e9, 1.17253e9, 1.09965e9, 1.03048e9,
9.64942e8, 9.02935e8, 8.4435e8, 7.89068e8, 7.36968e8,
6.87922e8, 6.41803e8, 5.98479e8, 5.5782e8, 5.19697e8,
4.83982e8, 4.50548e8, 4.19274e8, 3.90041e8, 
3.62732e8, 3.37236e8, 3.13447e8, 2.91262e8, 2.70583e8,
2.51317e8, 2.33375e8, 2.16671e8, 2.01128e8, 
1.86668e8, 1.73222e8, 1.60721e8, 1.49102e8, 1.38306e8,
1.28277e8, 1.18962e8, 1.10313e8, 1.02283e8, 
9.48298e7, 8.79126e7, 8.14939e7, 7.55386e7, 7.00142e7,
6.48899e7, 6.01374e7, 5.57301e7, 5.16434e7, 
4.78543e7, 4.43415e7, 4.10849e7, 3.80662e7, 3.52682e7,
3.26749e7, 3.02714e7, 2.8044e7, 2.59799e7, 2.40672e7,
2.22949e7, 2.06526e7, 1.91311e7, 1.77213e7, 
1.64152e7, 1.52051e7, 1.40841e7, 1.30455e7, 1.20835e7,
1.11922e7, 1.03666e7, 9.60182e6, 8.89339e6, 
8.23717e6, 7.62932e6, 7.06628e6, 6.54476e6, 6.0617e6, 
5.61427e6, 5.19985e6, 4.816e6, 4.46047e6, 4.13117e6, 
3.82618e6, 3.54369e6, 3.28206e6, 3.03973e6, 2.8153e6, 
2.60743e6, 2.41491e6, 2.23661e6, 2.07146e6, 1.91852e6,
1.77686e6, 1.64567e6, 1.52417e6, 1.41164e6, 
1.30742e6, 1.21089e6, 1.1215e6, 
1.03871e6, 962028., 891015., 825246., 764336., 707925., 655681., 
607295., 562484., 520984., 482549., 446953., 413987., 383456., 
355180., 328993., 304741., 282280., 261479., 242215., 224373., 
207850., 192547., 178375., 165250., 153095., 141837., 131412., 
121756., 112814., 104532., 96862.7, 89759.6, 83181.3, 77089., 
71446.8, 66221.5, 61382.2, 56900.4, 52749.8, 48905.7, 45345.7, 
42048.7, 38995.3, 36167.5, 33548.6, 31123.2, 28876.9, 26796.7, 
24870.1, 23085.8, 21433.4, 19903.1, 18485.8, 17173.2, 15957.6, 
14831.9, 13789.2, 12823.7, 11929.4, 11101.2, 10334.3, 9623.93, 
8966.09, 8356.84, 7792.61, 7270.07, 6786.13, 6337.95, 5922.88, 
5538.47, 5182.46, 4852.76, 4547.42, 4264.63, 4002.74, 3760.2, 
3535.57, 3327.54, 3134.88, 2956.46, 2791.21, 2638.18, 2496.45, 
2365.19, 2243.63, 2131.05, 2026.79, 1930.23, 1840.81, 1757.99, 
1681.29, 1610.26, 1544.48, 1483.55, 1427.13, 1374.87, 1326.48, 
1281.66, 1240.15, 1201.71, 1166.11, 1133.14, 1102.61, 1074.33, 
1048.14, 1023.89, 1001.42, 980.62, 961.354, 943.512, 926.987, 
911.684, 897.511, 884.385, 872.229, 860.971, 850.545, 840.889, 
831.947, 823.665, 815.995, 808.892, 802.314, 796.221, 790.579, 
785.353, 780.514, 776.032, 771.881, 768.037, 764.477, 761.18, 
758.127, 755.299, 752.68, 750.255, 748.008, 745.928, 744.001, 
742.217, 740.565, 739.034, 737.617, 736.305, 735.089, 733.963, 
732.921, 731.955, 731.061, 730.233, 729.466, 728.755, 728.097, 
727.488, 726.924, 726.401, 725.917, 725.469, 725.054, 724.67, 
724.314, 723.984, 723.679, 723.396, 723.134, 722.891, 722.667, 
722.459, 722.266, 722.088, 721.922, 721.769, 721.628, 721.496, 
721.375, 721.262, 721.158, 721.061, 720.972, 720.889, 720.812, 
720.741, 720.676, 720.615, 720.558, 720.506, 720.458, 720.413, 
720.371, 720.333, 720.297, 720.264, 720.233, 720.205, 720.179, 
720.154, 720.132, 720.111, 720.091, 720.073])



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