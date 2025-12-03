import numpy as np
from scipy.constants import pi, hbar, epsilon_0, Boltzmann

from dataclasses import dataclass, field

@dataclass
class BaseSystemParams:
    """
    Ion/nanoparticle properties and trap parameters common to ALL simulations.
    Parameters in the simulations scripts are inherited from this dataclass.
    """
    # Physical conditions
    Temp: float = 300
    Pres: float = 1e-8
    m0: float = 4.65e-26
    
    # Ion parameters
    Q_i: float = 1.6e-19
    M_i: float = 40 * 1.6e-27

    # Nanoparticle parameters
    Q_p: float = 750 * 1.6e-19
    M_p: float = 2e-17
    R_p: float = field(init=False)
    rho_p: float = 2000
    
    # Trap parameters
    dx: float = 0.9e-3
    dy: float = 0.9e-3
    dz: float = 1.7e-3

    alpha_x: float = 0.93
    alpha_y: float = 0.93
    alpha_z: float = 0.38

    V_0_z: float = 56.5
    V_slow_z: float = 0
    V_fast_z: float = 0

    V_0_x: float = field(init=False)
    V_slow_x: float = 80
    V_fast_x: float = 1350

    V_0_y: float = field(init=False)
    V_slow_y: float = 80
    V_fast_y: float = 1350

    omega_slow: float = 2*pi*7e3
    omega_fast: float = 2*pi*17.5e6
    l: float = field(init=False)    

    # Ion and nanoparticle dissipation
    gamma_gas: float = field(init=False)
    E_dot_gas: float = field(init=False)
    E_dot_td: float = 0 #2.8 * 10**(-26)

    gamma_dop: float = 2*pi*10e3
    E_dot_dop: float = 3.8e-22

    Gamma_ba_to_gamma_fb: np.ndarray = field(
    default_factory=lambda: np.array([561.39, 750.09, 842.08])
    )

    def __post_init__(self):
        self.R_p = np.cbrt((self.M_p/self.rho_p) * 3 / (4*pi))

        self.gamma_gas = 0.619 * ((6 * pi * self.R_p**2) / self.M_p) * self.Pres * np.sqrt(2 * self.m0 / (pi * Boltzmann * self.Temp))
        self.E_dot_gas = 11.5e-28 #(Boltzmann * self.Temp) * self.gamma_gas
        
        self.V_0_x = 0.5 * (self.alpha_z/self.alpha_x) * (self.dx/self.dz)**2 * self.V_0_z
        self.V_0_y = -0.5 * (self.alpha_z/self.alpha_x) * (self.dx/self.dz)**2 * self.V_0_z
        
        self.l = self.omega_slow / self.omega_fast  # Dimensionless ratio
        self.T_slow = (2 * pi) / self.omega_slow    # Time period of slow voltage
        self.T_fast = (2 * pi) / self.omega_fast    # Time period of fast voltage

        # Mathieu parameters (a, q)
        # Dimensionless parameters characterising the dynamics in the Paul trap.
        # Ion Mathieu parameters
        self.a_x_i = (4 * self.Q_i * self.V_0_x * self.alpha_x) / (self.M_i * self.dx**2 * self.omega_fast**2)
        self.q_slow_x_i = -(2 * self.Q_i * self.V_slow_x * self.alpha_x) / (self.M_i * self.dx**2 * self.omega_slow**2)
        self.q_fast_x_i = -(2 * self.Q_i * self.V_fast_x * self.alpha_x) / (self.M_i * self.dx**2 * self.omega_fast**2)

        self.a_y_i = (4 * self.Q_i * self.V_0_y * self.alpha_y) / (self.M_i * self.dy**2 * self.omega_fast**2)
        self.q_slow_y_i = -(2 * self.Q_i * self.V_slow_y * self.alpha_y) / (self.M_i * self.dy**2 * self.omega_slow**2)
        self.q_fast_y_i = -(2 * self.Q_i * self.V_fast_y * self.alpha_y) / (self.M_i * self.dy**2 * self.omega_fast**2)

        self.a_z_i = (4 * self.Q_i * self.V_0_z * self.alpha_z) / (self.M_i * self.dz**2 * self.omega_fast**2)
        self.q_slow_z_i = -(2 * self.Q_i * self.V_slow_z * self.alpha_z) / (self.M_i * self.dz**2 * self.omega_slow**2)
        self.q_fast_z_i = -(2 * self.Q_i * self.V_fast_z * self.alpha_z) / (self.M_i * self.dz**2 * self.omega_fast**2)
        
        # Nanoparticle Mathieu parameters
        self.a_x_p = (4 * self.Q_p * self.V_0_x * self.alpha_x) / (self.M_p * self.dx**2 * self.omega_fast**2)
        self.q_slow_x_p = -(2 * self.Q_p * self.V_slow_x * self.alpha_x) / (self.M_p * self.dx**2 * self.omega_slow**2)
        self.q_fast_x_p = -(2 * self.Q_p * self.V_fast_x * self.alpha_x) / (self.M_p * self.dx**2 * self.omega_fast**2)

        self.a_y_p = (4 * self.Q_p * self.V_0_y * self.alpha_y) / (self.M_p * self.dy**2 * self.omega_fast**2)
        self.q_slow_y_p = -(2 * self.Q_p * self.V_slow_y * self.alpha_y) / (self.M_p * self.dy**2 * self.omega_slow**2)
        self.q_fast_y_p = -(2 * self.Q_p * self.V_fast_y * self.alpha_y) / (self.M_p * self.dy**2 * self.omega_fast**2)

        self.a_z_p = (4 * self.Q_p * self.V_0_z * self.alpha_z) / (self.M_p * self.dz**2 * self.omega_fast**2)
        self.q_slow_z_p = -(2 * self.Q_p * self.V_slow_z * self.alpha_z) / (self.M_p * self.dz**2 * self.omega_slow**2)
        self.q_fast_z_p = -(2 * self.Q_p * self.V_fast_z * self.alpha_z) / (self.M_p * self.dz**2 * self.omega_fast**2)

        # Effective secular trap frequencies
        # Computed using the modified Lindstedt-Poincare method for the two-tone Mathieu equation.
        # Ion secular frequencies
        self.Omega_x_i = (1/np.sqrt(2)) * np.sqrt(((self.omega_fast/2)*np.sqrt(self.a_x_i + (self.q_fast_x_i**2)/2))**2 + (self.omega_slow/2)**2 + np.sqrt((((self.omega_fast/2)*np.sqrt(self.a_x_i + (self.q_fast_x_i**2)/2))**2 - (self.omega_slow/2)**2)**2 - (self.q_slow_x_i**2 * self.omega_slow**4/8)))
        self.Omega_y_i = (1/np.sqrt(2)) * np.sqrt(((self.omega_fast/2)*np.sqrt(self.a_y_i + (self.q_fast_y_i**2)/2))**2 + (self.omega_slow/2)**2 + np.sqrt((((self.omega_fast/2)*np.sqrt(self.a_y_i + (self.q_fast_y_i**2)/2))**2 - (self.omega_slow/2)**2)**2 - (self.q_slow_y_i**2 * self.omega_slow**4/8)))
        self.Omega_z_i = (1/np.sqrt(2)) * np.sqrt(((self.omega_fast/2)*np.sqrt(self.a_z_i + (self.q_fast_z_i**2)/2))**2 + (self.omega_slow/2)**2 + np.sqrt((((self.omega_fast/2)*np.sqrt(self.a_z_i + (self.q_fast_z_i**2)/2))**2 - (self.omega_slow/2)**2)**2 - (self.q_slow_z_i**2 * self.omega_slow**4/8)))

        # Nanoparticle secular frequencies
        self.Omega_x_p = (self.omega_fast/2) * np.sqrt(self.a_x_p + (self.q_slow_x_p * self.l)**2/2 + self.q_fast_x_p**2/2)
        self.Omega_y_p = (self.omega_fast/2) * np.sqrt(self.a_y_p + (self.q_slow_y_p * self.l)**2/2 + self.q_fast_y_p**2/2)
        self.Omega_z_p = (self.omega_fast/2) * np.sqrt(self.a_z_p + (self.q_slow_z_p * self.l)**2/2 + self.q_fast_z_p**2/2)

        # Distance between ion and nanoparticle along the trap axis
        self.dist = np.cbrt(((self.Q_i * self.Q_p)/(4 * pi * epsilon_0)) * ((1/(self.M_i * self.Omega_z_i**2)) + (1/(self.M_p * self.Omega_z_p**2))))

        # Renormalised effective secular frequencies of the ion and the nanoparticle
        self.new_Omega_x_i = np.sqrt(self.Omega_x_i**2 - ((self.Q_i * self.Q_p)/(4 * pi * epsilon_0 * self.M_i * self.dist**3)))
        self.new_Omega_y_i = np.sqrt(self.Omega_y_i**2 - ((self.Q_i * self.Q_p)/(4 * pi * epsilon_0 * self.M_i * self.dist**3)))
        self.new_Omega_z_i = np.sqrt(self.Omega_z_i**2 + ((self.Q_i * self.Q_p)/(2 * pi * epsilon_0 * self.M_i * self.dist**3)))
        
        self.new_Omega_x_p = np.sqrt(self.Omega_x_p**2 - ((self.Q_i * self.Q_p)/(4 * pi * epsilon_0 * self.M_p * self.dist**3)))
        self.new_Omega_y_p = np.sqrt(self.Omega_y_p**2 - ((self.Q_i * self.Q_p)/(4 * pi * epsilon_0 * self.M_p * self.dist**3)))
        self.new_Omega_z_p = np.sqrt(self.Omega_z_p**2 + ((self.Q_i * self.Q_p)/(2 * pi * epsilon_0 * self.M_p * self.dist**3)))

        # Zero-point fluctuations in the position and momentum of the ion and the nanoparticle 
        # Computed from the renormalised frequencies of the ion and nanoparticle.
        self.x_zpf_i = np.sqrt(hbar / (2 * self.M_i * self.new_Omega_x_i))
        self.y_zpf_i = np.sqrt(hbar / (2 * self.M_i * self.new_Omega_y_i))
        self.z_zpf_i = np.sqrt(hbar / (2 * self.M_i * self.new_Omega_z_i))
        
        self.px_zpf_i = np.sqrt(hbar * self.M_i * self.new_Omega_x_i / 2)
        self.py_zpf_i = np.sqrt(hbar * self.M_i * self.new_Omega_y_i / 2)
        self.pz_zpf_i = np.sqrt(hbar * self.M_i * self.new_Omega_z_i / 2)

        self.x_zpf_p = np.sqrt(hbar / (2 * self.M_p * self.new_Omega_x_p))
        self.y_zpf_p = np.sqrt(hbar / (2 * self.M_p * self.new_Omega_y_p))
        self.z_zpf_p = np.sqrt(hbar / (2 * self.M_p * self.new_Omega_z_p))
        
        self.px_zpf_p = np.sqrt(hbar * self.M_p * self.new_Omega_x_p / 2)
        self.py_zpf_p = np.sqrt(hbar * self.M_p * self.new_Omega_y_p / 2)
        self.pz_zpf_p = np.sqrt(hbar * self.M_p * self.new_Omega_z_p / 2)

        # Coupling rate between ion and nanoparticle
        self.g_x = (1/hbar) * ((self.Q_i * self.Q_p)/(4 * pi * epsilon_0)) * ((self.x_zpf_i * self.x_zpf_p)/self.dist**3)
        self.g_y = (1/hbar) * ((self.Q_i * self.Q_p)/(4 * pi * epsilon_0)) * ((self.y_zpf_i * self.y_zpf_p)/self.dist**3)
        self.g_z = -(1/hbar) * ((self.Q_i * self.Q_p)/(2 * pi * epsilon_0)) * ((self.z_zpf_i * self.z_zpf_p)/self.dist**3)
        
        # hbar * g_r / (r_zpf_i * r_zpf_p)
        # Coefficient K_r from the coupling term K_r * deltaR_i * deltaR_p in the Hamiltonian.
        self.K_x = (self.Q_i * self.Q_p)/(4 * pi * epsilon_0 * self.dist**3)
        self.K_y = (self.Q_i * self.Q_p)/(4 * pi * epsilon_0 * self.dist**3)
        self.K_z = -(self.Q_i * self.Q_p)/(2 * pi * epsilon_0 * self.dist**3)


        # Dissipation rates of ion and nanoparticle
        self.Gamma_gas_x = self.E_dot_gas / (hbar * self.new_Omega_x_p)
        self.Gamma_gas_y = self.E_dot_gas / (hbar * self.new_Omega_y_p)
        self.Gamma_gas_z = self.E_dot_gas / (hbar * self.new_Omega_z_p)

        self.Gamma_td_x = self.E_dot_td / (hbar * self.new_Omega_x_p)
        self.Gamma_td_y = self.E_dot_td / (hbar * self.new_Omega_y_p)
        self.Gamma_td_z = self.E_dot_td / (hbar * self.new_Omega_z_p)

        self.Gamma_dop_x = self.E_dot_dop / (hbar * self.new_Omega_x_i)
        self.Gamma_dop_y = self.E_dot_dop / (hbar * self.new_Omega_y_i)
        self.Gamma_dop_z = self.E_dot_dop / (hbar * self.new_Omega_z_i)


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
        r_zpf_vec_3d = np.sqrt(hbar / (2 * mass_vec_3d * Omega_vec_3d.flatten()))
        pr_zpf_vec_3d = np.sqrt((hbar * mass_vec_3d * Omega_vec_3d.flatten()) / 2)
        
        return charge_vec, mass_vec, Omega_vec_1d, mass_vec_3d, Omega_vec_3d, q_zpf_vec_3d, pr_zpf_vec_3d
        