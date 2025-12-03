import matplotlib.pyplot as plt
import numpy as np

def plot_steady_state_energy(t_list, potential_joules, kinetic_joules, T_slow):
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
    T_slow : float
        The period of the slow drive [s] used for time normalization.
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
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

    return fig1


def plot_zoomed_energy(t_list, kinetic_history, T_slow, slice_start=5000, slice_end=5050):
    """
    Generates a zoomed-in view of the kinetic energy to visualize micromotion.

    Parameters
    ----------
    t_list : ndarray
        Array of time evaluation points.
    kinetic_history : list or ndarray
        History of kinetic energy values (in Joules).
    T_slow : float
        The period of the slow drive [s] used for time normalization.
    slice_start : int
        Start index for the zoom window.
    slice_end : int
        End index for the zoom window.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    """
    # Convert slice to eV
    # Note: kinetic_history is assumed to be raw Joules here. 
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
    
    return fig2


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

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
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
    ax3.set_xlabel(r'Nanoparticle Damping $\gamma_p / 2\pi$ [Hz]', fontsize=12)
    ax3.set_ylabel(r'Phonon number $\bar{n}$', fontsize=12)

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
    
    return fig3