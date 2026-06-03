from gain_control.utils_gc import *
from libraries.proportional_constant_rate_change import GC_prop_cons

gain_v = [1.0]              # Vector of gains
s_model = 'DoornSTD'        # Synaptic model to use: TM, MSSM, or Doorn variations (DoornSTD, DoornSTF)
n_model = "HH"              # Neuron model to use: LIF (Leaky Integrate-and-Fire), HH (Hodgkin Huxley)
ind = 0                     # Index to recover params of a given synaptic and neuron model (See table below)
sfreq = 30e3                # Sampling frequency of the simulation  16.8KHz
max_freq = 3701             # Maximum baseline rate of the experiment
tau_m_lif = 1               # If LIF neuron is used, this specifies the time constant (in milliseconds)
folder_vars = "../gain_control/variables/high_freq_30k/"  # Folder to save results
folder_plots = '../gain_control/plots/'                   # Folder to save plots

# ******************************************************************************************************************
# COMBINATION OF SYNAPTIC AND NEURON MODELS (INDICES)
# DoornSTD + HH
# (Ind 1) DoornSTD control
# (Ind 2 and 3) DoornSTD strong STD (a and b)
# (Ind 4 and 5) DoornSTD + Asynchronous release (low and high)
# (Ind 6) DoornSTD + strong NMDA current
# (Ind 7) DoornSTF
# MSSM or TM + LIF
# (Ind 2) freq. response decay around 100Hz (depression)
# (Ind 3) freq. response decay around 10Hz (depression)
# (Ind 4) freq. response from Gain Control paper (depression)
# (Ind 5) freq. response decay around 100Hz (depression)
# (Ind 6) freq. response decay around 10Hz (depression)
# (Ind 7) freq. response (facilitation) ONLY FOR MSSM
# (Ind 8) freq. response (facilitation) ONLY FOR TM

# ******************************************************************************************************************
# GLOBAL VARIABLES
save_vars = True            # Save results in folders
force_experiment = False    # Run pipeline even if file with results is saved (For refining the code)
stoch_input = False         # Whether to use stochastic inputs (from Poisson processes) or deterministic ones

plot_ind_memPot = False     # Plot temporal dynamics
save_figs = False           # Save temporal dynamics in folders

dyn_synapse = True          # Use Synaptic Dynamics or a simple static synapse (a weight)
n_noise = True              # Activate noise in neuron model if available

# tr_st_time variables
# # th_percentage=1e-3 for not filtering (Doorn 1, 2, 3, 6, 7)
filtering_tr = False        # Use filtering to detect time of steady-state
cutoff_filt = 5             # Cut-off frequency of the filter if used
threshold_per = 1e-3        # Threshold factor to detect time of steady-state

total_realizations = 104    # Number of stochastic realisations if activated  104
num_realizations = 8        # Number of parallel realisations                 8
# **********************************************************************************************************************
# Time conditions
max_t = 6                               # Time of simulation (in seconds)
dt = 1 / sfreq                          # Time step
time_vector = np.arange(0, max_t, dt)   # Time vector
L = time_vector.shape[0]                # Length of time vector
sim_params = {'sfreq': sfreq, 'max_t': max_t, 'L': L, 'time_vector': time_vector}

# **********************************************************************************************************************
# STP model
num_syn = 1
syn_params, description, name_params = get_params_stp(s_model, ind)

# Neuron model
neuron_params = get_neuron_params(n_model=n_model, tau_m=tau_m_lif, ind=ind, y_lim_ind_plot=True, num_syn=num_syn)

dict_params = {'stp_model': s_model, 'stp_name_params': name_params, 'stp_value_params': syn_params, 'num_syn': num_syn,
               'neuron_model': n_model, 'neuron_params': neuron_params, 'sim_params': sim_params, 'gain_vector': gain_v,
               'folder_vars': folder_vars, 'folder_plots': folder_plots, 'save_vars': save_vars, 'save_figs': save_figs,
               'force_experiment': force_experiment, 'stoch_input': stoch_input, 'dynamic_synapse': dyn_synapse,
               'description': description, 'num_realizations': num_realizations,
               'total_realizations': total_realizations, 'neuron_noise': n_noise}

# Instance of Gain-Control class
initial_frequencies = np.array([10, 1000]) if force_experiment else None
gc_prop_cons = GC_prop_cons(dict_params)
_ = gc_prop_cons.set_experiment_vars(gain_v, f_vec=initial_frequencies, max_freq=max_freq)

# Running Gain-Control process for different gains
file_name = ""
dr = {}
for gain in gain_v:
    # if not gc_prop_cons.stoch_input:
    print("Loading/computing deterministic experiments")
    file_name = gc_prop_cons.get_folder_file_name(s_model, n_model, gain, ind, tau_n=tau_m_lif)
    file_loaded, dr_aux = gc_prop_cons.load_set_simulation_params()
    gc_prop_cons.models_creation()
    dr = gc_prop_cons.run(gain=gain, fixed_rate_change=5, soft_stop_cond=(not file_loaded),
                          plot_ind_figs=plot_ind_memPot, y_lims_ind_plot=neuron_params['y_lim_plot'],
                          th_percentage=threshold_per, filtering=filtering_tr, cutoff=cutoff_filt)
    """
    else:
        print("Loading/computing stochastic experiments. First deterministic experiment")
        # First load or compute deterministic response
        # Forcing flags
        gc_prop_cons.stoch_input = False
        old_save_vars = save_vars
        gc_prop_cons.save_vars = True

        # running gain_control
        aux_max_freq = int(np.min([max_freq * gain + max_freq + 50, (sfreq / 6) - 10]))
        _ = gc_prop_cons.set_experiment_vars(gain_v, f_vec=initial_frequencies, max_freq=aux_max_freq)
        file_name = gc_prop_cons.get_folder_file_name(s_model, n_model, gain, ind, tau_n=tau_m_lif)
        file_loaded, dr_aux = gc_prop_cons.load_set_simulation_params()
        gc_prop_cons.models_creation()
        dr_det = gc_prop_cons.run(gain=gain, fixed_rate_change=5, soft_stop_cond=(not file_loaded or force_experiment),
                                  plot_ind_figs=plot_ind_memPot, y_lims_ind_plot=neuron_params['y_lim_plot'],
                                  th_percentage=1e-5)
        # Default setting of flags
        gc_prop_cons.stoch_input = True
        gc_prop_cons.save_vars = old_save_vars
        _ = gc_prop_cons.set_experiment_vars(gain_v, f_vec=initial_frequencies, max_freq=max_freq)
        print("Loading/computing stochastic experiments. Second stochastic experiment")

        # Now compute the stochastic response
        st_k = ['st_ini_prop_q5', 'st_ini_prop_q10', 'st_ini_prop_q90', 'st_ini_prop_q95',
                'st_ini_prop_min', 'st_ini_prop_max', 'st_ini_prop_mean', 'st_ini_prop_med']
        st_prior = np.array([dr_det['initial_frequencies']] + [dr_det[k][0, :] for k in st_k])
        file_name = gc_prop_cons.get_folder_file_name(s_model, n_model, gain, ind, tau_n=tau_m_lif)
        file_loaded, dr_aux = gc_prop_cons.load_set_simulation_params()
        gc_prop_cons.models_creation()
        dr = gc_prop_cons.run(gain=gain, fixed_rate_change=5, soft_stop_cond=(not file_loaded or force_experiment),
                              plot_ind_figs=plot_ind_memPot, y_lims_ind_plot=neuron_params['y_lim_plot'],
                              th_percentage=1e-5, st_prior=st_prior)
    # """