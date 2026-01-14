from gain_control.utils_gc import *
from libraries.proportional_constant_rate_change import GC_prop_cons

# ******************************************************************************************************************
# Global variables
# (Experiment 2) freq. response decay around 100Hz
# (Experiment 3) freq. response decay around 10Hz
# (Experiment 4) freq. response from Gain Control paper
# (Experiment 5) freq. response decay around 100Hz
# (Experiment 6) freq. response decay around 10Hz
model = 'MSSM'
ind = 7
save_vars = False
force_experiment = True
plot_ind_memPot = True
save_figs = False
imputations = True
stoch_input = True
lif_output = True
dyn_synapse = True
tau_m_lif = 1  # ms
total_realizations = 100
num_realizations = 8
gain_v = [0.1, 0.2, 0.5]
folder_vars = "../gain_control/variables/"
folder_plots = '../gain_control/plots/'

# ******************************************************************************************************************
# Time conditions
sfreq = 6e3
max_t = 6
dt = 1 / sfreq
time_vector = np.arange(0, max_t, dt)
L = time_vector.shape[0]
sim_params = {'sfreq': sfreq, 'max_t': max_t, 'L': L, 'time_vector': time_vector}

# ******************************************************************************************************************
# STP model
num_syn = 1
syn_params, description, name_params = get_params_stp(model, ind)

# Neuron model
n_model = "LIF"
lif_params = get_neuron_params(tau_m=tau_m_lif)

dict_params = {'stp_model': model, 'stp_name_params': name_params, 'stp_value_params': syn_params, 'num_syn': num_syn,
               'neuron_model': n_model, 'neuron_params': lif_params, 'sim_params': sim_params, 'gain_vector': gain_v,
               'folder_vars': folder_vars, 'folder_plots': folder_plots, 'save_vars': save_vars, 'save_figs': save_figs,
               'force_experiment': force_experiment, 'imputations': imputations, 'stoch_input': stoch_input,
               'lif_output': lif_output, 'dynamic_synapse': dyn_synapse, 'description': description,
               'num_realizations': num_realizations, 'total_realizations': total_realizations}

# Instance of Gain-Control class
initial_frequencies = np.array([10, 100, 500])
gc_prop_cons = GC_prop_cons(dict_params)
_ = gc_prop_cons.set_experiment_vars(gain_v, f_vec=initial_frequencies)

# Running Gain-Control process for different gains
file_name = ""
dr = {}
for gain in gain_v:
    file_name = gc_prop_cons.get_folder_file_name(model, gain, ind)
    file_loaded, dr_aux = gc_prop_cons.load_set_simulation_params()
    gc_prop_cons.models_creation()
    dr = gc_prop_cons.run(gain=gain, fixed_rate_change=5, soft_stop_cond=(not file_loaded or force_experiment),
                          plot_ind_figs=plot_ind_memPot)

# **********************************************************************************************************************
# PLOTS
# PLOT OF CHARACTERISTICS FOR INI, MID, AND END WINDOWS. SPLITTED BY PROPORTIONAL AND CONSTANT INPUT RATE CHANGES
# """
initial_frequencies = dr['initial_frequencies']
# Steady-state
lbl = ['st_ini_prop', 'st_mid_prop', 'st_end_prop', 'st_ini_fix', 'st_mid_fix', 'st_end_fix']
st_lbl = ['_mean', '_med', '_max', '_min', '_q10', '_q90', '_q5', '_q95']
cols = ['tab:orange', 'tab:blue', 'tab:red', 'tab:red', 'tab:green', 'tab:green', 'tab:olive', 'tab:olive']
title = description + " " + str(tau_m_lif) + "ms for steady-state"
path_save = folder_plots + file_name + '_windows_statistics.png'
plot_features_windows_prop_fix(initial_frequencies, dr, lbl, st_lbl, cols, title, path_save, save_figs)

# PLOT CHARACTERISTICS OF MID AND INI WINDOWS IN THE SAME PLOT, FOR PROPORTIONAL AND CONSTANT INPUT RATE CHANGES
lbl = ['st_ini_prop', 'mtr_ini_prop', 'st_ini_fix', 'mtr_ini_fix']
lbl2 = ['st_mid_prop', 'mtr_mid_prop', 'st_mid_fix', 'mtr_mid_fix']
st_lbl = ['_mean', '_med', '_max', '_min', '_q10', '_q90']  # , '_q5', '_q95']
t_ = ['Steady-state, ini/mid windows (prop)', 'Transition-state, ini/mid windows (prop)',
      'Steady-state, ini/mid windows (cons)', 'Transition-state, ini/mid windows (cons)']
cols = ['tab:orange', 'tab:blue', 'tab:red', 'tab:red', 'tab:green', 'tab:green']  # , 'tab:olive', 'tab:olive']
name_save = folder_plots + file_name + '_windows_statistics3.png'
title = description + " " + str(tau_m_lif) + "ms"
plot_features_2windows_prop_fix(initial_frequencies, dr, lbl, lbl2, st_lbl, cols, t_, title, path_save, save_figs)
# """

# SIMPLE PLOT OF DIFFERENCES OF STEADY-STATE BETWEEN MID AND INI WINDOWS FOR PROPORTIONAL AND CONSTANT CHANGE OF RATES
# AND THE DIFFERENCES BETWEEN MAX OF MID WINDOW AND MEDIAN OF INI WINDOW
# """
lbl = ['st_mid_prop', 'st_mid_fix']
lbl2 = ['st_ini_prop', 'st_ini_fix']
st_lbl = ['_max', '_min', '_q10', '_q90', '_mean', '_med']
cols_ = ['tab:red', 'tab:red', 'tab:green', 'tab:green', 'tab:orange', 'tab:blue']
t_ = [r"$mid_{st} - ini_{st}$ (Prop)", r"$mid_{st} - ini_{st}$ (Cons)",
      "Max Transient (Proportional)", "Max Transient (Constant)"]
# ylims = [-62.5, -54.0]  # [-70.05, -52]
name_save = folder_plots + file_name + '_' + 'diff_st_log.png'
title = description + " " + str(tau_m_lif) + "ms"
plot_diff_windows(initial_frequencies, dr, lbl, lbl2, st_lbl, cols_, t_, title_graph=title, name_save=name_save,
                  save_figs=save_figs)
# """
