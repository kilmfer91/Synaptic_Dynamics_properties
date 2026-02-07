from gain_control.utils_gc import *

# ******************************************************************************************************************
# STP model and extra global variables
model = 'MSSM'
# (Experiment 2) freq. response decay around 100Hz
# (Experiment 3) freq. response decay around 10Hz
# (Experiment 4) freq. response from Gain Control paper
# (Experiment 5) freq. response decay around 100Hz
# (Experiment 6) freq. response decay around 10Hz
ind = 4
# save_vars = True
run_experiment = False
save_figs = False
imputations = True
lif_output = True
num_syn = 100
gain = 0.5

# Sampling frequency and conditions for running parallel or single LIF neurons
sfreq = 5e3
tau_lif = 30  # ms
total_realizations = 1  # 100
num_realizations = 1  # 8 for server, 4 for macbook air
t_tra = None  # 0.25

# Path variables
path_vars = "../gain_control/variables/"
check_create_folder(path_vars)
folder_plots = '../gain_control/plots/'
check_create_folder(folder_plots)

# File names
dr_syn_filtering_file = get_name_file(sfreq, model, ind, num_syn, lif_output, tau_lif, False, imputations, gain)
dr_gain_control_file = get_name_file(sfreq, model, ind, num_syn, lif_output, tau_lif, True, imputations, gain)

# Auxiliar variables
description = ""
dr_filt = None
dr_gain = None
initial_frequencies = []

print("For gain control, file %s and index %d" % (dr_gain_control_file, ind))
print("For synaptic filtering, file %s and index %d" % (dr_syn_filtering_file, ind))

# ******************************************************************************************************************
# Trying to load freq. response of Gain Control
if os.path.isfile(path_vars + dr_syn_filtering_file) and not run_experiment:
    dr_filt = loadObject(dr_syn_filtering_file, path_vars)

    # Auxiliar variables
    initial_frequencies, model = dr_filt['initial_frequencies'], dr_filt['stp_model']
    dyn_synapse, num_synapses = dr_filt['dyn_synapse'], dr_filt['num_synapses']
    num_realizations, sim_params = dr_filt['realizations'], dr_filt['sim_params']
    # prop_rate_change_a = dr_filt['prop_rate_change_a']
    fix_rate_change_a, num_changes_rate, = dr_filt['fix_rate_change_a'], dr_filt['num_changes_rate'],
    description = dr_filt['description']
    seeds = dr_filt['seeds']
    total_realizations = dr_filt['t_realizations']
    # lif_params2, syn_params, lif_params, name_params = dr_filt['lif_params2'], dr_filt['syn_params'], dr_filt['lif_params'], dr_filt['name_params']

    """
    # Time conditions
    max_t, sfreq, time_vector, L = sim_params['max_t'], sim_params['sfreq'], sim_params['time_vector'], sim_params['L']
    dt = 1 / sfreq
    Le_time_win = int(max_t / num_changes_rate)

    # time transition
    t_tra = dr_filt['time_transition']

    # Parameters in dict format
    params = dict(zip(name_params, syn_params))
    # """

if os.path.isfile(path_vars + dr_gain_control_file) and not run_experiment:
    dr_gain = loadObject(dr_gain_control_file, path_vars)

# ******************************************************************************************************************
# Plots
lbl = ['st_mid_prop']
lbl2 = ['st_ini_prop']
# st_lbl = ['_max',  '_q95', '_q90', '_med', '_min', '_q5', '_q10', '_mean']
# cols_ = ['tab:red', 'tab:olive', 'tab:green', 'tab:blue', 'tab:red', 'tab:olive', 'tab:green', 'tab:orange']
st_lbl = ['_max',  '_q90', '_med', '_min', '_q10', '_mean']
cols_ = ['tab:red', 'tab:green', 'tab:blue', 'tab:red', 'tab:green', 'tab:orange']
# ylims = [-62.5, -54.0]  # [-70.05, -52]
title = ("Steady-state: " + description.split(",")[0] + r', $\tau_{lif}$ ' +
         str(tau_lif) + "ms, gain " + str(int(gain * 100)) + "%")
path_save = folder_plots + dr_gain_control_file + '_phase_portrait.png'
plot_gain_filtering(dr_gain, dr_filt, lbl, lbl2, st_lbl, cols_, title, path_save, True)
# title = "transition-state: " + title.split(":")[1]
# lbl = ['mtr_mid_prop']
# lbl2 = ['mtr_ini_prop']
# plot_gain_filtering(dr_gain, dr_filt, lbl, lbl2, st_lbl, cols_, title, path_save, True)

# PLOT OF CHARACTERISTICS FOR INI, MID, AND END WINDOWS. SPLITTED BY PROPORTIONAL AND CONSTANT INPUT RATE CHANGES
# """
# Steady-state
lbl = ['st_ini_prop', 'st_mid_prop', 'st_end_prop', 'st_ini_fix', 'st_mid_fix', 'st_end_fix']
# st_lbl = ['_mean', '_med', '_max', '_min', '_q10', '_q90', '_q5', '_q95']
# cols = ['tab:orange', 'tab:blue', 'tab:red', 'tab:red', 'tab:green', 'tab:green', 'tab:olive', 'tab:olive']
st_lbl = ['_max',  '_q90', '_med', '_min', '_q1', '_mean']
cols_ = ['tab:red', 'tab:green', 'tab:blue', 'tab:red', 'tab:green', 'tab:orange']
title = ("Steady-state: " + description.split(",")[0] + r', $\tau_{lif}$ ' + str(tau_lif) + "ms, gain " +
         str(int(gain * 100)) + "%")
path_save = folder_plots + dr_gain_control_file + '_windows_statistics.png'
plot_features_windows_prop_fix(initial_frequencies, dr_gain, lbl, st_lbl, cols, title, path_save, save_figs)

# Transition-state
lbl = ['mtr_ini_prop', 'mtr_mid_prop', 'mtr_end_prop', 'mtr_ini_fix', 'mtr_mid_fix', 'mtr_end_fix']
title = ("Transition-state: " + description.split(",")[0] + r', $\tau_{lif}$ ' + str(tau_lif) + "ms, gain " +
         str(int(gain * 100)) + "%")
path_save = folder_plots + dr_gain_control_file + '_windows_statistics.png'
plot_features_windows_prop_fix(initial_frequencies, dr_gain, lbl, st_lbl, cols, title, path_save, save_figs)

# PLOT CHARACTERISTICS OF MID AND INI WINDOWS IN THE SAME PLOT, FOR PROPORTIONAL AND CONSTANT INPUT RATE CHANGES
lbl = ['st_ini_prop', 'mtr_ini_prop', 'st_ini_fix', 'mtr_ini_fix']
lbl2 = ['st_mid_prop', 'mtr_mid_prop', 'st_mid_fix', 'mtr_mid_fix']
st_lbl = ['_mean', '_med', '_max', '_min', '_q1', '_q90']  # , '_q5', '_q95']
t_ = ['Steady-state, ini/mid windows (prop)', 'Transition-state, ini/mid windows (prop)',
      'Steady-state, ini/mid windows (cons)', 'Transition-state, ini/mid windows (cons)']
cols = ['tab:orange', 'tab:blue', 'tab:red', 'tab:red', 'tab:green', 'tab:green']  # , 'tab:olive', 'tab:olive']
name_save = folder_plots + dr_gain_control_file + '_windows_statistics3.png'
title = description.split(",")[0] + r', $\tau_{lif}$ ' + str(tau_lif) + "ms, gain " + str(int(gain * 100)) + "%"
plot_features_2windows_prop_fix(initial_frequencies, dr_gain, lbl, lbl2, st_lbl, cols, t_, title, path_save, save_figs)
# """

# SIMPLE PLOT OF DIFFERENCES OF STEADY-STATE BETWEEN MID AND INI WINDOWS FOR PROPORTIONAL AND CONSTANT CHANGE OF RATES
# AND THE DIFFERENCES BETWEEN MAX OF MID WINDOW AND MEDIAN OF INI WINDOW
# """
# ylims = [-62.5, -54.0]  # [-70.05, -52]
lbl = ['st_mid_prop', 'st_mid_fix']
lbl2 = ['st_ini_prop', 'st_ini_fix']
st_lbl = ['_max', '_min', '_q10', '_q90', '_mean', '_med']
cols_ = ['tab:red', 'tab:red', 'tab:green', 'tab:green', 'tab:orange', 'tab:blue']
t_ = [r"$mid_{st} - ini_{st}$ (Prop)", r"$mid_{st} - ini_{st}$ (Cons)",
      "Max Transient (Proportional)", "Max Transient (Constant)"]
name_save = folder_plots + dr_gain_control_file + '_' + 'diff_st_log.png'
title = ("Steady-state: " + description.split(",")[0] + r', $\tau_{lif}$ ' + str(tau_lif) + "ms, gain " +
         str(int(gain * 100)) + "%")
plot_diff_windows(initial_frequencies, dr_gain, lbl, lbl2, st_lbl, cols_, t_, title_graph=title, name_save=name_save,
                  save_figs=save_figs)
# """

