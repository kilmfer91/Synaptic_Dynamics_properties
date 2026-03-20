from gain_control.utils_gc import *

# ******************************************************************************************************************
# STP model and extra global variables
# (Experiment 2) freq. response decay around 100Hz
# (Experiment 3) freq. response decay around 10Hz
# (Experiment 4) freq. response from Gain Control paper
# (Experiment 5) freq. response decay around 100Hz
# (Experiment 6) freq. response decay around 10Hz
s_model = 'MSSM'
n_model = "LIF"
ind = 4
# save_vars = True
run_experiment = False
save_figs = True
imputations = True
lif_output = True
n_noise = True
num_syn = 1
gain = 1

# Sampling frequency and conditions for running parallel or single LIF neurons
sfreq = 10e3
tau_lif = 1  # ms
total_realizations = 1  # 100
num_realizations = 1  # 8 for server, 4 for macbook air
t_tra = None  # 0.25

# Path variables
path_vars = "../gain_control/variables/"
check_create_folder(path_vars)
folder_plots = '../gain_control/plots/'
check_create_folder(folder_plots)

# **********************************************************************************************************************
# MULTIPLE GAINS
# ""
# Plot
lbl = ['st_mid_prop']
lbl2 = ['st_ini_prop']
lbl_syn = ['syn_st_mid_prop']
lbl2_syn = ['syn_st_ini_prop']
st_lbl = ['_max',  '_q95', '_q90', '_med', '_min', '_q5', '_q10', '_mean']
cols_ = ['tab:red', 'tab:olive', 'tab:green', 'tab:blue', 'tab:red', 'tab:olive', 'tab:green', 'tab:orange']
# st_lbl = ['_max',  '_q90', '_med', '_min', '_q10', '_mean']
# cols_ = ['tab:red', 'tab:green', 'tab:blue', 'tab:red', 'tab:green', 'tab:orange']
# ylims = [-62.5, -54.0]  # [-70.05, -52]
title = ("Steady-state of model " + s_model + 'r multiple gains')
if n_model == 'LIF':
    title = ("Steady-state of model " + s_model + r', $\tau_{lif}$ ' + str(tau_lif) + "ms, multiple gains")

fig2 = plt.figure(figsize=(15, 8))  # 6.5, 5
plt.suptitle(title + ". Mem. pot.")
ax_ = [fig2.add_subplot(2, 4, j + 1) for j in range(len(st_lbl))]
fig3 = plt.figure(figsize=(15, 8))  # 6.5, 5
plt.suptitle(title + ". Synapse")
ax_s = [fig3.add_subplot(2, 4, j + 1) for j in range(len(st_lbl))]

alpha = 0.3
markers = ['+', '*']
alphas = [1.0, 0.5]

# **********************************************************************************************************************
gain_v = [1.0]  # [1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
filt_dict_loaded = False

# Auxiliar variables
description = ""          
dr_filt = None            
dr_gain = None            
initial_frequencies = []  

for gain in gain_v:
    # File names
    dr_syn_filtering_file = get_name_file(sfreq, s_model, n_model, ind, num_syn, lif_output, tau_lif, False,
                                          imputations, 1.0, n_noise=n_noise)
    dr_gain_control_file = get_name_file(sfreq, s_model, n_model, ind, num_syn, lif_output, tau_lif, True,
                                         imputations, gain, n_noise=n_noise)

    print("For gain control, file %s and index %d" % (dr_gain_control_file, ind))
    print("For synaptic filtering, file %s and index %d" % (dr_syn_filtering_file, ind))

    # ******************************************************************************************************************
    # Trying to load freq. response of Gain Control
    if os.path.isfile(path_vars + dr_syn_filtering_file) and not filt_dict_loaded:
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

        filt_dict_loaded = True
        # lif_params2, syn_params, lif_params, name_params = dr_filt['lif_params2'], dr_filt['syn_params'],
        # dr_filt['lif_params'], dr_filt['name_params']

        '''
        # Time conditions
        max_t, sfreq, time_vector,  = sim_params['max_t'], sim_params['sfreq'], sim_params['time_vector']
        L = sim_params['L']
        dt = 1 / sfreq
        Le_time_win = int(max_t / num_changes_rate)
    
        # time transition
        t_tra = dr_filt['time_transition']
    
        # Parameters in dict format
        params = dict(zip(name_params, syn_params))
        # '''

    if os.path.isfile(path_vars + dr_gain_control_file) and not run_experiment:
        dr_gain = loadObject(dr_gain_control_file, path_vars)

    # dr_gain = dr_filt
    # ******************************************************************************************************************
    # Plots
    for i in [0]:  # range(len(lbl)):
        for j in range(len(st_lbl)):
            aux_gain = np.copy(dr_gain[lbl[i] + st_lbl[j]] - dr_gain[lbl2[i] + st_lbl[j]])
            aux_filt = np.copy(dr_gain[lbl2[i] + st_lbl[j]])
            aux_det_gain = np.copy(dr_filt[lbl[i] + st_lbl[j]] - dr_filt[lbl2[i] + st_lbl[j]])[0, :]
            aux_det_filt = np.copy(dr_filt[lbl2[i] + st_lbl[j]][0, :])
            if n_model == 'HH':
                aux_gain *= 1e3
                aux_filt *= 1e3
                aux_det_gain *= 1e3
                aux_det_filt *= 1e3

            # Stochastic plots
            ax_[j].scatter(np.median(aux_filt, axis=0), np.median(aux_gain, axis=0), marker=markers[i],  # c=cols_[j], Automatic colors for trajectories of specific gain
                           alpha=alphas[i])
            ax_[j].plot(np.median(aux_filt, axis=0), np.median(aux_gain, axis=0),  # c=cols_[j], Automatic colors for trajectories of specific gain
                        alpha=alphas[i], label=gain)
            if i == 0: ax_[j].fill_between(np.median(aux_filt, axis=0), np.quantile(aux_gain, 0.1, axis=0),
                                           np.quantile(aux_gain, 0.9, axis=0), color=cols_[j], alpha=0.1)
            ax_[j].scatter(np.median(aux_filt, axis=0)[0], np.median(aux_gain, axis=0)[0], c='black')

            # Deterministic plots
            ax_[j].plot(aux_det_filt, aux_det_gain, c='gray', alpha=alphas[i], label='det')
            ax_[j].scatter(aux_det_filt, aux_det_gain, c='gray', marker=markers[i], alpha=alphas[i])
            ax_[j].scatter(aux_det_filt[0], aux_det_gain[0], c='black')

            # ax_3.set_xscale('log')
            # if (i + 1) % 3 == 0: ax_[j].legend(loc='upper right')
            # ax_st.set_ylim(ylims)

            # For synaptic contributions
            aux_gain = np.copy(dr_gain[lbl_syn[i] + st_lbl[j]] - dr_gain[lbl2_syn[i] + st_lbl[j]])
            aux_filt = np.copy(dr_gain[lbl2_syn[i] + st_lbl[j]])
            aux_det_gain = np.copy(dr_filt[lbl_syn[i] + st_lbl[j]] - dr_filt[lbl2_syn[i] + st_lbl[j]])[0, :]
            aux_det_filt = np.copy(dr_filt[lbl2_syn[i] + st_lbl[j]][0, :])
            if n_model == 'HH':
                aux_gain *= 1e3
                aux_filt *= 1e3
                aux_det_gain *= 1e3
                aux_det_filt *= 1e3

            # Stochastic plots
            ax_s[j].scatter(np.median(aux_filt, axis=0), np.median(aux_gain, axis=0), marker=markers[i],  # c=cols_[j], Automatic colors for trajectories of specific gain
                           alpha=alphas[i])
            ax_s[j].plot(np.median(aux_filt, axis=0), np.median(aux_gain, axis=0),  # c=cols_[j], Automatic colors for trajectories of specific gain
                        alpha=alphas[i], label=gain)
            if i == 0: ax_s[j].fill_between(np.median(aux_filt, axis=0), np.quantile(aux_gain, 0.1, axis=0),
                                           np.quantile(aux_gain, 0.9, axis=0), color=cols_[j], alpha=0.1)
            ax_s[j].scatter(np.median(aux_filt, axis=0)[0], np.median(aux_gain, axis=0)[0], c='black')

            # Deterministic plots
            ax_s[j].plot(aux_det_filt, aux_det_gain, c='gray', alpha=alphas[i], label='det')
            ax_s[j].scatter(aux_det_filt, aux_det_gain, c='gray', marker=markers[i], alpha=alphas[i])
            ax_s[j].scatter(aux_det_filt[0], aux_det_gain[0], c='black')


xlims = [[-69.9, -69.9, -69.95, -70.01, -70.01, -70.01, -70.01, -70.01],
         [-67.9, -69.1, -69.40, -69.82, -69.82, -69.82, -69.82, -69.82]]
# ylims = [[-0.6, -0.15, -0.1, -0.01, -0.01, -0.01, -0.01, -0.01],
#          [0.20, 0.250, 0.30, 0.100, 0.100, 0.075, 0.070, 0.030]]
ylims = [[-0.6, -0.6, -0.6, -0.6, -0.6, -0.6, -0.6, -0.6],
         [0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40]]
xlims_s = [[-69.9, -69.9, -69.95, -70.01, -70.01, -70.01, -70.01, -70.01],
           [0.07, -69.1, -69.40, -69.82, -69.82, -69.82, -69.82, -69.82]]
ylims_s = [[-0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02],
           [0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015]]

path_save = (folder_plots + s_model + '_ind_' + str(ind) + '_' + str(len(gain_v)) + '_gains_sf_' +
             str(int(sfreq * 1e-3)) + 'k_tauLIF_' + str(tau_lif) + 'ms' + '_phase_portrait.png')
for j in range(len(st_lbl)):
    # neuron contributions
    ax_[j].set_xlabel("Synaptic filtering (mV)", color='gray')
    ax_[j].set_ylabel("Gain-Control (mV)", color='gray')
    if n_model != 'HH':
        ax_[j].set_ylim(ylims[0][j], ylims[1][j])
        ax_[j].set_xlim(xlims[0][j], xlims[1][j])
    # ax_[j].set_ylim(ylims)
    ax_[j].set_title(st_lbl[j][1:], c="gray")
    ax_[j].grid()

    # Synaptic contributions
    ax_s[j].set_xlabel("Synaptic filtering (mV)", color='gray')
    ax_s[j].set_ylabel("Gain-Control (mV)", color='gray')
    if n_model != 'HH':
        ax_s[j].set_ylim(ylims_s[0][j], ylims_s[1][j])
        # ax_s[j].set_xlim(xlims[0][j], xlims[1][j])
    # ax_[j].set_ylim(ylims)
    ax_s[j].set_title(st_lbl[j][1:], c="gray")
    ax_s[j].grid()

ax_[int(len(st_lbl) / 2) - 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
fig2.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
fig3.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
if save_figs: fig2.savefig(path_save, format='png')

    # plot_gain_filtering(dr_gain, dr_filt, lbl, lbl2, st_lbl, cols_, title, path_save, False, fig2=fig)

    # title = "transition-state: " + title.split(":")[1]
    # lbl = ['mtr_mid_prop']
    # lbl2 = ['mtr_ini_prop']
    # plot_gain_filtering(dr_gain, dr_filt, lbl, lbl2, st_lbl, cols_, title, path_save, True)
# """

# PLOT OF CHARACTERISTICS FOR INI, MID, AND END WINDOWS. SPLITTED BY PROPORTIONAL AND CONSTANT INPUT RATE CHANGES
"""
# Steady-state
lbl = ['st_ini_prop', 'st_mid_prop', 'st_end_prop', 'st_ini_fix', 'st_mid_fix', 'st_end_fix']
st_lbl = ['_mean', '_med', '_max', '_min', '_q10', '_q90', '_q5', '_q95']
cols = ['tab:orange', 'tab:blue', 'tab:red', 'tab:red', 'tab:green', 'tab:green', 'tab:olive', 'tab:olive']
# st_lbl = ['_max',  '_q90', '_med', '_min', '_q1', '_mean']
# cols_ = ['tab:red', 'tab:green', 'tab:blue', 'tab:red', 'tab:green', 'tab:orange']
title = ("Steady-state: " + description.split(",")[0] + r', $\tau_{lif}$ ' + str(tau_lif * 1e3) + "ms, gain " +
         str(int(gain * 100)) + "%")
path_save = folder_plots + dr_gain_control_file + '_windows_statistics.png'
plot_features_windows_prop_fix(initial_frequencies, dr_gain, lbl, st_lbl, cols, title, path_save, save_figs)

# Transition-state
lbl = ['mtr_ini_prop', 'mtr_mid_prop', 'mtr_end_prop', 'mtr_ini_fix', 'mtr_mid_fix', 'mtr_end_fix']
title = ("Transition-state: " + description.split(",")[0] + r', $\tau_{lif}$ ' + str(tau_lif * 1e3) + "ms, gain " +
         str(int(gain * 100)) + "%")
path_save = folder_plots + dr_gain_control_file + '_windows_statistics.png'
plot_features_windows_prop_fix(initial_frequencies, dr_gain, lbl, st_lbl, cols, title, path_save, save_figs)
# """

# PLOT OF CHARACTERISTICS FOR INI, MID, AND END WINDOWS. FOR PROPORTIONAL AND CONSTANT INPUT RATE CHANGES
# Steady-state
"""
lbl = ['st_ini_prop', 'st_mid_prop', 'st_end_prop', 'mtr_ini_prop', 'mtr_mid_prop', 'mtr_end_prop']
st_lbl = ['_mean', '_med', '_max', '_min', '_q10', '_q90', '_q5', '_q95']
cols = ['tab:orange', 'tab:blue', 'tab:red', 'tab:red', 'tab:green', 'tab:green', 'tab:olive', 'tab:olive']
# st_lbl = ['_max',  '_q90', '_med', '_min', '_q1', '_mean']
# cols_ = ['tab:red', 'tab:green', 'tab:blue', 'tab:red', 'tab:green', 'tab:orange']
title = ("Transition- and steady-state: " + description.split(",")[0] + r', $\tau_{lif}$ ' + str(tau_lif * 1e3) + 
         "ms, gain " + str(int(gain * 100)) + "%")
path_save = folder_plots + dr_gain_control_file + '_windows_statistics.png'
plot_features_windows_prop_fix(initial_frequencies, dr_gain, lbl, st_lbl, cols, title, path_save, save_figs)
# """

# PLOT CHARACTERISTICS OF MID AND INI WINDOWS IN THE SAME PLOT, FOR PROPORTIONAL AND CONSTANT INPUT RATE CHANGES
"""
lbl = ['st_ini_prop', 'mtr_ini_prop', 'st_ini_fix', 'mtr_ini_fix']
lbl2 = ['st_mid_prop', 'mtr_mid_prop', 'st_mid_fix', 'mtr_mid_fix']
st_lbl = ['_mean', '_med', '_max', '_min', '_q1', '_q90']  # , '_q5', '_q95']
t_ = ['Steady-state, ini/mid windows (prop)', 'Transition-state, ini/mid windows (prop)',
      'Steady-state, ini/mid windows (cons)', 'Transition-state, ini/mid windows (cons)']
cols = ['tab:orange', 'tab:blue', 'tab:red', 'tab:red', 'tab:green', 'tab:green']  # , 'tab:olive', 'tab:olive']
name_save = folder_plots + dr_gain_control_file + '_windows_statistics3.png'
title = description.split(",")[0] + r', $\tau_{lif}$ ' + str(tau_lif * 1e3) + "ms, gain " + str(int(gain * 100)) + "%"
plot_features_2windows_prop_fix(initial_frequencies, dr_gain, lbl, lbl2, st_lbl, cols, t_, title, path_save, save_figs)
# """

# SIMPLE PLOT OF DIFFERENCES OF STEADY-STATE BETWEEN MID AND INI WINDOWS FOR PROPORTIONAL AND CONSTANT CHANGE OF RATES
# AND THE DIFFERENCES BETWEEN MAX OF MID WINDOW AND MEDIAN OF INI WINDOW
"""
# ylims = [-62.5, -54.0]  # [-70.05, -52]
lbl = ['st_mid_prop', 'st_mid_fix']
lbl2 = ['st_ini_prop', 'st_ini_fix']
st_lbl = ['_max', '_min', '_q10', '_q90', '_mean', '_med']
cols_ = ['tab:red', 'tab:red', 'tab:green', 'tab:green', 'tab:orange', 'tab:blue']
t_ = [r"$mid_{st} - ini_{st}$ (Prop)", r"$mid_{st} - ini_{st}$ (Cons)",
      "Max Transient (Proportional)", "Max Transient (Constant)"]
name_save = folder_plots + dr_gain_control_file + '_' + 'diff_st_log.png'
title = ("Steady-state: " + description.split(",")[0] + r', $\tau_{lif}$ ' + str(tau_lif * 1e3) + "ms, gain " +
         str(int(gain * 100)) + "%")
plot_diff_windows(initial_frequencies, dr_gain, lbl, lbl2, st_lbl, cols_, t_, title_graph=title, name_save=name_save,
                  save_figs=save_figs)
# """

# **********************************************************************************************************************
# SINGLE GAIN
# **********************************************************************************************************************

# File names
"""
dr_syn_filtering_file = get_name_file(sfreq, s_model, n_model, ind, num_syn, lif_output, tau_lif, False, imputations,.5)
dr_gain_control_file = get_name_file(sfreq, s_model, n_model, ind, num_syn, lif_output, tau_lif, True, imputations,gain)
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
    # lif_params2, syn_params, lif_params, name_params = dr_filt['lif_params2'], dr_filt['syn_params'], 
    # dr_filt['lif_params'], dr_filt['name_params']

    '''
    # Time conditions
    max_t, sfreq, time_vector, L = sim_params['max_t'], sim_params['sfreq'], sim_params['time_vector'], sim_params['L']
    dt = 1 / sfreq
    Le_time_win = int(max_t / num_changes_rate)

    # time transition
    t_tra = dr_filt['time_transition']

    # Parameters in dict format
    params = dict(zip(name_params, syn_params))
    # '''

if os.path.isfile(path_vars + dr_gain_control_file) and not run_experiment:
    dr_gain = loadObject(dr_gain_control_file, path_vars)
# """

# ******************************************************************************************************************
# Plots
"""
lbl = ['st_mid_prop']
lbl2 = ['st_ini_prop']
st_lbl = ['_max',  '_q95', '_q90', '_med', '_min', '_q5', '_q10', '_mean']
cols_ = ['tab:red', 'tab:olive', 'tab:green', 'tab:blue', 'tab:red', 'tab:olive', 'tab:green', 'tab:orange']
# st_lbl = ['_max',  '_q90', '_med', '_min', '_q10', '_mean']
# cols_ = ['tab:red', 'tab:green', 'tab:blue', 'tab:red', 'tab:green', 'tab:orange']
# ylims = [-62.5, -54.0]  # [-70.05, -52]
title = ("Steady-state: " + description.split(",")[0] + r', $\tau_{lif}$ ' +
         str(tau_lif * 1e3) + "ms, gain " + str(int(gain * 100)) + "%")
path_save = folder_plots + dr_gain_control_file + '_phase_portrait.png'
plot_gain_filtering(dr_gain, dr_filt, lbl, lbl2, st_lbl, cols_, title, path_save, True)
# title = "transition-state: " + title.split(":")[1]
# lbl = ['mtr_mid_prop']
# lbl2 = ['mtr_ini_prop']
# plot_gain_filtering(dr_gain, dr_filt, lbl, lbl2, st_lbl, cols_, title, path_save, True)
# """

# PLOT OF CHARACTERISTICS FOR INI, MID, AND END WINDOWS. SPLITTED BY PROPORTIONAL AND CONSTANT INPUT RATE CHANGES
"""
# Steady-state
lbl = ['st_ini_prop', 'st_mid_prop', 'st_end_prop', 'st_ini_fix', 'st_mid_fix', 'st_end_fix']
st_lbl = ['_mean', '_med', '_max', '_min', '_q10', '_q90', '_q5', '_q95']
cols = ['tab:orange', 'tab:blue', 'tab:red', 'tab:red', 'tab:green', 'tab:green', 'tab:olive', 'tab:olive']
# st_lbl = ['_max',  '_q90', '_med', '_min', '_q1', '_mean']
# cols_ = ['tab:red', 'tab:green', 'tab:blue', 'tab:red', 'tab:green', 'tab:orange']
title = ("Steady-state: " + description.split(",")[0] + r', $\tau_{lif}$ ' + str(tau_lif * 1e3) + "ms, gain " +
         str(int(gain * 100)) + "%")
path_save = folder_plots + dr_gain_control_file + '_windows_statistics.png'
plot_features_windows_prop_fix(initial_frequencies, dr_gain, lbl, st_lbl, cols, title, path_save, save_figs)

# Transition-state
lbl = ['mtr_ini_prop', 'mtr_mid_prop', 'mtr_end_prop', 'mtr_ini_fix', 'mtr_mid_fix', 'mtr_end_fix']
title = ("Transition-state: " + description.split(",")[0] + r', $\tau_{lif}$ ' + str(tau_lif * 1e3) + "ms, gain " +
         str(int(gain * 100)) + "%")
path_save = folder_plots + dr_gain_control_file + '_windows_statistics.png'
plot_features_windows_prop_fix(initial_frequencies, dr_gain, lbl, st_lbl, cols, title, path_save, save_figs)
# """

# PLOT OF CHARACTERISTICS FOR INI, MID, AND END WINDOWS. FOR PROPORTIONAL AND CONSTANT INPUT RATE CHANGES
"""
lbl = ['st_ini_prop', 'st_mid_prop', 'st_end_prop', 'mtr_ini_prop', 'mtr_mid_prop', 'mtr_end_prop']
st_lbl = ['_mean', '_med', '_max', '_min', '_q10', '_q90', '_q5', '_q95']
cols = ['tab:orange', 'tab:blue', 'tab:red', 'tab:red', 'tab:green', 'tab:green', 'tab:olive', 'tab:olive']
# st_lbl = ['_max',  '_q90', '_med', '_min', '_q1', '_mean']
# cols_ = ['tab:red', 'tab:green', 'tab:blue', 'tab:red', 'tab:green', 'tab:orange']
title = ("Transition- and steady-state: " + description.split(",")[0] + r', $\tau_{lif}$ ' + str(tau_lif * 1e3) + 
         "ms, gain " + str(int(gain * 100)) + "%")
path_save = folder_plots + dr_gain_control_file + '_windows_statistics.png'
plot_features_windows_prop_fix(initial_frequencies, dr_gain, lbl, st_lbl, cols, title, path_save, save_figs)
# """

# PLOT CHARACTERISTICS OF MID AND INI WINDOWS IN THE SAME PLOT, FOR PROPORTIONAL AND CONSTANT INPUT RATE CHANGES
"""
lbl = ['st_ini_prop', 'mtr_ini_prop', 'st_ini_fix', 'mtr_ini_fix']
lbl2 = ['st_mid_prop', 'mtr_mid_prop', 'st_mid_fix', 'mtr_mid_fix']
st_lbl = ['_mean', '_med', '_max', '_min', '_q1', '_q90']  # , '_q5', '_q95']
t_ = ['Steady-state, ini/mid windows (prop)', 'Transition-state, ini/mid windows (prop)',
      'Steady-state, ini/mid windows (cons)', 'Transition-state, ini/mid windows (cons)']
cols = ['tab:orange', 'tab:blue', 'tab:red', 'tab:red', 'tab:green', 'tab:green']  # , 'tab:olive', 'tab:olive']
name_save = folder_plots + dr_gain_control_file + '_windows_statistics3.png'
title = description.split(",")[0] + r', $\tau_{lif}$ ' + str(tau_lif * 1e3) + "ms, gain " + str(int(gain * 100)) + "%"
plot_features_2windows_prop_fix(initial_frequencies, dr_gain, lbl, lbl2, st_lbl, cols, t_, title, path_save, save_figs)
# """

# SIMPLE PLOT OF DIFFERENCES OF STEADY-STATE BETWEEN MID AND INI WINDOWS FOR PROPORTIONAL AND CONSTANT CHANGE OF RATES
# AND THE DIFFERENCES BETWEEN MAX OF MID WINDOW AND MEDIAN OF INI WINDOW
"""
# ylims = [-62.5, -54.0]  # [-70.05, -52]
lbl = ['st_mid_prop', 'st_mid_fix']
lbl2 = ['st_ini_prop', 'st_ini_fix']
st_lbl = ['_max', '_min', '_q10', '_q90', '_mean', '_med']
cols_ = ['tab:red', 'tab:red', 'tab:green', 'tab:green', 'tab:orange', 'tab:blue']
t_ = [r"$mid_{st} - ini_{st}$ (Prop)", r"$mid_{st} - ini_{st}$ (Cons)",
      "Max Transient (Proportional)", "Max Transient (Constant)"]
name_save = folder_plots + dr_gain_control_file + '_' + 'diff_st_log.png'
title = ("Steady-state: " + description.split(",")[0] + r', $\tau_{lif}$ ' + str(tau_lif * 1e3) + "ms, gain " +
         str(int(gain * 100)) + "%")
plot_diff_windows(initial_frequencies, dr_gain, lbl, lbl2, st_lbl, cols_, t_, title_graph=title, name_save=name_save,
                  save_figs=save_figs)
# """
