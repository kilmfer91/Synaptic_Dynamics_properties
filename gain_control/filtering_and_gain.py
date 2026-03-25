from gain_control.utils_gc import *


def avg_f(vec):
    return np.mean(vec, axis=0)


# ******************************************************************************************************************
# STP model and extra global variables
# (Experiment 2) freq. response decay around 100Hz
# (Experiment 3) freq. response decay around 10Hz
# (Experiment 4) freq. response from Gain Control paper
# (Experiment 5) freq. response decay around 100Hz
# (Experiment 6) freq. response decay around 10Hz
s_model = 'DoornSTD'
n_model = "HH"
ind = 1
# save_vars = True
run_experiment = False
save_figs = True
imputations = True
lif_output = True
n_noise = True
num_syn = 1
gain = 0.1

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
lbl_h_tr = ['H_PSR_tr']
lbl_h_st = ['H_PSR_st']
titles_H = ['tr-ini win.', 'tr-mid win.', 'tr-end win.', 'st-ini win.', 'st-mid win.', 'st-end win.']
st_lbl = ['_max',  '_q95', '_q90', '_med', '_min', '_q5', '_q10', '_mean']
cols_ = ['tab:red', 'tab:olive', 'tab:green', 'tab:blue', 'tab:red', 'tab:olive', 'tab:green', 'tab:orange']
c_g = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
       'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
# st_lbl = ['_max',  '_q90', '_med', '_min', '_q10', '_mean']
# cols_ = ['tab:red', 'tab:green', 'tab:blue', 'tab:red', 'tab:green', 'tab:orange']
# ylims = [-62.5, -54.0]  # [-70.05, -52]
title = ("Steady-state of model " + s_model + ' multiple gains')
if n_model == 'LIF':
    title = ("Steady-state of model " + s_model + r', $\tau_{lif}$ ' + str(tau_lif) + "ms, multiple gains")

figNeuron = plt.figure(figsize=(15, 8))  # 6.5, 5
plt.suptitle(title + ". Mem. pot.")
ax_ = [figNeuron.add_subplot(2, 4, j + 1) for j in range(len(st_lbl))]
figSynapse = plt.figure(figsize=(15, 8))  # 6.5, 5
plt.suptitle(title + ". Synapse")
ax_s = [figSynapse.add_subplot(2, 4, j + 1) for j in range(len(st_lbl))]
figEntropy = plt.figure(figsize=(15, 8))
plt.suptitle(title + ". Information Theory analysis")
ax_h = [figEntropy.add_subplot(2, 3, j + 1) for j in range(6)]

alpha = 0.3
markers = ['+', '*']
alphas = [1.0, 0.5]

# **********************************************************************************************************************
gain_v = [0.1, 0.5, 1.0]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
filt_dict_loaded = False

# Auxiliar variables
description = ""          
dr_filt = None            
dr_gain = None            
initial_frequencies = []  
i_g = 0
for gain in gain_v:
    # File names
    dr_syn_filtering_file = get_name_file(sfreq, s_model, n_model, ind, num_syn, lif_output, tau_lif, False,
                                          imputations, gain, n_noise=n_noise)
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

        # filt_dict_loaded = True
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
        # Information theory analysis
        aux_H_iw_tr = np.copy(dr_gain[lbl_h_tr[i]][0, :])
        aux_H_mw_tr = np.copy(dr_gain[lbl_h_tr[i]][1, :])
        aux_H_ew_tr = np.copy(dr_gain[lbl_h_tr[i]][2, :])
        aux_H_iw_st = np.copy(dr_gain[lbl_h_st[i]][0, :])
        aux_H_mw_st = np.copy(dr_gain[lbl_h_st[i]][1, :])
        aux_H_ew_st = np.copy(dr_gain[lbl_h_st[i]][2, :])
        aux_H_tr = [aux_H_iw_tr, aux_H_mw_tr, aux_H_ew_tr, aux_H_iw_st, aux_H_mw_st, aux_H_ew_st]
        aux_det_H_iw_tr = np.copy(dr_filt[lbl_h_tr[i]][0, :])
        aux_det_H_mw_tr = np.copy(dr_filt[lbl_h_tr[i]][1, :])
        aux_det_H_ew_tr = np.copy(dr_filt[lbl_h_tr[i]][2, :])
        aux_det_H_iw_st = np.copy(dr_filt[lbl_h_st[i]][0, :])
        aux_det_H_mw_st = np.copy(dr_filt[lbl_h_st[i]][1, :])
        aux_det_H_ew_st = np.copy(dr_filt[lbl_h_st[i]][2, :])
        aux_det_H_tr = [aux_det_H_iw_tr, aux_det_H_mw_tr, aux_det_H_ew_tr, aux_det_H_iw_st, aux_det_H_mw_st, aux_det_H_ew_st]
        f_vector = dr_gain['initial_frequencies']

        for j in range(len(st_lbl)):

            # **********************************************************************************************************
            # For Membrane potential analysis
            aux_gain = np.copy(dr_gain[lbl[i] + st_lbl[j]] - dr_gain[lbl2[i] + st_lbl[j]])
            aux_filt = np.copy(dr_gain[lbl2[i] + st_lbl[j]])
            aux_det_gain = np.copy(dr_filt[lbl[i] + st_lbl[j]] - dr_filt[lbl2[i] + st_lbl[j]])[0, :]
            aux_det_filt = np.copy(dr_filt[lbl2[i] + st_lbl[j]][0, :])
            if n_model == 'HH':
                aux_gain *= 1e3
                aux_filt *= 1e3
                aux_det_gain *= 1e3
                aux_det_filt *= 1e3

            # Deterministic plots
            """
            if i_g == 0:
                ax_[j].plot(aux_det_filt, aux_det_gain, c='gray', alpha=alphas[i], label='Det')
            else:
                ax_[j].plot(aux_det_filt, aux_det_gain, c='gray', alpha=alphas[i])
            ax_[j].scatter(aux_det_filt, aux_det_gain, c=c_g[i_g], marker=markers[i], alpha=alphas[i])
            ax_[j].scatter(aux_det_filt[0], aux_det_gain[0], c='black')
            # """
            # Stochastic plots
            ax_[j].scatter(avg_f(aux_filt), avg_f(aux_gain), marker=markers[i], alpha=alphas[i])
            ax_[j].plot(avg_f(aux_filt), avg_f(aux_gain), alpha=alphas[i], label=gain)
            if i == 0: ax_[j].fill_between(avg_f(aux_filt), np.quantile(aux_gain, 0.1, axis=0),
                                           np.quantile(aux_gain, 0.9, axis=0), color=cols_[j], alpha=0.1)
            ax_[j].scatter(avg_f(aux_filt)[0], avg_f(aux_gain)[0], c='black')

            # **********************************************************************************************************
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

            # Deterministic plots
            """
            if i_g == 0:
                ax_s[j].plot(aux_det_filt, aux_det_gain, c='gray', alpha=alphas[i], label='Det')
            else:
                ax_s[j].plot(aux_det_filt, aux_det_gain, c='gray', alpha=alphas[i])
            ax_s[j].scatter(aux_det_filt, aux_det_gain, c=c_g[i_g], marker=markers[i], alpha=alphas[i])
            ax_s[j].scatter(aux_det_filt[0], aux_det_gain[0], c='black')
            # """
            # Stochastic plots
            ax_s[j].scatter(avg_f(aux_filt), avg_f(aux_gain), marker=markers[i], alpha=alphas[i])
            ax_s[j].plot(avg_f(aux_filt), avg_f(aux_gain), alpha=alphas[i], label=gain)
            if i == 0: ax_s[j].fill_between(avg_f(aux_filt), np.quantile(aux_gain, 0.1, axis=0),
                                           np.quantile(aux_gain, 0.9, axis=0), color=cols_[j], alpha=0.1)
            ax_s[j].scatter(avg_f(aux_filt)[0], avg_f(aux_gain)[0], c='black')

            # **********************************************************************************************************
            # Information theory analysis
            if j < 6:
                x_vec, x_det_vec = f_vector, f_vector
                # if j >= 3:
                #     aux_filt = np.copy(dr_gain[lbl2[i] + st_lbl[7]])  # mean of filtering property (index 7)
                #     aux_det_filt = np.copy(dr_filt[lbl2[i] + st_lbl[7]][0, :])  # mean of filtering property (index 7)
                #     x_vec, x_det_vec = avg_f(aux_filt), aux_det_filt
                ax_h[j].plot(x_vec, aux_H_tr[j], c=c_g[i_g], marker=markers[i], label=gain)
                ax_h[j].scatter(x_vec, aux_H_tr[j], c=c_g[i_g], marker=markers[i])
                ax_h[j].scatter(x_vec[0], aux_H_tr[j][0], c='black')
                # ax_h[j].plot(x_det_vec, aux_det_H_tr[j][:len(aux_det_filt)], marker=markers[i], c=c_g[i_g])
                # ax_h[j].scatter(x_det_vec, aux_det_H_tr[j][:len(aux_det_filt)], marker=markers[i], c=c_g[i_g], label=gain)
                # ax_h[j].scatter(x_det_vec[0], aux_det_H_tr[j][0], c='black')
    i_g += 1

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
             str(int(sfreq * 1e-3)) + 'k_tauLIF_' + str(tau_lif) + 'ms' + '_phase_portrait')
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

    # Information theory analysis
    if j < 6:
        ax_h[j].set_xlabel("Rate (Hz)", color='gray')
        ax_h[j].set_ylabel("Information entropy (bits)", color='gray')
        if n_model != 'HH':
            ax_h[j].set_ylim(ylims_s[0][j], ylims_s[1][j])
            # ax_h[j].set_xlim(xlims[0][j], xlims[1][j])
        # ax_h[j].set_ylim(ylims)
        ax_h[j].set_title(titles_H[j], c="gray")
        ax_h[j].grid()

ax_[int(len(st_lbl) / 2) - 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax_s[int(len(st_lbl) / 2) - 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax_h[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
figNeuron.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
figSynapse.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
figEntropy.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
if save_figs:
    figNeuron.savefig(path_save + "_neuron.png", format='png')
    figSynapse.savefig(path_save + "_synase.png", format='png')
    figEntropy.savefig(path_save + "_information.png", format='png')

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
