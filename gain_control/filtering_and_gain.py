from gain_control.utils_gc import *

# Neuron: min (-0.050) => 1: -0.0395, 2: -0.0473, 3: -0.0472, 4: -0.0467, 5: -0.0458, 6: -0.050, 7: -0.0473
# AMPA:  min (0.0000) => 1: 0.0000, 2: 0.0000, 3: 0.0000, 4: 0.0000, 5: 0.0000, 6: 0.0000, 7: 0.0000
#        max (3.0967) => 1: 0.8504, 2: 1.1592, 3: 0.9177, 4: 1.6524, 5: 2.1846, 6: 3.0967, 7: 2.9327
# bNMDA: min (0.0000) => 1: 0.0000, 2: 0.0000, 3: 0.0000, 4: 0.0000, 5: 0.0000, 6: 0.0000, 7: 0.0000
#        max (0.7989) => 1: 0.2157, 2: 0.2374, 3: 0.2083, 4: 0.5646, 5: 0.6419, 6: 0.7989, 7: 0.51747
# ******************************************************************************************************************
# STP model and extra global variables
# (Experiment 2) freq. response decay around 100Hz
# (Experiment 3) freq. response decay around 10Hz
# (Experiment 4) freq. response from Gain Control paper
# (Experiment 5) freq. response decay around 100Hz
# (Experiment 6) freq. response decay around 10Hz
s_model = 'DoornSTD'
n_model = "HH"
ind = 3
# save_vars = True
run_experiment = False
save_figs = True
imputations = True
lif_output = True
n_noise = True
num_syn = 1

# Sampling frequency and conditions for running parallel or single LIF neurons
sfreq = 6e3
tau_lif = 1  # ms
total_realizations = 1  # 100
num_realizations = 1  # 8 for server, 4 for macbook air
t_tra = None  # 0.25

# Path variables
path_vars = "../gain_control/variables/synaptic_entropy_high_freq/"
check_create_folder(path_vars)
folder_plots = '../gain_control/plots/'
check_create_folder(folder_plots)

# Normalization
norm_neuron = True
min_n, max_n = -0.05, 0.0
# **********************************************************************************************************************
# MULTIPLE GAINS
# **********************************************************************************************************************
gain_v = [1.0]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
filt_dict_loaded = False

# Titles graphs
title = "Model " + s_model + ', ind ' + str(ind)
titleH = "Model " + s_model + ', ind ' + str(ind)
if len(gain_v) == 1:
    title += ', gain ' + str(int(gain_v[0] * 100)) + '%'
    titleH += ', gain ' + str(int(gain_v[0] * 100)) + '%'
else:
    title += ', multiple gains'
    titleH += ', multiple gains'

# ""
# Plot
lbl = ['st_mid_prop']
lbl2 = ['st_ini_prop']
lbl_syn = ['syn_st_mid_prop']
lbl2_syn = ['syn_st_ini_prop']
lbl_synb = ['syn_b_st_mid_prop']
lbl2_synb = ['syn_b_st_ini_prop']
lbl_h_tr = ['H_PSR_tr']
lbl_h_st = ['H_PSR_st']
lbl_hI_tr = ['H_ISI_tr']
lbl_hI_st = ['H_ISI_st']
lbl_h2_tr = ['H_PSR_tr_100']
lbl_h2_st = ['H_PSR_st_100']
lbl_h_s_tr = ['H_PSR_syn_tr']
lbl_h_s_st = ['H_PSR_syn_st']
lbl_h_sb_tr = ['H_PSR_syn_b_tr']
lbl_h_sb_st = ['H_PSR_syn_b_st']
titles_H = ['ini win.', 'mid win.', 'end win.', 'st-ini win.', 'st-mid win.', 'st-end win.']
titles_H_syn = ['AMPA ini win.', 'AMPA mid win.', 'AMPA end win.', 'NMDA ini win.', 'NMDA mid win.', 'NMDA end win.']
labels_H = ['tr-', 'tr-', 'tr-', 'st-', 'st-', 'st-']
st_lbl = ['_max',  '_q95', '_q90', '_med', '_min', '_q5', '_q10', '_mean']
cols_ = ['tab:red', 'tab:olive', 'tab:green', 'tab:blue', 'tab:red', 'tab:olive', 'tab:green', 'tab:orange']
c_g = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
       'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

if n_model == 'LIF':
    title = ("Steady-state of model " + s_model + r', $\tau_{lif}$ ' + str(tau_lif) + "ms, multiple gains")

# Synaptic filtering vs. Gain-Control for Neuron
plt.rcParams['figure.constrained_layout.use'] = True
figNeuron = plt.figure(figsize=(15, 8))  # 6.5, 5
plt.suptitle(title + ". Mem. pot.")
ax_ = [figNeuron.add_subplot(2, 4, j + 1) for j in range(len(st_lbl))]

# Synaptic filtering vs. Gain-Control for Synapse A
figSynapse = plt.figure(figsize=(15, 8))  # 6.5, 5
plt.suptitle(title + ". Synapse")
ax_s = [figSynapse.add_subplot(2, 4, j + 1) for j in range(len(st_lbl))]
# figEntropyInput = plt.figure(figsize=(15, 4))
# plt.suptitle(titleH + ". Information Theory (Input)")
# ax_hI = [figEntropyInput.add_subplot(1, 3, j + 1) for j in range(3)]

# Synaptic entropy in frequency for Input
figEntropyInput = plt.figure(figsize=(15, 4))
plt.suptitle(titleH + ". Information Theory (Input)")
ax_hI = [figEntropyInput.add_subplot(1, 3, j + 1) for j in range(3)]

# Synaptic entropy in frequency for Neuron
figEntropy = plt.figure(figsize=(15, 4))
plt.suptitle(titleH + ". Information Theory (Neuron)")
ax_h = [figEntropy.add_subplot(1, 3, j + 1) for j in range(3)]

# Synaptic entropy in frequency for Synapse(s)
figEntropySyn = plt.figure(figsize=(15, 8))
plt.suptitle(titleH + ". Information Theory (Synapses) ")

# Plots of computational properties vs rates
st_lbl_b = ['H - filtering', 'H - Gain-control', 'Transition time', 'Synaptic Filtering', 'GC - max', 'GC - min',
            'GC - var', 'GC - med']
# Neuron
figCompPropNeuron = plt.figure(figsize=(25, 3.6))  # 15, 8
plt.suptitle(title + ". Mem. pot.", fontsize=23)
axb_ = [figCompPropNeuron.add_subplot(1, 8, j + 1) for j in range(8)]  # [0, 1, 4, 5, 6, 7, 8, 9]]
figCompPropSyn = plt.figure(figsize=(25, 3.6))  # 15, 8
plt.suptitle(title + ". AMPA synapse", fontsize=23)
axb_s = [figCompPropSyn.add_subplot(1, 8, j + 1) for j in range(8)]  # [0, 1, 4, 5, 6, 7, 8, 9]]

fig_syn_b = False
fig_H_100 = False

alpha = 0.3
markers = ['+', '*']
alphas = [1.0, 0.5]

# **********************************************************************************************************************
filt_dict_loaded = False

# Auxiliar variables
description = ""          
dr_filt = None            
dr_gain = None            
initial_frequencies = []  
i_g = 0
l_gain = len(gain_v)
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

    if os.path.isfile(path_vars + dr_gain_control_file) and not run_experiment:
        dr_gain = loadObject(dr_gain_control_file, path_vars)

    f_vec = dr_gain['initial_frequencies']
    f_vecD = dr_filt['initial_frequencies']
    # **********************************************************************************************************
    # For Entropy analysis
    # Setting figure for synaptic entropy in case there are one or two synapses
    if not fig_syn_b:
        if 'H_PSR_syn_b_tr' in dr_gain.keys():
            # Synaptic filtering vs. Gain-Control for Synapse B
            ax_hs = [figEntropySyn.add_subplot(2, 3, j + 1) for j in range(6)]
            fig_syn_b = True
            figSynapseb = plt.figure(figsize=(15, 8))  # 6.5, 5
            plt.suptitle(title + ". Synapse B")
            ax_sb = [figSynapseb.add_subplot(2, 4, j + 1) for j in range(len(st_lbl))]
            # Plots of computational properties vs rates
            figCompPropSynb = plt.figure(figsize=(25, 3.6))  # 6.5, 5
            plt.suptitle(title + ". NMDA synapse", fontsize=23)
            axb_sb = [figCompPropSynb.add_subplot(1, 8, j + 1) for j in range(8)]  # [0, 1, 4, 5, 6, 7, 8, 9]]

        else:
            ax_hs = [figEntropySyn.add_subplot(1, 3, j + 1) for j in range(3)]

    # Information theory analysis for Inputs (ISI)
    H_i_tr, H_m_tr, H_e_tr = dr_gain[lbl_hI_tr[0]][0, :], dr_gain[lbl_hI_tr[0]][1, :], dr_gain[lbl_hI_tr[0]][2, :]
    H_i_st, H_m_st, H_e_st = dr_gain[lbl_hI_st[0]][0, :], dr_gain[lbl_hI_st[0]][1, :], dr_gain[lbl_hI_st[0]][2, :]
    aux_HI = [H_i_tr, H_m_tr, H_e_tr, H_i_st, H_m_st, H_e_st]
    # Information theory analysis for neurons - fixed bin size
    H_iw_tr, H_mw_tr, H_ew_tr = dr_gain[lbl_h_tr[0]][0, :], dr_gain[lbl_h_tr[0]][1, :], dr_gain[lbl_h_tr[0]][2, :]
    H_iw_st, H_mw_st, H_ew_st = dr_gain[lbl_h_st[0]][0, :], dr_gain[lbl_h_st[0]][1, :], dr_gain[lbl_h_st[0]][2, :]
    aux_H = [H_iw_tr, H_mw_tr, H_ew_tr, H_iw_st, H_mw_st, H_ew_st]
    H_iw_tr, H_mw_tr, H_ew_tr = dr_filt[lbl_h_tr[0]][0, :], dr_filt[lbl_h_tr[0]][1, :], dr_filt[lbl_h_tr[0]][2, :]
    H_iw_st, H_mw_st, H_ew_st = dr_filt[lbl_h_st[0]][0, :], dr_filt[lbl_h_st[0]][1, :], dr_filt[lbl_h_st[0]][2, :]
    aux_det_H = [H_iw_tr, H_mw_tr, H_ew_tr, H_iw_st, H_mw_st, H_ew_st]
    # Information theory analysis for synapses - fixed bin size
    H_iw_tr, H_mw_tr, H_ew_tr = (dr_gain[lbl_h_s_tr[0]][0, :], dr_gain[lbl_h_s_tr[0]][1, :],
                                 dr_gain[lbl_h_s_tr[0]][2, :])
    H_iw_st, H_mw_st, H_ew_st = (dr_gain[lbl_h_s_st[0]][0, :], dr_gain[lbl_h_s_st[0]][1, :],
                                 dr_gain[lbl_h_s_st[0]][2, :])
    aux_H_s = [H_iw_tr, H_mw_tr, H_ew_tr, H_iw_st, H_mw_st, H_ew_st]
    H_iw_tr, H_mw_tr, H_ew_tr = (dr_filt[lbl_h_s_tr[0]][0, :], dr_filt[lbl_h_s_tr[0]][1, :],
                                 dr_filt[lbl_h_s_tr[0]][2, :])
    H_iw_st, H_mw_st, H_ew_st = (dr_filt[lbl_h_s_st[0]][0, :], dr_filt[lbl_h_s_st[0]][1, :],
                                 dr_filt[lbl_h_s_st[0]][2, :])
    aux_det_H_s = [H_iw_tr, H_mw_tr, H_ew_tr, H_iw_st, H_mw_st, H_ew_st]
    if fig_syn_b:
        H_iw_tr, H_mw_tr, H_ew_tr = (dr_gain[lbl_h_sb_tr[0]][0, :], dr_gain[lbl_h_sb_tr[0]][1, :],
                                     dr_gain[lbl_h_sb_tr[0]][2, :])
        H_iw_st, H_mw_st, H_ew_st = (dr_gain[lbl_h_sb_st[0]][0, :], dr_gain[lbl_h_sb_st[0]][1, :],
                                     dr_gain[lbl_h_sb_st[0]][2, :])
        aux_H_sb = [H_iw_tr, H_mw_tr, H_ew_tr, H_iw_st, H_mw_st, H_ew_st]
        H_iw_tr, H_mw_tr, H_ew_tr = (dr_filt[lbl_h_sb_tr[0]][0, :], dr_filt[lbl_h_sb_tr[0]][1, :],
                                     dr_filt[lbl_h_sb_tr[0]][2, :])
        H_iw_st, H_mw_st, H_ew_st = (dr_filt[lbl_h_sb_st[0]][0, :], dr_filt[lbl_h_sb_st[0]][1, :],
                                     dr_filt[lbl_h_sb_st[0]][2, :])
        aux_det_H_sb = [H_iw_tr, H_mw_tr, H_ew_tr, H_iw_st, H_mw_st, H_ew_st]

    # ******************************************************************************************************************
    # Plots 1

    # **********************************************************************************************************
    dr_ = dr_gain
    # For Membrane potential analysis
    var_ = ['st_mid_prop_max', 'mtr_mid_prop_max', 'st_ini_prop_max', 'mtr_ini_prop_max', 'st_mid_prop_q90',
                     'st_mid_prop_q10', 'st_ini_prop_q90', 'st_ini_prop_q10', 'st_mid_prop_min', 'mtr_mid_prop_min',
                     'st_ini_prop_min', 'mtr_ini_prop_min', 'st_mid_prop_med', 'st_ini_prop_med']
    # Plotting properties
    axb_ = plot_properties_in_freq(dr_, var_, f_vec, aux_H, gain, axb_, dr_['time_transition'], min_n=min_n,
                                   max_n=max_n, c_g=c_g[i_g], plot_filt=i_g == 0, norm_neuron=True)

    # For Synaptic analysis
    lbl_prop_freq = ['syn_st_mid_prop_max', 'syn_mtr_mid_prop_max', 'syn_st_ini_prop_max', 'syn_mtr_ini_prop_max',
                     'syn_st_mid_prop_q90', 'syn_st_mid_prop_q10', 'syn_st_ini_prop_q90', 'syn_st_ini_prop_q10',
                     'syn_st_mid_prop_min', 'syn_mtr_mid_prop_min', 'syn_st_ini_prop_min', 'syn_mtr_ini_prop_min',
                     'syn_st_mid_prop_med', 'syn_st_ini_prop_med']
    # Plotting properties
    axb_s = plot_properties_in_freq(dr_, lbl_prop_freq, f_vec, aux_H_s, gain, axb_s, dr_['time_transition_syn'],
                                    c_g=c_g[i_g], plot_filt=i_g == 0, norm_neuron=False)

    # For Synaptic analysis (second synapse)
    lbl_b_prop_freq = ['syn_b_st_mid_prop_max', 'syn_b_mtr_mid_prop_max', 'syn_b_st_ini_prop_max',
                       'syn_b_mtr_ini_prop_max', 'syn_b_st_mid_prop_q90', 'syn_b_st_mid_prop_q10',
                       'syn_b_st_ini_prop_q90', 'syn_b_st_ini_prop_q10', 'syn_b_st_mid_prop_min',
                       'syn_b_mtr_mid_prop_min', 'syn_b_st_ini_prop_min', 'syn_b_mtr_ini_prop_min',
                       'syn_b_st_mid_prop_med', 'syn_b_st_ini_prop_med']
    # H_list = aux_H
    # Plotting properties
    if fig_syn_b:
        axb_sb = plot_properties_in_freq(dr_, lbl_b_prop_freq, f_vec, aux_H_sb, gain, axb_sb,
                                         dr_['time_transition_syn_b'], c_g=c_g[i_g], plot_filt=i_g == 0,
                                         norm_neuron=False)

    # Plots 2
    for i in [0]:  # range(len(lbl)):
        for j in range(len(st_lbl)):
            # **********************************************************************************************************
            # For Membrane potential analysis
            n_sto_m_st = norm_array(dr_gain[lbl[i] + st_lbl[j]], compute_norm=norm_neuron, min_n=min_n, max_n=max_n)
            n_sto_i_st = norm_array(dr_gain[lbl2[i] + st_lbl[j]], compute_norm=norm_neuron, min_n=min_n, max_n=max_n)
            n_det_m_st = norm_array(dr_filt[lbl[i] + st_lbl[j]], compute_norm=norm_neuron, min_n=min_n, max_n=max_n)
            n_det_i_st = norm_array(dr_filt[lbl2[i] + st_lbl[j]], compute_norm=norm_neuron, min_n=min_n, max_n=max_n)
            aux_gain = np.copy(n_sto_m_st - n_sto_i_st)
            aux_filt = np.copy(n_sto_i_st)
            aux_det_gain = np.copy(n_det_m_st - n_det_i_st)[0, :]
            aux_det_filt = np.copy(n_det_i_st[0, :])
            # if n_model == 'HH':
            #     aux_gain *= 1e3
            #     aux_filt *= 1e3
            #     aux_det_gain *= 1e3
            #     aux_det_filt *= 1e3

            # Deterministic plots
            # """
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
            # First synapse
            aux_gain = np.copy(dr_gain[lbl_syn[i] + st_lbl[j]] - dr_gain[lbl2_syn[i] + st_lbl[j]])
            aux_filt = np.copy(dr_gain[lbl2_syn[i] + st_lbl[j]])
            aux_det_gain = np.copy(dr_filt[lbl_syn[i] + st_lbl[j]] - dr_filt[lbl2_syn[i] + st_lbl[j]])[0, :]
            aux_det_filt = np.copy(dr_filt[lbl2_syn[i] + st_lbl[j]][0, :])
            # if n_model == 'HH':
            #     aux_gain *= 1e3
            #     aux_filt *= 1e3
            #     aux_det_gain *= 1e3
            #     aux_det_filt *= 1e3

            # Deterministic plots
            # """
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

            if fig_syn_b:
                # First synapse
                aux_gain = np.copy(dr_gain[lbl_synb[i] + st_lbl[j]] - dr_gain[lbl2_synb[i] + st_lbl[j]])
                aux_filt = np.copy(dr_gain[lbl2_synb[i] + st_lbl[j]])
                aux_det_gain = np.copy(dr_filt[lbl_synb[i] + st_lbl[j]] - dr_filt[lbl2_synb[i] + st_lbl[j]])[0, :]
                aux_det_filt = np.copy(dr_filt[lbl2_synb[i] + st_lbl[j]][0, :])
                # if n_model == 'HH':
                #     aux_gain *= 1e3
                #     aux_filt *= 1e3
                #     aux_det_gain *= 1e3
                #     aux_det_filt *= 1e3

                # Deterministic plots
                # """
                if i_g == 0:
                    ax_sb[j].plot(aux_det_filt, aux_det_gain, c='gray', alpha=alphas[i], label='Det')
                else:
                    ax_sb[j].plot(aux_det_filt, aux_det_gain, c='gray', alpha=alphas[i])
                ax_sb[j].scatter(aux_det_filt, aux_det_gain, c=c_g[i_g], marker=markers[i], alpha=alphas[i])
                ax_sb[j].scatter(aux_det_filt[0], aux_det_gain[0], c='black')
                # """
                # Stochastic plots
                ax_sb[j].scatter(avg_f(aux_filt), avg_f(aux_gain), marker=markers[i], alpha=alphas[i])
                ax_sb[j].plot(avg_f(aux_filt), avg_f(aux_gain), alpha=alphas[i], label=gain)
                if i == 0: ax_sb[j].fill_between(avg_f(aux_filt), np.quantile(aux_gain, 0.1, axis=0),
                                                np.quantile(aux_gain, 0.9, axis=0), color=cols_[j], alpha=0.1)
                ax_sb[j].scatter(avg_f(aux_filt)[0], avg_f(aux_gain)[0], c='black')

            # **********************************************************************************************************
            # Information theory analysis
            if j < 3:
                # FOR INPUTS
                # Stochastic - transition
                ax_hI[j].plot(f_vec, aux_HI[j], c=c_g[i_g], marker=markers[0], label=labels_H[i_g] + str(gain))
                ax_hI[j].scatter(f_vec[0], aux_HI[j][0], c='black')
                # Stochastic - stationary
                ax_hI[j].plot(f_vec, aux_HI[j + 3], c=c_g[i_g + 3], marker=markers[0],
                             label=labels_H[i_g + 3] + str(gain))
                ax_hI[j].scatter(f_vec[0], aux_HI[j + 3][0], c='black')
                # FOR NEURON
                # For only one computation of entropy (# bins fixed if only one, or fix bin size if two entropies)
                # Stochastic - transition
                ax_h[j].plot(f_vec, aux_H[j], c=c_g[i_g], marker=markers[0], label=labels_H[i_g] + str(gain))
                ax_h[j].scatter(f_vec[0], aux_H[j][0], c='black')
                # Stochastic - stationary
                ax_h[j].plot(f_vec, aux_H[j + 3], c=c_g[i_g + 3], marker=markers[0],
                             label=labels_H[i_g + 3] + str(gain))
                ax_h[j].scatter(f_vec[0], aux_H[j + 3][0], c='black')
                # Deterministic - transition
                ax_h[j].plot(f_vecD, aux_det_H[j][:len(aux_det_filt)], marker=markers[1], c=c_g[i_g], alpha=0.4)  # ,
                             # label=str(gain) + " (det)")
                ax_h[j].scatter(f_vec[0], aux_det_H[j][0], c='gray')
                # Deterministic - stationary
                ax_h[j].plot(f_vecD, aux_det_H[j + 3][:len(aux_det_filt)], marker=markers[1], c=c_g[i_g + 3], alpha=0.4)
                             # label=str(gain) + " (det)")
                ax_h[j].scatter(f_vecD[0], aux_det_H[j + 3][0], c='gray')

                # FOR SYNAPSES
                # """
                # For only one computation of entropy (# bins fixed if only one, or fix bin size if two entropies)
                # Stochastic - transition
                ax_hs[j].plot(f_vec, aux_H_s[j], c=c_g[i_g], marker=markers[0], label=labels_H[i_g] + str(gain))
                ax_hs[j].scatter(f_vec[0], aux_H_s[j][0], c='black')
                # Stochastic - stationary
                ax_hs[j].plot(f_vec, aux_H_s[j + 3], c=c_g[i_g + 3], marker=markers[0],
                             label=labels_H[i_g + 3] + str(gain))
                ax_hs[j].scatter(f_vec[0], aux_H_s[j + 3][0], c='black')
                # Deterministic - transition
                ax_hs[j].plot(f_vecD, aux_det_H_s[j][:len(aux_det_filt)], marker=markers[1], c=c_g[i_g], alpha=0.4)  # ,
                              # label=str(gain) + " (det)")
                ax_hs[j].scatter(f_vecD[0], aux_det_H_s[j][0], c='gray')
                # Deterministic - stationary
                ax_hs[j].plot(f_vecD, aux_det_H_s[j + 3][:len(aux_det_filt)], marker=markers[1], c=c_g[i_g + 3],
                              alpha=0.4)  # , label=str(gain) + " (det)")
                ax_hs[j].scatter(f_vecD[0], aux_det_H_s[j + 3][0], c='gray')
                # """

            if 3 <= j < 6 and fig_syn_b:
                # FOR SECOND SYNAPSE
                # For only one computation of entropy (# bins fixed if only one, or fix bin size if two entropies)
                # Stochastic - transition
                ax_hs[j].plot(f_vec, aux_H_sb[j-3], c=c_g[i_g], marker=markers[0], label=labels_H[i_g] + str(gain))
                ax_hs[j].scatter(f_vec[0], aux_H_sb[j-3][0], c='black')
                # Stochastic - stationary
                ax_hs[j].plot(f_vec, aux_H_sb[j], c=c_g[i_g + 3], marker=markers[0],
                             label=labels_H[i_g] + str(gain))
                ax_hs[j].scatter(f_vec[0], aux_H_sb[j][0], c='black')
                # Deterministic - transition
                ax_hs[j].plot(f_vecD, aux_det_H_sb[j-3][:len(aux_det_filt)], marker=markers[1], c=c_g[i_g], alpha=0.4)
                              # label=str(gain) + " (det)")
                ax_hs[j].scatter(f_vecD[0], aux_det_H_sb[j-3][0], c='gray')
                # Deterministic - stationary
                ax_hs[j].plot(f_vecD, aux_det_H_sb[j][:len(aux_det_filt)], marker=markers[1], c=c_g[i_g + 3], alpha=0.4)
                             # label=str(gain) + " (det)")
                ax_hs[j-3].scatter(f_vecD[0], aux_det_H_sb[j][0], c='gray')

    i_g += 1

xlims = [[-69.9, -69.9, -69.95, -70.01, -70.01, -70.01, -70.01, -70.01],
         [-67.9, -69.1, -69.40, -69.82, -69.82, -69.82, -69.82, -69.82]]
ylims = [[-0.6, -0.6, -0.6, -0.6, -0.6, -0.6, -0.6, -0.6],
         [0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40]]
xlims_s = [[-69.9, -69.9, -69.95, -70.01, -70.01, -70.01, -70.01, -70.01],
           [0.07, -69.1, -69.40, -69.82, -69.82, -69.82, -69.82, -69.82]]
ylims_s = [[-0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02],
           [0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015]]

path_save = (folder_plots + s_model + '_ind_' + str(ind) + '_' + str(len(gain_v)) + '_gains_sf_' +
             str(int(sfreq * 1e-3)) + 'k_tauLIF_' + str(tau_lif) + 'ms' + '_phase_portrait')

# For plot neuron b
sizeF = 20
for j in range(8):
    # For Computational properties in rate
    # axb_[j].set_xlabel("Rate (Hz)", color='gray', fontsize=sizeF)
    # axb_s[j].set_xlabel("Rate (Hz)", color='gray', fontsize=sizeF)
    if fig_syn_b: axb_sb[j].set_xlabel("Rate (Hz)", color='gray', fontsize=sizeF)
    if j < 2:
        axb_[j].set_ylabel("(bits)", color='gray', fontsize=sizeF)
        axb_s[j].set_ylabel("(bits)", color='gray', fontsize=sizeF)
        if fig_syn_b: axb_sb[j].set_ylabel("Entropy (bits)", color='gray', fontsize=sizeF)
    elif j == 2:
        axb_[j].set_ylabel("Time (s)", color='gray', fontsize=sizeF)
        axb_s[j].set_ylabel("Time (s)", color='gray', fontsize=sizeF)
        if fig_syn_b: axb_sb[j].set_ylabel("Time (s)", color='gray', fontsize=sizeF)
    else:
        axb_[j].set_ylabel("(mV)", color='gray', fontsize=sizeF)
        axb_s[j].set_ylabel("(mV)", color='gray', fontsize=sizeF)
        if fig_syn_b: axb_sb[j].set_ylabel("(mV)", color='gray', fontsize=sizeF)
    axb_[j].set_title(st_lbl_b[j], c="gray", fontsize=sizeF)
    axb_[j].grid()
    axb_[j].set_xscale('log')
    # axb_s[j].set_title(st_lbl_b[j], c="gray", fontsize=sizeF)
    axb_s[j].grid()
    axb_s[j].set_xscale('log')
    if fig_syn_b:
        # axb_sb[j].set_title(st_lbl_b[j], c="gray", fontsize=sizeF)
        axb_sb[j].set_xlabel("Rate (Hz)", color='gray', fontsize=sizeF)
        axb_sb[j].grid()
        axb_sb[j].set_xscale('log')

for j in range(len(st_lbl)):
    # Synaptic filtering vs. Gain-Control for Neuron
    ax_[j].set_xlabel("Synaptic filtering (mV)", color='gray')
    ax_[j].set_ylabel("Gain-Control (mV)", color='gray')
    if n_model != 'HH':
        ax_[j].set_ylim(ylims[0][j], ylims[1][j])
        ax_[j].set_xlim(xlims[0][j], xlims[1][j])
    # ax_[j].set_ylim(ylims)
    ax_[j].set_title(st_lbl[j][1:], c="gray")
    ax_[j].grid()

    # Synaptic filtering vs. Gain-Control for Synapse A
    ax_s[j].set_xlabel("Synaptic filtering (mV)", color='gray')
    ax_s[j].set_ylabel("Gain-Control (mV)", color='gray')
    if n_model != 'HH':
        ax_s[j].set_ylim(ylims_s[0][j], ylims_s[1][j])
        # ax_s[j].set_xlim(xlims[0][j], xlims[1][j])
    ax_s[j].set_title(st_lbl[j][1:], c="gray")
    ax_s[j].grid()
    # Synaptic filtering vs. Gain-Control for Synapse B
    if fig_syn_b:
        ax_sb[j].set_xlabel("Synaptic filtering (mV)", color='gray')
        ax_sb[j].set_ylabel("Gain-Control (mV)", color='gray')
        if n_model != 'HH':
            ax_sb[j].set_ylim(ylims_s[0][j], ylims_s[1][j])
            # ax_sb[j].set_xlim(xlims[0][j], xlims[1][j])
        ax_sb[j].set_title(st_lbl[j][1:], c="gray")
        ax_sb[j].grid()

    # Information theory analysis
    if j < 3:
        # INPUT
        ax_hI[j].set_xlabel("Rate (Hz)", color='gray')
        ax_hI[j].set_ylabel("H (bits)", color='gray')
        ax_hI[j].set_ylim((-0.1, 6.7))
        ax_hI[j].set_title(titles_H[j], c="gray")
        ax_hI[j].grid()
        ax_hI[j].set_xscale('log')
        # NEURON
        ax_h[j].set_xlabel("Rate (Hz)", color='gray')
        ax_h[j].set_ylabel("H (bits)", color='gray')
        # if n_model != 'HH':
        #     ax_h[j].set_ylim(ylims_s[0][j], ylims_s[1][j])
        ax_h[j].set_ylim((-0.1, 6.7))
        ax_h[j].set_title(titles_H[j], c="gray")
        ax_h[j].grid()
        ax_h[j].set_xscale('log')
        # SYNAPSE
        ax_hs[j].set_xlabel("Rate (Hz)", color='gray')
        ax_hs[j].set_ylabel("H (bits)", color='gray')
        if n_model != 'HH':
            ax_hs[j].set_ylim(ylims_s[0][j], ylims_s[1][j])
            # ax_hs[j].set_xlim(xlims[0][j], xlims[1][j])
        ax_hs[j].set_ylim((-0.1, 6.7))
        ax_hs[j].set_title(titles_H_syn[j], c="gray")
        ax_hs[j].grid()
        ax_hs[j].set_xscale('log')
    if 3 <= j < 6 and fig_syn_b:
        ax_hs[j].set_xlabel("Rate (Hz)", color='gray')
        ax_hs[j].set_ylabel("H (bits)", color='gray')
        if n_model != 'HH':
            ax_hs[j].set_ylim(ylims_s[0][j], ylims_s[1][j])
        ax_hs[j].set_title(titles_H_syn[j], c="gray")
        ax_hs[j].grid()
        ax_hs[j].set_ylim((-0.1, 6.7))
        ax_hs[j].set_xscale('log')

# Synaptic filtering vs. Gain-Control for Neuron
ax_[int(len(st_lbl) / 2) - 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# figNeuron.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
# Synaptic filtering vs. Gain-Control for Synapse A
ax_s[int(len(st_lbl) / 2) - 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# figSynapse.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
# Plots of computational properties vs rates. Neuron
axb_[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=sizeF - 10)  # 1
axb_[7].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=sizeF - 10)  # 7
# figCompPropNeuron.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
# Plots of computational properties vs rates. Synapse A
axb_s[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=sizeF - 10)  # 1
axb_s[7].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=sizeF - 10)  # 7
# figCompPropSyn.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)

# Synaptic filtering vs. Gain-Control for Synapse B
if fig_syn_b:
    # Synaptic filtering vs. Gain-Control for Synapse B
    ax_sb[int(len(st_lbl) / 2) - 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # figSynapseb.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    # Plots of computational properties vs rates. Synapse B
    axb_sb[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=sizeF - 10)  # 1
    axb_sb[7].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=sizeF - 10)  # 7
    # figCompPropSynb.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)

# Synaptic entropy in frequency for Neuron
ax_h[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# figEntropy.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
# Synaptic entropy in frequency for Input
ax_hI[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# figEntropyInput.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
# Synaptic entropy in frequency for Synapses
ax_hs[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# figEntropySyn.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)

if save_figs:
    figNeuron.savefig(path_save + "_filt_vs_gc_neuron.png", format='png')
    figSynapse.savefig(path_save + "_filt_vs_gc_synapse.png", format='png')
    figCompPropNeuron.savefig(path_save + "_filt_entropy_gc_vs_rate_neuron.png", format='png')
    figCompPropSyn.savefig(path_save + "_filt_entropy_gc_vs_rate_synapse.png", format='png')
    if fig_syn_b:
        figSynapseb.savefig(path_save + "_filt_vs_gc_synase_b.png", format='png')
        figCompPropSynb.savefig(path_save + "_filt_entropy_gc_vs_rate_synase_b.png", format='png')
    figEntropyInput.savefig(path_save + "_information_input.png", format='png')
    figEntropy.savefig(path_save + "_information_neuron.png", format='png')
    figEntropySyn.savefig(path_save + "_information_synapses.png", format='png')
# """

# PLOT OF CHARACTERISTICS FOR INI, MID, AND END WINDOWS. SPLIT BY PROPORTIONAL AND CONSTANT INPUT RATE CHANGES
# """
# Steady-state
# lbl = ['st_ini_prop', 'st_mid_prop', 'st_end_prop', 'st_ini_fix', 'st_mid_fix', 'st_end_fix']
# st_lbl = ['_mean', '_med', '_max', '_min', '_q10', '_q90', '_q5', '_q95']
# cols = ['tab:orange', 'tab:blue', 'tab:red', 'tab:red', 'tab:green', 'tab:green', 'tab:olive', 'tab:olive']
# title = ("Steady-state: " + description.split(",")[0] + r', $\tau_{lif}$ ' + str(tau_lif * 1e3) + "ms, gain " +
#          str(int(gain * 100)) + "%. Neuronal response")
# path_save = folder_plots + dr_gain_control_file + '_windows_statistics.png'
# plot_features_windows_prop_fix(initial_frequencies, dr_gain, lbl, st_lbl, cols, title, path_save, save_figs)
# Transition-state
# For considering proportional and constant change of rates
# lbl = ['mtr_ini_prop', 'mtr_mid_prop', 'mtr_end_prop', 'mtr_ini_fix', 'mtr_mid_fix', 'mtr_end_fix']
# title = ("Transition-state: " + description.split(",")[0] + r', $\tau_{lif}$ ' + str(tau_lif * 1e3) + "ms, gain " +
#          str(int(gain * 100)) + "%. Neuronal response")
# path_save = folder_plots + dr_gain_control_file + '_windows_statistics.png'
# plot_features_windows_prop_fix(initial_frequencies, dr_gain, lbl, st_lbl, cols, title, path_save, save_figs)

# For Neuron responses: Stationary and transitory states
lbl = ['mtr_ini_prop', 'mtr_mid_prop', 'mtr_end_prop']
lbl2 = ['st_ini_prop', 'st_mid_prop', 'st_end_prop']
st_lbl = ['_mean', '_med', '_max', '_min', '_q10', '_q90']
ls = ['-', '-', '-', '--', '--', '-']
cols = ['tab:orange', 'tab:blue', 'tab:red', 'tab:red', 'tab:green', 'tab:green']
t_ = ['ini-window', 'mid-window', 'end-window']
title = ("Transitory and stationary, " + description.split(",")[0] + ", gain " + str(int(gain * 100)) +
         "%. Neuronal response")
path_save = folder_plots + dr_gain_control_file + '_windows_tr_st.png'
plot_features_tr_st_3windows(f_vec, dr_gain, lbl, lbl2, st_lbl, cols, t_, title, path_save, save_figs, ls=ls,
                             normalise=norm_neuron, min_n=min_n, max_n=max_n)

# FOR SYNAPSES
# First synapse
lbl = ['syn_mtr_ini_prop', 'syn_mtr_mid_prop', 'syn_mtr_end_prop']
lbl2 = ['syn_st_ini_prop', 'syn_st_mid_prop', 'syn_st_end_prop']
t_ = ['ini-window', 'mid-window', 'end-window']
path_save = folder_plots + dr_gain_control_file + '_windows_syn_tr_st.png'
title = ("Transitory and stationary, " + description.split(",")[0] + ", gain " + str(int(gain * 100)) +
         "%. AMPA Synaptic response")
plot_features_tr_st_3windows(f_vec, dr_gain, lbl, lbl2, st_lbl, cols, t_, title, path_save, save_figs, ls=ls)

# Second synapse
lbl = ['syn_b_mtr_ini_prop', 'syn_b_mtr_mid_prop', 'syn_b_mtr_end_prop']
lbl2 = ['syn_b_st_ini_prop', 'syn_b_st_mid_prop', 'syn_b_st_end_prop']
t_ = ['ini-window', 'mid-window', 'end-window']
path_save = folder_plots + dr_gain_control_file + '_windows_syn_b_tr_st.png'
title = ("Transitory and stationary, " + description.split(",")[0] + ", gain " + str(int(gain * 100)) +
         "%. NMDA Synaptic response")
plot_features_tr_st_3windows(f_vec, dr_gain, lbl, lbl2, st_lbl, cols, t_, title, path_save, save_figs, ls=ls)
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
# """
# For Neuron responses
lbl = ['mtr_mid_prop', 'st_mid_prop', 'st_end_prop']
lbl2 = ['st_ini_prop', 'st_ini_prop', 'st_ini_prop']
st_lbl = ['_max', '_min', '_q10', '_q90', '_mean', '_med']
ls = ['-', '--', '--', '-', '-', '-']
cols_ = ['tab:red', 'tab:red', 'tab:green', 'tab:green', 'tab:orange', 'tab:blue']
t_ = [r"$mid_{tr} - ini_{st}$", r"$mid_{st} - ini_{st}$", r"$end_{st} - ini_{st}$"]
name_save = folder_plots + dr_gain_control_file + '_' + 'diff_tr_st_log.png'
title = (description.split(",")[0] + r', $\tau_{lif}$ ' + str(tau_lif * 1e3) + "ms, gain " +
         str(int(gain * 100)) + "%. Neuron response")
# plot_diff_windows(f_vec, dr_gain, lbl, lbl2, st_lbl, cols_, t_, title_graph=title, name_save=name_save, ls=ls,
#                   save_figs=save_figs)
mid_st_lbl = ['st_mid_prop']
mid_tr_lbl = ['mtr_mid_prop']
ini_st_lbl = ['st_ini_prop']
lbls = [r'$m_{st}$ - $i_{st}$(', r'$m_{tr}$ - $i_{st}$(', r'$m_{st}$ - $i_{st}$(', r'$m_{tr}$ - $i_{st}$(',
        r'$m_{st}$ - $i_{st}$(', '', r'$m_{st}$ - $i_{st}$(', '', r'$m_{st}$ - $i_{st}$(', '', r'$m_{st}$ - $i_{st}$(', '']
plot_diff_windows_tr_st(f_vec, dr_gain, mid_st_lbl, mid_tr_lbl, ini_st_lbl, st_lbl, cols_, t_, title_graph=title,
                        name_save=name_save, ls=ls, save_figs=save_figs, lbls=lbls, fillBetween=False,
                        normalise=norm_neuron, min_n=min_n, max_n=max_n)

# For synapse A
lbl = ['syn_mtr_mid_prop', 'syn_st_mid_prop', 'syn_st_end_prop']
lbl2 = ['syn_st_ini_prop', 'syn_st_ini_prop', 'syn_st_ini_prop']
name_save = folder_plots + dr_gain_control_file + '_' + 'diff_syn_tr_st_log.png'
title = (description.split(",")[0] + r', $\tau_{lif}$ ' + str(tau_lif * 1e3) + "ms, gain " +
         str(int(gain * 100)) + "%. AMPA synaptic response")
# plot_diff_windows(f_vec, dr_gain, lbl, lbl2, st_lbl, cols_, t_, title_graph=title, name_save=name_save, ls=ls,
#                   save_figs=save_figs)
mid_st_lbl = ['syn_st_mid_prop']
mid_tr_lbl = ['syn_mtr_mid_prop']
ini_st_lbl = ['syn_st_ini_prop']
plot_diff_windows_tr_st(f_vec, dr_gain, mid_st_lbl, mid_tr_lbl, ini_st_lbl, st_lbl, cols_, t_, title_graph=title,
                        name_save=name_save, ls=ls, save_figs=save_figs, lbls=lbls, fillBetween=False)
# For synapse B
lbl = ['syn_b_mtr_mid_prop', 'syn_b_st_mid_prop', 'syn_b_st_end_prop']
lbl2 = ['syn_b_st_ini_prop', 'syn_b_st_ini_prop', 'syn_b_st_ini_prop']
name_save = folder_plots + dr_gain_control_file + '_' + 'diff_syn_b_tr_st_log.png'
title = (description.split(",")[0] + r', $\tau_{lif}$ ' + str(tau_lif * 1e3) + "ms, gain " +
         str(int(gain * 100)) + "%. NMDA synaptic response")
# plot_diff_windows(f_vec, dr_gain, lbl, lbl2, st_lbl, cols_, t_, title_graph=title, name_save=name_save, ls=ls,
#                   save_figs=save_figs)
mid_st_lbl = ['syn_b_st_mid_prop']
mid_tr_lbl = ['syn_b_mtr_mid_prop']
ini_st_lbl = ['syn_b_st_ini_prop']
plot_diff_windows_tr_st(f_vec, dr_gain, mid_st_lbl, mid_tr_lbl, ini_st_lbl, st_lbl, cols_, t_, title_graph=title,
                        name_save=name_save, ls=ls, save_figs=save_figs, lbls=lbls, fillBetween=False)
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
