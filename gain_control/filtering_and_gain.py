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
name_n_state_variables = ['v', 'm', 'h', 'n']  #  ['v']  #
name_syn_state_variables = ['s_ampa', 's_nmda', 'x_nmda', 'xd']  # ['R', 'U', 'epsc']  #
s_model = 'DoornSTF'
n_model = "HH"
ind = 7
run_experiment = False
save_figs = False
imputations = True
lif_output = True
n_noise = True
plot_figs = True
num_syn = 1

# Sampling frequency and conditions for running parallel or single LIF neurons
sfreq = 10e3
tau_lif = 30  # ms
total_realizations = 1  # 100
num_realizations = 1  # 8 for server, 4 for macbook air
t_tra = None  # 0.25

# Path variables
# path_vars = "../gain_control/variables/high_freq_" + str(int(sfreq/1e3)) + "k_2/"
path_vars = "../gain_control/variables/high_freq_30k_2/"
check_create_folder(path_vars)
folder_plots = '../gain_control/plots/'
check_create_folder(folder_plots)

# Normalization
norm_neuron = False  # True
min_n, max_n = None, None
if n_model == "HH":
    norm_neuron = False
    min_n, max_n = -0.05, 0.0
if n_model == "LIF":
    norm_neuron = False
    min_n, max_n = -70, -55
# **********************************************************************************************************************
# MULTIPLE GAINS
# **********************************************************************************************************************
gain_v = [0.1, 0.5, 1.0]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
filt_dict_loaded = False

# Titles graphs
title = "Model " + s_model + ', ind ' + str(ind)
if n_model == 'LIF': title += r', $\tau_{lif}$ ' + str(tau_lif) + "ms"
if len(gain_v) == 1:
    title += ', gain ' + str(int(gain_v[0] * 100)) + '%'
else:
    title += ', multiple gains'

# ""
# Plot
lbl = ['st_mid_prop']
lbl2 = ['st_ini_prop']
lbl_syn = ['syn_st_mid_prop']
lbl2_syn = ['syn_st_ini_prop']
lbl_synb = ['syn_b_st_mid_prop']
lbl2_synb = ['syn_b_st_ini_prop']
lbl_h_tr = ['H_v_neu_tr']
lbl_h_st = ['H_v_neu_st']
lbl_hI_tr = ['H_ISI_tr']
lbl_hI_st = ['H_ISI_st']
lbl_h_s_tr = ['H_epsc_syn_tr'] if s_model == 'TM' else ['H_s_ampa_syn_tr']
lbl_h_s_st = ['H_epsc_syn_st'] if s_model == 'TM' else ['H_s_ampa_syn_st']
lbl_h_sb_tr = ['H_PSR_syn_b_tr'] if s_model == 'TM' else ['H_x_nmda_syn_tr']
lbl_h_sb_st = ['H_PSR_syn_b_st'] if s_model == 'TM' else ['H_x_nmda_syn_st']
titles_H = ['ini win.', 'mid win.', 'end win.', 'st-ini win.', 'st-mid win.', 'st-end win.']
titles_H_syn = ['AMPA ini win.', 'AMPA mid win.', 'AMPA end win.', 'NMDA ini win.', 'NMDA mid win.', 'NMDA end win.']
labels_H = ['tr-', 'tr-', 'tr-', 'st-', 'st-', 'st-']
st_lbl = ['_max',  '_q95', '_q90', '_med', '_min', '_q5', '_q10', '_mean']
title_mp = ['Amplitude in steady-state', 'Varibility in steady-state', 'Median in steady-state',
            'Amplitude in transition-state', 'Varibility in transition-state', 'Median in transition-state']
x_label_ax_p = [r'$E_{ff_{i,st}}^{amp}$ (mV)', r'$E_{ff_{i,st}}^{var}$ (mV)', r'$E_{ff_{i,st}}^{med}$ (mV)',
               r'$E_{ff_{i,st}}^{amp}$ (mV)', r'$E_{ff_{i,st}}^{var}$ (mV)', r'$E_{ff_{i,st}}^{med}$ (mV)']
y_label_ax_p = [r'$G_{m-i,st}^{amp} (mV)$', r'$G_{m-i,st}^{var} (mV)$', r'$G_{m-i,st}^{med} (mV)$',
               r'$G_{m-i,tr}^{amp} (mV)$', r'$G_{m-i,tr}^{var} (mV)$', r'$G_{m-i,tr}^{med} (mV)$']
x_label_ax_n = [r'$E_{ff_{m,st}}^{amp}$ (mV)', r'$E_{ff_{m,st}}^{var}$ (mV)', r'$E_{ff_{m,st}}^{med}$ (mV)',
               r'$E_{ff_{m,st}}^{amp}$ (mV)', r'$E_{ff_{m,st}}^{var}$ (mV)', r'$E_{ff_{m,st}}^{med}$ (mV)']
y_label_ax_n = [r'$G_{e-m,st}^{amp} (mV)$', r'$G_{e-m,st}^{var} (mV)$', r'$G_{e-m,st}^{med} (mV)$',
               r'$G_{e-m,tr}^{amp} (mV)$', r'$G_{e-m,tr}^{var} (mV)$', r'$G_{e-m,tr}^{med} (mV)$']
[r'$E_{ff_{%s}}$', r'$E_{ff{var_{%s}}}$', r'$E_{ff{med_{%s}}}$']
t_ = [r'$G_{m-i,tr}(r,\delta)$ and $G_{m-i,st}(r,\delta)$', r'$G_{e-m,tr}(r,\delta)$ and $G_{e-m,st}(r,\delta)$']
cols_ = ['tab:red', 'tab:olive', 'tab:green', 'tab:blue', 'tab:red', 'tab:olive', 'tab:green', 'tab:orange']
c_g = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
       'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

axb_s, figEntropySyn, aux_H_sb, axb_, ax_, alphas, markers, axb_sb = None, None, None, None, None, None, None, None
st_lbl_b, xl_neu, xl_syn, xl_syb, ax_s, ax_sb, ax_hI, ax_h, ax_h, ax_hs = [None for _ in range(10)]
figNeur_pos_gc, figNeur_neg_gc, figSynapse, figCompPropNeuron, figCompPropSyn = None, None, None, None, None
figSynapseb, figCompPropSynb, figEntropyInput, figEntropy, ax_p, ax_n = None, None, None, None, None, None

if plot_figs:
    plt.rcParams['figure.constrained_layout.use'] = True
    # Synaptic filtering vs. Gain-Control for Neuron
    dr_gain_control_file = get_name_file(sfreq, s_model, n_model, ind, num_syn, lif_output, tau_lif, True,
                                         imputations, 0.1, n_noise=n_noise)
    if os.path.isfile(path_vars + dr_gain_control_file) and not filt_dict_loaded:
        figNeur_pos_gc = [plt.figure(figsize=(15, 6)) for _ in range(len(name_n_state_variables))]  # 6.5, 5
        title_ = 'Frequency portrait for short-term depression - %s(t) - positive changes of rate'
        for j in range(len(name_n_state_variables)):
            figNeur_pos_gc[j].suptitle(title_ % name_n_state_variables[j], fontsize=22)
        ax_p = [[figNeur_pos_gc[i].add_subplot(2, 3, j + 1) for j in range(len(title_mp))] for i in
               range(len(name_n_state_variables))]
        figNeur_neg_gc = [plt.figure(figsize=(15, 6)) for _ in range(len(name_n_state_variables))]  # 6.5, 5
        title_ = 'Frequency portrait for short-term depression - %s(t) - negative changes of rate'
        for j in range(len(name_n_state_variables)):
            figNeur_neg_gc[j].suptitle(title_ % name_n_state_variables[j], fontsize=22)
        ax_n = [[figNeur_neg_gc[i].add_subplot(2, 3, j + 1) for j in range(len(title_mp))] for i in
               range(len(name_n_state_variables))]

    # figNeuron = plt.figure(figsize=(15, 8))  # 6.5, 5
    # plt.suptitle(title + ". Mem. pot.")
    # ax_ = [figNeuron.add_subplot(2, 3, j + 1) for j in range(len(title_mp))]

    # Synaptic filtering vs. Gain-Control for Synapse A
    figSynapse = plt.figure(figsize=(15, 8))  # 6.5, 5
    plt.suptitle(title + ". Synapse")
    ax_s = [figSynapse.add_subplot(2, 4, j + 1) for j in range(len(st_lbl))]
    # figEntropyInput = plt.figure(figsize=(15, 4))
    # plt.suptitle(title + ". Information Theory (Input)")
    # ax_hI = [figEntropyInput.add_subplot(1, 3, j + 1) for j in range(3)]

    # Synaptic entropy in frequency for Input
    figEntropyInput = plt.figure(figsize=(15, 4))
    plt.suptitle(title + ". Information Theory (Input)")
    ax_hI = [figEntropyInput.add_subplot(1, 3, j + 1) for j in range(3)]

    # Synaptic entropy in frequency for Neuron
    figEntropy = plt.figure(figsize=(15, 4))
    plt.suptitle(title + ". Information Theory (Neuron)")
    ax_h = [figEntropy.add_subplot(1, 3, j + 1) for j in range(3)]

    # Synaptic entropy in frequency for Synapse(s)
    figEntropySyn = plt.figure(figsize=(15, 8))
    plt.suptitle(title + ". Information Theory (Synapses) ")

    # Plots of computational properties vs rates
    st_lbl_b = ['H - filtering', 'H - Gain-control', 'Transition time', 'Synaptic Filtering', 'GC - amp', 'GC - var',
                'GC - med']
    # Neuron
    figCompPropNeuron = plt.figure(figsize=(20, 3.6))  # 15, 8
    plt.suptitle(title + ". Neuron v(t)", fontsize=23)
    axb_ = [figCompPropNeuron.add_subplot(1, 7, j + 1) for j in range(7)]  # [0, 1, 4, 5, 6, 7, 8]]
    figCompPropSyn = plt.figure(figsize=(20, 3.6))  # 15, 8
    aux_t = ""
    if n_model == "HH": aux_t = ". AMPA synapse"
    if n_model == "LIF": aux_t = ". Synapse"
    plt.suptitle(title + aux_t, fontsize=23)
    axb_s = [figCompPropSyn.add_subplot(1, 7, j + 1) for j in range(7)]  # [0, 1, 4, 5, 6, 7, 8, 9]]

    alpha = 0.3
    markers = ['+', '*']
    alphas = [1.0, 0.5]

fig_syn_b = False
fig_H_100 = False

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

        # Name state variables
        name_n_state_variables = dr_filt['name_neuron_state_variables']
        name_syn_state_variables = dr_filt['name_syn_state_variables']

    if os.path.isfile(path_vars + dr_gain_control_file) and not run_experiment:
        dr_gain = loadObject(dr_gain_control_file, path_vars)

    f_vec = dr_gain['initial_frequencies']
    f_vecD = dr_filt['initial_frequencies']
    # **********************************************************************************************************
    # For Entropy analysis
    if plot_figs:
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

    # Getting arrays of Synaptic Information
    a = get_info_arrays(dr_gain, dr_filt, lbl_hI_tr, lbl_hI_st, lbl_h_tr, lbl_h_st, lbl_h_s_tr, lbl_h_s_st,
                        lbl_h_sb_tr, lbl_h_sb_st, fig_syn_b)
    aux_HI, aux_H, aux_det_H, aux_H_s, aux_det_H_s, aux_H_sb, aux_det_H_sb = a

    # ******************************************************************************************************************
    # Plots 1
    # FREQUENCY RESPONSES OF SYNAPTIC PROPERTIES
    dr_ = dr_gain
    if plot_figs:
        if 'name_neuron_state_variables' not in dr_:
            # For Membrane potential analysis
            var_ = organise_keys_dr_gc(sufix='')
            # Plotting properties
            axb_ = plot_properties_in_freq(dr_, var_, f_vec, aux_H, gain, axb_, dr_['time_transition'], min_n=min_n,
                                       max_n=max_n, c_g=c_g[i_g], plot_filt=i_g == 0, norm_neuron=norm_neuron)
        else:
            for name_ in dr_['name_neuron_state_variables']:
                var_ = organise_keys_dr_gc(sufix=name_ + '_')
                if name_ == 'v':
                    var_ = organise_keys_dr_gc(sufix='')
                    # Plotting properties
                    axb_ = plot_properties_in_freq(dr_, var_, f_vec, aux_H, gain, axb_, dr_['time_transition'], min_n=min_n,
                                                   max_n=max_n, c_g=c_g[i_g], plot_filt=i_g == 0, norm_neuron=norm_neuron)

        if 'name_syn_state_variables' not in dr_:
            # For Synaptic analysis
            var_ = organise_keys_dr_gc(sufix='syn_')
            # Plotting properties
            axb_s = plot_properties_in_freq(dr_, var_, f_vec, aux_H_s, gain, axb_s, dr_['time_transition_syn'],
                                        c_g=c_g[i_g], plot_filt=i_g == 0, norm_neuron=False)

            # For Synaptic analysis (second synapse)
            var_ = organise_keys_dr_gc(sufix='syn_b_')
            # Plotting properties
            if fig_syn_b:
                axb_sb = plot_properties_in_freq(dr_, var_, f_vec, aux_H_sb, gain, axb_sb,
                                                 dr_['time_transition_syn_b'], c_g=c_g[i_g], plot_filt=i_g == 0,
                                                 norm_neuron=False)
    # **********************************************************************************************************
    # Plots 2

    if plot_figs:
        # **********************************************************************************************************
        # For Membrane potential analysis
        #
        plot_freq_portrait(name_n_state_variables, dr_filt, dr_gain, gain, ax_p, 'ini', 'mid',
                           norm_neuron, title_mp, markers, alphas)
        plot_freq_portrait(name_n_state_variables, dr_filt, dr_gain, gain, ax_n, 'mid', 'end',
                           norm_neuron, title_mp, markers, alphas)

        """
        for i in [0]:  # range(len(lbl)):
            for j in range(len(st_lbl)):
                # **********************************************************************************************************
                # For Membrane potential analysis
                n_sto_m_st = norm_array(dr_gain[lbl[i] + st_lbl[j]], compute_norm=norm_neuron, min_n=min_n, max_n=max_n)
                n_sto_i_st = norm_array(dr_gain[lbl2[i] + st_lbl[j]], compute_norm=norm_neuron, min_n=min_n,max_n=max_n)
                n_det_m_st = norm_array(dr_filt[lbl[i] + st_lbl[j]], compute_norm=norm_neuron, min_n=min_n, max_n=max_n)
                n_det_i_st = norm_array(dr_filt[lbl2[i] + st_lbl[j]], compute_norm=norm_neuron, min_n=min_n,max_n=max_n)
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
                if i_g == 0:
                    ax_[j].plot(aux_det_filt, aux_det_gain, c='gray', alpha=alphas[i], label='Det')
                else:
                    ax_[j].plot(aux_det_filt, aux_det_gain, c='gray', alpha=alphas[i])
                ax_[j].scatter(aux_det_filt, aux_det_gain, c=c_g[i_g], marker=markers[i], alpha=alphas[i])
                ax_[j].scatter(aux_det_filt[0], aux_det_gain[0], c='black')
                
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
                if i_g == 0:
                    ax_s[j].plot(aux_det_filt, aux_det_gain, c='gray', alpha=alphas[i], label='Det')
                else:
                    ax_s[j].plot(aux_det_filt, aux_det_gain, c='gray', alpha=alphas[i])
                ax_s[j].scatter(aux_det_filt, aux_det_gain, c=c_g[i_g], marker=markers[i], alpha=alphas[i])
                ax_s[j].scatter(aux_det_filt[0], aux_det_gain[0], c='black')
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
                    if i_g == 0:
                        ax_sb[j].plot(aux_det_filt, aux_det_gain, c='gray', alpha=alphas[i], label='Det')
                    else:
                        ax_sb[j].plot(aux_det_filt, aux_det_gain, c='gray', alpha=alphas[i])
                    ax_sb[j].scatter(aux_det_filt, aux_det_gain, c=c_g[i_g], marker=markers[i], alpha=alphas[i])
                    ax_sb[j].scatter(aux_det_filt[0], aux_det_gain[0], c='black')
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
                    ax_h[j].plot(f_vecD, aux_det_H[j][:len(aux_det_filt)], marker=markers[1], c=c_g[i_g], alpha=0.4)  
                                 # ,label=str(gain) + " (det)")
                    ax_h[j].scatter(f_vec[0], aux_det_H[j][0], c='gray')
                    # Deterministic - stationary
                    ax_h[j].plot(f_vecD, aux_det_H[j + 3][:len(aux_det_filt)], marker=markers[1], c=c_g[i_g + 3], 
                                 alpha=0.4)  # label=str(gain) + " (det)")
                    ax_h[j].scatter(f_vecD[0], aux_det_H[j + 3][0], c='gray')

                    # FOR SYNAPSES
                    # For only one computation of entropy (# bins fixed if only one, or fix bin size if two entropies)
                    # Stochastic - transition
                    ax_hs[j].plot(f_vec, aux_H_s[j], c=c_g[i_g], marker=markers[0], label=labels_H[i_g] + str(gain))
                    ax_hs[j].scatter(f_vec[0], aux_H_s[j][0], c='black')
                    # Stochastic - stationary
                    ax_hs[j].plot(f_vec, aux_H_s[j + 3], c=c_g[i_g + 3], marker=markers[0],
                                 label=labels_H[i_g + 3] + str(gain))
                    ax_hs[j].scatter(f_vec[0], aux_H_s[j + 3][0], c='black')
                    # Deterministic - transition
                    ax_hs[j].plot(f_vecD, aux_det_H_s[j][:len(aux_det_filt)], marker=markers[1], c=c_g[i_g], alpha=0.4)  
                                  # , label=str(gain) + " (det)")
                    ax_hs[j].scatter(f_vecD[0], aux_det_H_s[j][0], c='gray')
                    # Deterministic - stationary
                    ax_hs[j].plot(f_vecD, aux_det_H_s[j + 3][:len(aux_det_filt)], marker=markers[1], c=c_g[i_g + 3],
                                  alpha=0.4)  # , label=str(gain) + " (det)")
                    ax_hs[j].scatter(f_vecD[0], aux_det_H_s[j + 3][0], c='gray')
        

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
                    ax_hs[j].plot(f_vecD, aux_det_H_sb[j-3][:len(aux_det_filt)], marker=markers[1], c=c_g[i_g], 
                                  alpha=0.4)  # label=str(gain) + " (det)")
                    ax_hs[j].scatter(f_vecD[0], aux_det_H_sb[j-3][0], c='gray')
                    # Deterministic - stationary
                    ax_hs[j].plot(f_vecD, aux_det_H_sb[j][:len(aux_det_filt)], marker=markers[1], c=c_g[i_g + 3], 
                                  alpha=0.4)  # label=str(gain) + " (det)")
                    ax_hs[j-3].scatter(f_vecD[0], aux_det_H_sb[j][0], c='gray')
        # """
    i_g += 1

# For PhD Figure "methodology - Frequency portrait"
# xlims = [[-0.01, -0.01, -70.005, -0.01, -0.01, -70.005], [1.52, 1.52, -69.88, 1.52, 1.52, -69.88]]
# ylims = [[-0.48, -0.48, -0.48, -0.48, -0.48, -0.48], [0.15, 0.15, 0.15, 0.15, 0.15, 0.15]]
xlims = [[-0.05, -0.05, -0.01, -0.05, -0.05, -0.01],
         [2.42, 2.42, 0.34, 2.42, 2.42, 0.34]]
ylims = [[-0.75, -0.75, -0.75, -0.75, -0.75, -0.75],
         [1., 1., 1., 1., 1., 1.]]

xlims_s = [[-69.9, -69.9, -69.95, -70.01, -70.01, -70.01, -70.01, -70.01],
           [0.07, -69.1, -69.40, -69.82, -69.82, -69.82, -69.82, -69.82]]
ylims_s = [[-0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02],
           [0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015]]

# H-filt, H-GC, tr_st_time, Filt, GC-max, GC-min, GC-var, GC-med
if n_model == "HH":
    xl_neu = {0: [[None for _ in range(10)],
                  [None for _ in range(10)]],
              1: [[-0.05, -0.65, -0.01, 0.2,   -0.06,  -5e-3,  -0.01,  -3e-3,  0.20, -0.02],
                  [8.0,   0.2,   1.6,   0.5,   0.06,   0.02,   0.015,  8e-3,   0.35, 0.01]],
              2: [[-0.05, -0.25, -0.01, -0.01, -0.5,   -0.3,   -0.04,  -0.01,  0.05, -0.25],
                  [8.0,   0.75,  1.6,   1.05,   0.6,    0.3,    0.04,   0.06,   1.02, 0.55]],
              3: [[-0.05, -1.25, -0.01, -0.01, -0.02,  -1e-3,  -0.028, -1e-3,  0.2,  -0.02],
                  [8.0,   0.65,  1.6,   1.05,  0.04,   0.03,   0.05,   0.022,  0.4,  0.031]],
              4: [[None for _ in range(10)],
                  [None for _ in range(10)]],
              5: [[-0.05, -0.22, -0.01, 0.05,  -5e-3,  -2e-3,  -1e-5,  -1e-3,  0.23, -4e-3],
                  [8.0,   0.45,  1.6,   1.05,  9e-3,   9e-3,   5e-4,   8e-3,   0.27, 9e-3]],
              6: [[-0.05, -0.7,  -0.01, -0.01, -0.075, -5e-3,  -15e-3, -3e-3,  0.2,  -0.05],
                  [8.0,   0.15,  1.6,   1.05,  0.05,   0.013,  0.02,   4e-3,   0.5,  0.02]],
              7: [[-0.05, -1.3,  -0.01, -0.01, -0.1,   -15e-3, -0.04,  -12e-3, 0.05, -0.06],
                  [8.0,   0.4,   1.6,   1.1,   0.2,    12e-3,  65e-3,  0.02,   0.47, 0.08]],
              8: [[-0.05, -1.3, -0.01, -0.01, -0.075, -5e-3,  -0.028,  -0.01,  0.05, -0.06],
                  [8.0,   0.65,  1.6,   1.02,   0.06,    0.3,  0.025,   0.02,   0.5,  0.08]]}

    xl_syn = {0: [[None for _ in range(10)],
                  [None for _ in range(10)]],
              1: [[-0.05, -0.8, -0.01, -0.01, -0.2,  -1e-3, -0.01,  -1e-3, -0.01, -0.07],
                  [8.0,   0.2,  1.6,   0.8,   0.15,  0.012, 0.027,  75e-4, 0.4,   0.025]],
              2: [[-0.05, -1.0, -0.01, -0.05, -0.2,  -0.02, -0.1,   -0.01, -0.01, -0.12],
                  [8.0,   1.2,  1.6,   1.3,   0.4,   0.12,  0.1,    0.08,  0.7,   0.15]],
              3: [[-0.05, -1.0, -0.01, 0.08,  -0.06, -1e-3, -0.027, -1e-3, -0.01, -0.06],
                  [8.0,   0.5,  1.6,   1.05,  0.08,  0.03,  0.04,   0.02,  0.4,   0.07]],
              4: [[None for _ in range(10)],
                  [None for _ in range(10)]],
              5: [[-0.05, -1.0, -0.01, -0.01, -8e-3, -1e-3, -2e-4,  -1e-3, 5e-3,  -75e-4],
                  [8.0,   1.0,  1.6,   2.0,   8e-3,  0.01,  8e-4,   6e-3,  0.05,  0.01]],
              6: [[-0.05, -1.0, -0.01, -0.05, -0.2,  -1e-4, -12e-3, -1e-4, -0.05, -0.19],
                  [8.0,   0.3,  1.6,   2.6,   0.1,   0.01,  26e-3,  7e-3,  0.8,   0.03]],
              7: [[-0.05, -1.4, -0.01, -0.05, -0.4,  -5e-4, -0.04,  -2e-3, -0.05, -0.15],
                  [8.0,   1.4,  1.6,   2.5,   0.7,   5e-3,  65e-3,  6e-3,  0.8,   0.25]],
              8: [[-0.05, -1.4, -0.01, -0.05, -0.4,  -0.02, -0.1,   -0.01, -0.05, -0.19],
                  [8.0,   1.4,  1.6,   2.6,   0.7,   0.12,  0.1,    0.08,  0.8,   0.25]]}
    if fig_syn_b:
        xl_syb = {0: [[None for _ in range(10)],
                      [None for _ in range(10)]],
                  1: [[-0.05, -1.5, -0.01, -0.01, -0.07, -0.07, -0.09,  -0.04,  -0.01, -0.04],
                      [8.0,   0.27, 1.6,   0.25,  0.03,  0.07,  0.05,   0.04,   0.25,  0.04]],
                  2: [[-0.05, -1.2, -0.01, -0.05, -0.05, -0.05, -0.08,  -0.05,  -0.01, -0.045],
                      [8.0,   1.0,  1.6,   0.25,  0.05,  0.1,   0.04,   0.08,   0.25,  0.06]],
                  3: [[-0.05, -1.2, -0.01, -0.05, -0.04, -0.03, -0.05,  -0.04,  -0.01, -0.04],
                      [8.0,   0.5,  1.6,   0.22,  0.02,  0.04,  0.01,   0.05,   0.22,  0.048]],
                  4: [[None for _ in range(10)],
                      [None for _ in range(10)]],
                  5: [[-0.05, -0.7, -0.01, -0.01, -5e-3, -3e-4, -12e-4, -11e-4, 0.0,   -5e-3],
                      [8.0,   0.6,  1.6,   0.6,   1e-3,  4e-4,  5e-4,   1e-4,   0.018, 1e-3]],
                  6: [[-0.05, -1.2, -0.01, -0.01, -0.11, -0.05, -0.13,  -0.05,  -0.01, -0.11],
                      [8.0,   0.21, 1.6,   0.81,  0.04,  0.04,  0.01,   0.01,   0.5,   0.05]],
                  7: [[-0.05, -1.7, -0.01, -0.05, -0.15, -0.15, -0.2,   -0.1,   -0.01, -0.12],
                      [8.0,   0.25, 1.6,   0.6,   0.22,  0.12,  0.1,    0.2,    0.43,  0.1]],
                  8: [[-0.05, -1.7, -0.01, -0.05, -0.15, -0.15, -0.2,   -0.1,   -0.01, -0.12],
                      [8.0,   1.0,  1.6,   0.81,  0.22,  0.12,  0.1,    0.2,    0.5,   0.1]]}
elif n_model == "LIF":
    if s_model == "MSSM":
        xl_neu = {4: [[-0.05, -2.5, -0.01, -0.01, -0.05, -5e-4, -0.01, -1e-4, -0.01, -0.035],
                      [13.0,  1.05, 1.6,   0.3,   0.025, 8e-3,  0.025, 7e-3,  0.17,  0.02]],
                  5: [[None for _ in range(10)],
                      [None for _ in range(10)]]}
        xl_syn = {4: [[-0.05, -1.2, -0.01, -0.01, -0.02, -1e-4, -4e-3, -1e-4, -0.01, -0.017],
                      [13.0,  0.2,  1.6,   0.14,  0.01,  3e-3,  8e-3,  25e-4, 0.08,  0.008]],
                  5: [[None for _ in range(10)],
                      [None for _ in range(10)]]}
    if s_model == "TM":
        xl_neu = {4: [[-0.05, -3.0,  -0.01, -0.01, -0.05,  -5e-4, -7e-3, -1e-4, -0.01, -0.04],
                      [13.0,  1.25,  1.6,   0.3,   0.025,  55e-4, 0.015, 4e-3,  0.15,  0.017]],
                  8: [[-0.05, -2.6,  -0.01, -0.01, -0.1,   -5e-4, -0.02, -5e-4, -0.01, -0.05],
                      [13.0,  3.0,   1.6,   0.35,  0.15,   0.015, 0.05,  45e-3, 0.22,  0.06]]}
        xl_syn = {4: [[-0.05, -1.5,  -0.01, -0.01, -0.025, -5e-4, -3e-3, -1e-4, -0.01, -0.02],
                      [13.0,  1.0,   1.6,   0.175, 0.025,  2e-3,  6e-3,  13e-4, 0.09,  5e-3]],
                  8: [[-0.05, -1.67, -0.01, -0.01, -0.05,  -5e-4, -7e-3, -5e-4, -0.01, -0.035],
                      [13.0,  1.05,  1.6,   0.175, 0.05,   5e-3,  0.015, 4e-3,  0.15,  0.035]]}

path_save = (folder_plots + s_model + '_ind_' + str(ind) + '_' + str(len(gain_v)) + '_gains_sf_' +
             str(int(sfreq * 1e-3)) + 'k_tauLIF_' + str(tau_lif) + 'ms' + '_phase_portrait')

# For plot neuron b
# ind = 10
if plot_figs:
    sizeF = 20
    for j in range(8):
        # For Computational properties vs. rate
        # axb_[j].set_xlabel("Rate (Hz)", color='gray', fontsize=sizeF)
        # axb_s[j].set_xlabel("Rate (Hz)", color='gray', fontsize=sizeF)
        if j < 2:
            axb_[j].set_ylabel("Entropy (bits)", color='gray', fontsize=sizeF)
            axb_s[j].set_ylabel("Entropy (bits)", color='gray', fontsize=sizeF)
            if fig_syn_b: axb_sb[j].set_ylabel("Entropy (bits)", color='gray', fontsize=sizeF)
        elif j == 2:
            axb_[j].set_ylabel("Time (s)", color='gray', fontsize=sizeF)
            axb_s[j].set_ylabel("Time (s)", color='gray', fontsize=sizeF)
            if fig_syn_b: axb_sb[j].set_ylabel("Time (s)", color='gray', fontsize=sizeF)
        else:
            axb_[j].set_ylabel("Mem. pot. (mV)", color='gray', fontsize=sizeF)
            axb_s[j].set_ylabel("Syn. strength", color='gray', fontsize=sizeF)
            if fig_syn_b: axb_sb[j].set_ylabel("Syn. strength", color='gray', fontsize=sizeF)
        axb_[j].set_title(st_lbl_b[j], c="gray", fontsize=sizeF)
        axb_[j].grid()
        axb_[j].set_xscale('log')
        axb_[j].set_ylim(xl_neu[ind][0][j], xl_neu[ind][1][j])
        # axb_s[j].set_title(st_lbl_b[j], c="gray", fontsize=sizeF)
        axb_s[j].grid()
        axb_s[j].set_xscale('log')
        axb_s[j].set_ylim(xl_syn[ind][0][j], xl_syn[ind][1][j])
        if fig_syn_b:
            # axb_sb[j].set_title(st_lbl_b[j], c="gray", fontsize=sizeF)
            axb_sb[j].set_xlabel("Rate (Hz)", color='gray', fontsize=sizeF)
            axb_sb[j].grid()
            axb_sb[j].set_xscale('log')
            if n_model == "HH": axb_sb[j].set_ylim(xl_syb[ind][0][j], xl_syb[ind][1][j])
        else:
            axb_s[j].set_xlabel("Rate (Hz)", color='gray', fontsize=sizeF)

    # Neuronal state variables
    for n in range(len(name_n_state_variables)):
        for j in range(len(title_mp)):
            # Synaptic filtering vs. Gain-Control for Neuron - positive change of rate
            ax_p[n][j].set_xlabel(x_label_ax_p[j], color='gray')
            ax_p[n][j].set_ylabel(y_label_ax_p[j], color='gray')
            # if n_model != 'HH':
            #     ax_p[n][j].set_ylim(ylims[0][j], ylims[1][j])
            #     ax_p[n][j].set_xlim(xlims[0][j], xlims[1][j])
            # ax_[j].set_ylim(ylims)
            # ax_[j].set_title(st_lbl[j][1:], c="gray")
            ax_p[n][j].set_title(title_mp[j], color="black", alpha=0.7)
            ax_p[n][j].grid()

            # Synaptic filtering vs. Gain-Control for Neuron - negative change of rate
            ax_n[n][j].set_xlabel(x_label_ax_n[j], color='gray')
            ax_n[n][j].set_ylabel(y_label_ax_n[j], color='gray')
            # if n_model != 'HH':
            #     ax_n[n][j].set_ylim(ylims[0][j], ylims[1][j])
            #     ax_n[n][j].set_xlim(xlims[0][j], xlims[1][j])
            # ax_[j].set_ylim(ylims)
            # ax_[j].set_title(st_lbl[j][1:], c="gray")
            ax_n[n][j].set_title(title_mp[j], color="black", alpha=0.7)
            ax_n[n][j].grid()

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
    print("")
    for n in range(len(name_n_state_variables)):
        ax_p[n][int(len(title_mp) / 2) - 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,
                                                title='gain factor')
        ax_n[n][int(len(title_mp) / 2) - 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,
                                                   title='gain factor')
    # figNeuron.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    # Synaptic filtering vs. Gain-Control for Synapse A
    ax_s[int(len(st_lbl) / 2) - 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # figSynapse.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    # Plots of computational properties vs rates. Neuron
    axb_[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=sizeF - 10)  # 1
    axb_[6].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=sizeF - 10)  # 7
    # figCompPropNeuron.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    # Plots of computational properties vs rates. Synapse A
    axb_s[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=sizeF - 10)  # 1
    axb_s[6].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=sizeF - 10)  # 7
    # figCompPropSyn.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)

    # Synaptic filtering vs. Gain-Control for Synapse B
    if fig_syn_b:
        # Synaptic filtering vs. Gain-Control for Synapse B
        ax_sb[int(len(st_lbl) / 2) - 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        # figSynapseb.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
        # Plots of computational properties vs rates. Synapse B
        axb_sb[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=sizeF - 10)  # 1
        axb_sb[6].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=sizeF - 10)  # 7
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

if plot_figs and save_figs:
    for j in range(len(name_n_state_variables)):
        figNeur_pos_gc[j].savefig(path_save + "_freq_portrait_neuron_pos.png", format='png')
        figNeur_neg_gc[j].savefig(path_save + "_freq_portrait_neuron_neg.png", format='png')

    # figSynapse.savefig(path_save + "_filt_vs_gc_synapse.png", format='png')
    # figCompPropNeuron.savefig(path_save + "_filt_entropy_gc_vs_rate_neuron.png", format='png')
    # figCompPropSyn.savefig(path_save + "_filt_entropy_gc_vs_rate_synapse.png", format='png')
    # if fig_syn_b:
    #     figSynapseb.savefig(path_save + "_filt_vs_gc_synase_b.png", format='png')
    #     figCompPropSynb.savefig(path_save + "_filt_entropy_gc_vs_rate_synase_b.png", format='png')
    # figEntropyInput.savefig(path_save + "_information_input.png", format='png')
    # figEntropy.savefig(path_save + "_information_neuron.png", format='png')
    # figEntropySyn.savefig(path_save + "_information_synapses.png", format='png')
# """

# PLOT OF CHARACTERISTICS FOR INI, MID, AND END WINDOWS. SPLIT BY PROPORTIONAL AND CONSTANT INPUT RATE CHANGES
"""
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
st_lbl = ['_med', '_max', '_min', '_q10', '_q90']
ls = ['-', '-', '--', '--', '-']
legends = [r'$PSR^\mathrm{med}_{%s}$', r'$PSR^\mathrm{max}_{%s}$', r'$PSR^\mathrm{min}_{%s}$',
           r'$PSR^\mathrm{q10}_{%s}$', r'$PSR^\mathrm{q90}_{%s}$', r'$PSR^\mathrm{med}_{%s}$',
           r'$PSR^\mathrm{max}_{%s}$', r'$PSR^\mathrm{min}_{%s}$', r'$PSR^\mathrm{q10}_{%s}$',
           r'$PSR^\mathrm{q90}_{%s}$']
cols = ['tab:blue', 'tab:red', 'tab:red', 'tab:green', 'tab:green']
t_ = ['ini-window', 'mid-window', 'end-window']
y_lims = [-70.05, -68]  # [xl_neu[ind][0][8], xl_neu[ind][1][8]]
y_label = "mem. pot. (mV)"
title = "Frequency response of proportional schema for short-term "
title += "facilitation" if ind == 8 else "depression"
# "Transitory and stationary, " + description.split(",")[0] + ", gain " + str(int(gain * 100)) + "%. Neuron response"
path_save = folder_plots + dr_gain_control_file + '_windows_tr_st.png'
plot_features_tr_st_3windows(f_vec, dr_gain, lbl, lbl2, st_lbl, legends, cols, t_, title, path_save, save_figs, ls=ls,
                             normalise=False, min_n=min_n, max_n=max_n, y_lims_ind_plot=y_lims, y_lbl=y_label)

# FOR SYNAPSES
# First synapse
lbl = ['syn_mtr_ini_prop', 'syn_mtr_mid_prop', 'syn_mtr_end_prop']
lbl2 = ['syn_st_ini_prop', 'syn_st_mid_prop', 'syn_st_end_prop']
t_ = ['ini-window', 'mid-window', 'end-window']
y_lims = [xl_syn[ind][0][8], xl_syn[ind][1][8]]
y_label = "Syn. strength"
path_save = folder_plots + dr_gain_control_file + '_windows_syn_tr_st.png'
title = "Transitory and stationary, " + description.split(",")[0] + ", gain " + str(int(gain * 100))
if n_model == "HH": title += "%. AMPA synaptic response"
if n_model == "LIF": title += "%. Synaptic response"
plot_features_tr_st_3windows(f_vec, dr_gain, lbl, lbl2, st_lbl, legends, cols, t_, title, path_save, save_figs, ls=ls,
                             y_lims_ind_plot=y_lims, y_lbl=y_label)

# ****************************************************************************************************
# Figure PhD thesis (methodology / metrics temporal filtering)
dr = dr_gain
sg1 = [dr['mtr_ini_prop_max'] - dr['mtr_ini_prop_min'], dr['mtr_ini_prop_q90'] - dr['mtr_ini_prop_q10'],
       dr['syn_mtr_ini_prop_med']]
sg2 = [dr['st_ini_prop_max'] - dr['st_ini_prop_min'], dr['st_ini_prop_q90'] - dr['st_ini_prop_q10'],
       dr['st_ini_prop_med'] - dr['st_ini_prop_min']]
# sg1 = [dr['syn_mtr_ini_prop_max'] - dr['syn_mtr_ini_prop_min'], dr['syn_mtr_ini_prop_q90'] - dr['syn_mtr_ini_prop_q10'],
#       dr['syn_mtr_ini_prop_med']]
# sg2 = [dr['syn_st_ini_prop_max'] - dr['syn_st_ini_prop_min'], dr['syn_st_ini_prop_q90'] - dr['syn_st_ini_prop_q10'],
#       dr['syn_st_ini_prop_med']]
lbl_ = [r'$E_{ff_{%s}}$', r'$E_{ff{var_{%s}}}$', r'$E_{ff{med_{%s}}}$']
cols_ = ['tab:red', 'tab:green', 'tab:blue']
t_ = ['Transitory state', 'Stationary state']
title = "Frequency responses for short-term "
title += "facilitation" if ind == 8 else "depression"
y_label = r"$E_{psp}(t)$ (mV)"
path_save = folder_plots + dr_gain_control_file
path_save += '_freq_response_facilitation_phd.png' if ind == 8 else '_freq_response_depression_phd.png'
plot_features_tr_st_1window_phd(f_vec, sg1, sg2, lbl_, cols_, t_, title, path_save, True,
                                y_lims_ind_plot=y_lims, y_lbl=y_label, maxf=59)
title += r", $\delta = %.1f$" % gain
t_ = ['ini-window', 'mid-window', 'end-window']
cols_ = ['tab:red', 'tab:green', 'tab:blue']
legends = [r'$E_{ff_{[w],%s}}$', r'$E_{ff_{[w],%s}}^\mathrm{var}$', r'$E_{ff_{[w],%s}}^\mathrm{med}$']
prefix = ['mtr', 'st']
# prefix = ['syn_mtr', 'syn_st']
prefix_mid = ['ini', 'mid', 'end']
path_save = folder_plots + dr_gain_control_file
path_save += '_freq_response_3w_facilitation_phd.png' if ind == 8 else '_freq_response_3w_depression_phd.png'
y_lims = [-0.01, 3] if ind == 8 else [-0.005, 1.83]  # y_lims = [-0.005, 0.14] if ind == 8 else [-0.005, 0.08]
plot_features_tr_st_3windows_phd(f_vec, dr_gain, prefix, prefix_mid, lbl_, legends, cols_, t_, title, path_save, True,
                                 y_lims_ind_plot=y_lims, y_lbl=y_label)
t_ = ['ini-window (zoom)', 'mid-window (zoom)', 'end-window (zoom)']
path_save = folder_plots + dr_gain_control_file
path_save += '_freq_response_3w_facilitation_phd_zoom.png' if ind == 8 else '_freq_response_3w_depression_phd_zoom.png'
y_lims = [-0.01, 1] if ind == 8 else [-0.01, 0.4]  # y_lims = [-0.0005, 0.027] if ind == 8 else [-0.0005, 0.012]
plot_features_tr_st_3windows_phd(f_vec, dr_gain, prefix, prefix_mid, lbl_, None, cols_, t_, title, path_save, True,
                                 y_lims_ind_plot=y_lims, y_lbl=y_label)
# Difference between mid-ini and end-mid windows (For efficacy)
eff_i_tr = [dr['%s_%s_prop_max' % (prefix[0], prefix_mid[0])] - dr['%s_%s_prop_min' % (prefix[0], prefix_mid[0])],
            dr['%s_%s_prop_q90' % (prefix[0], prefix_mid[0])] - dr['%s_%s_prop_q10' % (prefix[0], prefix_mid[0])],
            dr['%s_%s_prop_med' % (prefix[0], prefix_mid[0])]]
eff_m_tr = [dr['%s_%s_prop_max' % (prefix[0], prefix_mid[1])] - dr['%s_%s_prop_min' % (prefix[0], prefix_mid[1])],
            dr['%s_%s_prop_q90' % (prefix[0], prefix_mid[1])] - dr['%s_%s_prop_q10' % (prefix[0], prefix_mid[1])],
            dr['%s_%s_prop_med' % (prefix[0], prefix_mid[1])]]
eff_e_tr = [dr['%s_%s_prop_max' % (prefix[0], prefix_mid[2])] - dr['%s_%s_prop_min' % (prefix[0], prefix_mid[2])],
            dr['%s_%s_prop_q90' % (prefix[0], prefix_mid[2])] - dr['%s_%s_prop_q10' % (prefix[0], prefix_mid[2])],
            dr['%s_%s_prop_med' % (prefix[0], prefix_mid[2])]]
eff_i_st = [dr['%s_%s_prop_max' % (prefix[1], prefix_mid[0])] - dr['%s_%s_prop_min' % (prefix[1], prefix_mid[0])],
            dr['%s_%s_prop_q90' % (prefix[1], prefix_mid[0])] - dr['%s_%s_prop_q10' % (prefix[1], prefix_mid[0])],
            dr['%s_%s_prop_med' % (prefix[1], prefix_mid[0])]]
eff_m_st = [dr['%s_%s_prop_max' % (prefix[1], prefix_mid[1])] - dr['%s_%s_prop_min' % (prefix[1], prefix_mid[1])],
            dr['%s_%s_prop_q90' % (prefix[1], prefix_mid[1])] - dr['%s_%s_prop_q10' % (prefix[1], prefix_mid[1])],
            dr['%s_%s_prop_med' % (prefix[1], prefix_mid[1])]]
eff_e_st = [dr['%s_%s_prop_max' % (prefix[1], prefix_mid[2])] - dr['%s_%s_prop_min' % (prefix[1], prefix_mid[2])],
            dr['%s_%s_prop_q90' % (prefix[1], prefix_mid[2])] - dr['%s_%s_prop_q10' % (prefix[1], prefix_mid[2])],
            dr['%s_%s_prop_med' % (prefix[1], prefix_mid[2])]]
f_ = 1  # f_vec
pc_m_i = [(eff_m_tr[i] - eff_i_st[i]) * f_ for i in range(len(eff_m_tr))] + [(eff_m_st[i] - eff_i_st[i]) * f_ for i in range(len(eff_m_st))]
pc_e_m = [(eff_e_tr[i] - eff_m_st[i]) * f_ for i in range(len(eff_e_tr))] + [(eff_e_st[i] - eff_m_st[i]) * f_ for i in range(len(eff_e_st))]
title = "Frequency responses for Proportional Changes (short-term "
title += "facilitation)" if ind == 8 else "depression)"
y_label = r"$E_{psp}$ (mV)"
path_save = folder_plots + dr_gain_control_file
path_save += '_freq_response_pc_facilitation_phd.png' if ind == 8 else '_freq_response_pc_depression_phd.png'
title += r", $\delta = %.1f$" % gain
legends = [r'$PC_{%s,tr}$', r'$PC_{%s,tr}^\mathrm{var}$', r'$PC_{%s,tr}^\mathrm{med}$',
           r'$PC_{%s,st}$', r'$PC_{%s,st}^\mathrm{var}$', r'$PC_{%s,st}^\mathrm{med}$']  # %s = ['m-i', 'e-m']
cols_ = ['tab:red', 'tab:green', 'tab:blue', 'tab:red', 'tab:green', 'tab:blue']
ls = ['--', '--', '--', '-', '-', '-']
t_ = [r'$G_{m-i,tr}(r,\delta)$ and $G_{m-i,st}(r,\delta)$', r'$G_{e-m,tr}(r,\delta)$ and $G_{e-m,st}(r,\delta)$']
y_lims = [-0.61, 0.81] if ind == 8 else [-0.5, 0.35]  # y_lims = [-0.03, 0.04] if ind == 8 else [-0.02, 0.02] # for syn
plot_diff_windows_tr_st_phd(f_vec, dr_gain, pc_m_i, pc_e_m, lbl_, legends, cols_, t_, title, path_save,
                            True, y_lims_ind_plot=y_lims, y_lbl=y_label, ls=ls)
# Difference between mid-ini and end-mid windows (for statistical descriptors)
# Difference between mid-ini and end-mid windows (For efficacy)
eff_i_tr = [dr['%s_%s_prop_max' % (prefix[0], prefix_mid[0])], dr['%s_%s_prop_min' % (prefix[0], prefix_mid[0])],
            dr['%s_%s_prop_q90' % (prefix[0], prefix_mid[0])], dr['%s_%s_prop_q10' % (prefix[0], prefix_mid[0])],
            dr['%s_%s_prop_med' % (prefix[0], prefix_mid[0])]]
eff_m_tr = [dr['%s_%s_prop_max' % (prefix[0], prefix_mid[1])], dr['%s_%s_prop_min' % (prefix[0], prefix_mid[1])],
            dr['%s_%s_prop_q90' % (prefix[0], prefix_mid[1])], dr['%s_%s_prop_q10' % (prefix[0], prefix_mid[1])],
            dr['%s_%s_prop_med' % (prefix[0], prefix_mid[1])]]
eff_e_tr = [dr['%s_%s_prop_max' % (prefix[0], prefix_mid[2])], dr['%s_%s_prop_min' % (prefix[0], prefix_mid[2])],
            dr['%s_%s_prop_q90' % (prefix[0], prefix_mid[2])], dr['%s_%s_prop_q10' % (prefix[0], prefix_mid[2])],
            dr['%s_%s_prop_med' % (prefix[0], prefix_mid[2])]]
eff_i_st = [dr['%s_%s_prop_max' % (prefix[1], prefix_mid[0])], dr['%s_%s_prop_min' % (prefix[1], prefix_mid[0])],
            dr['%s_%s_prop_q90' % (prefix[1], prefix_mid[0])], dr['%s_%s_prop_q10' % (prefix[1], prefix_mid[0])],
            dr['%s_%s_prop_med' % (prefix[1], prefix_mid[0])]]
eff_m_st = [dr['%s_%s_prop_max' % (prefix[1], prefix_mid[1])], dr['%s_%s_prop_min' % (prefix[1], prefix_mid[1])],
            dr['%s_%s_prop_q90' % (prefix[1], prefix_mid[1])], dr['%s_%s_prop_q10' % (prefix[1], prefix_mid[1])],
            dr['%s_%s_prop_med' % (prefix[1], prefix_mid[1])]]
eff_e_st = [dr['%s_%s_prop_max' % (prefix[1], prefix_mid[2])], dr['%s_%s_prop_min' % (prefix[1], prefix_mid[2])],
            dr['%s_%s_prop_q90' % (prefix[1], prefix_mid[2])], dr['%s_%s_prop_q10' % (prefix[1], prefix_mid[2])],
            dr['%s_%s_prop_med' % (prefix[1], prefix_mid[2])]]
f_ = 1  # f_vec
pc_m_i = [(eff_m_tr[i] - eff_i_st[i]) * f_ for i in range(len(eff_m_tr))] + [(eff_m_st[i] - eff_i_st[i]) * f_ for i in range(len(eff_m_st))]
pc_e_m = [(eff_e_tr[i] - eff_m_st[i]) * f_ for i in range(len(eff_e_tr))] + [(eff_e_st[i] - eff_m_st[i]) * f_ for i in range(len(eff_e_st))]
cols_ = ['tab:red', 'tab:red', 'tab:green', 'tab:green', 'tab:blue',
         'tab:red', 'tab:red', 'tab:green', 'tab:green', 'tab:blue']
legends = [r'$PC_{%s,tr}^\mathrm{max}$', r'$PC_{%s,tr}^\mathrm{min}$', r'$PC_{%s,tr}^\mathrm{q90}$',
           r'$PC_{%s,tr}^\mathrm{q10}$', r'$PC_{%s,tr}^\mathrm{med}$',
           r'$PC_{%s,st}^\mathrm{max}$', r'$PC_{%s,st}^\mathrm{min}$', r'$PC_{%s,st}^\mathrm{q90}$',
           r'$PC_{%s,st}^\mathrm{q10}$', r'$PC_{%s,st}^\mathrm{med}$']  # %s = ['m-i', 'e-m']
ls = ['-', '--', '-', '--', '-', '-', '--', '-', '--', '-']
plot_diff_windows_tr_st_phd(f_vec, dr_gain, pc_m_i, pc_e_m, lbl_, legends, cols_, t_, title, path_save,
                            False, y_lims_ind_plot=y_lims, y_lbl=y_label, ls=ls)

# ****************************************************************************************************

# Second synapse
if fig_syn_b:
    lbl = ['syn_b_mtr_ini_prop', 'syn_b_mtr_mid_prop', 'syn_b_mtr_end_prop']
    lbl2 = ['syn_b_st_ini_prop', 'syn_b_st_mid_prop', 'syn_b_st_end_prop']
    t_ = ['ini-window', 'mid-window', 'end-window']
    y_lims = [xl_syb[ind][0][8], xl_syb[ind][1][8]] if n_model == "HH" else None
    y_label = "Syn. strength"
    path_save = folder_plots + dr_gain_control_file + '_windows_syn_b_tr_st.png'
    title = ("Transitory and stationary, " + description.split(",")[0] + ", gain " + str(int(gain * 100)) +
             "%. NMDA Synaptic response")
    plot_features_tr_st_3windows(f_vec, dr_gain, lbl, lbl2, st_lbl, legends, cols, t_, title, path_save, save_figs,
                                 ls=ls, y_lims_ind_plot=y_lims, y_lbl=y_label)
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
# For Neuron responses
lbl = ['mtr_mid_prop', 'st_mid_prop', 'st_end_prop']
lbl2 = ['st_ini_prop', 'st_ini_prop', 'st_ini_prop']
st_lbl = ['_max', '_min', '_q10', '_q90', '_med']
ls = ['-', '--', '--', '-', '-']
cols_ = ['tab:red', 'tab:red', 'tab:green', 'tab:green', 'tab:blue']
t_ = [r"$mid_{tr} - ini_{st}$", r"$mid_{st} - ini_{st}$", r"$end_{st} - ini_{st}$"]
y_lims = [xl_neu[ind][0][9], xl_neu[ind][1][9]]
y_label = "mem. pot. (mV)"
name_save = folder_plots + dr_gain_control_file + '_' + 'diff_tr_st_log.png'
title = (description.split(",")[0] + r', $\tau_{lif}$ ' + str(tau_lif) + "ms, gain " +
         str(int(gain * 100)) + "%. Neuron response")
mid_st_lbl = ['st_mid_prop']
mid_tr_lbl = ['mtr_mid_prop']
ini_st_lbl = ['st_ini_prop']
lbls = [r'$m_{st}$ - $i_{st}$(', r'$m_{tr}$ - $i_{st}$(', r'$m_{st}$ - $i_{st}$(', r'$m_{tr}$ - $i_{st}$(',
        r'$m_{st}$ - $i_{st}$(', r'$m_{st}$ - $i_{st}$(', r'$m_{st}$ - $i_{st}$(', r'$m_{st}$ - $i_{st}$(',
        r'$m_{st}$ - $i_{st}$(', r'$m_{st}$ - $i_{st}$(']
plot_diff_windows_tr_st(f_vec, dr_gain, mid_st_lbl, mid_tr_lbl, ini_st_lbl, st_lbl, cols_, t_, title_graph=title,
                        name_save=name_save, ls=ls, save_figs=save_figs, lbls=lbls, fillBetween=True,
                        normalise=norm_neuron, min_n=min_n, max_n=max_n, y_lims_ind_plot=y_lims, y_lbl=y_label)

# For synapse A
lbl = ['syn_mtr_mid_prop', 'syn_st_mid_prop', 'syn_st_end_prop']
lbl2 = ['syn_st_ini_prop', 'syn_st_ini_prop', 'syn_st_ini_prop']
name_save = folder_plots + dr_gain_control_file + '_' + 'diff_syn_tr_st_log.png'
title = (description.split(",")[0] + r', $\tau_{lif}$ ' + str(tau_lif) + "ms, gain " +
         str(int(gain * 100)))
if n_model == "HH": title += "%. AMPA synaptic response"
if n_model == "LIF": title += "%. Synaptic response"
y_lims = [xl_syn[ind][0][9], xl_syn[ind][1][9]]
y_label = "Syn. strength"
# plot_diff_windows(f_vec, dr_gain, lbl, lbl2, st_lbl, cols_, t_, title_graph=title, name_save=name_save, ls=ls,
#                   save_figs=save_figs)
mid_st_lbl = ['syn_st_mid_prop']
mid_tr_lbl = ['syn_mtr_mid_prop']
ini_st_lbl = ['syn_st_ini_prop']
plot_diff_windows_tr_st(f_vec, dr_gain, mid_st_lbl, mid_tr_lbl, ini_st_lbl, st_lbl, cols_, t_, title_graph=title,
                        name_save=name_save, ls=ls, save_figs=save_figs, lbls=lbls, fillBetween=True,
                        y_lims_ind_plot=y_lims, y_lbl=y_label)
# For synapse B
if fig_syn_b:
    lbl = ['syn_b_mtr_mid_prop', 'syn_b_st_mid_prop', 'syn_b_st_end_prop']
    lbl2 = ['syn_b_st_ini_prop', 'syn_b_st_ini_prop', 'syn_b_st_ini_prop']
    name_save = folder_plots + dr_gain_control_file + '_' + 'diff_syn_b_tr_st_log.png'
    title = (description.split(",")[0] + r', $\tau_{lif}$ ' + str(tau_lif) + "ms, gain " +
             str(int(gain * 100)) + "%. NMDA synaptic response")
    y_lims = [xl_syb[ind][0][9], xl_syb[ind][1][9]] if n_model == "HH" else None
    y_label = "Syn. strength"
    # plot_diff_windows(f_vec, dr_gain, lbl, lbl2, st_lbl, cols_, t_, title_graph=title, name_save=name_save, ls=ls,
    #                   save_figs=save_figs)
    mid_st_lbl = ['syn_b_st_mid_prop']
    mid_tr_lbl = ['syn_b_mtr_mid_prop']
    ini_st_lbl = ['syn_b_st_ini_prop']
    plot_diff_windows_tr_st(f_vec, dr_gain, mid_st_lbl, mid_tr_lbl, ini_st_lbl, st_lbl, cols_, t_, title_graph=title,
                            name_save=name_save, ls=ls, save_figs=save_figs, lbls=lbls, fillBetween=False,
                            y_lims_ind_plot=y_lims, y_lbl=y_label)
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
