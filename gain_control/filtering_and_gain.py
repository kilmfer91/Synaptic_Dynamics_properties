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
s_model = 'TM'
n_model = 'LIF'
ind = 8
save_figs = True
plot_figs = True
num_syn = 1

# Sampling frequency and conditions for running parallel or single LIF neurons
sfreq = 10e3
tau_lif = 30  # ms

# Path variables
aux_p = ''  # '_2'
path_vars = "../gain_control/variables/high_freq_10k" + aux_p + "/"
check_create_folder(path_vars)
folder_plots = '../gain_control/plots/freq_portrait/'
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
gain_v = [0.5]  # [0.1, 0.5, 1.0]
filt_dict_loaded = False

# Titles graphs
title = "Model " + s_model + ', ind ' + str(ind)
if n_model == 'LIF': title += r', $\tau_{lif}$ ' + str(tau_lif) + "ms"
if len(gain_v) == 1: title += ', gain ' + str(int(gain_v[0] * 100)) + '%'
else: title += ', multiple gains'

# Plot
title_mp = ['Amplitude in steady-state', 'Varibility in steady-state', 'Median in steady-state',
            'Entropy in steady-state', 'Amplitude in transitory-state', 'Varibility in transitory-state',
            'Median in transitory-state', 'Entropy in transitory-state']
x_label_ax_p = [r'$E_{ff_{i,st}}^{amp}$ (mV)', r'$E_{ff_{i,st}}^{var}$ (mV)', r'$E_{ff_{i,st}}^{med}$ (mV)',
                r'$H_{i,st}$ (bits)', r'$E_{ff_{i,st}}^{amp}$ (mV)', r'$E_{ff_{i,st}}^{var}$ (mV)',
                r'$E_{ff_{i,st}}^{med}$ (mV)', r'$H_{i,st}$ (bits)']
y_label_ax_p = [r'$G_{m-i,st}^{amp} (mV)$', r'$G_{m-i,st}^{var} (mV)$', r'$G_{m-i,st}^{med} (mV)$',
                r'$GH_{m-i,st}$ (bits)', r'$G_{m-i,tr}^{amp} (mV)$', r'$G_{m-i,tr}^{var} (mV)$',
                r'$G_{m-i,tr}^{med} (mV)$', r'$GH_{m-i,tr}$ (bits)']
# x_label_ax_n = [r'$E_{ff_{m,st}}^{amp}$ (mV)', r'$E_{ff_{m,st}}^{var}$ (mV)', r'$E_{ff_{m,st}}^{med}$ (mV)',
#                 r'$H_{m,st}$ (bits)', r'$E_{ff_{m,st}}^{amp}$ (mV)', r'$E_{ff_{m,st}}^{var}$ (mV)',
#                 r'$E_{ff_{m,st}}^{med}$ (mV)', r'$H_{m,st}$ (bits)']
# y_label_ax_n = [r'$G_{e-m,st}^{amp} (mV)$', r'$G_{e-m,st}^{var} (mV)$', r'$G_{e-m,st}^{med} (mV)$',
#                 r'$GH_{e-m,st}$ (bits)', r'$G_{e-m,tr}^{amp} (mV)$', r'$G_{e-m,tr}^{var} (mV)$',
#                 r'$G_{e-m,tr}^{med} (mV)$', r'$GH_{e-m,tr}$ (bits)']
# title_freqres = ['H - filtering', 'H - Gain-control', 'Transitory time', 'Synaptic Filtering', 'GC - amp', 'GC - var',
#             'GC - med']
title_freqres = ['Temp. filtering', 'Transients', 'Entropy (stationary)', 'Entropy (transitory)', 'Gain effect (amp)',
                 'Gain effect (med)', 'Gain effect (Entropy)']
# ylabel_axb = ["Entropy (bits)", "Entropy (bits)", "Time (s)", "Mem. pot. (mV)", "Mem. pot. (mV)", "Mem. pot. (mV)",
#               "Mem. pot. (mV)"]
ylabel_axb = ["Mem. pot. (mV)", "Mem. pot. (mV)", "Entropy (bits)", "Entropy (bits)", "Mem. pot. (mV)",
              "Mem. pot. (mV)", "Entropy (bits)"]
c_g = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
       'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

name_n_state_variables, name_syn_state_variables = None, None
ax_f, ax_fs, alphas, markers = None, None, None, None
xl_neu, xl_syn, xl_syb, ax_s, ax_sb, ax_hI, ax_h, ax_h, ax_hs = [None for _ in range(9)]
n_freq_por, figNeur_neg_gc, figSynapse, n_freq_res, s_freq_res = None, None, None, None, None
figSynapseb, figCompPropSynb, figEntropyInput, figEntropy, ax_p, ax_n = None, None, None, None, None, None
s_freq_por, figSyn_neg_gc, ax_sp, ax_sn = None, None, None, None

if plot_figs:
    plt.rcParams['figure.constrained_layout.use'] = True
    # Synaptic filtering vs. Gain-Control for Neuron
    dr_gain_control_file = get_name_file(sfreq, s_model, n_model, ind, num_syn, tau_lif, True, 0.1)
    if os.path.isfile(path_vars + dr_gain_control_file) and not filt_dict_loaded:
        # Name state variables
        dr_filt = loadObject(dr_gain_control_file, path_vars)
        name_n_state_variables = dr_filt['name_neuron_state_variables']
        name_syn_state_variables = dr_filt['name_syn_state_variables']

        # Frequency portrait - Neuron
        title_ = 'Frequency portrait for Neuron - %s(t)'
        n_freq_por, ax_p = create_fig_freq_portrait(name_n_state_variables, title_)

        # Frequency portrait - Synapse
        title_ = 'Frequency portrait for Synapse - %s(t)'
        s_freq_por, ax_sp = create_fig_freq_portrait(name_syn_state_variables, title_)

        # Frequency responses - neuron
        title_ = 'Frequency responses for neuron - %s(t)'
        n_freq_res, ax_f = create_fig_freq_responses(name_n_state_variables, title_)

        # Frequency responses - synapse
        title_ = 'Frequency responses for synapse - %s(t)'
        s_freq_res, ax_fs = create_fig_freq_responses(name_syn_state_variables, title_)

    alpha = 0.3
    markers = ['+', '*']
    alphas = [1.0, 0.5]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

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
    dr_syn_filtering_file = get_name_file(sfreq, s_model, n_model, ind, num_syn, tau_lif, False, gain)
    dr_gain_control_file = get_name_file(sfreq, s_model, n_model, ind, num_syn, tau_lif, True, gain)

    print("For gain control, file %s and index %d" % (dr_gain_control_file, ind))
    print("For synaptic filtering, file %s and index %d" % (dr_syn_filtering_file, ind))

    # ******************************************************************************************************************
    # Trying to load freq. response of Gain Control
    if os.path.isfile(path_vars + dr_syn_filtering_file) and not filt_dict_loaded:
        dr_filt = loadObject(dr_syn_filtering_file, path_vars)
        # Auxiliar variables
        initial_frequencies, model = dr_filt['initial_frequencies'], dr_filt['stp_model']
        # dyn_synapse, num_synapses = dr_filt['dyn_synapse'], dr_filt['num_synapses']
        # num_realizations, sim_params = dr_filt['realizations'], dr_filt['sim_params']
        # prop_rate_change_a = dr_filt['prop_rate_change_a']
        # fix_rate_change_a, num_changes_rate, = dr_filt['fix_rate_change_a'], dr_filt['num_changes_rate'],
        # description = dr_filt['description']
        # seeds = dr_filt['seeds']
        total_realizations = dr_filt['t_realizations']

        # Name state variables
        name_n_state_variables = dr_filt['name_neuron_state_variables']
        name_syn_state_variables = dr_filt['name_syn_state_variables']

    if os.path.isfile(path_vars + dr_gain_control_file):
        dr_gain = loadObject(dr_gain_control_file, path_vars)

    f_vec = dr_gain['initial_frequencies']
    f_vecD = dr_filt['initial_frequencies']

    # ******************************************************************************************************************
    # Plots 1
    dr_ = dr_gain
    if plot_figs:
        # FREQUENCY RESPONSES OF NEURONS AND SYNAPSES
        # For Neurons
        plot_freq_responses(name_n_state_variables, dr_filt, dr_gain, dr_['time_transition'], gain, ax_f,
                            norm_neuron, title_mp, markers, alphas, c_g=c_g[i_g], plot_filt=i_g == 0, ode='n')
        # For synapses
        plot_freq_responses(name_syn_state_variables, dr_filt, dr_gain, dr_['time_transition'], gain, ax_fs,
                            norm_neuron, title_mp, markers, alphas, c_g=c_g[i_g], plot_filt=i_g == 0, ode='s')

        # FREQUENCY PORTRAITS OF NEURONS AND SYNAPSES
        # For neurons
        plot_freq_portrait2(name_n_state_variables, dr_filt, dr_gain, gain, ax_p, norm_neuron, title_mp,
                            colors[i_g], ode='n')  # , H_filt, H_gain)

        # For synapses
        plot_freq_portrait2(name_syn_state_variables, dr_filt, dr_gain, gain, ax_sp, norm_neuron, title_mp,
                            colors[i_g], ode='s')  # , H_filt, H_gain)
    # **********************************************************************************************************

    i_g += 1

path_save = (folder_plots + s_model + '_ind_' + str(ind) + '_' + str(len(gain_v)) + '_gains_sf_' +
             str(int(sfreq * 1e-3)) + 'k_tauLIF_' + str(tau_lif) + 'ms')

# Adjusting frequency portraits and frequency responses
if plot_figs:
    sizeF = 20
    # Neuronal state variables
    for n in range(len(name_n_state_variables)):
        for j in range(len(title_mp)):
            # Frequency portrait for Neuron
            adjust_freq_portraits(ax_p[n][j], x_label_ax_p[j], y_label_ax_p[j], title_mp[j])  # xl, yl
        for j in range(len(title_freqres)):
            # Frequency responses for ini window
            adjust_freq_portraits(ax_f[n][j], "Rate (Hz)", ylabel_axb[j], title_freqres[j], xscale='log',
                                  axes_=False, x_axis=False)
            # Frequency responses for mid window
            adjust_freq_portraits(ax_f[n][j + 7], "Rate (Hz)", ylabel_axb[j], title_freqres[j], xscale='log',
                                  axes_=False, x_axis=False, tit_=False)
            # Frequency responses for end window
            adjust_freq_portraits(ax_f[n][j + 14], "Rate (Hz)", ylabel_axb[j], title_freqres[j], xscale='log',
                                  axes_=False, tit_=False)

    for n in range(len(name_syn_state_variables)):
        for j in range(len(title_mp)):
            # Frequency portrait for Synapses
            adjust_freq_portraits(ax_sp[n][j], x_label_ax_p[j], y_label_ax_p[j], title_mp[j])  # xl, yl

        for j in range(len(title_freqres)):
            # Frequency responses for ini window
            adjust_freq_portraits(ax_fs[n][j], "Rate (Hz)", ylabel_axb[j], title_freqres[j], xscale='log',
                                  axes_=False, x_axis=False)
            # Frequency responses for mid window
            adjust_freq_portraits(ax_fs[n][j + 7], "Rate (Hz)", ylabel_axb[j], title_freqres[j], xscale='log',
                                  axes_=False, x_axis=False, tit_=False)
            # Frequency responses for end window
            adjust_freq_portraits(ax_fs[n][j + 14], "Rate (Hz)", ylabel_axb[j], title_freqres[j], xscale='log',
                                  axes_=False, tit_=False)

    # Legends
    # Frequency portraits
    for n in range(len(name_n_state_variables)):
        ax_p[n][int(len(title_mp) / 2) - 1].legend(bbox_to_anchor=(1.05, 0.7), loc='upper left', borderaxespad=0.,
                                                title='gain factor')
    for n in range(len(name_syn_state_variables)):
        ax_sp[n][int(len(title_mp) / 2) - 1].legend(bbox_to_anchor=(1.05, 0.7), loc='upper left', borderaxespad=0.,
                                                    title='gain factor')

    # Frequency responses
    lbl_ind = []
    if 0.1 in gain_v: lbl_ind.append([int(len(title_mp) / 2), len(title_freqres)])
    if 0.5 in gain_v: lbl_ind.append([7 + int(len(title_mp) / 2), 7 + len(title_freqres)])
    if 1.0 in gain_v: lbl_ind.append([14 + int(len(title_mp) / 2), 14 + len(title_freqres)])

    for n in range(len(name_n_state_variables)):
        adjust_legend_freq_res(lbl_ind, n_freq_res[n], ax_f[n], gain_v)
        # for ind_ in lbl_ind:
        #     ax_f[n][ind_].legend(bbox_to_anchor=(1.05, 0.7), loc='upper left', borderaxespad=0., title='windows')

    for n in range(len(name_syn_state_variables)):
        adjust_legend_freq_res(lbl_ind, s_freq_res[n], ax_fs[n], gain_v)
    #     for ind_ in lbl_ind:
    #         ax_fs[n][ind_].legend(bbox_to_anchor=(1.05, 0.7), loc='upper left', borderaxespad=0., title='windows')

if plot_figs and save_figs:
    for j in range(len(name_n_state_variables)):
        n = name_n_state_variables[j]
        n_freq_por[j].savefig(path_save + "_freq_portrait_neuron_" + n + "_pos" + aux_p + ".png", format='png')
        n_freq_res[j].savefig(path_save + "_freq_responses_neuron_" + n + aux_p + ".png", format='png')
    for j in range(len(name_syn_state_variables)):
        n = name_syn_state_variables[j]
        s_freq_por[j].savefig(path_save + "_freq_portrait_synapse_" + n + "_pos" + aux_p + ".png", format='png')
        s_freq_res[j].savefig(path_save + "_freq_responses_synapse_" + n + aux_p + ".png", format='png')
# """

# Figure PhD thesis (methodology / metrics temporal filtering)
"""
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
       dr['mtr_ini_prop_med']]
sg2 = [dr['st_ini_prop_max'] - dr['st_ini_prop_min'], dr['st_ini_prop_q90'] - dr['st_ini_prop_q10'],
       dr['st_ini_prop_med'] - dr['st_ini_prop_min']]
# sg1 = [dr['syn_mtr_ini_prop_max'] - dr['syn_mtr_ini_prop_min'], dr['syn_mtr_ini_prop_q90']-dr['syn_mtr_ini_prop_q10'],
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
pc_m_i = [(eff_m_tr[i] - eff_i_st[i]) * f_ for i in range(len(eff_m_tr))] + [(eff_m_st[i] - eff_i_st[i]) * f_ 
                                                                             for i in range(len(eff_m_st))]
pc_e_m = [(eff_e_tr[i] - eff_m_st[i]) * f_ for i in range(len(eff_e_tr))] + [(eff_e_st[i] - eff_m_st[i]) * f_ 
                                                                             for i in range(len(eff_e_st))]
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
pc_m_i = [(eff_m_tr[i] - eff_i_st[i]) * f_ for i in range(len(eff_m_tr))] + [(eff_m_st[i] - eff_i_st[i]) * f_ 
                                                                             for i in range(len(eff_m_st))]
pc_e_m = [(eff_e_tr[i] - eff_m_st[i]) * f_ for i in range(len(eff_e_tr))] + [(eff_e_st[i] - eff_m_st[i]) * 
                                                                             f_ for i in range(len(eff_e_st))]
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
