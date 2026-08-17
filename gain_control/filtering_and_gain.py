from gain_control.utils_gc import *

# STP model and extra global variables
# (Experiment 2) freq. response decay around 100Hz
# (Experiment 3) freq. response decay around 10Hz
# (Experiment 4) freq. response from Gain Control paper
# (Experiment 5) freq. response decay around 100Hz
# (Experiment 6) freq. response decay around 10Hz

SYSTEMS = {
    1: ["TM", "LIF", 4, 'TM STD (4)', 1e-3],
    2: ["TM", "LIF", 8, 'TM STF(8)', 1e-3],
    3: ["MSSM", "LIF", 4, 'MSSM STD(4)', 1e-3],
    4: ["MSSM", "LIF", 7, 'MSSM STF? (7)', 1e-3],
    5: ["DoornSTD", "HH", 0, 'Doorn STD (0) healthy', 1.0],
    6: ["DoornSTD", "HH", 1, 'Doorn STD (1) Dravet', 1.0],
    7: ["DoornSTF", "HH", 7, 'Doorn STF (7) Pers. Exc.', 1.0],
    8: ["DoornSTD", "HH", 8, 'Doorn STD (8) Dravet', 1.0],
}

ind_sys = 2
s_model, n_model, ind, sys_description, factor = SYSTEMS[ind_sys]

# Flags for plotting
save_figs = False
plot_figs = True
plot_phd_meth = True
plot_freq_res = False
freq_res_T = True
freq_res_single = False

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
gain_v = [0.1, 0.5, 1.0]
filt_dict_loaded = False

# Titles graphs
title = "Model " + s_model + ', ind ' + str(ind)
if n_model == 'LIF': title += r', $\tau_{lif}$ ' + str(tau_lif) + "ms"
if len(gain_v) == 1: title += ', gain ' + str(int(gain_v[0] * 100)) + '%'
else: title += ', multiple gains'

# Plot
# title_mp = ['Amplitude in steady-state', 'Varibility in steady-state', 'Median in steady-state',
#             'Entropy in steady-state', 'Amplitude in transitory-state', 'Varibility in transitory-state',
#             'Median in transitory-state', 'Entropy in transitory-state']
# title_mp = ['Amplitude in steady-state', 'Median in transitory-state', 'Amplitude in transient dynamics',
#             'Entropy in steady-state', 'Amplitude in transitory-state', 'Median in transient dynamics',
#             'Entropy in transient dynamics', 'Entropy in transitory-state']
title_mp = ['Amplitude in transitory-state', 'Median in transitory-state', 'Entropy in transitory-state',
            'Amplitude in transient dynamics', 'Median in transient dynamics', 'Entropy in transient dynamics']
# x_label_ax_p = [r'$E_{ff_{i,st}}^{amp}$ (mV)', r'$E_{ff_{i,st}}^{var}$ (mV)', r'$E_{ff_{i,st}}^{med}$ (mV)',
#                 r'$H_{i,st}$ (bits)', r'$E_{ff_{i,st}}^{amp}$ (mV)', r'$E_{ff_{i,st}}^{var}$ (mV)',
#                 r'$E_{ff_{i,st}}^{med}$ (mV)', r'$H_{i,st}$ (bits)']
# y_label_ax_p = [r'$G_{m-i,st}^{amp} (mV)$', r'$G_{m-i,st}^{var} (mV)$', r'$G_{m-i,st}^{med} (mV)$',
#                 r'$GH_{m-i,st}$ (bits)', r'$G_{m-i,tr}^{amp} (mV)$', r'$G_{m-i,tr}^{var} (mV)$',
#                 r'$G_{m-i,tr}^{med} (mV)$', r'$GH_{m-i,tr}$ (bits)']
# x_label_ax_p = [r'$E_{ff_{st}}^{amp}$ (mV)', r'$E_{ff_{st}}^{med}$ (mV)', r'$E_{ff_{tr}}^{amp}$ (mV)',
#                 r'$H_{st}$ (bits)', r'$E_{ff_{st}}^{amp}$ (mV)', r'$E_{ff_{tr}}^{med}$ (mV)',
#                 r'$H_{tr}$ (bits)', r'$H_{st}$ (bits)']
# y_label_ax_p = [r'$G_{st-st}^{amp} (mV)$', r'$G_{tr-st}^{med} (mV)$', r'$G_{tr-tr}^{amp} (mV)$',
 #                r'$GH_{st-st}$ (bits)', r'$G_{tr-st}^{amp} (mV)$', r'$G_{tr-tr}^{med} (mV)$',
#                 r'$GH_{tr-tr}$ (bits)', r'$GH_{tr-st}$ (bits)']
x_label_ax_p = [r'$E_{ff_{st}}^{amp}$ (mV)', r'$E_{ff_{st}}^{med}$ (mV)', r'$H_{st}$ (bits)',
                r'$E_{ff_{tr}}^{amp}$ (mV)', r'$E_{ff_{tr}}^{med}$ (mV)', r'$H_{tr}$ (bits)']
y_label_ax_p = [r'$G_{tr-st}^{amp} (mV)$', r'$G_{tr-st}^{med} (mV)$', r'$GH_{tr-st}$ (bits)',
                r'$G_{tr-tr}^{amp} (mV)$', r'$G_{tr-tr}^{med} (mV)$', r'$GH_{tr-tr}$ (bits)']
# x_label_ax_n = [r'$E_{ff_{m,st}}^{amp}$ (mV)', r'$E_{ff_{m,st}}^{var}$ (mV)', r'$E_{ff_{m,st}}^{med}$ (mV)',
#                 r'$H_{m,st}$ (bits)', r'$E_{ff_{m,st}}^{amp}$ (mV)', r'$E_{ff_{m,st}}^{var}$ (mV)',
#                 r'$E_{ff_{m,st}}^{med}$ (mV)', r'$H_{m,st}$ (bits)']
# y_label_ax_n = [r'$G_{e-m,st}^{amp} (mV)$', r'$G_{e-m,st}^{var} (mV)$', r'$G_{e-m,st}^{med} (mV)$',
#                 r'$GH_{e-m,st}$ (bits)', r'$G_{e-m,tr}^{amp} (mV)$', r'$G_{e-m,tr}^{var} (mV)$',
#                 r'$G_{e-m,tr}^{med} (mV)$', r'$GH_{e-m,tr}$ (bits)']
# title_freqres = ['H - filtering', 'H - Gain-control', 'Transitory time', 'Synaptic Filtering', 'GC - amp', 'GC - var',
#             'GC - med']
title_freqres = ['Transient dynamics', 'Temporal filtering', 'Entropy', 'Gain effect (amp)', 'Gain effect (med)',
           'Gain effect (Entropy)']
title_freq_save_fig = ['_transients', '_filtering', '_information', '_gain_amp', '_gain_med', '_gain_entropy']
if freq_res_single:
    s_d = sys_description
    tit_aux = [s_d + ', %s' % i + ' %s(t)' for i in title_freqres]
    title_freqres_sing = tit_aux
    title_freqres = [[r'$\delta$ = %.1f' % gain for gain in gain_v] for _ in range(len(title_freqres_sing))]

if freq_res_T and not freq_res_single: title_freqres = ['', 'Transient dynamics', '',
                                                        '', 'Temporal filtering', '',
                                                        '', 'Entropy', '',
                                                        '', 'Gain effect (amp)', '',
                                                        '', 'Gain effect (med)', '',
                                                        '', 'Gain effect (Entropy)', '']

# ylabel_axb = ["Entropy (bits)", "Entropy (bits)", "Time (s)", "Mem. pot. (mV)", "Mem. pot. (mV)", "Mem. pot. (mV)",
#               "Mem. pot. (mV)"]
ylabel_axb = ["Mem. pot. (mV)", "Mem. pot. (mV)", "Entropy (bits)", "Mem. pot. (mV)",
              "Mem. pot. (mV)", "Entropy (bits)"]
if freq_res_T: ylabel_axb = ['Mem. pot. (mV)', '', '',
                            'Mem. pot. (mV)', '', '',
                            'Entropy (bits)', '', '',
                            'Mem. pot. (mV)', '', '',
                            'Mem. pot. (mV)', '', '',
                            'Entropy (bits)', '', '']
c_g = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
       'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

name_n_state_variables, name_syn_state_variables = None, None
ax_f, ax_fs, alphas, markers = None, None, None, None
xl_neu, xl_syn, xl_syb, ax_s, ax_sb, ax_hI, ax_h, ax_h, ax_hs = [None for _ in range(9)]
n_freq_por, figNeur_neg_gc, figSynapse, n_freq_res, s_freq_res = None, None, None, None, None
figSynapseb, figCompPropSynb, figEntropyInput, figEntropy, ax_p, ax_n = None, None, None, None, None, None
s_freq_por, figSyn_neg_gc, ax_sp, ax_sn = None, None, None, None
alpha = 0.3
markers = ['+', '*']
alphas = [1.0, 0.5]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
colors = ['tab:gray', 'tab:purple', 'tab:cyan']
if plot_figs:
    plt.rcParams['figure.constrained_layout.use'] = True
    # Synaptic filtering vs. Gain-Control for Neuron
    dr_gain_control_file = get_name_file(sfreq, s_model, n_model, ind, 1, tau_lif, True, 0.1)
    if os.path.isfile(path_vars + dr_gain_control_file) and not filt_dict_loaded:
        # Name state variables
        dr_filt = loadObject(dr_gain_control_file, path_vars)
        name_n_state_variables = dr_filt['name_neuron_state_variables']
        name_syn_state_variables = dr_filt['name_syn_state_variables']

        # Frequency portrait - Neuron
        title_ = sys_description + '. Frequency portrait for Neuron - %s(t)'
        n_freq_por, ax_p = create_fig_freq_portrait(name_n_state_variables, title_, figsize=(20, 8))

        # Frequency portrait - Synapse
        title_ = sys_description + '. Frequency portrait for Synapse - %s(t)'
        s_freq_por, ax_sp = create_fig_freq_portrait(name_syn_state_variables, title_, figsize=(20, 8))

        if plot_freq_res:
            # Frequency responses - neuron
            title_ = ""
            if freq_res_single: title_ = title_freqres_sing
            else: title_ = sys_description + '. Frequency responses for neuron - %s(t)'
            n_freq_res, ax_f = create_fig_freq_responses(name_n_state_variables, title_, freq_res_T, freq_res_single)

            # Frequency responses - synapse
            if freq_res_single: title_ = title_freqres_sing
            else: title_ = sys_description + '. Frequency responses for synapse - %s(t)'
            s_freq_res, ax_fs = create_fig_freq_responses(name_syn_state_variables, title_, freq_res_T, freq_res_single)

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
    dr_syn_filtering_file = get_name_file(sfreq, s_model, n_model, ind, 1, tau_lif, False, gain)
    dr_gain_control_file = get_name_file(sfreq, s_model, n_model, ind, 1, tau_lif, True, gain)

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
    if plot_figs and plot_freq_res:
        # FREQUENCY RESPONSES OF NEURONS AND SYNAPSES
        # For Neurons
        plot_freq_responses(name_n_state_variables, dr_filt, dr_gain, dr_['time_transition'], gain, ax_f,
                            norm_neuron, title_mp, markers, alphas, c_g=c_g[i_g], plot_filt=i_g == 0, ode='n',
                            transpose=freq_res_T)
        # For synapses
        plot_freq_responses(name_syn_state_variables, dr_filt, dr_gain, dr_['time_transition'], gain, ax_fs,
                            norm_neuron, title_mp, markers, alphas, c_g=c_g[i_g], plot_filt=i_g == 0, ode='s',
                            transpose=freq_res_T, single_properties=freq_res_single)

    if plot_figs:
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
        for k in range(len(title_mp)):
            # Frequency portrait for Neuron
            adjust_freq_portraits(ax_p[n][k], x_label_ax_p[k], y_label_ax_p[k], title_mp[k])  # xl, yl

        if plot_freq_res:
            # Adjusting frequency responses for neurons
            adjust_freq_responses(ax_f[n], title_freqres, freq_res_T, freq_res_single, gain_v, ylabel_axb)

    for n in range(len(name_syn_state_variables)):
        for k in range(len(title_mp)):
            # Frequency portrait for Synapses
            adjust_freq_portraits(ax_sp[n][k], x_label_ax_p[k], y_label_ax_p[k], title_mp[k])  # xl, yl

        if plot_freq_res:
            # Adjusting frequency responses for synapses
            adjust_freq_responses(ax_fs[n], title_freqres, freq_res_T, freq_res_single, gain_v, ylabel_axb)

    # Legends
    # Frequency portraits
    for n in range(len(name_n_state_variables)):
        ax_p[n][int(len(title_mp) / 2) - 1].legend(bbox_to_anchor=(1.05, 0.7), loc='upper left', borderaxespad=0.,
                                                title='gain factor')
    for n in range(len(name_syn_state_variables)):
        ax_sp[n][int(len(title_mp) / 2) - 1].legend(bbox_to_anchor=(1.05, 0.7), loc='upper left', borderaxespad=0.,
                                                    title='gain factor')

    # Frequency responses
    if plot_freq_res:
        if not freq_res_single:
            lbl_ind = [7, 18] if freq_res_T else []
            l_ = len(title_freqres)
            if 0.1 in gain_v and not freq_res_T: lbl_ind.append([int(len(title_mp) / 2) - 1, l_])
            if 0.5 in gain_v and not freq_res_T: lbl_ind.append([6 + int(len(title_mp) / 2) - 1, 6 + l_])
            if 1.0 in gain_v and not freq_res_T: lbl_ind.append([12 + int(len(title_mp) / 2) - 1, 12 + l_])

            # For state variables of neurons
            for n in range(len(name_n_state_variables)):
                if freq_res_T: adjust_legend_freq_resT(lbl_ind, n_freq_res[n], ax_f[n], gain_v)
                else: adjust_legend_freq_res(lbl_ind, n_freq_res[n], ax_f[n], gain_v)
            # For state variables of synapses
            for n in range(len(name_syn_state_variables)):
                if freq_res_T: adjust_legend_freq_resT(lbl_ind, s_freq_res[n], ax_fs[n], gain_v)
                else: adjust_legend_freq_res(lbl_ind, s_freq_res[n], ax_fs[n], gain_v)

# Saving plots
if plot_figs and save_figs:
    for j in range(len(name_n_state_variables)):
        n = name_n_state_variables[j]
        n_freq_por[j].savefig(path_save + "_freq_portrait_neuron_" + n + "_pos" + aux_p + ".png", format='png')
        if plot_freq_res:
            if freq_res_single:
                for k in range(len(title_freq_save_fig)):
                    aux_p = title_freq_save_fig[k]
                    n_freq_res[j][k].savefig(path_save + "_freq_res_neuron_" + n + aux_p + ".png", format='png')
            else:
                n_freq_res[j].savefig(path_save + "_freq_responses_neuron_" + n + aux_p + ".png", format='png')
    for j in range(len(name_syn_state_variables)):
        n = name_syn_state_variables[j]
        s_freq_por[j].savefig(path_save + "_freq_portrait_synapse_" + n + "_pos" + aux_p + ".png", format='png')
        if plot_freq_res:
            if freq_res_single:
                for k in range(len(title_freq_save_fig)):
                    aux_p = title_freq_save_fig[k]
                    s_freq_res[j][k].savefig(path_save + "_freq_res_synapse_" + n + aux_p + ".png", format='png')
            else:
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
# y_lims = [xl_syn[ind][0][8], xl_syn[ind][1][8]]
y_label = "Syn. strength"
path_save = folder_plots + dr_gain_control_file + '_windows_syn_tr_st.png'
title = "Transitory and stationary, " + description.split(",")[0] + ", gain " + str(int(gain * 100))
if n_model == "HH": title += "%. AMPA synaptic response"
if n_model == "LIF": title += "%. Synaptic response"
plot_features_tr_st_3windows(f_vec, dr_gain, lbl, lbl2, st_lbl, legends, cols, t_, title, path_save, save_figs, ls=ls,
                             y_lims_ind_plot=y_lims, y_lbl=y_label)
# """

# ****************************************************************************************************
# Figure PhD thesis (methodology)
# """
if plot_phd_meth:
    color_stat = ["tab:purple", "tab:orange", "tab:green", "tab:cyan"]
    color_win = ["tab:red", "tab:olive", "tab:blue"]

    # Figure PhD thesis (methodology / Frequency response of transient dynamics and temporal filtering)
    dr = dr_gain
    sg1 = [dr['mtr_ini_prop_max'] - dr['mtr_ini_prop_min'], dr['mtr_ini_prop_med'] - dr['mtr_ini_prop_min']]
    sg2 = [dr['st_ini_prop_max'] - dr['st_ini_prop_min'], dr['st_ini_prop_med'] - dr['st_ini_prop_min']]
    lbl_ = [r'$E_{ff_{%s}}^{amp}$', r'$E_{ff_{%s}}^{amp}$']
    cols_ = [color_stat[0], color_stat[3]]
    t_ = ['Transient dynamics', 'Temporal filtering']
    tit_ = "Frequency response for short-term %s"
    title = tit_ % 'facilitation' if 'STF' in sys_description else tit_ % 'depression'
    y_label = r"$E_{psp}(t)$ (mV)"
    path_sav_ = folder_plots + dr_gain_control_file
    # For transient dynamics
    aux_t = '_freq_response_%s_transient_phd.png'
    aux_ti = aux_t % 'facilitation' if 'STF' in sys_description else aux_t % 'depression'
    path_save = path_sav_ + aux_ti
    plot_features_tr_st_1window_phd(f_vec, sg1, sg2, lbl_, cols_, t_[0], title, path_save, True, y_lbl=y_label,
                                    linesty='solid')
    # For temporal filtering
    aux_t = '_freq_response_%s_filtering_phd.png'
    aux_ti = aux_t % 'facilitation' if 'STF' in sys_description else aux_t % 'depression'
    path_save = path_sav_ + aux_ti
    plot_features_tr_st_1window_phd(f_vec, sg2, sg2, lbl_, cols_, t_[1], title, path_save, True, y_lbl=y_label,
                                    linesty='dashdot')

    # Figure PhD thesis (methodology / Frequency responses of amplitude and median for each window)
    title += r", $\delta = %.1f$" % gain
    t_ = ['ini-window', 'mid-window', 'end-window']
    # cols_ = ['tab:red', 'tab:green', 'tab:blue']
    legends = [r'$E_{ff_{[w],%s}}$', r'$E_{ff_{[w],%s}}^\mathrm{med}$']
    lbl_ = [r'$E_{ff_{%s}}^{amp}$', r'$E_{ff_{%s}}^{amp}$', r'$E_{ff_{%s}}^{amp}$']
    prefix = ['mtr', 'st']
    # prefix = ['syn_mtr', 'syn_st']
    prefix_mid = ['ini', 'mid', 'end']
    path_save = folder_plots + dr_gain_control_file
    path_save += '_freq_response_3w_facilitation_phd.png' if ind == 8 else '_freq_response_3w_depression_phd.png'
    y_lims = [-0.01, 0.20] if ind == 8 else [-0.005, 0.135]  # y_lims = [-0.005, 0.14] if ind == 8 else [-0.005, 0.08]
    plot_features_tr_st_3windows_phd(f_vec, dr_gain, prefix, prefix_mid, lbl_, legends, cols_, color_win, t_, title,
                                     path_save, True, y_lims_ind_plot=y_lims, y_lbl=y_label)

    # Figure PhD thesis (methodology / Frequency responses of pos and neg changes of rate (stat. desc. efficacy)
    # eff_i_tr = [dr['%s_%s_prop_max' % (prefix[0], prefix_mid[0])] - dr['%s_%s_prop_min' % (prefix[0], prefix_mid[0])],
    #             dr['%s_%s_prop_q90' % (prefix[0], prefix_mid[0])] - dr['%s_%s_prop_q10' % (prefix[0], prefix_mid[0])],
    #             dr['%s_%s_prop_med' % (prefix[0], prefix_mid[0])]]
    # eff_m_tr = [dr['%s_%s_prop_max' % (prefix[0], prefix_mid[1])] - dr['%s_%s_prop_min' % (prefix[0], prefix_mid[1])],
    #             dr['%s_%s_prop_q90' % (prefix[0], prefix_mid[1])] - dr['%s_%s_prop_q10' % (prefix[0], prefix_mid[1])],
    #             dr['%s_%s_prop_med' % (prefix[0], prefix_mid[1])]]
    # eff_e_tr = [dr['%s_%s_prop_max' % (prefix[0], prefix_mid[2])] - dr['%s_%s_prop_min' % (prefix[0], prefix_mid[2])],
    #             dr['%s_%s_prop_q90' % (prefix[0], prefix_mid[2])] - dr['%s_%s_prop_q10' % (prefix[0], prefix_mid[2])],
    #             dr['%s_%s_prop_med' % (prefix[0], prefix_mid[2])]]
    # eff_i_st = [dr['%s_%s_prop_max' % (prefix[1], prefix_mid[0])] - dr['%s_%s_prop_min' % (prefix[1], prefix_mid[0])],
    #             dr['%s_%s_prop_q90' % (prefix[1], prefix_mid[0])] - dr['%s_%s_prop_q10' % (prefix[1], prefix_mid[0])],
    #             dr['%s_%s_prop_med' % (prefix[1], prefix_mid[0])]]
    # eff_m_st = [dr['%s_%s_prop_max' % (prefix[1], prefix_mid[1])] - dr['%s_%s_prop_min' % (prefix[1], prefix_mid[1])],
    #             dr['%s_%s_prop_q90' % (prefix[1], prefix_mid[1])] - dr['%s_%s_prop_q10' % (prefix[1], prefix_mid[1])],
    #             dr['%s_%s_prop_med' % (prefix[1], prefix_mid[1])]]
    # eff_e_st = [dr['%s_%s_prop_max' % (prefix[1], prefix_mid[2])] - dr['%s_%s_prop_min' % (prefix[1], prefix_mid[2])],
    #             dr['%s_%s_prop_q90' % (prefix[1], prefix_mid[2])] - dr['%s_%s_prop_q10' % (prefix[1], prefix_mid[2])],
    #             dr['%s_%s_prop_med' % (prefix[1], prefix_mid[2])]]
    eff_m_tr = [dr['%s_%s_prop_max' % (prefix[0], prefix_mid[1])] - dr['%s_%s_prop_min' % (prefix[0], prefix_mid[1])],
                dr['%s_%s_prop_med' % (prefix[0], prefix_mid[1])]]
    eff_e_tr = [dr['%s_%s_prop_max' % (prefix[0], prefix_mid[2])] - dr['%s_%s_prop_min' % (prefix[0], prefix_mid[2])],
                dr['%s_%s_prop_med' % (prefix[0], prefix_mid[2])]]
    eff_i_st = [dr['%s_%s_prop_max' % (prefix[1], prefix_mid[0])] - dr['%s_%s_prop_min' % (prefix[1], prefix_mid[0])],
                dr['%s_%s_prop_med' % (prefix[1], prefix_mid[0])]]
    eff_m_st = [dr['%s_%s_prop_max' % (prefix[1], prefix_mid[1])] - dr['%s_%s_prop_min' % (prefix[1], prefix_mid[1])],
                dr['%s_%s_prop_med' % (prefix[1], prefix_mid[1])]]
    f_ = 1  # f_vec
    pc_m_i = [(eff_m_tr[i] - eff_i_st[i]) * f_ for i in range(len(eff_m_tr))]  # + [(eff_m_st[i] - eff_i_st[i]) * f_
                                                                               #    for i in range(len(eff_m_st))]
    pc_e_m = [(eff_e_tr[i] - eff_m_st[i]) * f_ for i in range(len(eff_e_tr))]  # + [(eff_e_st[i] - eff_m_st[i]) * f_
                                                                               #    for i in range(len(eff_e_st))]
    title = "Frequency responses for Proportional Changes (short-term "
    title += "facilitation)" if ind == 8 else "depression)"
    y_label = [r'$G_{pos}$ (mV)', r'$G_{neg}$ (mV)']
    path_save = folder_plots + dr_gain_control_file
    path_save += '_freq_response_pc_facilitation_phd.png' if ind == 8 else '_freq_response_pc_depression_phd.png'
    # cols_ = [color_stat[1], color_stat[2]]
    title += r", $\delta = %.1f$" % gain
    # legends = [r'$PC_{%s,tr}^\mathrm{amp}$', r'$PC_{%s,tr}^\mathrm{var}$', r'$PC_{%s,tr}^\mathrm{med}$',
    #            r'$PC_{%s,st}$', r'$PC_{%s,st}^\mathrm{var}$', r'$PC_{%s,st}^\mathrm{med}$']  # %s = ['m-i', 'e-m']
    legends = [r'$PC_{%s}^\mathrm{amp}$', r'$PC_{%s}^\mathrm{med}$']
    leg_2 = ['pos', 'neg']
    color_w = [color_stat[1], color_stat[2]]
    # cols_ = ['tab:red', 'tab:green', 'tab:blue', 'tab:red', 'tab:green', 'tab:blue']
    ls = ['solid', 'dashed']  # , '-', '-', '-']
    # t_ = [r'$G_{m-i,tr}(r,\delta)$ and $G_{m-i,st}(r,\delta)$', r'$G_{e-m,tr}(r,\delta)$ and $G_{e-m,st}(r,\delta)$']
    t_ = ['Positive changes of rate', 'Negative changes of rate']
    y_lims = [-0.075, 0.11] if ind == 8 else [-0.045, 0.04]
    plot_diff_windows_tr_st_phd(f_vec, pc_m_i, pc_e_m, leg_2, legends, cols_, color_w, t_, title, path_save, True,
                                y_lims_ind_plot=y_lims, y_lbl=y_label, ls=ls)

    # Figure PhD thesis (methodology / Frequency responses of Entropy for each window)
    title = "Frequency responses of Entropy (short-term "
    title += "facilitation)" if ind == 8 else "depression)"
    title += r", $\delta = %.1f$" % gain
    t_ = ['ini-window', 'mid-window', 'end-window']
    # cols_ = ['tab:red', 'tab:green', 'tab:blue']
    legends = r'$H_{%s,%s}$'
    lbl_ = ['i', 'm', 'e']
    y_label = 'Entropy (bits)'
    path_save = folder_plots + dr_gain_control_file
    path_save += '_freq_response_3w_H_facilitation_phd.png' if ind == 8 else '_freq_response_3w_H_depression_phd.png'
    y_lims = [2.9, 10.3] if ind == 8 else [1.25, 10.]  # y_lims = [-0.005, 0.14] if ind == 8 else [-0.005, 0.08]
    plot_features_H_3windows_phd(f_vec, dr_gain, legends, lbl_, color_win, t_, title, path_save, True,
                                 y_lims_ind_plot=y_lims, y_lbl=y_label)

    # Figure PhD thesis (methodology / Frequency responses of pos and neg changes of rate (Entropy)
    H_i_st, H_m_st, H_e_st = dr['H_v_neu_st'][0, :], dr['H_v_neu_st'][1, :], dr['H_v_neu_st'][2, :]
    H_i_tr, H_m_tr, H_e_tr = dr['H_v_neu_tr'][0, :], dr['H_v_neu_tr'][1, :], dr['H_v_neu_tr'][2, :]
    GH_mi_tr = H_m_tr - H_i_st
    GH_em_tr = H_e_tr - H_m_st
    title = "Frequency responses of Entropy (short-term "
    title += "facilitation)" if ind == 8 else "depression)"
    title += r", $\delta = %.1f$" % gain
    y_label = [r'$G_{pos}^\mathrm{H}$ (bits)', r'$G_{neg}^\mathrm{H}$ (bits)']
    path_save = folder_plots + dr_gain_control_file
    path_save += '_freq_response_pc_H_facilitation_phd.png' if ind == 8 else '_freq_response_pc_H_depression_phd.png'
    legends = [r'$PC_{pos}^\mathrm{H}$', r'$PC_{neg}^\mathrm{H}$']
    leg_2 = ['pos', 'neg']
    color_w = [color_stat[1], color_stat[2]]
    ls = ['solid', 'dashed']  # , '-', '-', '-']
    t_ = ['Positive changes of rate', 'Negative changes of rate']
    y_lims = [-1.3, 2.] if ind == 8 else [-1.1, 2.08]
    plot_diff_windows_tr_st_H_phd(f_vec, GH_mi_tr, GH_em_tr, leg_2, legends, cols_, color_w, t_, title, path_save, True,
                                y_lims_ind_plot=y_lims, y_lbl=y_label, ls=ls)
# """
