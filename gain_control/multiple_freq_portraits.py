from gain_control.utils_gc import *


def run_single_systems(s_model, n_model, ind_sys, sys_description, factor=1.0, ext_col=None, ax_global_freq_por=None,
                       plot_freq_res=False, legend_on=False):
    # Flags for plotting
    save_figs = False
    plot_figs = True
    plot_phd_meth = False
    plot_freq_res = False
    freq_res_T = True
    freq_port_T = True
    freq_res_single = False
    only_mem_pot = True
    num_single = 8  # 6

    # Returns
    handles, labels = [], []

    # Auxiliar frecuencies to plot in the transient dynamics responses
    f_auxs = ([[None, None, None], [None, None, None], [None, None, None]] for _ in range(num_single + 1))
    extra_f = True
    plt_iniw = True

    if ind_sys == 3:
        f_auxs = np.array(([[None, None, None], [None, None, None], [None, None, None]],  # transients amplitude
                           [[None, None, 200], [None, 110, 200], [None, 90, 200]],  # transients median
                           [[None, None, None], [None, None, None], [None, None, None]],  # filtering amplitude
                           [[None, None, 200], [None, 110, 200], [None, 90, 200]],  # filtering median
                           [[35, None, None], [35, None, None], [35, None, None]],  # entropy transitory
                           [[None, None, None], [None, None, None], [None, None, None]],  # Entropy stationary
                           [[None, None, None], [None, None, None], [None, None, None]],  # Amplitude gain effect
                           [[None, None, None], [None, None, None], [None, None, None]],  # Median gain effect
                           [[None, None, None], [None, None, None], [None, None, None]])  # Entropy gain effect
                          )
    if ind_sys == 4:
        f_auxs = np.array(([[None, 'max', 'max'], [None, 'max', 'max'], [None, 'max', 'max']],  # transients amplitude
                           [[None, None, None], [None, None, None], [None, None, None]],  # transients median
                           [[110, 'max', None], [110, 'max', None], [110, 'max', None]],  # filtering amplitude
                           [[None, None, None], [None, None, None], [None, None, None]],  # filtering median
                           [[360, None, None], [360, None, None], [360, None, None]],  # entropy transitory
                           [[None, None, None], [None, None, None], [None, None, None]],  # Entropy stationary
                           [[None, None, None], [None, None, None], [None, None, None]],  # Amplitude gain effect
                           [['max', None, None], ['max', None, None], ['max', None, None]],  # Median gain effect
                           [[120, None, None], [100, None, None], [90, None, None]])  # Entropy gain effect
                          )
    if ind_sys == 5:
        f_auxs = np.array(([[None, 'max', 'max'], [None, 'max', 'max'], [None, 'max', 'max']],  # transients amplitude
                           [[None, None, None], [None, None, None], [None, None, None]],  # transients median
                           [['max', 'max', None], ['max', 'max', None], [70, 'max', None]],  # filtering amplitude
                           [[None, None, None], [None, None, None], [None, None, None]],  # filtering median
                           [[None, None, None], [None, None, None], [None, None, None]],  # entropy transitory
                           [[None, None, None], [None, None, None], [None, None, None]],  # Entropy stationary
                           [[None, None, None], [None, None, None], [None, None, None]],  # Amplitude gain effect
                           [[None, None, None], [None, None, None], [None, None, None]],  # Median gain effect
                           [[None, None, None], [None, None, None], [None, None, None]])  # Entropy gain effect
                          )
    if ind_sys in [1, 2, 6, 7, 8]:
        f_auxs = np.array(([[None, None, None], [None, None, None], [None, None, None]],  # transients amplitude
                           [[None, None, None], [None, None, None], [None, None, None]],  # transients median
                           [[None, None, None], [None, None, None], [None, None, None]],  # filtering amplitude
                           [[None, None, None], [None, None, None], [None, None, None]],  # filtering median
                           [[None, None, None], [None, None, None], [None, None, None]],  # entropy transitory
                           [[None, None, None], [None, None, None], [None, None, None]],  # Entropy stationary
                           [[None, None, None], [None, None, None], [None, None, None]],  # Amplitude gain effect
                           [[None, None, None], [None, None, None], [None, None, None]],  # Median gain effect
                           [[None, None, None], [None, None, None], [None, None, None]])  # Entropy gain effect
                          )
    if ind_sys == 5: f_transient, f_temp_filt = 40, 70
    # if ind_sys == 6: f_transient, f_temp_filt = 100, 70
    f_temp_info = None if ind_sys != 4 else 360
    # Sampling frequency and conditions for running parallel or single LIF neurons
    sfreq = 10e3
    tau_lif = 30  # ms

    # Fontsizes
    fs_ax_portrait = 13
    fs_ttl_portrait = 15

    # Path variables
    aux_p = ''  # '_2'
    path_vars = "../gain_control/variables/high_freq_10k" + aux_p + "/"
    check_create_folder(path_vars)
    folder_plots = '../gain_control/plots/freq_portrait/'
    check_create_folder(folder_plots)
    ext_label = ''

    # Units
    u_v, u_mv = 'mV', 'mV'  # r'$\mu$V'
    factor_v = 1e0
    if "Doorn" in s_model: u_v, u_mv, factor_v = 'mV', 'mV', 1e3
    units_v = (u_v, u_mv)

    # **********************************************************************************************************************
    # MULTIPLE GAINS
    # **********************************************************************************************************************
    gain_v = [1.0]  # [0.1, 0.5, 1.0]
    ind_gain = {0.1: 0, 0.5: 1, 1.0: 2}
    filt_dict_loaded = False

    # Titles graphs
    title = "Model " + s_model + ', ind ' + str(ind)
    if n_model == 'LIF': title += r', $\tau_{lif}$ ' + str(tau_lif) + "ms"
    if len(gain_v) == 1: title += ', gain ' + str(int(gain_v[0] * 100)) + '%'
    else: title += ', multiple gains'

    # Plot
    # 2x2
    title_mp = ['Filtering vs gain effect (amp)', 'Filtering vs gain effect (med)',
                'Information vs gain effect (Entropy)', 'Transients vs filtering (amp)']
    x_label_ax_p = [r'$E_{ff_{st}}^{amp}$ (' + u_v + ')', r'$E_{ff_{st}}^{med}$ (' + u_v + ')',
                    r'$H_{st}$ (bits)', r' $E_{ff_{tr}}^{med}$ (' + u_v + ')']
    y_label_ax_p = [r'$G^{amp}$ (' + u_v + ')', r'$G^{med}$ (' + u_v + ')',
                    r'$PC^{H}$ (bits)', r'$E_{ff_{st}}^{amp}$ - $E_{ff_{tr}}^{amp}$ (' + u_v + ')']

    title_freqres = ['Transient dynamics', 'Transient dynamics -med-', 'Temporal filtering', 'Synaptic efficacy -med-',
                     'Entropy',
                     'Gain effect -amp-', 'Gain effect -med-', 'Gain effect -Entropy-']
    title_freq_save_fig = ['_transients', '_transients_med', '_filtering', '_eff_med', '_information', '_gain_amp',
                           '_gain_med', '_gain_entropy']
    ylabel_axb = ["Mem. pot. (mV)", "Mem. pot. (mV)", "Entropy (bits)", "Mem. pot. (mV)",
                  "Mem. pot. (mV)", "Entropy (bits)"]
    title_freqres_sing = ''
    if freq_res_single:
        s_d = sys_description
        tit_aux = [s_d + ', %s' % i + r', $%s(t)$' for i in title_freqres]
        title_freqres_sing = tit_aux
        title_freqres = [[r'$\delta$ = ' + str(int(g * 100)) + '%' for g in gain_v] for _ in
                         range(len(title_freqres_sing))]

    if freq_res_T and not freq_res_single: title_freqres = [r'$\delta$ = 10%',
                                                            r'Transient dynamics ' + os.linesep + ' $\delta$ = 50%',
                                                            r'$\delta$ = 100%',
                                                            '', 'Transient dynamics (med)', '',
                                                            '', 'Temporal filtering', '',
                                                            '', 'Synaptic efficacy (med)', '',
                                                            '', 'Entropy', '',
                                                            '', 'Gain effect (amp)', '',
                                                            '', 'Gain effect (med)', '',
                                                            '', 'Gain effect (Entropy)', '']

    ylabel_freqRes = ["Mem. pot. (" + u_v + ")", "Mem. pot. (" + u_v + ")", "Mem. pot. (" + u_v + ")",
                      "Mem. pot. (" + u_v + ")", "Entropy (bits)", "Mem. pot. (" + u_v + ")", "Mem. pot. (" + u_v + ")",
                      "Entropy (bits)"]
    # For state variables of neuron, only membrane potential
    min_max_mid_win = [True, False, False, False, False, False]
    if freq_res_T: ylabel_freqRes = ['Mem. pot. (' + u_v + ')', '', '',
                                     'Mem. pot. (' + u_v + ')', '', '',
                                     'Mem. pot. (' + u_v + ')', '', '',
                                     'Mem. pot. (' + u_v + ')', '', '',
                                     'Entropy (bits)', '', '',
                                     'Mem. pot. (' + u_v + ')', '', '',
                                     'Mem. pot. (' + u_v + ')', '', '',
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

            if ax_global_freq_por is None:
                name_n_state_variables = dr_filt['name_neuron_state_variables']
                name_syn_state_variables = dr_filt['name_syn_state_variables']

            # To reduce creation of graphics, debbuging
            if only_mem_pot: name_n_state_variables, name_syn_state_variables = ['v'], []

            # Frequency portrait - Neuron
            if ax_global_freq_por is None:
                title_ = sys_description + '. Frequency portrait for Neuron - %s(t)'
                n_freq_por, ax_p = create_fig_freq_portrait3(name_n_state_variables, title_, freq_port_T)
                # n_freq_por, ax_p = create_fig_freq_portrait(['v'], title_)
            else:
                n_freq_por, ax_p = ax_global_freq_por
                ext_label = sys_description
                colors = ext_col if ext_col is not None else colors
                name_n_state_variables = ['v']

            # Frequency portrait - Synapse
            # title_ = sys_description + '. Frequency portrait for Synapse - %s(t)'
            # s_freq_por, ax_sp = create_fig_freq_portrait(name_syn_state_variables, title_)

            if plot_freq_res:
                # Frequency responses - neuron
                title_ = ""
                if freq_res_single:
                    title_ = title_freqres_sing
                else:
                    title_ = sys_description + '. Frequency responses for neuron - %s(t)'
                n_freq_res, ax_f = create_fig_freq_responses(name_n_state_variables, title_, freq_res_T,
                                                             freq_res_single,
                                                             num_single=num_single)

                # Frequency responses - synapse
                if freq_res_single:
                    title_ = title_freqres_sing
                else:
                    title_ = sys_description + '. Frequency responses for synapse - %s(t)'
                s_freq_res, ax_fs = create_fig_freq_responses(name_syn_state_variables, title_, freq_res_T,
                                                              freq_res_single,
                                                              num_single=num_single)

    fig_syn_b = False
    fig_H_100 = False

    # ******************************************************************************************************************
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

        # **************************************************************************************************************
        # Trying to load freq. response of Gain Control
        if os.path.isfile(path_vars + dr_syn_filtering_file) and not filt_dict_loaded:
            dr_filt = loadObject(dr_syn_filtering_file, path_vars)
            # Auxiliar variables
            initial_frequencies, model = dr_filt['initial_frequencies'], dr_filt['stp_model']
            total_realizations = dr_filt['t_realizations']

            # Name state variables
            if ax_global_freq_por is None:
                name_n_state_variables = dr_filt['name_neuron_state_variables']
                name_syn_state_variables = dr_filt['name_syn_state_variables']

        if os.path.isfile(path_vars + dr_gain_control_file):
            dr_gain = loadObject(dr_gain_control_file, path_vars)

        f_vec = dr_gain['initial_frequencies']
        f_vecD = dr_filt['initial_frequencies']

        # **************************************************************************************************************
        # Plots 1
        dr_ = dr_gain
        if plot_figs and plot_freq_res:
            # FREQUENCY RESPONSES OF NEURONS AND SYNAPSES
            # For Neurons
            plot_freq_responses(name_n_state_variables, dr_filt, dr_gain, dr_['time_transition'], gain, ax_f,
                                title_mp, markers, alphas, c_g=c_g[i_g], factor_v=factor_v, units_v=units_v,
                                plot_filt=i_g == 0, ode='n', transpose=freq_res_T, min_max_mid_win=min_max_mid_win,
                                extra_f=extra_f, plt_iniw=plt_iniw, f_temp_info=f_temp_info,
                                f_auxs=f_auxs[:, ind_gain[gain], :])
            # For synapses
            # plot_freq_responses(name_syn_state_variables, dr_filt, dr_gain, dr_['time_transition'], gain, ax_fs,
            #                     title_mp, markers, alphas, c_g=c_g[i_g], factor_v=factor_v, units_v=units_v,
            #                     plot_filt=i_g == 0, ode='s', transpose=freq_res_T, single_properties=freq_res_single)

        if plot_figs:
            # FREQUENCY PORTRAITS OF NEURONS AND SYNAPSES
            # For neurons
            plot_freq_portrait3(name_n_state_variables, dr_filt, dr_gain, gain, ax_p, title_mp, colors[i_g],
                                ode='n', freq_port_T=freq_port_T, factor=factor_v, plt_transient=i_g == 0,
                                ext_col_transients=colors[i_g], ext_lbl_transients=sys_description)

            # For synapses
            # plot_freq_portrait2(name_syn_state_variables, dr_filt, dr_gain, gain, ax_sp, title_mp,
            #                     colors[ind_gain[gain]], ode='s', freq_port_T=freq_port_T)  # , H_filt, H_gain)
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
                # ax_y = True if k != 4 else False
                # Frequency portrait for Neuron
                adjust_freq_portraits(ax_p[n][k], x_label_ax_p[k], y_label_ax_p[k], title_mp[k], ax_x=False,
                                      axis_fontsize=fs_ax_portrait, title_fontsize=fs_ttl_portrait)

            if plot_freq_res:
                # Adjusting frequency responses for neurons
                adjust_freq_responses(ax_f[n], title_freqres, freq_res_T, freq_res_single, gain_v, ylabel_freqRes)
        """
        for n in range(len(name_syn_state_variables)):
            for k in range(len(title_mp)):
                ax_x = True if "Entropy" not in title_mp[k] else False
                # Frequency portrait for Synapses
                adjust_freq_portraits(ax_sp[n][k], x_label_ax_p[k], y_label_ax_p[k], title_mp[k], ax_x=False,
                                      axis_fontsize=fs_ax_portrait, title_fontsize=fs_ttl_portrait)  # xl, yl
    
            if plot_freq_res:
                # Adjusting frequency responses for synapses
                adjust_freq_responses(ax_fs[n], title_freqres, freq_res_T, freq_res_single, gain_v, ylabel_freqRes)
        # """

        # Legends
        # Frequency portraits
        # """
        # if legend_on:
        for n in range(len(name_n_state_variables)):
            handles, labels = [], []
            h_, l_ = ax_p[n][2].get_legend_handles_labels()
            for h_i in h_: handles.append(h_i)
            for l_i in l_: labels.append(l_i)
            # h_, l_ = ax_p[n][int(len(title_mp) / 2)].get_legend_handles_labels()
            # h_, l_ = ax_p[n][3].get_legend_handles_labels()
            # for h_i in h_: handles.append(h_i)
            # for l_i in l_: labels.append(l_i)
            # n_freq_por[n].legend(handles, labels, loc='outside lower center', ncol=5, frameon=True,
            #                      title='gain factor')
        # """

        # for n in range(len(name_syn_state_variables)):
        #     ax_sp[n][int(len(title_mp) / 2) - 1].legend(bbox_to_anchor=(1.05, 0.7), loc='upper left',borderaxespad=0.,
        #                                                 title='gain factor')

        # Frequency responses
        if plot_freq_res:
            if not freq_res_single:
                lbl_ind = []

                if freq_res_T:
                    if num_single == 6: lbl_ind = [7, 18]
                    if num_single == 7: lbl_ind = [10, 21]
                    if num_single == 8: lbl_ind = [13, 24]

                l_ = len(title_freqres)
                if 0.1 in gain_v and not freq_res_T: lbl_ind.append([int(len(title_mp) / 2) - 1, l_])
                if 0.5 in gain_v and not freq_res_T: lbl_ind.append([6 + int(len(title_mp) / 2) - 1, 6 + l_])
                if 1.0 in gain_v and not freq_res_T: lbl_ind.append([12 + int(len(title_mp) / 2) - 1, 12 + l_])

                # For state variables of neurons
                for n in range(len(name_n_state_variables)):
                    if freq_res_T:
                        adjust_legend_freq_resT(lbl_ind, n_freq_res[n], ax_f[n], gain_v)
                    else:
                        adjust_legend_freq_res(lbl_ind, n_freq_res[n], ax_f[n], gain_v)
                # For state variables of synapses
                for n in range(len(name_syn_state_variables)):
                    if freq_res_T:
                        adjust_legend_freq_resT(lbl_ind, s_freq_res[n], ax_fs[n], gain_v)
                    else:
                        adjust_legend_freq_res(lbl_ind, s_freq_res[n], ax_fs[n], gain_v)

            # for n in range(len(name_syn_state_variables)):
            #     adjust_legend_freq_res(lbl_ind, s_freq_res[n], ax_fs[n], gain_v)

    """
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

    return handles, labels


# **********************************************************************************************************************
# "Doorn", "MSSM", "TM", "MSSM/TM"
s_model = "Doorn"
title = ""
SYSTEMS = {
    1: ["TM", "LIF", 4, 'TM STD, ', 1e-3],
    2: ["TM", "LIF", 8, 'TM STF, ', 1e-3],
    3: ["MSSM", "LIF", 4, 'MSSM STD, ', 1.0],
    4: ["MSSM", "LIF", 7, 'MSSM STF, ', 1.0],
    5: ["DoornSTD", "HH", 0, 'Doorn Control, ', 1.0],
    6: ["DoornSTD", "HH", 1, 'Doorn Superbursts, ', 1.0],
    # 7: ["DoornSTF", "HH", 7, 'DoornSTF(7) Dravet, ', 1.0],
    # 8: ["DoornSTD", "HH", 8, 'DoornSTD(8) Dravet, ', 1.0],
}
systems = SYSTEMS.copy()

# Name of state variables
name_n_state_variables = ['v']

# Optionally, define colors / styles per system
colors = plt.cm.tab10(range(len(SYSTEMS) * 2))

if s_model == "Doorn":
    systems = {5: SYSTEMS[5], 6: SYSTEMS[6]}
    title = "Frequency portraits for Doorn models - %s(t)"
elif s_model == "MSSM":
    systems = {4: SYSTEMS[4], 3: SYSTEMS[3]}
    title = "Frequency portraits for Prototype models with MSSM - %s(t)"
elif s_model == "TM":
    systems = {1: SYSTEMS[1], 2: SYSTEMS[2]}
    title = "Frequency portraits for Prototype models with TM model - %s(t)"
elif s_model == "MSSM/TM":
    systems = {1: SYSTEMS[1], 2: SYSTEMS[2], 3: SYSTEMS[3], 4: SYSTEMS[4]}
    title = "Frequency portraits for Prototype models MSSM and TM - %s(t)"

# **********************************************************************************************************************
# Creating figures
global_fre_por, ax_global_freq_por = create_fig_freq_portrait3(name_n_state_variables, title, freq_port_T=True)
handles, labels = [], []


# **********************************************************************************************************************
# Running loop
for i, (sys_id, (s_model, n_model, ind, lbl, factor)) in enumerate(systems.items()):
    # Run your existing pipeline for this system, but only compute what you need
    # You may want to modify run_single_system to return the portrait data
    # instead of plotting internally, or to accept an external axis to plot on.
    # Here I assume you adapt it to plot on ax_fp directly when requested.

    h_, l_ = run_single_systems(s_model=s_model,
                                n_model=n_model,
                                ind_sys=ind,
                                sys_description=lbl,
                                factor=factor,
                                ext_col=[colors[i]],
                                ax_global_freq_por=[global_fre_por, ax_global_freq_por],
                                plot_freq_res=False,
                                legend_on=i == 0,
                                )
    handles.append(h_)
    labels.append(l_)

# Legend
h_, l_ = handles[-1], labels[-1]
handles_, labels_ = [], []
j = 0
for i, (sys_id, (s_model, n_model, ind, lbl, factor)) in enumerate(systems.items()):
    # labels_.append(l_[j][:6] + lbl[:-2])
    labels_.append(l_[j] + "|" + lbl.split(" ")[1][:-1])
    handles_.append(h_[j])
    j += 1
    # labels_.append(l_[j][:6] + lbl[:-2])
    labels_.append(l_[j] + "|" + lbl.split(" ")[1][:-1])
    handles_.append(h_[j])
    j += 1

folder_plots = '../gain_control/plots/freq_portrait/'
path_save = (folder_plots + s_model)

for fig in global_fre_por:
    fig.legend(handles_, labels_, loc='outside lower center', ncol=4, frameon=True)
    fig.tight_layout()

    fig.savefig(path_save + "_multiple_freq_portrait.png", format='png')

"""
# =============================================================================
# SYSTEMS DICTIONARY - Define all systems to compare
# =============================================================================
systems = {
    1: ['TM', 'LIF', 4],
    2: ['TM', 'LIF', 8],
    3: ['MSSM', 'LIF', 4],
    4: ['MSSM', 'LIF', 7],
    5: ['DoornSTD', 'HH', 0],
    6: ['DoornSTD', 'HH', 1],
    7: ['DoornSTF', 'HH', 7]
}

# Color scheme for different systems
system_colors = {
    1: 'tab:blue',
    2: 'tab:orange',
    3: 'tab:green',
    4: 'tab:red',
    5: 'tab:purple',
    6: 'tab:brown',
    7: 'tab:pink'
}

# System labels for legend
system_labels = {
    1: 'TM+LIF (STD)',
    2: 'TM+LIF (STF)',
    3: 'MSSM+LIF (STD)',
    4: 'MSSM+LIF (STF)',
    5: 'DoornSTD+HH (Ctrl)',
    6: 'DoornSTD+HH (Dravet)',
    7: 'DoornSTF+HH (STF)'
}

save_figs = True
plot_figs = True
num_syn = 1

# Sampling frequency and conditions
sfreq = 10e3
tau_lif = 30  # ms

# Path variables
aux_p = ''  # '_multi_system'  # Changed to indicate multi-system plot
path_vars = "../gain_control/variables/high_freq_10k" + aux_p + "/"
check_create_folder(path_vars)
folder_plots = '../gain_control/plots/freq_portrait/'
check_create_folder(folder_plots)

# Normalization (will be set per system)
norm_neuron = False
min_n, max_n = None, None

# **********************************************************************************************************************
# MULTIPLE GAINS
# **********************************************************************************************************************
gain_v = [0.5]
filt_dict_loaded = False

# Titles graphs
title_mp = ['Amplitude in steady-state', 'Varibility in steady-state', 'Median in steady-state',
            'Entropy in steady-state', 'Amplitude in transitory-state', 'Varibility in transitory-state',
            'Median in transitory-state', 'Entropy in transitory-state']
x_label_ax_p = [r'$E_{ff_{i,st}}^{amp}$ (mV)', r'$E_{ff_{i,st}}^{var}$ (mV)', r'$E_{ff_{i,st}}^{med}$ (mV)',
                r'$H_{i,st}$ (bits)', r'$E_{ff_{i,st}}^{amp}$ (mV)', r'$E_{ff_{i,st}}^{var}$ (mV)',
                r'$E_{ff_{i,st}}^{med}$ (mV)', r'$H_{i,st}$ (bits)']
y_label_ax_p = [r'$G_{m-i,st}^{amp} (mV)$', r'$G_{m-i,st}^{var} (mV)$', r'$G_{m-i,st}^{med} (mV)',
                r'$GH_{m-i,st}$ (bits)', r'$G_{m-i,tr}^{amp} (mV)$', r'$G_{m-i,tr}^{var} (mV)',
                r'$G_{m-i,tr}^{med} (mV)', r'$GH_{m-i,tr}$ (bits)']
title_freqres = ['Temp. filtering', 'Transients', 'Entropy (stationary)', 'Entropy (transitory)',
                 'Gain effect (amp)', 'Gain effect (med)', 'Gain effect (Entropy)']
ylabel_axb = ["Mem. pot. (mV)", "Mem. pot. (mV)", "Entropy (bits)", "Entropy (bits)",
              "Mem. pot. (mV)", "Mem. pot. (mV)", "Entropy (bits)"]

# **********************************************************************************************************************
# FIGURE CREATION - Frequency Portrait (COMBINED for all systems)
# **********************************************************************************************************************
name_n_state_variables, name_syn_state_variables = None, None
n_freq_por, figNeur_combined, s_freq_por, figSyn_combined = None, None, None, None
n_freq_res_dict, s_freq_res_dict = {}, {}  # Separate frequency responses per system
ax_p, ax_sp = None, None

if plot_figs:
    plt.rcParams['figure.constrained_layout.use'] = True

    # ==========================================================================
    # COMBINED FREQUENCY PORTRAIT (All systems on same figure)
    # ==========================================================================
    # Neuron - Combined Frequency Portrait
    title_ = 'Frequency Portrait - Neuron Comparison (All Systems) %s(t)'
    figNeur_combined, ax_p = create_fig_freq_portrait(['v'], title_)
    # figNeur_combined = [plt.figure(figsize=(18, 12)) for _ in range(len(['v']))]  # Just membrane potential
    # for j in range(len(['v'])):
    #     figNeur_combined[j].suptitle(title_, fontsize=18)
    #     # 2 rows (st/tr) × 2 columns (pos/neg) × 4 metrics = 16 subplots
    #     ax_p = [[figNeur_combined[0].add_subplot(2, 4, j + 1 + k * 4) for j in range(4)] for k in range(2)]
    #     ax_p = [item for sublist in ax_p for item in sublist]

    # Synapse - Combined Frequency Portrait
    title_ = 'Frequency Portrait - Synapse Comparison (All Systems) %s(t)'
    figSyn_combined, ax_sp = create_fig_freq_portrait(['epsc'], title_)
    # figSyn_combined = [plt.figure(figsize=(18, 12)) for _ in range(len(['epsc']))]
    # for j in range(len(['epsc'])):
    #     figSyn_combined[j].suptitle(title_, fontsize=18)
    #     ax_sp = [[figSyn_combined[0].add_subplot(2, 4, j + 1 + k * 4) for j in range(4)] for k in range(2)]
    #     ax_sp = [item for sublist in ax_sp for item in sublist]

    alpha = 0.6  # Slightly more transparent for overlapping trajectories
    markers = ['o']  # Use circles for all systems
    alphas = [0.7]

# **********************************************************************************************************************
# LOOP THROUGH ALL SYSTEMS
# **********************************************************************************************************************
for sys_id, (s_model, n_model, ind) in systems.items():
    print(f"\n{'=' * 80}")
    print(f"Processing System {sys_id}: {s_model} + {n_model} (ind={ind})")
    print(f"{'=' * 80}")

    # Set normalization per system
    norm_neuron = False
    min_n, max_n = None, None
    if n_model == "HH":
        norm_neuron = False
        min_n, max_n = -0.05, 0.0
    if n_model == "LIF":
        norm_neuron = False
        min_n, max_n = -70, -55

    # Title for this system
    title = "Model " + s_model + ', ind ' + str(ind)
    if n_model == 'LIF':
        title += r', $\tau_{lif}$ ' + str(tau_lif) + "ms"
    if len(gain_v) == 1:
        title += ', gain ' + str(int(gain_v[0] * 100)) + '%'

    # Create separate frequency response figures for each system
    if plot_figs:
        # Frequency responses - neuron (SEPARATE per system)
        title_ = f'Frequency responses for neuron - {s_model}+{n_model} (ind={ind}) %s'
        n_freq_res, ax_f = create_fig_freq_responses(['v'], title_)
        n_freq_res_dict[sys_id] = n_freq_res

        # Frequency responses - synapse (SEPARATE per system)
        title_ = f'Frequency responses for synapse - {s_model}+{n_model} (ind={ind}) %s'
        s_freq_res, ax_fs = create_fig_freq_responses(['epsc'], title_)
        s_freq_res_dict[sys_id] = s_freq_res

    # ******************************************************************************************************************
    # LOAD DATA FOR THIS SYSTEM
    # ******************************************************************************************************************
    filt_dict_loaded = False
    dr_filt = None
    dr_gain = None

    for gain in gain_v:
        # File names
        dr_syn_filtering_file = get_name_file(sfreq, s_model, n_model, ind, num_syn, tau_lif, False, gain)
        dr_gain_control_file = get_name_file(sfreq, s_model, n_model, ind, num_syn, tau_lif, True, gain)

        print(f"  Loading gain control: {dr_gain_control_file}")
        print(f"  Loading synaptic filtering: {dr_syn_filtering_file}")

        # Load filtering data
        if os.path.isfile(path_vars + dr_syn_filtering_file):
            dr_filt = loadObject(dr_syn_filtering_file, path_vars)
            name_n_state_variables = dr_filt['name_neuron_state_variables']
            name_syn_state_variables = dr_filt['name_syn_state_variables']
            filt_dict_loaded = True
        else:
            print(f"  WARNING: File not found: {dr_syn_filtering_file}")
            continue

        # Load gain control data
        if os.path.isfile(path_vars + dr_gain_control_file):
            dr_gain = loadObject(dr_gain_control_file, path_vars)
        else:
            print(f"  WARNING: File not found: {dr_gain_control_file}")
            continue

        f_vec = dr_gain['initial_frequencies']

        # **********************************************************************************************************
        # PLOT FREQUENCY PORTRAIT (COMBINED - All systems on same axes)
        # **********************************************************************************************************
        if plot_figs and filt_dict_loaded:
            system_color = system_colors[sys_id]
            system_label = system_labels[sys_id]

            # For Neurons - COMBINED
            plot_freq_portrait2(name_n_state_variables, dr_filt, dr_gain, gain, ax_p, norm_neuron, title_mp,
                                system_color, ode='n')  # , system_label=system_label, alpha=alpha)

            # For Synapses - COMBINED
            plot_freq_portrait2(name_syn_state_variables, dr_filt, dr_gain, gain, ax_sp, norm_neuron, title_mp,
                                system_color, ode='s')  # , system_label=system_label, alpha=alpha)

            # **********************************************************************************************************
            # PLOT FREQUENCY RESPONSES (SEPARATE - One figure per system)
            # **********************************************************************************************************
            # For Neurons
            plot_freq_responses(name_n_state_variables, dr_filt, dr_gain, dr_gain['time_transition'], gain, ax_f,
                                norm_neuron, title_mp, markers, alphas, c_g=[system_color], plot_filt=True, ode='n')

            # For Synapses
            plot_freq_responses(name_syn_state_variables, dr_filt, dr_gain, dr_gain['time_transition'], gain, ax_fs,
                                norm_neuron, title_mp, markers, alphas, c_g=[system_color], plot_filt=True, ode='s')

# **********************************************************************************************************************
# ADJUST AND SAVE FIGURES
# **********************************************************************************************************************
if plot_figs:
    sizeF = 20

    # ==========================================================================
    # COMBINED FREQUENCY PORTRAIT (All systems)
    # ==========================================================================
    for n in range(len(['v'])):  # Just membrane potential
        for j in range(len(title_mp)):
            # Neuron portrait
            adjust_freq_portraits(ax_p[n][j], x_label_ax_p[j], y_label_ax_p[j], title_mp[j])

            # Synapse portrait
            adjust_freq_portraits(ax_sp[n][j], x_label_ax_p[j], y_label_ax_p[j], title_mp[j])

    # Add combined legend for all systems
    for n in range(len(['v'])):
        # Create legend handles for all systems
        from matplotlib.lines import Line2D

        legend_handles = []
        for sys_id, color in system_colors.items():
            legend_handles.append(Line2D([0], [0], color=color, linewidth=2, label=system_labels[sys_id]))

        ax_p[n][int(len(title_mp) / 2) - 1].legend(handles=legend_handles,
                                                   bbox_to_anchor=(1.05, 0.7),
                                                   loc='upper left',
                                                   borderaxespad=0.,
                                                   title='Systems',
                                                   fontsize=10)

        ax_sp[n][int(len(title_mp) / 2) - 1].legend(handles=legend_handles,
                                                    bbox_to_anchor=(1.05, 0.7),
                                                    loc='upper left',
                                                    borderaxespad=0.,
                                                    title='Systems',
                                                    fontsize=10)

    # ==========================================================================
    # SEPARATE FREQUENCY RESPONSES (One per system)
    # ==========================================================================
    for sys_id in systems.keys():
        n = 0  # Just membrane potential
        for j in range(len(title_freqres)):
            adjust_freq_portraits(ax_f[n][j], "Rate (Hz)", ylabel_axb[j], title_freqres[j], xscale='log',
                                  axes_=False, x_axis=False)
            adjust_freq_portraits(ax_f[n][j + 7], "Rate (Hz)", ylabel_axb[j], title_freqres[j], xscale='log',
                                  axes_=False, x_axis=False, tit_=False)
            adjust_freq_portraits(ax_f[n][j + 14], "Rate (Hz)", ylabel_axb[j], title_freqres[j], xscale='log',
                                  axes_=False, tit_=False)

        # Add legend for frequency responses
        lbl_ind = []
        if 0.1 in gain_v: lbl_ind.append([int(len(title_mp) / 2), len(title_freqres)])
        if 0.5 in gain_v: lbl_ind.append([7 + int(len(title_mp) / 2), 7 + len(title_freqres)])
        if 1.0 in gain_v: lbl_ind.append([14 + int(len(title_mp) / 2), 14 + len(title_freqres)])

        adjust_legend_freq_res(lbl_ind, n_freq_res_dict[sys_id][n], ax_f, gain_v)

# **********************************************************************************************************************
# SAVE FIGURES
# **********************************************************************************************************************
if plot_figs and save_figs:
    # Combined frequency portraits (ALL SYSTEMS)
    for j in range(len(['v'])):
        n = 'v'
        figNeur_combined[j].savefig(folder_plots + "COMBINED_freq_portrait_neuron_" + n + "_pos" + aux_p + ".png",
                                    format='png', dpi=300, bbox_inches='tight')
        figSyn_combined[j].savefig(folder_plots + "COMBINED_freq_portrait_synapse_" + n + "_pos" + aux_p + ".png",
                                   format='png', dpi=300, bbox_inches='tight')

    # Separate frequency responses (ONE PER SYSTEM)
    for sys_id, (s_model, n_model, ind) in systems.items():
        for j in range(len(['v'])):
            n = 'v'
            n_freq_res_dict[sys_id][j].savefig(
                folder_plots + f"SYS{sys_id}_{s_model}_ind{ind}_freq_responses_neuron_" + n + aux_p + ".png",
                format='png', dpi=300, bbox_inches='tight')
            s_freq_res_dict[sys_id][j].savefig(
                folder_plots + f"SYS{sys_id}_{s_model}_ind{ind}_freq_responses_synapse_" + n + aux_p + ".png",
                format='png', dpi=300, bbox_inches='tight')

    print(f"\n{'=' * 80}")
    print("FIGURES SAVED:")
    print(f"  - Combined Frequency Portraits: 2 files (neuron + synapse)")
    print(f"  - Separate Frequency Responses: {len(systems) * 2} files ({len(systems)} systems × 2 types)")
    print(f"{'=' * 80}\n")
# """