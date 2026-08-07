from gain_control.utils_gc import *


def run_single_systems(s_model, n_model, ind, ext_lbl, ext_col=None, ax_global_freq_por=None, plot_freq_res=False):
    # s_model = 'TM'
    # n_model = 'LIF'
    # ind = 8
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
    ext_label = ''

    # Normalization
    norm_neuron = True  # True
    min_n, max_n = None, None
    if n_model == "HH":
        # norm_neuron = False
        min_n, max_n = -0.05, 0.0
    if n_model == "LIF":
        # norm_neuron = False
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
    # title_freqres = ['H - filtering', 'H - Gain-control', 'Transitory time', 'Synaptic Filtering', 'GC - amp',
    #                  'GC - var', 'GC - med']
    title_freqres = ['Temp. filtering', 'Transients', 'Entropy (stationary)', 'Entropy (transitory)',
                     'Gain effect (amp)', 'Gain effect (med)', 'Gain effect (Entropy)']
    # ylabel_axb = ["Entropy (bits)", "Entropy (bits)", "Time (s)", "Mem. pot. (mV)", "Mem. pot. (mV)",
    #               "Mem. pot. (mV)", "Mem. pot. (mV)"]
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
    alpha = 0.3
    markers = ['+', '*']
    alphas = [1.0, 0.5]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    if plot_figs:
        plt.rcParams['figure.constrained_layout.use'] = True
        # Synaptic filtering vs. Gain-Control for Neuron
        dr_gain_control_file = get_name_file(sfreq, s_model, n_model, ind, num_syn, tau_lif, True, 0.1)
        if os.path.isfile(path_vars + dr_gain_control_file) and not filt_dict_loaded:
            # Name state variables
            dr_filt = loadObject(dr_gain_control_file, path_vars)

            if ax_global_freq_por is None:
                name_n_state_variables = dr_filt['name_neuron_state_variables']
                name_syn_state_variables = dr_filt['name_syn_state_variables']

            # Frequency portrait - Neuron
            if ax_global_freq_por is None:
                title_ = 'Frequency portrait for Neuron - %s(t)'
                n_freq_por, ax_p = create_fig_freq_portrait(name_n_state_variables, title_)
                # n_freq_por, ax_p = create_fig_freq_portrait(['v'], title_)
            else:
                n_freq_por, ax_p = ax_global_freq_por
                ext_label = ext_lbl
                colors = ext_col if ext_col is not None else colors
                name_n_state_variables = ['v']

            # Frequency portrait - Synapse
            # title_ = 'Frequency portrait for Synapse - %s(t)'
            # s_freq_por, ax_sp = create_fig_freq_portrait(name_syn_state_variables, title_)

            if plot_freq_res:
                # Frequency responses - neuron
                title_ = 'Frequency responses for neuron - %s(t)'
                n_freq_res, ax_f = create_fig_freq_responses(name_n_state_variables, title_)

                # Frequency responses - synapse
                title_ = 'Frequency responses for synapse - %s(t)'
                s_freq_res, ax_fs = create_fig_freq_responses(name_syn_state_variables, title_)

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
        dr_syn_filtering_file = get_name_file(sfreq, s_model, n_model, ind, num_syn, tau_lif, False, gain)
        dr_gain_control_file = get_name_file(sfreq, s_model, n_model, ind, num_syn, tau_lif, True, gain)

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
                                norm_neuron, title_mp, markers, alphas, c_g=c_g[i_g], plot_filt=i_g == 0, ode='n')
            # For synapses
            # plot_freq_responses(name_syn_state_variables, dr_filt, dr_gain, dr_['time_transition'], gain, ax_fs,
            #                     norm_neuron, title_mp, markers, alphas, c_g=c_g[i_g], plot_filt=i_g == 0, ode='s')

        if plot_figs:
            # FREQUENCY PORTRAITS OF NEURONS AND SYNAPSES
            # For neurons
            plot_freq_portrait2(name_n_state_variables, dr_filt, dr_gain, gain, ax_p, norm_neuron, title_mp,
                                colors[i_g], ode='n', ext_label=ext_label)  # , H_filt, H_gain)

            # For synapses
            # plot_freq_portrait2(name_syn_state_variables, dr_filt, dr_gain, gain, ax_sp, norm_neuron, title_mp,
            #                     colors[i_g], ode='s')  # , H_filt, H_gain)
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

            if plot_freq_res:
                for j in range(len(title_freqres)):
                    # Frequency responses for ini window
                    adjust_freq_portraits(ax_f[n][j], "Rate (Hz)", ylabel_axb[j], title_freqres[j], xscale='log',
                                          axes_=False, x_axis=False)
                    # Frequency responses for mid window
                    adjust_freq_portraits(ax_f[n][j + 7], "Rate (Hz)", ylabel_axb[j], title_freqres[j],
                                          xscale='log', axes_=False, x_axis=False, tit_=False)
                    # Frequency responses for end window
                    adjust_freq_portraits(ax_f[n][j + 14], "Rate (Hz)", ylabel_axb[j], title_freqres[j],
                                          xscale='log', axes_=False, tit_=False)
        """
        for n in range(len(name_syn_state_variables)):
            for j in range(len(title_mp)):
                # Frequency portrait for Synapses
                adjust_freq_portraits(ax_sp[n][j], x_label_ax_p[j], y_label_ax_p[j], title_mp[j])  # xl, yl

            if plot_freq_res:
                for j in range(len(title_freqres)):
                    # Frequency responses for ini window
                    adjust_freq_portraits(ax_fs[n][j], "Rate (Hz)", ylabel_axb[j], title_freqres[j],
                                          xscale='log', axes_=False, x_axis=False)
                    # Frequency responses for mid window
                    adjust_freq_portraits(ax_fs[n][j + 7], "Rate (Hz)", ylabel_axb[j], title_freqres[j],
                                          xscale='log', axes_=False, x_axis=False, tit_=False)
                    # Frequency responses for end window
                    adjust_freq_portraits(ax_fs[n][j + 14], "Rate (Hz)", ylabel_axb[j], title_freqres[j],
                                          xscale='log', axes_=False, tit_=False)
        # """

        # Legends
        # Frequency portraits
        for n in range(len(name_n_state_variables)):
            ax_p[n][int(len(title_mp) / 2) - 1].legend(bbox_to_anchor=(1.05, 0.7), loc='upper left', borderaxespad=0.,
                                                    title='gain factor')
        # for n in range(len(name_syn_state_variables)):
        #     ax_sp[n][int(len(title_mp) / 2) - 1].legend(bbox_to_anchor=(1.05, 0.7), loc='upper left', borderaxespad=0.,
        #                                                 title='gain factor')

        # Frequency responses
        if plot_freq_res:
            lbl_ind = []
            if 0.1 in gain_v: lbl_ind.append([int(len(title_mp) / 2), len(title_freqres)])
            if 0.5 in gain_v: lbl_ind.append([7 + int(len(title_mp) / 2), 7 + len(title_freqres)])
            if 1.0 in gain_v: lbl_ind.append([14 + int(len(title_mp) / 2), 14 + len(title_freqres)])

            for n in range(len(name_n_state_variables)):
                adjust_legend_freq_res(lbl_ind, n_freq_res[n], ax_f[n], gain_v)

            # for n in range(len(name_syn_state_variables)):
            #     adjust_legend_freq_res(lbl_ind, s_freq_res[n], ax_fs[n], gain_v)

    # if plot_figs and save_figs:
    #     for j in range(len(name_n_state_variables)):
    #         n = name_n_state_variables[j]
    #         n_freq_por[j].savefig(path_save + "_freq_portrait_neuron_" + n + "_pos" + aux_p + ".png", format='png')
    #         if plot_freq_res:
    #             n_freq_res[j].savefig(path_save + "_freq_responses_neuron_" + n + aux_p + ".png", format='png')

        """
        for j in range(len(name_syn_state_variables)):
            n = name_syn_state_variables[j]
            s_freq_por[j].savefig(path_save + "_freq_portrait_synapse_" + n + "_pos" + aux_p + ".png", format='png')
            if plot_freq_res:
                s_freq_res[j].savefig(path_save + "_freq_responses_synapse_" + n + aux_p + ".png", format='png')
        # """
    # """


SYSTEMS = {
    1: ["TM", "LIF", 4, 'TM/LIF(4)'],
    2: ["TM", "LIF", 8, 'TM/LIF(8)'],
    # 3: ["MSSM", "LIF", 4, 'MSSM/LIF(4)'],
    # 4: ["MSSM", "LIF", 7, 'MSSM/LIF(7)'],
    3: ["DoornSTD", "HH", 0, 'Doorn/HH(0) healthy'],
    4: ["DoornSTD", "HH", 1, 'Doorn/HH(1) Dravet'],
    # 7: ["DoornSTF", "HH", 7],
}

# Optionally, define colors / styles per system
colors = plt.cm.tab10(range(len(SYSTEMS) * 2))

global_fre_por, ax_global_freq_por = create_fig_freq_portrait(['v'],
                                                              "Frequency portrait all systems %s(t)",
                                                              figsize=(20, 8))

for i, (sys_id, (s_model, n_model, ind, lbl)) in enumerate(SYSTEMS.items()):
    # Run your existing pipeline for this system, but only compute what you need
    # You may want to modify run_single_system to return the portrait data
    # instead of plotting internally, or to accept an external axis to plot on.
    # Here I assume you adapt it to plot on ax_fp directly when requested.

    run_single_systems(
        s_model=s_model,
        n_model=n_model,
        ind=ind,
        ext_lbl=lbl,
        ext_col=[colors[i]],
        ax_global_freq_por=[global_fre_por, ax_global_freq_por],      # don’t create a new figure inside
        plot_freq_res=False,
    )

    # Inside run_single_system, you would now compute the portrait data
    # and then plot onto ax_fp, e.g.:
    # ax_fp.plot(freq, gain, label=f"{sys_id}: {s_model}-{n_model}-ind{ind}",
    #            color=colors[i])
global_fre_por[0].tight_layout()


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