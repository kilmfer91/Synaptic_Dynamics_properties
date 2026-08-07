from gain_control.utils_gc import *

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