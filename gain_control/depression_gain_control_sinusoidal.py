from gain_control.utils_gc import *

# ******************************************************************************************************************
# Depression using the MSSM
model = 'MSSM'
# (Experiment 4) freq. response from Gain Control paper
# (Experiment 5) slow-decay frequency response
ind = 7

# For gain control, 100 inputs to a single LIF neuron
plots_net = False
dyn_synapse = True
gaincontrol_sinusoidal = False

# Hyperparameters for frequency analysis and poisson input spike
freq_analysis = True
Poisson = False
num_syn = 200

# Model parameters
val_params, description, name_params = get_params_stp(model, ind)

out_ylim_min, out_ylim_max, description_2 = -70, -50, ""
if ind == 4: out_ylim_min, out_ylim_max, description_2 = -67, -57, r'Fast-decay synapse with $freq_{st}$ of efficacy=260Hz'
if ind == 5: out_ylim_min, out_ylim_max, description_2 = -70, -50, r'Slow-decay synapse with $freq_{st}$ of efficacy=560Hz'

# time conditions
max_t, min_imp, max_imp, sfreq = 0.8, 0.0, 0.8, 5e3  # 15, 0.0, 10, 5e3
dt = 1 / sfreq
time_vector = np.arange(0, max_t, dt)
L = time_vector.shape[0]

# Parameters definition
params = dict(zip(name_params, val_params))
sim_params = {'sfreq': sfreq, 'max_t': max_t, 'L': L, 'time_vector': time_vector}

# Creating STP model
stp_model = MSSM_model(n_syn=num_syn)
if model == "TM": stp_model = TM_model(n_syn=num_syn)

stp_model.set_model_params(params)
stp_model.set_simulation_params(sim_params)

# Creating simple depression model
s_dep = Simple_Depression(n_syn=num_syn)
s_dep.set_simulation_params(sim_params)

# Frequency ranges for Frequency response of efficacy
range_f = [10, 20, 30, 40, 50, 60, 70, 80]  # [i for i in range(10, 801, 10)]
loop_frequencies = np.array(range_f)

# ******************************************************************************************************************
# LIF MODEL
lif = LIF_model(n_neu=1)  # (n_neu=100)
lif_params = get_neuron_params(tau_m=10, y_lim_ind_plot=True, num_syn=1)

lif.set_model_params(lif_params)
lif.set_simulation_params(sim_params)

# Params for sinusoidal envelope of input stimuli
mean_rates, max_oscils, fix_rates = [], [], []

if gaincontrol_sinusoidal:
    mean_rates = [[50, 10, 50], [100, 10, 100], [300, 10,  300], [500, 10,  500]]
    max_oscils = [[25, 5,  5],  [50,  5,  5],   [150, 5,   5],   [250, 5,   5]]
    fix_rates = [[10,  50, 10], [10, 100, 10],  [10,  300, 10], [10,   500, 10]]

# Results variable
res_per_reali = np.zeros((10, 3, len(mean_rates)))  # statistical descriptors, num. scenarios, num. ref rate

# Aux variables for plotting
fig, fig3, fig_esann, output_mp_esann, output_mp_low_filt_esann = None, None, None, None, None

seeds = []

if plots_net:
    # Plotting
    fig_size = (10, 5)
    if ind == 4 or ind == 2 or ind == 5: fig_size = (12, 1.6)
    fig_esann = plt.figure(figsize=fig_size)
    fig_esann.suptitle(description_2, fontsize=18)

# ******************************************************************************************************************
# SIMULATION GAIN CONTROL SINUSOIDAL INPUT (200 SYNAPSES TO ONE LIF NEURON)
ind_exp = 0
while ind_exp < len(mean_rates):  # len(mean_rates): # for ind_exp in range(len(mean_rates)):
    ini_loop_time = m_time()
    mean_rate = mean_rates[ind_exp]
    max_oscil = max_oscils[ind_exp]
    fix_rate = fix_rates[ind_exp]
    # Input
    time_vector_sin = np.arange(0, max_t, 1 / sfreq)  # 3e3

    if plots_net:
        fig3 = plt.figure(figsize=(6.5, 5))
        fig3.suptitle("Types of input")

    for i in range(len(mean_rate)):

        se = int(time.time())
        seeds.append(se)
        seeds1 = [j + se for j in range(int(L / 2))]
        seeds2 = [j + se + 2 for j in range(int(L / 2))]

        # Signals with firing rate modulation
        modulation_signal1 = mean_rate[i] + max_oscil[i] * np.sin(2 * np.pi * (1 / 10) * time_vector_sin)
        modulation_signal2 = fix_rate[i] * np.ones(L)

        # Sinusoidal modulated firing rate signal
        modulated_signal1 = oscillatory_spike_train(sfreq, modulation_signal1, num_realizations=int(num_syn / 2),
                                                    poisson=True, seeds=seeds1, correction=True)
        # Constant firing rate signal
        modulated_signal2 = simple_spike_train(sfreq, modulation_signal2[0], len(modulation_signal2),
                                               num_realizations=int(num_syn / 2), poisson=True,
                                               seeds=seeds2)

        # Organising input to correspond to the paper
        if i == 1: Input_test = np.concatenate((modulated_signal2, modulated_signal1), axis=0)
        else: Input_test = np.concatenate((modulated_signal1, modulated_signal2), axis=0)

        # Running STP model
        if dyn_synapse:
            model_stp(stp_model, lif, params, Input_test)
        else:
            static_synapse(lif, Input_test, 0.0125)

        # Filtering membrane potential, lowpass for getting the sinusoidal trend, high pass for the variance without
        # seasonality
        coff = 1
        res_per_reali[:, i, ind_exp], lp_mp, hp_mp = aux_statistics_sin(lif.membrane_potential[0, :], coff, sfreq)

        if i == 0:
            output_mp_esann = np.copy(lif.membrane_potential[0, :])
            output_mp_low_filt_esann = np.copy(lp_mp)
            output_mp_high_filt_esann = np.copy(hp_mp)

        # Plots
        if plots_net:
            plot_gc_sin_three_scenarios(fig3, i, time_vector, mean_rate, max_oscil, lif, coff, sfreq,
                                        modulation_signal1, modulation_signal2,
                                        modulated_signal1, modulated_signal2)

            if mean_rate[0] == 100 and i == 0:
                plot_gc_sin_input_example(time_vector, dt, ind_exp, modulation_signal1, modulation_signal2,
                                          modulated_signal1[0, :], modulated_signal2[0, :])

    if plots_net:
        plot_gc_sin_mp_high_rates(fig_esann, ind, ind_exp, time_vector, mean_rate, output_mp_esann, out_ylim_min,
                                  out_ylim_max, output_mp_low_filt_esann)
        fig3.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)

    time_desc = (f'[%dsin(0.2pit) + %d, %d], [%d, %dsin(0.2pit) + %d], [%dsin(0.2pit) + %d, %d]' %
                 (max_oscil[0], mean_rate[0], fix_rate[0], fix_rate[1], max_oscil[1], mean_rate[1],
                  max_oscil[2], mean_rate[2], fix_rate[2]))
    print_time(m_time() - ini_loop_time, "Experiment " + str(ind_exp) + ":" + time_desc)

    ind_exp += 1
# if plots_net: plot_gc_sin_statistics(res_per_reali, mean_rates)
# ******************************************************************************************************************
# FREQUENCY ANALYSIS
ini_loop_time = m_time()
if freq_analysis:
    fa = Freq_analysis(sim_params=stp_model.sim_params, loop_f=loop_frequencies, n_syn=num_syn)  # loop_frequencies
    fa.set_model(model_str=model, sim_params=sim_params, name_params=list(params.keys()),
                 model_params=list(params.values()))
    fa.run()
    # plot_freq_analysis(fa, " " + model + " a")
    title = ""
    if ind == 4: title = "Efficacy for fast-decay synapse"
    if ind == 5: title = "Efficacy for slow-decay synapse"

    # Plotting frequency response of efficacy
    plot_gc_sin_freq_response_efficacy(loop_frequencies, fa, title)
    print_time(m_time() - ini_loop_time, "Time for frequency analysis")
    # plot_net_depolarization(fa, loop_frequencies)

    # """
    for i in range(len(loop_frequencies)):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax.plot([0.1, 0.8], [phasic_st[0], phasic_st[0]], c='tab:red', alpha=0.5)
        ax.plot(loop_frequencies[0: i + 1], fa.eff_st[0, :i + 1], c='tab:blue')  # , label="phasic effect")
        ax.scatter(loop_frequencies[0: i + 1], fa.eff_st[0, :i + 1], c='black')  # , label="phasic effect")
        ax.scatter(loop_frequencies[i], fa.eff_st[0, i], c='tab:red')
        ax.set_xlabel("Rate (Hz)")
        ax.set_ylabel(r"$E_{f}$(r)")
        ax.grid()
        ax.set_title("Frequency response of efficacy")
        ax.set_xlim(0, loop_frequencies[-1] + 20)
        ax.set_ylim([-0.001, 0.075])
        # x.legend()
        # fig.savefig("../gain_control/plots/MSSM_fac_freq_res_" + str(loop_frequencies[i]) + "_2.png", format='png')
    # """
