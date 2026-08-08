import pylab as pl

from gain_control.utils_gc import *
import cProfile, pstats, tracemalloc, os, psutil

# ******************************************************************************************************************
# Depression using the MSSM
s_model = 'TM'
n_model = 'LIF'

# (Experiment 4) freq. response from Gain Control paper
# (Experiment 5) slow-decay frequency response
ind = 4
tau_m = 30
max_freq = 500

# For gain control, 100 inputs to a single LIF neuron
plots_net = False
plots_phd = False
dyn_synapse = True
gaincontrol_sinusoidal = True

# Hyperparameters for frequency analysis and poisson input spike
freq_analysis = False
Poisson = False
num_syn = 200

# Results
dict_results = {}
folder_vars = "../gain_control/variables/high_freq_10k/"  # Folder to save results
file_name = None
save_vars = True
num_realisations = 100
aux_ = [[] for _ in range(num_realisations)]
P_signal, P_noise = [[] for _ in range(num_realisations)], [[] for _ in range(num_realisations)]
SNR, gc_metric = [[] for _ in range(num_realisations)], [[] for _ in range(num_realisations)]
und_amp, und_amp_max = [[] for _ in range(num_realisations)], [[] for _ in range(num_realisations)]
und_q10, und_amp_min = [[] for _ in range(num_realisations)], [[] for _ in range(num_realisations)]
und_q90, und_var = [[] for _ in range(num_realisations)], [[] for _ in range(num_realisations)]

# Profiling
profiling = False
profiler = None

# Model parameters
val_params, description, name_params = get_params_stp(s_model, ind)

out_ylim_min, out_ylim_max, description_2 = -70, -50, ""
# if ind == 4: out_ylim_min, out_ylim_max, description_2 = -67, -57, r'Fast-decay synapse with $freq_{st}$ of efficacy=260Hz ($\tau_m$ ' + str (tau_m) + 'ms)'
# if ind == 5: out_ylim_min, out_ylim_max, description_2 = -70, -50, r'Slow-decay synapse with $freq_{st}$ of efficacy=560Hz ($\tau_m$ ' + str (tau_m) + 'ms)'
if ind == 4: out_ylim_min, out_ylim_max, description_2 = -70, -35, r'Fast-decay synapse with $freq_{st}$ of efficacy=260Hz ($\tau_m$ ' + str (tau_m) + 'ms)'
if ind == 5: out_ylim_min, out_ylim_max, description_2 = -70, -35, r'Slow-decay synapse with $freq_{st}$ of efficacy=560Hz ($\tau_m$ ' + str (tau_m) + 'ms)'
if ind == 7: out_ylim_min, out_ylim_max, description_2 = -60, 400, r'Facilitation ($\tau_m$ ' + str (tau_m) + 'ms)'
if ind == 8: out_ylim_min, out_ylim_max, description_2 = -70, -20, r'Diff. signaling synapse from Tsodyks, et al. ($\tau_m$ ' + str (tau_m) + 'ms)'

# time conditions
max_t, min_imp, max_imp, sfreq = 15, 5, 15, 10e3  # 10.2, 0.2, 10.2, 10e3  #
dt = 1 / sfreq
time_vector = np.arange(0, max_t, dt)
L = time_vector.shape[0]

# Parameters definition
params = dict(zip(name_params, val_params))
sim_params = {'sfreq': sfreq, 'max_t': max_t, 'L': L, 'time_vector': time_vector}

# **********************************************************************************************************************
# STP model
syn_params, description, name_params = get_params_stp(s_model, ind)
s_params = dict(zip(name_params, syn_params))
# Neuron model
neuron_params = get_neuron_params(n_model=n_model, tau_m=tau_m, ind=ind, y_lim_ind_plot=True, num_syn=1)

# Reducing synaptic strength of neurotransmitters in case of Doorn models
if 'g_nmda' in neuron_params: neuron_params['g_nmda'] = neuron_params['g_nmda'] * 5e-2
if 'g_ampa' in neuron_params: neuron_params['g_ampa'] = neuron_params['g_ampa'] * 5e-2

# Creating STP and neuron models
stp_model, neuron_model = models_creation_gc_sin(s_model, n_model, s_params, neuron_params, sim_params,
                                                 n_neu=1, n_syn=num_syn)
# stp_model = MSSM_model(n_syn=num_syn)
# if s_model == "TM": stp_model = TM_model(n_syn=num_syn)

# stp_model.set_model_params(params)
# stp_model.set_simulation_params(sim_params)

# Creating simple depression model
s_dep = Simple_Depression(n_syn=num_syn)
s_dep.set_simulation_params(sim_params)
# **********************************************************************************************************************

# **********************************************************************************************************************
# Frequency ranges for Frequency response of efficacy
# range_f = [10, 20, 30, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]  # [i for i in range(10, 801, 10)]
# [i for i in range(1, 11, 1)] + [i for i in range(15, 101, 5)] + [i for i in range(100, 501, 50)]
range_f, range_f2, range_f3, range_f4 = [], [], [], []
if 100 < max_freq:
    range_f = [i for i in range(10, 100, 5)]
    if 500 < max_freq:
        range_f2 = [i for i in range(100, 500, 10)]
        if 1000 < max_freq:
            range_f3 = [i for i in range(500, 1000, 50)]
            range_f4 = [i for i in range(1000, max_freq, 100)]
        else:
            range_f3 = [i for i in range(500, max_freq, 50)]
    else:
        range_f2 = [i for i in range(100, max_freq, 10)]
else:
    range_f = [i for i in range(10, max_freq, 5)]
# range_f = [10, 20, 500]
# range_f2, range_f3, range_f4 = [], [] , []
f_vector = np.array(range_f + range_f2 + range_f3 + range_f4)
loop_frequencies = np.array(f_vector)
# **********************************************************************************************************************

# **********************************************************************************************************************
# Params for sinusoidal envelope of input stimuli
mean_rates, max_oscils, fix_rates = [], [], []
delta = 0.5

if gaincontrol_sinusoidal:
    # mean_rates = [[50, 10, 50], [300, 10, 300], [1000, 10, 1000]]  # [[10, 10, 10], [20, 10, 20], [50, 10, 50], [100, 10, 100], [300, 10,  300], [500, 10,  500]]
    # max_oscils = [[25,  5,  5], [150,  5,  5], [500, 5, 5]]  # [[5, 5,  5],  [10, 5,  5],   [25, 5,  5],  [50,  5,  5],   [150, 5,   5],   [250, 5,   5]]  #
    # fix_rates = [[10, 50, 10], [10, 300, 10], [10, 1000, 10]]  # [[10, 10, 10], [10, 20, 10],  [10, 50, 10], [10, 100, 10],  [10,  300, 10], [10,   500, 10]]  #
    for i in f_vector:
        mean_rates.append([i, 10, i])
        max_oscils.append([i - i*delta, 5, 5])
        fix_rates.append([10, i, 10])
# **********************************************************************************************************************


# Results variable
res_per_reali = np.zeros((10, 3, len(mean_rates)))  # statistical descriptors, num. scenarios, num. ref rate

# Aux variables for plotting
fig, fig3, fig_esann, output_mp_esann, output_mp_low_filt_esann = None, None, None, None, None

seeds = []

# ******************************************************************************************************************
# Final dictionary
dict_results = {'initial_frequencies': f_vector, 'num_synapses': num_syn, 'sfreq': sfreq,
                'tau_lif': tau_m, 'gain_v': delta, 'stp_name_params': name_params, 'stp_value_params': syn_params,
                'sim_params': sim_params, 'n_params': neuron_params, 'dyn_synapse': dyn_synapse}
aux_name = "_ind_" + str(ind) + "_sf_" + str(
            int(sfreq / 1000)) + "k_syn_" + str(num_syn)
if neuron_model == 'LIF': aux_name += "_tau" + n_model + "_" + str(tau_m) + "ms"
aux_name += "_sinusoidal_q95"
file_name = s_model + aux_name
# ******************************************************************************************************************

if plots_net:
    # Plotting
    fig_size = (10, 5)
    if ind in [2, 4, 5, 7, 8]:
        fig_size = (12, 2)  # 1.6)
        if len(mean_rates) > 6:
            fig_size = (12, 3.5)
    fig_esann = plt.figure(figsize=fig_size)
    fig_esann.suptitle(description_2, fontsize=18)

# **************************************************************************************************************
# PROFILING
if profiling:
    profiler = cProfile.Profile()
    profiler.enable()
# **************************************************************************************************************

# ******************************************************************************************************************
# SIMULATION GAIN CONTROL SINUSOIDAL INPUT (200 SYNAPSES TO ONE LIF NEURON)
if os.path.isfile(folder_vars + file_name):
    dict_results = loadObject(file_name, folder_vars)
    if plots_phd:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        SNR_mean = np.mean(np.array(dict_results['SNR']), axis=0)
        SNR_std = np.std(np.array(dict_results['SNR']), axis=0)
        gc_mean = np.mean(np.array(dict_results['gc_metric']), axis=0)
        gc_std = np.std(np.array(dict_results['gc_metric']), axis=0)
        ax.plot(dict_results['initial_frequencies'], SNR_mean, label='SNR', color='tab:red')
        ax.fill_between(dict_results['initial_frequencies'], SNR_mean-SNR_std, SNR_mean+SNR_std, color='tab:red',
                         alpha=0.5)
        ax.plot(dict_results['initial_frequencies'], gc_mean, label='GC metric', color='tab:blue')
        ax.fill_between(dict_results['initial_frequencies'], gc_mean - gc_std, gc_mean + gc_std, color='tab:blue',
                         alpha=0.5)
        # ax.grid()
        ax.legend()
        # ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        # ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid()

if gaincontrol_sinusoidal and not os.path.isfile(folder_vars + file_name):
    ini_sin_time = m_time()
    for reali in range(num_realisations):
        ind_exp = 0
        ini_reali_time = m_time()
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

            for i in [0]:  # range(len(mean_rate)):

                se = int(time.time())
                seeds.append(se)
                seeds1 = [j + se for j in range(int(L / 2))]
                seeds2 = [j + se + 2 for j in range(int(L / 2))]

                # Signals with firing rate modulation
                modulation_signal1 = mean_rate[i] + max_oscil[i] * np.sin(2 * np.pi * (1 / 10) * time_vector_sin)
                modulation_signal2 = fix_rate[i] * np.ones(L)

                # Sinusoidal modulated firing rate signal
                modulated_signal1 = oscillatory_spike_train(sfreq, modulation_signal1,
                                                            num_realizations=int(num_syn / 2), poisson=True,
                                                            seeds=seeds1, correction=True)
                # Constant firing rate signal
                modulated_signal2 = simple_spike_train(sfreq, modulation_signal2[0], len(modulation_signal2),
                                                       num_realizations=int(num_syn / 2), poisson=True,
                                                       seeds=seeds2)

                # Organising input to correspond to the paper
                if i == 1: Input_test = np.concatenate((modulated_signal2, modulated_signal1), axis=0)
                else: Input_test = np.concatenate((modulated_signal1, modulated_signal2), axis=0)

                # Running STP model
                if dyn_synapse:
                    model_stp(stp_model, neuron_model, params, Input_test)
                else:
                    static_synapse(neuron_model, Input_test, 0.0125)

                # Filtering mem. pot., lowpass for getting the sinusoidal trend, high pass for the variance without
                # seasonality
                # coff = 1
                # res_per_reali[:, i, ind_exp], lp_mp, hp_mp = aux_statistics_sin(
                # neuron_model.membrane_potential[0, :], coff, sfreq, min_imp, max_imp)

                # """
                # ******************************************************************************************************
                # Plot for the PhD Thesis
                if i == 0:  # Only for high rate signal
                    mp_signal = neuron_model.membrane_potential[0, :]
                    low_pass_mempot = lowpass(mp_signal, 1, sfreq)
                    high_pass_mempot = highpass(mp_signal, 1, sfreq)
                    mem_pot_low_filt = low_pass_mempot[int(min_imp / dt): int(max_imp / dt)]
                    mem_pot_high_filt = high_pass_mempot[int(min_imp / dt): int(max_imp / dt)]
                    hp_mp_q90, hp_mp_q10 = np.quantile(mem_pot_high_filt, q=0.95), np.quantile(mem_pot_high_filt, q=0.15)
                    lp_mp_max, lp_mp_min = np.max(mem_pot_low_filt), np.min(mem_pot_low_filt)

                    P_signal_i = np.mean(np.abs(mem_pot_low_filt - np.mean(mem_pot_low_filt))**2)
                    P_noise_i = np.mean(np.abs(mem_pot_high_filt)**2)
                    SNR_ = 10 * np.log10(P_signal_i / P_noise_i)

                    amplitude = lp_mp_max - lp_mp_min
                    variability = hp_mp_q90 - hp_mp_q10
                    limit_gc = amplitude - variability

                    SNR[reali].append(SNR_)
                    P_signal[reali].append(P_signal_i)
                    P_noise[reali].append(P_noise_i)
                    und_amp_max[reali].append(lp_mp_max)
                    und_amp_min[reali].append(lp_mp_min)
                    und_amp[reali].append(amplitude)
                    und_q10[reali].append(hp_mp_q10)
                    und_q90[reali].append(hp_mp_q90)
                    und_var[reali].append(variability)
                    gc_metric[reali].append(limit_gc)

                    aux_ = (file_name + ". Realisation %s, rate %d, SNR %.2fdB, GC met. %.2f, amp %.2f, var %.2f"
                            % (reali, f_vector[ind_exp], SNR_, limit_gc, amplitude, variability))
                    print_time(m_time() - ini_loop_time, aux_)

                    if plots_phd:
                        fig_filters = plt.figure(figsize=(9, 4))
                        plt.suptitle(r"Gain Control measurements in sinusoidal schema for $\delta=0.5$",
                                     color="black", fontsize=14)
                        ax_v = fig_filters.add_subplot(2, 1, 1)
                        ax_s = fig_filters.add_subplot(2, 2, 3)
                        ax_n = fig_filters.add_subplot(2, 2, 4)
                        ax_v.plot(time_vector[int(min_imp / dt): int(max_imp / dt)],
                                  mp_signal[int(min_imp / dt): int(max_imp / dt)], color='tab:blue')
                        ax_s.plot(time_vector[int(min_imp / dt): int(max_imp / dt)],
                                  mem_pot_low_filt - np.mean(mp_signal[int(min_imp / dt): int(max_imp / dt)]),
                                  color='black')
                        ax_n.plot(time_vector[int(min_imp / dt): int(max_imp / dt)], mem_pot_high_filt, color='gray')
                        ax_n.plot([min_imp, max_imp], [hp_mp_q90, hp_mp_q90], color='tab:green', label='q90')
                        ax_n.plot([min_imp, max_imp], [hp_mp_q10, hp_mp_q10], color='tab:green', linestyle='--',
                                  label='q10')
                        ax_s.plot([min_imp, max_imp], [hp_mp_q90, hp_mp_q90], color='tab:green')
                        ax_s.plot([min_imp, max_imp], [hp_mp_q10, hp_mp_q10], color='tab:green', linestyle='--')
                        ax_s.set_ylim([-1., 2.])
                        ax_n.set_ylim([-1., 2.])
                        ax_v.set_ylim([-66, -62.9])
                        ax_v.grid(), ax_s.grid(), ax_n.grid()
                        ax_v.set_xlabel('Time (s)', color='gray')
                        ax_s.set_xlabel('Time (s)', color='gray')
                        ax_n.set_xlabel('Time (s)', color='gray')
                        ax_v.set_ylabel(r'$v(t) (mV)$', color='gray')
                        ax_s.set_ylabel(r'$v_{\mathrm{LP}}(t)$ (mV)', color='gray')
                        ax_n.set_ylabel(r'$v_{\mathrm{HP}}(t)$ (mV)', color='gray')
                        ax_v.set_title('Membrane potential of output neuron for baseline rate %dHz' % mean_rate[0],
                                       color="black", alpha=0.7)
                        ax_s.set_title('Underlined sinusoidal amplitude A(r) = %.2fmV' % (lp_mp_max - lp_mp_min),
                                       color="black", alpha=0.7)
                        ax_n.set_title('Noise of membrane potential $\eta(r)=$%.2fmV' % (hp_mp_q90 - hp_mp_q10),
                                       color="black", alpha=0.7)
                        ax_n.legend()
                        plt.tight_layout()
                        path_save = (r'../gain_control/plots/gain_control_sin_' + s_model + '_ind_' + str(ind) +
                                     '_high_and_low_filters_br_' + str(mean_rate[0]) + '_taum_' + str(tau_m) + 'ms.png')
                        fig_filters.savefig(path_save, format='png')
                # ******************************************************************************************************************
                # """

                if i == 0:
                    output_mp_esann = np.copy(neuron_model.membrane_potential[0, :])
                    output_mp_low_filt_esann = np.copy(low_pass_mempot)
                    output_mp_high_filt_esann = np.copy(high_pass_mempot)

                # Plots
                if plots_net:
                    plot_gc_sin_three_scenarios(fig3, i, time_vector, mean_rate, max_oscil, neuron_model, coff, sfreq,
                                                modulation_signal1, modulation_signal2,
                                                modulated_signal1, modulated_signal2)
                    fig3.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
                    if mean_rate[0] == 100 and i == 0:
                        fig_essan3 = plot_gc_sin_input_example(time_vector, dt, ind_exp, modulation_signal1,
                                                               modulation_signal2, modulated_signal1[0, :],
                                                               modulated_signal2[0, :])

            if plots_net:
                # plot_gc_sin_mp_high_rates_esann(fig_esann, ind, ind_exp, time_vector, mean_rate, output_mp_esann,
                #                                 out_ylim_min, out_ylim_max, output_mp_low_filt_esann)
                path_save = (r'../gain_control/plots/gain_control_sin_' + s_model + '_ind_' + str(ind) +
                             '_high_rate_v(t)_taum_' + str(tau_m) + 'ms.png')
                plot_gc_sin_mp_high_rates(fig_esann, ind, ind_exp, time_vector, mean_rate, output_mp_esann,
                                          out_ylim_min, out_ylim_max, output_mp_low_filt_esann,
                                          num_graphs=len(mean_rates), pathsave=path_save, savefig=False)
                fig_esann.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
                fig_esann.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
                fig_esann.savefig(path_save, format='png')
                path_save = (r'../gain_control/plots/gain_control_sin_' + s_model + '_ind_' + str(ind) +
                             '_input_example.png')
                # fig_essan3.savefig(path_save, format='png')

            # time_desc = (f'[%dsin(0.2pit) + %d, %d], [%d, %dsin(0.2pit) + %d], [%dsin(0.2pit) + %d, %d]' %
            #              (max_oscil[0], mean_rate[0], fix_rate[0], fix_rate[1], max_oscil[1], mean_rate[1],
            #               max_oscil[2], mean_rate[2], fix_rate[2]))
            # print_time(m_time() - ini_loop_time, "Experiment " + str(ind_exp) + ":" + time_desc)

            ind_exp += 1
        print_time(m_time() - ini_reali_time, file_name)

    # Saving final dictionary if file does not exist
    if not os.path.isfile(folder_vars + file_name):
        dict_results['SNR'] = SNR
        dict_results['P_signal'] = P_signal
        dict_results['P_noise'] = P_noise
        dict_results['noise_q10'] = und_q10
        dict_results['noise_q90'] = und_q90
        dict_results['variability_noise'] = und_var
        dict_results['und_amplitude_max'] = und_amp_max
        dict_results['und_amplitude_min'] = und_amp_min
        dict_results['und_amplitude'] = und_amp
        dict_results['gc_metric'] = gc_metric
        if save_vars:
            saveObject(dict_results, file_name, folder_vars)
    print_time(m_time() - ini_sin_time, "Total time for " + file_name)

# if plots_net: plot_gc_sin_statistics(res_per_reali, mean_rates)

# **************************************************************************************************************
# PROFILING
if profiling:
    profiler.disable()
    pstats.Stats(profiler).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(50)
# **************************************************************************************************************

# ******************************************************************************************************************
# FREQUENCY ANALYSIS

ini_loop_time = m_time()
if freq_analysis:
    ax = None
    # """
    # Figure PhD thesis (methodology / metrics temporal filtering)
    fig_phd = plt.figure(figsize=[10, 4])
    plt.suptitle("Frequency response for short-term facilitation", fontsize=16)
    ax = [fig_phd.add_subplot(3, 2, i) for i in range(1, 7)]
    # """

    fa = Freq_analysis(sim_params=stp_model.sim_params, loop_f=loop_frequencies, n_syn=1)  # n_syn=num_syn
    fa.set_model(model_str=s_model, sim_params=sim_params, name_params=list(params.keys()),
                 model_params=list(params.values()))
    fa.run(ax=ax)
    # plot_freq_analysis(fa, " " + model + " a")
    title = ""
    if ind == 4: title = "Efficacy for fast-decay synapse"
    if ind == 5: title = "Efficacy for slow-decay synapse"

    # Plotting frequency response of efficacy
    plot_gc_sin_freq_response_efficacy(loop_frequencies, fa, title, freqst=True)
    print_time(m_time() - ini_loop_time, "Time for frequency analysis")
    # plot_net_depolarization(fa, loop_frequencies)
    # """

    # ******************************************************************************************************************
    # Figure PhD thesis (methodology / Synaptic dynamics cases of study)
    title = ""
    if ind == 4: title = "Frequency response for STD"
    if ind == 8: title = "Frequency response for STF"
    path = "../gain_control/plots/freq_response_" + title[-3:] + ".png"
    plot_gc_sin_freq_response_efficacy(loop_frequencies, fa, title, freqst=False, savefig=True, path=path, log_sc=True)
    # ******************************************************************************************************************

    c_ax = 1
    for i in range(len(loop_frequencies)):
        """
        fig = plt.figure(figsize=[8, 2])
        ax_ = fig.add_subplot(111)
        # ax_.plot([0.1, 0.8], [phasic_st[0], phasic_st[0]], c='tab:red', alpha=0.5)
        ax_.plot(loop_frequencies[0: i + 1], fa.time_ss[0, :i + 1], c='tab:blue')  # , label="phasic effect")
        ax_.scatter(loop_frequencies[0: i + 1], fa.time_ss[0, :i + 1], c='black')  # , label="phasic effect")
        ax_.scatter(loop_frequencies[i], fa.time_ss[0, i], c='tab:red')
        # ax_.set_xlabel("Rate (Hz)", color="gray")
        ax_.set_ylabel(r"$E_{f}$(r)", color="gray")
        ax_.grid()
        ax_.set_title(r"Frequency response of $t_{st}$", c='black', alpha=0.7, fontsize=16)
        ax_.set_xlim(0, loop_frequencies[-1] + 20)
        ax_.set_ylim([-0.001, 0.085])
        fig.tight_layout()
        # x.legend()
        # fig.savefig("../gain_control/plots/MSSM_dep_freq_res_" + str(loop_frequencies[i]) + "_2.png", format='png')
        # """

        # """
        # **************************************************************************************************************
        # Figure PhD thesis (methodology / metrics temporal filtering)
        if loop_frequencies[i] in [10, 50, 500]:
            ax[c_ax].plot(loop_frequencies[0: i + 1], fa.eff_st[0, :i + 1], c='tab:blue')  # , label="phasic effect")
            # ax[c_ax].scatter(loop_frequencies[0: i + 1], fa.eff_st[0, :i + 1], c='black')  # , label="phasic effect")
            ax[c_ax].scatter(loop_frequencies[i], fa.eff_st[0, i], c='tab:red')
            if loop_frequencies[i] == 500: ax[c_ax].set_xlabel("Rate (Hz)", color="gray")
            ax[c_ax].set_ylabel("(mV)", color="gray")
            ax[c_ax].grid()
            ax[c_ax].set_title(r"Frequency response of $E_{psp_{st}}(r)$", c='black', alpha=0.7, fontsize=14)
            ax[c_ax].set_ylim([-0.001, 0.085])
            ax[c_ax].set_xscale('log')
            ax[c_ax].set_xlim(9, loop_frequencies[-1] + 50)
            c_ax += 2
    fig_phd.tight_layout()
    # fig_phd.savefig("../gain_control/plots/MSSM_fac_temp_freq_res.png", format='png')
    # plt.close(fig_phd)
    # ******************************************************************************************************************
    # """
