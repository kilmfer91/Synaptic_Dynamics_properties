import matplotlib.pyplot as plt
import scipy.signal
import numpy as np

from synaptic_dynamic_models.TM import TM_model
from synaptic_dynamic_models.MSSM import MSSM_model
from spiking_neuron_models.LIF import LIF_model
from synaptic_dynamic_models.simple_depression import Simple_Depression
from libraries.frequency_analysis import Freq_analysis
from utils import *
from gain_control.utils_gc import *


def oscillatory_spike_train(sfreq, modulation_signal, num_realizations=1, poisson=False, seeds=None, correction=False,
                            imputation=True):

    seed = None
    if seeds is not None:
        assert isinstance(seeds, list), "seeds must be a list"
        seed = np.random.choice(seeds)
    seed_print = []
    Input_test = np.array([[]])
    L = len(modulation_signal)
    i = 0
    while Input_test.shape[1] < L:
        if poisson:
            if seeds is not None:
                seed = np.random.choice(seeds)
                seed_print.append(seed)
            # aux_s = poisson_generator2(1 / sfreq, int(3 * sfreq / modulation_signal[i]), modulation_signal[i],
            #                            num_realizations, myseed=seed)
            aux_s = poisson_generator2(1 / sfreq, int(0.03 * sfreq), modulation_signal[i],
                                       num_realizations, myseed=seed)

            # Correcting Poisson spike train to avoid consecutive spikes
            if correction:
                aux_s = correct_poisson_spike_trains(aux_s, num_realizations, seed=seed, imputation=imputation)

            desired_len_aux_s = aux_s.shape[1]
        else:
            aux_s = input_spike_train(sfreq, modulation_signal[i], 3 / modulation_signal[i], min_time=0.0)
            aux_s = np.repeat(np.expand_dims(aux_s, axis=0), num_realizations, axis=0)

            if i == 0: aux_s = np.roll(aux_s, 1)
            desired_len_aux_s = np.where(aux_s == 1)[1][-1]
            aux_s = aux_s[:, :desired_len_aux_s]
        #
        if i == 0:
            Input_test = aux_s
        else:
            # Input_test = np.concatenate((Input_test, aux_s))
            Input_test = np.hstack((Input_test, aux_s))
        i += desired_len_aux_s
    # print(seed_print)
    return Input_test[:, :L]


# ******************************************************************************************************************
# Depression using the MSSM
model = 'MSSM'
# (Experiment 2) freq. response decay around 100Hz
# (Experiment 3) freq. response decay around 10Hz
# (Experiment 4) freq. response from Gain Control paper
# (Experiment 5) freq. response decay around 100Hz
# (Experiment 6) freq. response decay around 10Hz
ind = 4

# For gain control, 100 inputs to a single LIF neuron
plots_net = True
dyn_synapse = True
gaincontrol_sinusoidal = True

# Hyperparameters for frequency analysis and poisson input spike
freq_analysis = False
Poisson = True
num_syn = 200

# Model parameters
name_params = ['tau_c', 'alpha', 'V0', 'tau_v', 'P0', 'k_NtV', 'k_Nt', 'tau_Nt', 'k_EPSP', 'tau_EPSP']
# Model parameters
val_params, description, name_params = get_params_stp(model, ind)

"""
# freq. response decay around 100Hz (Experiment 2) {5: 0.99, 10: 0.9, 20: 0.8, 50: 0.7, 100: 0.5, 200: 0.2}
if ind == 2:
    description = "Experiment 2, decay around 100Hz"
    val_params = [5.42451963e-02, 2.92925980e+00, 6.67821125e+01, 1.80143000e-02, 7.54167519e-01,
                  5.99119322e+01, 9.94215228e-01, 1.03825167e-03,
                  (4.52507712e-01 * 0.075) / 779.1984, 1.00243185e-03]  # 4.52507712e-01, g_L = 5e-6 homogeneous / 4.6e-3 poisson [2.3e-3 for subthreshold]
    # g_L = 1e0 # [1.9e-3 for sfreq 2e3] [8.6e-4 for sfreq 4e3] [8.4e-4 for sfreq 5e3]
# freq. response decay around 100Hz (Experiment 5) {5: 0.99, 10: 0.9, 20: 0.8, 50: 0.7, 100: 0.5, 200: 0.2}
if ind == 5:
    description = "Experiment 5, decay around 100Hz"
    val_params = [3.03503488e-02, 1.49534243e-01, 1.27807004e+00, 8.32749189e-02,
                  1.03652329e-03, 7.06246475e+01, 9.84186889e-01, 1.00258903e-03,
                  (4.42926340e-01 * 0.075) / 2.975, 1.00046291e-03]  # g_L = 2.55e-1/2.46e-1 poisson [1.1e-1 for subthreshold]
    # g_L = 1e0 # [1.1e-1 for sfreq = 2e3] [6.9e-2 for sfreq 3e3] [4.9e-2 for sfreq 4e3] [4e-2 for sfreq 5e3]
# freq. response decay around 10Hz (Experiment 3) {2: 0.88946682, 5: 0.70155924, 10: 0.57552428, 15: 0.2, 20: 0.1, 50: 0.07}
if ind == 3:
    description = "Experiment 3, decay around 10Hz"
    val_params = [6.53659368e-03, 1.75660742e-01, 3.17123087e+01, 1.78659320e-01, 2.50362727e-01,
                  9.12004545e+01, 9.13420672e-01, 2.14204288e-03,
                  5.20907890e-03 / 4.29, 4.32890680e-03]  # 5.20907890e-01, g_L = 2.6e-2 homogeneous / 4.6e-2 poisson [2.0e-2 for subthreshold]
    # g_L = 3.7e-2  # [3.7e-2 for sfreq 2e3] [1.8e-2 for sfreq 4e3] [1.4e-2 for sfreq 5e3]
# freq. response decay around 10Hz (Experiment 6) {[2:0.88946682, 4:0.70155924, 5.4:0.57552428, 10:0.2, 15:0.1, 30:0.06, 50:0.035]
if ind == 6:
    description = "Experiment 6, decay around 10Hz"
    val_params = [3.61148253e-03, 9.98782883e-02, 9.99236857e+00, 2.81436921e-01,
                  1.97666651e-02, 1.00657445e+01, 7.39059950e-01, 1.07519099e-03,
                  3.11356220e-01, 1.74116605e-03]  # g_L = 1.4e-1 poisson [5.8e-2 for subthreshold]
    # g_L = 5.8e-2  # [5.8e-2 for sfreq 2e3] [4.4e-2 for sfreq 4e3] [3.9e-2 for sfreq 5e3]
# freq. response from Gain Control paper (Experiment 4) {5: 0.785, 10: 0.6, 20: 0.415, 50: 0.205, 100: 0.115}
if ind == 4:
    description = "Experiment 4, Gain-Control paper"
    val_params = [7.85735182e-02, 4.56599128e-01, 1.46835212e+00, 1.63998958e-01, 2.41885797e-04,
                  5.84619146e+01, 8.00871281e-01, 1.50280526e-03,
                  (5.94890729e-02 * 0.075) / 1.0217, 1.75609424e-03]  # 5.94890729e-01, g_L = 5.4e-4 homogeneous / 3.21e-3 poisson [1.3e-3 for subthreshold]  # g_L = 5.4e-5 for static synapse
    # g_L = 1.0e0  # [1.3e-3 for sfreq 2e3] [1.0e-3 for sfreq 4e3] [ for sfreq 5e3]
# params_s_dep = {'tau_g': 2e-3, 'tau_alpha': 300e-3, 'g0': 0.075, 'f': 0.75}
# """
g_L = 7.5e-2

out_ylim_min, out_ylim_max, description_2 = -70, -50, ""
if ind == 4: out_ylim_min, out_ylim_max, description_2 = -65, -60.5, r'Fast-decay synapse with $freq_{st}$ of efficacy=260Hz'
if ind == 5: out_ylim_min, out_ylim_max, description_2 = -64.5, -50, r'Slow-decay synapse with $freq_{st}$ of efficacy=560Hz'

# For experiment 4
# prop_high_rate_max_pos = array([-62.58562452, -62.21529441, -61.93138295, -61.72693519]) -61.5
# prop_high_rate_min_neg = array([-63.69040866, -63.1157654 , -62.38389251, -62.23640865])

# For experiment 5
# prop_high_rate_max_pos = array([-58.23917046, -55.58041238, -52.93823224, -52.23514712]) - 52
# prop_high_rate_min_neg = array([-63.17012234, -60.16772678, -55.65855235, -54.09690444]) -63.5

# time conditions
max_t, min_imp, max_imp, sfreq = 15, 0.0, 10, 5e3  # .6, 0.0, .6, 1e5
if plots_net: max_t = 15

dt = 1 / sfreq
time_vector = np.arange(0, max_t, dt)
L = time_vector.shape[0]

# Parameters definition
params = dict(zip(name_params, val_params))
sim_params = {'sfreq': sfreq, 'max_t': max_t, 'L': L, 'time_vector': time_vector}

# Creating MSSM
mssm = MSSM_model(n_syn=num_syn)
mssm.set_model_params(params)
mssm.set_simulation_params(sim_params)
# Creating simple depression model
s_dep = Simple_Depression(n_syn=num_syn)
s_dep.set_simulation_params(sim_params)

# Frequency ranges
range_f0 = [1, 2, 3, 4]
range_f1 = [i for i in range(10, 100, 5)]
range_f2 = [i for i in range(100, 800, 20)]  # [i for i in range(100, 500, 10)] [i for i in range(100, 321, 10)]
loop_frequencies = np.array(range_f0 + range_f1 + range_f2)


# ******************************************************************************************************************
# PARAMS FOR LIF MODEL
lif_params = {'V_threshold': np.array([50 for _ in range(1)]), 'V_reset': np.array([-70 for _ in range(1)]),  # V_th = -55
              'tau_m': np.array([30e-3 for _ in range(1)]),
              'g_L': np.array([g_L for _ in range(1)]),
              'V_init': np.array([-70 for _ in range(1)]), 'V_equilibrium': np.array([-70 for _ in range(1)]),
              't_refractory': np.array([0.01 for _ in range(1)])}

lif_params2 = {'V_threshold': np.array([50 for _ in range(1)]), 'V_reset': np.array([-70 for _ in range(1)]),  # V_th = -55
               'tau_m': np.array([30e-3 for _ in range(1)]),
               'g_L': np.array([g_L for _ in range(1)]),  # 3.21e-3
               'V_init': np.array([-70 for _ in range(1)]), 'V_equilibrium': np.array([-70 for _ in range(1)]),
               't_refractory': np.array([0.01 for _ in range(1)])}


# Creating LIF model
lif = LIF_model(n_neu=1)  # (n_neu=100)
lif.set_model_params(lif_params)
lif.set_simulation_params(sim_params)

# 2nd LIF model
lif2 = LIF_model(n_neu=1)
lif2.set_model_params(lif_params2)
lif2.set_simulation_params(sim_params)

# ******************************************************************************************************************
# SIMULATION GAIN CONTROL (200 SYNAPSES TO ONE LIF NEURON)
# """
# ind_exp = 10
# iterator = 12
# mean_rates = [[50,8,50],[100,10,100],[150,12,150],[200,14,200],[250,16,250],[300,18,300],[350,20,350],[400,22,400],[450,24,450],[500,26,500],[550,28,550],[600,30,600]]
# max_oscils = [[25,4,4], [50,5,5],    [75,6,6],    [100,7,7],   [125,8,8],   [150,9,9],   [175,10,10], [200,11,11], [225,12,12], [250,13,13], [275,14,14], [300,15,15]]
# fix_rates =  [[8,50,8], [10,100,10], [12,150,12], [14,200,14], [16,250,16], [18,300,18], [20,350,20], [22,400,22], [24,450,24], [26,500,26], [28,550,28], [30,600,30]]

# Params for sinusoidal envelope of input stimuli
ini_high_rate = 50 # 50
step_high_rate = 50 # 10 # 50
ini_low_rate = 8 # 8
step_low_rate = 2 # 1  # 2
proportion = 0.5  # 0.5
iterator = 10 # 56  # 12
# auxiliars
ihr = ini_high_rate
shr = step_high_rate
ilr = ini_low_rate
slr = step_low_rate
pr = proportion
mean_rates = [[(shr * i) + ihr, (slr * i) + ilr, (shr * i) + ihr] for i in range(iterator)]
max_oscils = [[((shr * i) + ihr) * pr, ((slr * i) + ilr) * pr, ((slr * i) + ilr) * pr] for i in range(iterator)]
fix_rates =  [[(slr * i) + ilr, (shr * i) + ihr, (slr * i) + ilr] for i in range(iterator)]
mean_rates = [[50,10,50], [100,10,100],[300,10,300],[500,10,500]] # [[100,10,100]]
max_oscils = [[25,5,5],   [50,5,5],    [150,5,5],   [250,5,5]]  # [[50,5,5]]
fix_rates =  [[10,50,10], [10,100,10], [10,300,10], [10,500,10]]  # [[10,100,10]]

# Overall variables
vec_median_mempot, vec_mean_mempot, vec_std_mempot = [], [], []
vec_median_mempot_highI, vec_mean_mempot_highI, vec_std_mempot_highI = [], [], []
vec_median_mempot_lowI, vec_mean_mempot_lowI, vec_std_mempot_lowI = [], [], []
vec_max_mp_pos, vec_min_mp_neg, vec_q1_mp_pos, vec_q90_mp_pos, vec_q1_mp_neg, vec_q90_mp_neg, vec_q90_mp, vec_q1_mp, vec_max_mp, vec_min_mp = [], [], [], [], [], [], [], [], [], []
high_rate_diff_medians, ref_diff_medians, fig, fig3, fig4 = 20, 1, None, None, None

# Aux variables for plotting
ax2_s, fig_esann, input_mod1_esann, input_mod2_esann, output_mp_esann, output_mp_low_filt_esann = None, None, None, None, None, None

# Stop condition
stop_condition = False
cond_while = len(mean_rates)
# if plots_net: cond_while = 1
seeds = []

if gaincontrol_sinusoidal:
    if plots_net:
        # Plotting
        fig_size = (10, 5)
        if ind == 4 or ind == 2 or ind == 5: fig_size = (12, 1.6)
        # if ind == 5 or ind == 2: fig_size = (10,1.8)
        fig_esann = plt.figure(figsize=fig_size)
        fig_esann.suptitle(description_2, fontsize=18)

    ind_exp = 0
    while ind_exp < cond_while:  # len(mean_rates): # for ind_exp in range(len(mean_rates)):
        ini_loop_time = m_time()
        mean_rate = mean_rates[ind_exp]
        max_oscil = max_oscils[ind_exp]
        fix_rate = fix_rates[ind_exp]
        # Input
        sub_sfreq = 1  # 4e3
        time_vector_sin = np.arange(0, max_t, 1 / (sfreq / sub_sfreq))  # 3e3

        if plots_net:
            # Plotting
            fig = plt.figure(figsize=(10, 5))
            fig.suptitle("MSSM")
            fig3 = plt.figure(figsize=(6.5, 5))  # 3.5, 7)))
            # fig3 = plt.figure(figsize=(15, 3.2))
            fig3.suptitle("Types of input")  # , fontsize=21) #  (using windows of 30ms for each change of rate)", fontsize=21)
            fig4 = plt.figure(figsize=(15, 2))

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
            if i == 1:
                Input_test = np.concatenate((modulated_signal2, modulated_signal1), axis=0)
            else:
                Input_test = np.concatenate((modulated_signal1, modulated_signal2), axis=0)

            # legends
            legend1 = "osc rate " + str(mean_rate[i]) + "Hz +/-" + str(max_oscil[i]) + "Hz"
            legend2 = "fixed at rate " + str(fix_rate[i]) + "Hz"

            # Running STP model
            if dyn_synapse:
                model_stp(mssm, lif, params, Input_test)
            else:
                static_synapse(lif, Input_test, 0.0125)

            # Running simple depression model
            # model_simple_dep(s_dep, lif2, params_s_dep, Input_test)

            """
            # Analysis of membrane potential changes due to input change
            median_mempot = np.array([np.median(lif.membrane_potential[0, :]) for _ in range(L)])
            mean_mempot = np.array([np.mean(lif.membrane_potential[0, :]) for _ in range(L)])
            std_mempot = np.array([np.std(lif.membrane_potential[0, :]) for _ in range(L)])
            low_pass_mempot = lowpass(lif.membrane_potential[0, :], 3, sfreq)
            median_segments = np.zeros(lif.membrane_potential.shape[1])
            mean_segments = np.zeros(lif.membrane_potential.shape[1])
            std_segments = np.zeros(lif.membrane_potential.shape[1])
            for k in range(int(max_t/5)):
                aux = lif.membrane_potential[0, int(k * 5 / dt): int((k + 1) * 5 / dt)]
                median_segments[int(k * 5 / dt): int((k + 1) * 5 / dt)] = np.median(aux)
                mean_segments[int(k * 5 / dt): int((k + 1) * 5 / dt)] = np.mean(aux)
                std_segments[int(k * 5 / dt): int((k + 1) * 5 / dt)] = np.std(aux)
            sec_lowI = 1
            if max_t % 2 == 1: sec_lowI == 11
            a = median_segments[int(sec_lowI / dt)]  # (median_segments[int(16 / dt)] + median_segments[int(6 / dt)]) /2
            b = median_segments[int(6 / dt)]  # (median_segments[int(1 / dt)] + median_segments[int(11 / dt)]) / 2
            c = mean_segments[int(sec_lowI / dt)]
            d = mean_segments[int(6 / dt)]
            e = std_segments[int(sec_lowI / dt)]
            f = std_segments[int(6 / dt)]
    
            vec_median_mempot.append(median_mempot[0])
            vec_median_mempot_highI.append(a)
            vec_median_mempot_lowI.append(b)
            vec_mean_mempot.append(mean_mempot[0])
            vec_mean_mempot_highI.append(c)
            vec_mean_mempot_lowI.append(d)
            vec_std_mempot.append(std_mempot[0])
            vec_std_mempot_highI.append(e)
            vec_std_mempot_lowI.append(f)
            
    
            # Stop condition
            if i == 0:
                prop_high_rate_lowI = b
                prop_high_rate_median = a  # vec_median_mempot[-1]
                high_rate_diff_medians = prop_high_rate_lowI - prop_high_rate_median
                # print("high rate diff medians " + str(high_rate_diff_medians))
            if ind_exp == 0 and i == 2:
                small_high_rate_lowI = b
                small_high_rate_median = a  # vec_median_mempot[-1]
                ref_diff_medians = small_high_rate_lowI - small_high_rate_median
                # print("ref diff medians " + str(ref_diff_medians))
    
            if i == 2:
                if high_rate_diff_medians < ref_diff_medians:
                    cutoff_freq_lowI = mean_rate[0]
                    stop_condition = True
                    break
            # """

            # Filtering membrane potential, lowpass for getting the sinusoidal trend, high pass for the variance without
            # seasonality
            coff = 1
            low_pass_mempot = lowpass(lif.membrane_potential[0, :], coff, sfreq)
            high_pass_mempot = highpass(lif.membrane_potential[0, :], coff, sfreq)

            #
            pos_mempot = None
            mem_pot_low_filt = None
            mem_pot_high_filt = None
            neg_mempot = lif.membrane_potential[0, int(5 / dt): int(10 / dt)]
            pos_mempot_low_filt = None
            neg_mempot_low_filt = low_pass_mempot[int(5 / dt): int(10 / dt)]
            pos_mempot_high_filt = None
            neg_mempot_high_filt = high_pass_mempot[int(5 / dt): int(10 / dt)]

            if max_t % 2 == 0:
                pos_mempot = lif.membrane_potential[0, int(1 / dt): int(5 / dt)]
                pos_mempot_low_filt = low_pass_mempot[int(1 / dt): int(5 / dt)]
                pos_mempot_high_filt = high_pass_mempot[int(1 / dt): int(5 / dt)]
                mem_pot_low_filt = low_pass_mempot[: int(10 / dt)]
                mem_pot_high_filt = high_pass_mempot[: int(10 / dt)]
            else:
                pos_mempot = lif.membrane_potential[0, int(10 / dt): int(15 / dt)]
                pos_mempot_low_filt = low_pass_mempot[int(10 / dt): int(15 / dt)]
                pos_mempot_high_filt = high_pass_mempot[int(10 / dt): int(15 / dt)]
                mem_pot_low_filt = low_pass_mempot[int(5 / dt): int(15 / dt)]
                mem_pot_high_filt = high_pass_mempot[int(5 / dt): int(15 / dt)]

            # Saving statistics
            vec_max_mp_pos.append(np.max(pos_mempot_low_filt))  # pos_mempot_low_filt
            vec_min_mp_neg.append(np.min(neg_mempot_low_filt))  # neg_mempot_low_filt
            vec_q1_mp_pos.append(np.quantile(pos_mempot_high_filt, 0.1))  # pos_mempot_high_filt
            vec_q90_mp_pos.append(np.quantile(pos_mempot_high_filt, 0.9))  # pos_mempot_high_filt
            vec_q1_mp_neg.append(np.quantile(neg_mempot_high_filt, 0.1))  # neg_mempot_high_filt
            vec_q90_mp_neg.append(np.quantile(neg_mempot_high_filt, 0.9))  # neg_mempot_high_filt
            vec_max_mp.append(np.max(mem_pot_low_filt))
            vec_min_mp.append(np.min(mem_pot_low_filt))
            vec_q1_mp.append(np.quantile(mem_pot_high_filt, 0.1))  # neg_mempot_high_filt
            vec_q90_mp.append(np.quantile(mem_pot_high_filt, 0.9))  # neg_mempot_high_filt

            # Plots
            if plots_net:
                # Organising signals for plotting
                plot_mod1 = modulation_signal1
                plot_mod2 = modulation_signal2
                plot_s1 = modulated_signal1[0, :]
                plot_s2 = modulated_signal2[0, :]
                plot_SD1 = mssm.get_output()[0, :]
                plot_SD2 = mssm.get_output()[100, :]
                legend_1 = legend1
                legend_2 = legend2

                if i == 1:
                    plot_mod1 = modulation_signal2
                    plot_mod2 = modulation_signal1
                    plot_s1 = modulated_signal2[0, :]
                    plot_s2 = modulated_signal1[0, :]
                    plot_SD1 = mssm.get_output()[100, :]
                    plot_SD2 = mssm.get_output()[0, :]
                    legend_1 = legend2
                    legend_2 = legend1

                ax1 = fig.add_subplot(3, 3, (i * 3) + 1)
                ax1.plot(time_vector, 1.1 + plot_s1, label=legend_1, alpha=0.8)
                ax1.plot(time_vector, plot_s2, label=legend_2, alpha=0.8)
                ax1.grid()
                ax1.legend()
                ax1.get_yaxis().set_visible(False)
                if i == 0: ax1.set_title("Input", c='gray')

                ax2 = fig.add_subplot(3, 3, (i * 3) + 2)
                ax2.plot(time_vector, np.mean(mssm.get_output()[:100, :], axis=0), alpha=0.8)
                ax2.plot(time_vector, np.mean(mssm.get_output()[100:, :], axis=0), alpha=0.8)
                ax2.grid()
                if i == 0: ax2.set_title(r'Sine of $\mu$ %d and A %d' % (mean_rate[i], max_oscil[i]), c='gray')

                ax3 = fig.add_subplot(3, 3, (i * 3) + 3)
                ax3.plot(time_vector, lif.membrane_potential[0, :], c='black', alpha=0.8)
                ax3.plot(time_vector, lowpass(lif.membrane_potential[0, :], coff, sfreq), c='tab:red', alpha=0.8)
                ax3.grid()
                # ax3.set_ylim(-65.5, -62)
                ax3.set_ylabel("mV")
                if i == 0: ax3.set_title("Membrane potential LIF", c='gray')

                col = "#71BE56"
                if i == 1: col = "#0192C8"
                ax7 = fig3.add_subplot(3, 2, (i * 2) + 1)
                ax7.plot(time_vector, plot_mod1, c="#71BE56")
                ax7.set_ylabel("Rate (Hz)")
                if i == 0 or i == 2: ax7.grid()
                ax7.set_ylim(10, 850)
                ax7.set_title(r'%dsin(0.2$\pi$t) + %d' % (mean_rate[i], max_oscil[i]), c=col)
                if i == 2: ax7.set_ylim(mean_rate[0] - 20, mean_rate[0] + 20)
                ax7.tick_params(axis='y', labelcolor="#71BE56")
                ax7b = ax7.twinx()
                ax7b.plot(time_vector, plot_mod2, c="#0192C8")
                if i == 1: ax7b.grid()
                ax7b.tick_params(axis='y', labelcolor="#0192C8")
                ax7b.set_ylim(0, 50)

                ax8 = fig3.add_subplot(3, 2, (i * 2) + 2)
                ax8.plot(time_vector, lif.membrane_potential[0, :], c='black', alpha=0.8)
                ax8.plot(time_vector, lowpass(lif.membrane_potential[0, :], coff, sfreq), c='tab:red', alpha=0.8)
                ax8.grid()
                # ax8.set_ylim(-65.5, -60)
                ax8.set_ylabel("mV")
                # ax8.set_title(f"LIF, diff median {diff_median_mempot:.3f}mV", c='gray')

                if i == 0:
                    input_mod1_esann = np.copy(plot_mod1)
                    input_mod2_esann = np.copy(plot_mod2)
                    output_mp_esann = np.copy(lif.membrane_potential[0, :])
                    output_mp_low_filt_esann = np.copy(lowpass(lif.membrane_potential[0, :], coff, sfreq))
                    output_mp_high_filt_esann = np.copy(highpass(lif.membrane_potential[0, :], coff, sfreq))

                    if mean_rate[0] == 100:
                        fonts = 12
                        fig_esann3 = plt.figure(figsize=(11, 3))
                        ax1s3 = fig_esann3.add_subplot(211)
                        ax1s3.plot(time_vector, input_mod1_esann, label="high-firing rates", c="#000000")  # , c="#71BE56"
                        if ind_exp == 0: ax1s3.set_ylabel("Rate (Hz)", fontsize=fonts)
                        ax1s3.set_ylim(0, 160)
                        ax1s3.set_title('Sinusoidal pattern for proportional change of firing rate at baseline rate %dHz' % mean_rate[0], fontsize=fonts + 6, c="gray")
                        ax1s3.plot(time_vector, input_mod2_esann, label="low-firing rates", c="#AFAFAF")  # , c="#0192C8"
                        # ax1s3.grid()
                        ax1s3.set_ylabel("Rate (Hz)", fontsize=fonts - 1)
                        ax1s3.yaxis.set_tick_params(labelsize=fonts)
                        ax1s3.set_xlabel("Time  (s)", fontsize=fonts)
                        ax1s3.xaxis.set_tick_params(labelsize=fonts)
                        ax1s3.legend(framealpha=0.3)
                        ax2s3 = fig_esann3.add_subplot(234)
                        ax2s3.plot(time_vector[:int(0.5 / dt)], plot_s1[:int(0.5 / dt)] + 1.1, c="#000000")  # , c="#71BE56")
                        ax2s3.plot(time_vector[:int(0.5 / dt)], plot_s2[:int(0.5 / dt)], c="#AFAFAF")  # , c="#0192C8")
                        ax2s3.set_xlabel("Time (s)", fontsize=fonts)
                        ax2s3.xaxis.set_tick_params(labelsize=fonts)
                        # ax2s3.grid()
                        ax2s3.get_yaxis().set_visible(False)
                        ax2s3.set_title("Firing rates around 100Hz", c="gray", fontsize=fonts + 4)
                        ax3s3 = fig_esann3.add_subplot(235)
                        ax3s3.plot(time_vector[int(7 / dt):int(7.5 / dt)], plot_s1[int(7 / dt):int(7.5 / dt)] + 1.1, c="#000000")  # , c="#71BE56")
                        ax3s3.plot(time_vector[int(7 / dt):int(7.5 / dt)], plot_s2[int(7 / dt):int(7.5 / dt)], c="#AFAFAF")  # , c="#0192C8")
                        ax3s3.set_xlabel("Time (s)", fontsize=fonts)
                        ax3s3.xaxis.set_tick_params(labelsize=fonts)
                        # ax3s3.grid()
                        ax3s3.get_yaxis().set_visible(False)
                        ax3s3.set_title("Firing rates around 50Hz", c="gray", fontsize=fonts + 4)
                        ax4s3 = fig_esann3.add_subplot(236)
                        ax4s3.plot(time_vector[int(12 / dt):int(12.5 / dt)], plot_s1[int(12 / dt):int(12.5 / dt)] + 1.1, c="#000000")  # , c="#71BE56")
                        ax4s3.plot(time_vector[int(12 / dt):int(12.5 / dt)], plot_s2[int(12 / dt):int(12.5 / dt)], c="#AFAFAF")  # , c="#0192C8")
                        ax4s3.set_xlabel("Time (s)", fontsize=fonts)
                        ax4s3.xaxis.set_tick_params(labelsize=fonts)
                        # ax4s3.grid()
                        ax4s3.get_yaxis().set_visible(False)
                        ax4s3.set_title("Firing rates around 150Hz", c="gray", fontsize=fonts + 4)
                        fig_esann3.tight_layout(pad=0.5, w_pad=1.0, h_pad=0.1)


                ax9 = fig4.add_subplot(1, 3, i + 1)
                ax9.plot(time_vector, 1.1 + plot_s1, alpha=0.8, c="#000000")  # , c="#71BE56")
                ax9.plot(time_vector, plot_s2, alpha=0.8, c="#AFAFAF")  # , c="#0192C8")
                ax9.set_xlabel("time (s)", fontsize=18)
                ax9.set_title("Example input for group " + str(i + 1), fontsize=18, c="gray")
                # plt.xticks(fontsize=15)
                # plt.yticks(fontsize=15, c="white")
                ax9.grid()

                if i == 2:
                    ax1.set_xlabel("time (s)")
                    ax2.set_xlabel("time (s)")
                    ax3.set_xlabel("time (s)")
                    ax7.set_xlabel("time (s)") # , fontsize=18)
                    ax8.set_xlabel("time (s)")

        """
        # Figure for ESANN paper, 2026
        if ind == 4:
            ax1_s = fig_esann.add_subplot(2, 4, ind_exp + 1)
            ax1_s.plot(time_vector, input_mod1_esann, c="#71BE56")
            if ind_exp == 0: ax1_s.set_ylabel("Rate (Hz)")
            ax1_s.set_ylim(0, 760)
            # ax1_s.set_title(r'%dsin(0.2$\pi$t) + %d' % (mean_rate[0], max_oscil[0]), c='gray')
            ax1_s.set_title('baseline rate %dHz' % mean_rate[0], c='gray')
            # ax1_s.tick_params(axis='y', labelcolor="#71BE56")
            ax1_s.plot(time_vector, input_mod2_esann, c="#0192C8")
            ax1_s.grid()
            ax2_s = fig_esann.add_subplot(2, 4, cond_while + ind_exp + 1)
        if ind == 5 or ind == 2:
            ax2_s = fig_esann.add_subplot(1, 4, ind_exp + 1)

        ax2_s.plot(time_vector, output_mp_esann, c="black")
        if ind_exp == 0: ax2_s.set_ylabel("mem. pot. (mv)")
        ax2_s.set_ylim(out_ylim_min, out_ylim_max)
        # ax2_s.set_title('membrane potential', c='gray')
        ax2_s.plot(time_vector, output_mp_filt_esann, c="tab:red")
        ax2_s.grid()
        if ind == 5: ax2_s.set_xlabel("time (s)")
        fig_esann.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
        # """
        # Figure for ESANN paper, 2026
        fonts = 12
        ax1_s = fig_esann.add_subplot(1, 4, ind_exp + 1)
        ax1_s.plot(time_vector, output_mp_esann, c="#AFAFAF")
        if ind_exp == 0: ax1_s.set_ylabel("mem. pot. (mv)")
        ax1_s.yaxis.set_tick_params(labelsize=fonts - 2)
        ax1_s.set_ylim(out_ylim_min, out_ylim_max)
        # ax1_s.set_title(r'%dsin(0.2$\pi$t) + %d' % (mean_rate[0], max_oscil[0]), c='gray')
        if ind == 4: ax1_s.set_title('For baseline rate %dHz' % mean_rate[0], c='gray')
        if ind == 5: ax1_s.set_xlabel("time (s)")
        # ax1_s.tick_params(axis='y', labelcolor="#71BE56")
        ax1_s.plot(time_vector, output_mp_low_filt_esann, c="black")
        ax1_s.xaxis.set_tick_params(labelsize=fonts - 2)
        # ax1_s.grid()

        # ax2_s = fig_esann.add_subplot(2, 4, cond_while + ind_exp + 1)
        # aux_s = output_mp_high_filt_esann[int(5/dt):int(max_t/dt)]
        # ax2_s.fill_between([5, max_t], np.quantile(aux_s, 0.1), np.quantile(aux_s, 0.9), color="#CFCFCF")
        # if ind_exp == 0: ax2_s.set_ylabel("mem. pot. (mv)")
        # ax2_s.yaxis.set_tick_params(labelsize=fonts - 2)
        # ax2_s.set_ylim(-1.2, 1.2)
        # aux_s = output_mp_low_filt_esann - np.mean(lif.membrane_potential[0, int(5/dt):])
        # ax2_s.plot(time_vector[int(5/dt):int(max_t/dt)], aux_s[int(5/dt):int(max_t/dt)], c="black")
        # if ind == 5: ax2_s.set_xlabel("time (s)")
        # ax2_s.xaxis.set_tick_params(labelsize=fonts - 2)
        fig_esann.tight_layout(pad=0.1, w_pad=0.5, h_pad=0.3)


        time_desc = (f'[%dsin(0.2pit) + %d, %d], [%d, %dsin(0.2pit) + %d], [%dsin(0.2pit) + %d, %d]' %
                     (max_oscil[0], mean_rate[0], fix_rate[0], fix_rate[1], max_oscil[1], mean_rate[1],
                      max_oscil[2], mean_rate[2], fix_rate[2]))
        print_time(m_time() - ini_loop_time, "Experiment " + str(ind_exp) + ":" + time_desc)

        if plots_net:
            fig.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
            fig3.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
            fig4.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)

            plt.close(fig4)

        # if stop_condition:
        #     ind_exp  = len(mean_rates)

        ind_exp += 1

    L_freqs = int(len(vec_max_mp_pos) / 3)
    ind_a = [i for i in range(0, L_freqs * 3, 3)]  # prop change high-rate
    ind_b = [i for i in range(1, L_freqs * 3, 3)]  # prop change low-rate
    ind_c = [i for i in range(2, L_freqs * 3, 3)]  # small change high-rate

    # Statistics for proportional changes of high-firing rates
    prop_high_rate_max_pos = np.array(vec_max_mp_pos)[ind_a]
    prop_high_rate_min_neg = np.array(vec_min_mp_neg)[ind_a]
    prop_high_rate_q1_pos = np.array(vec_q1_mp_pos)[ind_a]
    prop_high_rate_q90_pos = np.array(vec_q90_mp_pos)[ind_a]
    prop_high_rate_q1_neg = np.array(vec_q1_mp_neg)[ind_a]
    prop_high_rate_q90_neg = np.array(vec_q90_mp_neg)[ind_a]
    prop_high_freq_vector = np.array(mean_rates)[:L_freqs, 0]
    prop_high_rate_amplitude = prop_high_rate_max_pos - prop_high_rate_min_neg
    prop_high_rate_var_pos = prop_high_rate_q90_pos - prop_high_rate_q1_pos
    prop_high_rate_var_neg = prop_high_rate_q90_neg - prop_high_rate_q1_neg
    prop_high_rate_max = np.array(vec_max_mp)[ind_a]
    prop_high_rate_min = np.array(vec_min_mp)[ind_a]
    prop_high_rate_q1 = np.array(vec_q1_mp)[ind_a]
    prop_high_rate_q90 = np.array(vec_q90_mp)[ind_a]
    prop_high_rate_amplitude2 = prop_high_rate_max - prop_high_rate_min
    prop_high_rate_var = prop_high_rate_q90 - prop_high_rate_q1

    # Statistics for proportional changes of low-firing rates
    prop_low_rate_max_pos = np.array(vec_max_mp_pos)[ind_b]
    prop_low_rate_min_neg = np.array(vec_min_mp_neg)[ind_b]
    prop_low_rate_q1_pos = np.array(vec_q1_mp_pos)[ind_b]
    prop_low_rate_q90_pos = np.array(vec_q90_mp_pos)[ind_b]
    prop_low_rate_q1_neg = np.array(vec_q1_mp_neg)[ind_b]
    prop_low_rate_q90_neg = np.array(vec_q90_mp_neg)[ind_b]
    prop_low_freq_vector = np.array(mean_rates)[:L_freqs, 1]
    prop_low_rate_amplitude = prop_low_rate_max_pos - prop_low_rate_min_neg
    prop_low_rate_var_pos = prop_low_rate_q90_pos - prop_low_rate_q1_pos
    prop_low_rate_var_neg = prop_low_rate_q90_neg - prop_low_rate_q1_neg
    prop_low_rate_max = np.array(vec_max_mp)[ind_b]
    prop_low_rate_min = np.array(vec_min_mp)[ind_b]
    prop_low_rate_q1 = np.array(vec_q1_mp)[ind_b]
    prop_low_rate_q90 = np.array(vec_q90_mp)[ind_b]
    prop_low_rate_amplitude2 = prop_low_rate_max - prop_low_rate_min
    prop_low_rate_var = prop_low_rate_q90 - prop_low_rate_q1

    # Statistics for small changes of high-firing rates
    small_high_rate_max_pos = np.array(vec_max_mp_pos)[ind_c]
    small_high_rate_min_neg = np.array(vec_min_mp_neg)[ind_c]
    small_high_rate_q1_pos = np.array(vec_q1_mp_pos)[ind_c]
    small_high_rate_q90_pos = np.array(vec_q90_mp_pos)[ind_c]
    small_high_rate_q1_neg = np.array(vec_q1_mp_neg)[ind_c]
    small_high_rate_q90_neg = np.array(vec_q90_mp_neg)[ind_c]
    small_high_freq_vector = np.array(mean_rates)[:L_freqs, 2]
    small_high_rate_amplitude = small_high_rate_max_pos - small_high_rate_min_neg
    small_high_rate_var_pos = small_high_rate_q90_pos - small_high_rate_q1_pos
    small_high_rate_var_neg = small_high_rate_q90_neg - small_high_rate_q1_neg
    small_high_rate_max = np.array(vec_max_mp)[ind_c]
    small_high_rate_min = np.array(vec_min_mp)[ind_c]
    small_high_rate_q1 = np.array(vec_q1_mp)[ind_c]
    small_high_rate_q90 = np.array(vec_q90_mp)[ind_c]
    small_high_rate_amplitude2 = small_high_rate_max - small_high_rate_min
    small_high_rate_var = small_high_rate_q90 - small_high_rate_q1

    # For experiment 4
    # prop_high_rate_max_pos = array([-62.58562452, -62.21529441, -61.93138295, -61.72693519]) -61.5
    # prop_high_rate_min_neg = array([-63.69040866, -63.1157654 , -62.38389251, -62.23640865]) -64

    # For experiment 5
    # prop_high_rate_max_pos = array([-58.23917046, -55.58041238, -52.93823224, -52.23514712]) - 52
    # prop_high_rate_min_neg = array([-63.17012234, -60.16772678, -55.65855235, -54.09690444]) -63.5

    fig = plt.figure(figsize=(10, 3))
    fig.suptitle(description)
    ax11 = fig.add_subplot(131)
    ax12 = fig.add_subplot(132)
    ax13 = fig.add_subplot(133)
    ax11.plot(prop_high_freq_vector, prop_high_rate_amplitude, c="black")
    ax11.fill_between(prop_high_freq_vector, 0, prop_high_rate_var_pos, color="tab:red", alpha=0.3)
    ax11.fill_between(prop_high_freq_vector, 0, prop_high_rate_var_neg, color="tab:blue", alpha=0.3)
    ax11.grid()
    ax11.set_title("High-rate Proportional", c="gray")
    ax11.set_ylabel("mV")
    ax11.set_xlabel("Frequency (Hz)")
    # ax11.set_ylim(-66.7, -59.5)

    ax12.plot(prop_low_freq_vector, prop_low_rate_amplitude, c="black")
    ax12.fill_between(prop_low_freq_vector, 0, prop_low_rate_var_pos, color="tab:red", alpha=0.3)
    ax12.fill_between(prop_low_freq_vector, 0, prop_low_rate_var_neg, color="tab:blue", alpha=0.3)
    ax12.grid()
    ax12.set_ylabel("mV")
    ax12.set_xlabel("Frequency (Hz)")
    ax12.set_title("Low-rate Proportional", c="gray")
    # ax12.set_ylim(-66.7, -59.5)

    ax13.plot(small_high_freq_vector, small_high_rate_amplitude, c="black")
    ax13.fill_between(small_high_freq_vector, 0, small_high_rate_var_pos, color="tab:red", alpha=0.3)
    ax13.fill_between(small_high_freq_vector, 0, small_high_rate_var_neg, color="tab:blue", alpha=0.3)
    ax13.grid()
    ax13.set_ylabel("mV")
    ax13.set_xlabel("Frequency (Hz)")
    ax13.set_title("High-rate small variation", c="gray")
    # ax13.set_ylim(-66.7, -59.5)

    fig.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)

"""
L_freqs = int(len(vec_median_mempot_lowI) / 3)
ind_a = [i for i in range(0, L_freqs * 3, 3)]  # median_mempot
ind_b = [i for i in range(1, L_freqs * 3, 3)]  # median_highI
ind_c = [i for i in range(2, L_freqs * 3, 3)]  # median_lowI

prop_high_rate_median = np.array(vec_mean_mempot)[ind_a]
prop_high_rate_highI = np.array(vec_mean_mempot_highI)[ind_a]
prop_high_rate_lowI = np.array(vec_mean_mempot_lowI)[ind_a]
prop_high_freq_vector = np.array(mean_rates)[:L_freqs, 0]

prop_low_rate_median = np.array(vec_mean_mempot)[ind_b]
prop_low_rate_highI = np.array(vec_mean_mempot_highI)[ind_b]
prop_low_rate_lowI = np.array(vec_mean_mempot_lowI)[ind_b]
prop_low_freq_vector = np.array(mean_rates)[:L_freqs, 1]

small_high_rate_median = np.array(vec_mean_mempot)[ind_c]
small_high_rate_highI = np.array(vec_mean_mempot_highI)[ind_c]
small_high_rate_lowI = np.array(vec_mean_mempot_lowI)[ind_c]
small_high_freq_vector = np.array(mean_rates)[:L_freqs, 2]

fig = plt.figure(figsize=(10, 3))
fig.suptitle(description)
ax11 = fig.add_subplot(131)
ax12 = fig.add_subplot(132)
ax13 = fig.add_subplot(133)
ax11.plot(prop_high_freq_vector, prop_high_rate_median, c="black")
ax11.fill_between(prop_high_freq_vector, prop_high_rate_highI, prop_high_rate_lowI, color="lightgray", alpha=0.8)
# ax11.set_ylim(-66.7, -59.5)
ax11.grid()
ax11.set_title("High-rate Proportional", c="gray")
ax12.plot(prop_low_freq_vector, prop_low_rate_median, c="black")
ax12.fill_between(prop_low_freq_vector, prop_low_rate_highI, prop_low_rate_lowI, color="lightgray", alpha=0.8)
# ax12.set_ylim(-66.7, -59.5)
ax12.grid()
ax12.set_title("Low-rate Proportional", c="gray")
ax13.plot(small_high_freq_vector, small_high_rate_median, c="black")
ax13.fill_between(small_high_freq_vector, small_high_rate_highI, small_high_rate_lowI, color="lightgray", alpha=0.8)
# ax13.set_ylim(-66.7, -59.5)
ax13.grid()
ax13.set_title("High-rate small variation", c="gray")
fig.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)

fig2 = plt.figure(figsize=(10, 3))
ax21 = fig2.add_subplot(131)
sign1 = small_high_rate_median - small_high_rate_highI
ax21.plot(small_high_freq_vector, sign1, c="tab:blue", alpha=0.8, label="small high freq")
ax21.fill_between(small_high_freq_vector, np.min(sign1), np.max(sign1), color="tab:blue", alpha=0.2)
sign = prop_high_rate_median - prop_high_rate_highI
ax21.plot(prop_high_freq_vector, sign, c="tab:red", alpha=0.8, label="prop high freq")
ax21.grid()
ax21.legend()
ax21.set_title("Positive wave of sin", c='gray')
cutoff_freq_highI = small_high_freq_vector[np.where(sign < sign1[0])[0][0]]

ax22 = fig2.add_subplot(132)
sign1 = small_high_rate_lowI - small_high_rate_median
ax22.plot(small_high_freq_vector, sign1, c="tab:blue", alpha=0.8, label="small high freq")
ax22.fill_between(small_high_freq_vector, np.min(sign1), np.max(sign1), color="tab:blue", alpha=0.2)
sign = prop_high_rate_lowI - prop_high_rate_median
ax22.plot(prop_high_freq_vector, sign, c="tab:red", alpha=0.8, label="prop high freq")
ax22.grid()
ax22.legend()
ax22.set_title("Negative wave of sin", c='gray')
cutoff_freq_lowI = small_high_freq_vector[np.where(sign < sign1[0])[0][0]]

ax23 = fig2.add_subplot(133)
sign1 = small_high_rate_lowI - small_high_rate_highI
ax23.plot(small_high_freq_vector, sign1, c="tab:blue", alpha=0.8, label="small rates")
ax23.fill_between(small_high_freq_vector, np.min(sign1), np.max(sign1), color="tab:blue", alpha=0.2)
sign = prop_high_rate_lowI - prop_high_rate_highI
ax23.plot(prop_high_freq_vector, sign, c="tab:red", alpha=0.8, label="prop rates")
ax23.grid()
ax23.legend()
ax23.set_title("Difference of medians (lowI - highI)", c='gray')
cutoff_freq_diff = small_high_freq_vector[np.where(sign < sign1[0])[0][0]]

fig2.suptitle(description + " highI %d lowI %d diff_med %d" % (cutoff_freq_highI, cutoff_freq_lowI, cutoff_freq_diff))
fig2.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
# """

# print("Cutoff frequency in highI %d, Cutoff frequency in lowI %d, and difference of medians %d" %
#       (cutoff_freq_highI, cutoff_freq_lowI, cutoff_freq_diff))
# ******************************************************************************************************************
# """
# FREQUENCY ANALYSIS
ini_loop_time = m_time()
if freq_analysis:
    fa = Freq_analysis(max_t=max_t, end_t=max_imp, sfreq=sfreq, loop_f=loop_frequencies, n_syn=num_syn)  # loop_frequencies
    fa.set_model(model_str="MSSM", sim_params=sim_params, name_params=list(params.keys()),
                 model_params=list(params.values()))
    fa.run()
    plot_freq_analysis(fa, " MSSM a")
    print_time(m_time() - ini_loop_time, "Time for frequency analysis")

    netmem = fa.efficacy[0, :] * loop_frequencies
    inv_freq = 1 / np.array(loop_frequencies)
    plt.figure()
    plt.plot(loop_frequencies, netmem)  #  * 60e-3)
    plt.grid()
    plt.xlabel("f (Hz)")
    plt.ylabel("Net depolarization (mV)")
    plt.title("EPSP_st * 1/f", c='gray')

    plt.figure()
    fac_mul = 8.7e-2
    plt.plot(loop_frequencies, fa.efficacy_2[0, :] * fac_mul, label='EPSP_st')
    plt.plot(loop_frequencies, inv_freq, "--", label="1/f", c='orange')
    plt.grid()
    plt.xlabel("f (Hz)")
    plt.ylabel("EPSPst * " + str(fac_mul))
    plt.ylim((0, 40e-3))
    plt.title("EPSP_st", c='gray')
    plt.legend()

    fig_esann2 = plt.figure(figsize=(5, 2))
    fonts = 12
    ax1s2 = fig_esann2.add_subplot(1, 1, 1)
    ax1s2.plot(loop_frequencies, fa.efficacy[0, :], color="black")
    # ax1s2.set_xscale('log')
    # ax1s2.grid()
    ax1s2.set_xlabel("Frequency (Hz)", fontsize=fonts)
    ax1s2.xaxis.set_tick_params(labelsize=fonts)
    ax1s2.set_ylabel("Current (pA)", fontsize=fonts)
    ax1s2.yaxis.set_tick_params(labelsize=fonts)
    ax1s2.set_title("Efficacy for fast-decay synapse", fontsize=fonts + 4)
    range_eff = np.max(fa.efficacy[0]) - np.min(fa.efficacy[0])
    ind_eff = np.where(fa.efficacy[0] < (0.01 * range_eff) + np.min(fa.efficacy[0]))
    freq_st = loop_frequencies[ind_eff[0][0]]
    ax1s2.plot([freq_st, freq_st], [np.min(fa.efficacy[0]), np.max(fa.efficacy[0])], color="#AFAFAF")
    empty_patch = mpatches.Patch(color='none', label=r'$freq_{st}=$%dHz' % freq_st)
    ax1s2.legend(handles=[empty_patch], loc='upper right', fontsize=fonts)
    fig_esann2.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)



# """

# ******************************************************************************************************************
# FIGURES

"""
plot_title = "Simulation MSSM"
subtitle_size = 12
subtitle_color = 'gray'
org_plots = 330
ind_plots = [org_plots + 1, org_plots + 2, org_plots + 3, org_plots + 4, org_plots + 5, org_plots + 6, org_plots + 7]
time_vectors = [time_vector for _ in range(len(ind_plots))]
plots = [Input[0, :], np.mean(mssm.C, axis=0), np.mean(mssm.V, axis=0), np.mean(mssm.P, axis=0),
         np.mean(mssm.N, axis=0), np.mean(mssm.EPSP, axis=0), np.mean(lif.membrane_potential, axis=0)]
subplot_title = ["Input of MSSM", "Calcium", "Vesicles", "Probability", "Neurotransmitters", "EPSP", 
                 "membrane potential"]
ylabels = [None, None, None, None, None, None, None]
xlabels = ['time (ms)', 'time (ms)', 'time (ms)', 'time (ms)', 'time (ms)', 'time (ms)', 'time (ms)']
plot_syn_dyn(time_vectors, plot_title, ind_plots, plots, subplot_title=subplot_title, xlabels=xlabels, ylabels=ylabels,
             plot=True)
# """


"""
freq_fac_vec = [10, 15, 22, 28, 30]
freq_dep_vec = [10, 15, 22, 28, 80]
eff_fac_vec = [4.15 / 8.7,  4.4 / 3.7,  4.6 / 3.7,  4.55 / 3.7,  4.5 / 3.7]
eff_dep_vec = [2.03 / 7.95, 1.7 / 7.95, 1.4 / 7.95, 1.1 / 7.95, 0.4 / 7.95]
aux_path = '../../../../PhD Leipzig/Uni Leipzig/Kolloquium Uni/Kolloquium Präsentation - 2025-07-01'
fontsize_axis = 18
fontsize_title = 24

for i in range(1, 6):
    plt.figure(figsize=(7.5, 5))
    plt.plot(freq_dep_vec[:i], eff_dep_vec[:i], c='black')
    plt.scatter(freq_dep_vec[:i], eff_dep_vec[:i], c='gray')
    plt.scatter(freq_dep_vec[i - 1], eff_dep_vec[i - 1], c='red')
    plt.grid()
    plt.xlabel("Frequency (Hz)", fontsize=fontsize_axis)
    plt.ylabel("Efficacy", fontsize=fontsize_axis)
    plt.xlim(8, 82)
    plt.ylim(0, 0.27)
    plt.title("Frequency response of depressive synapse", fontsize=fontsize_title)
    plt.savefig(aux_path + '/freq_res_dep_' + str(i) + ".png", format='png')
    plt.close()

    plt.figure(figsize=(7.5, 5))
    plt.plot(freq_fac_vec[:i], eff_fac_vec[:i], c='black')
    plt.scatter(freq_fac_vec[:i], eff_fac_vec[:i], c='gray')
    plt.scatter(freq_fac_vec[i - 1], eff_fac_vec[i - 1], c='red')
    plt.grid()
    plt.xlabel("Frequency (Hz)", fontsize=fontsize_axis)
    plt.ylabel("Efficacy", fontsize=fontsize_axis)
    plt.xlim(8, 32)
    plt.ylim(0, 1.3)
    plt.title("Frequency response of facilitating synapse", fontsize=fontsize_title)
    plt.savefig(aux_path + '/freq_res_fac_' + str(i) + ".png", format='png')
    plt.close()
# """