import os.path
# import matplotlib.pyplot as plt
import scipy.signal
# import numpy as np

from synaptic_dynamic_models.TM import TM_model
from synaptic_dynamic_models.MSSM import MSSM_model
from synaptic_dynamic_models.DoornSynSTD import DoornSTD_model
from synaptic_dynamic_models.DoornSynSTF import DoornSTF_model
from synaptic_dynamic_models.DoornSynAsyn import DoornAsyn_model
from spiking_neuron_models.LIF import LIF_model
from spiking_neuron_models.HH_doorn import HH_AHP_model
from spiking_neuron_models.HH_simplified_Doorn import HH_Simple_model
from synaptic_dynamic_models.simple_depression import Simple_Depression
from libraries.frequency_analysis import Freq_analysis
from libraries.SlidingWindowTransitoryAnalyser import SlidingWindowTransitoryAnalyser
from utils import *

# ******************************************************************************************************************
# Local variables
stat_list = ['st_ini_prop_mean', 'st_ini_prop_med', 'st_ini_prop_q5', 'st_ini_prop_q10',
             'st_ini_prop_q90', 'st_ini_prop_q95', 'st_ini_prop_min', 'st_ini_prop_max',
             'st_mid_prop_mean', 'st_mid_prop_med', 'st_mid_prop_q5', 'st_mid_prop_q10',
             'st_mid_prop_q90', 'st_mid_prop_q95', 'st_mid_prop_min', 'st_mid_prop_max',
             'st_end_prop_mean', 'st_end_prop_med', 'st_end_prop_q5', 'st_end_prop_q10',
             'st_end_prop_q90', 'st_end_prop_q95', 'st_end_prop_min', 'st_end_prop_max',
             'st_ini_fix_mean', 'st_ini_fix_med', 'st_ini_fix_q5', 'st_ini_fix_q10',
             'st_ini_fix_q90', 'st_ini_fix_q95', 'st_ini_fix_min', 'st_ini_fix_max',
             'st_mid_fix_mean', 'st_mid_fix_med', 'st_mid_fix_q5', 'st_mid_fix_q10',
             'st_mid_fix_q90', 'st_mid_fix_q95', 'st_mid_fix_min', 'st_mid_fix_max',
             'st_end_fix_mean', 'st_end_fix_med', 'st_end_fix_q5', 'st_end_fix_q10',
             'st_end_fix_q90', 'st_end_fix_q95', 'st_end_fix_min', 'st_end_fix_max',
             'mtr_ini_prop_max', 'mtr_mid_prop_max', 'mtr_end_prop_max',
             'mtr_ini_prop_min', 'mtr_mid_prop_min', 'mtr_end_prop_min',
             'initial_frequencies', 'stp_model', 'name_params', 'dyn_synapse', 'num_synapses', 'syn_params',
             'sim_params', 'lif_params', 'lif_params2', 'prop_rate_change_a', 'fix_rate_change_a', 'num_changes_rate',
             'description', 'seeds', 'realizations', 't_realizations', 'time_transition']
stat_list_sin = ['vec_max_mp_pos', 'vec_min_mp_neg', 'vec_q1_mp_pos', 'vec_q90_mp_pos',
                 'vec_q1_mp_neg', 'vec_q90_mp_neg', 'vec_max_mp', 'vec_min_mp', 'vec_q1_mp', 'vec_q90_mp']


# Parameters for neuron
def get_neuron_params(n_model, tau_m, ind, y_lim_ind_plot=False, num_syn=1, num_neurons=1):
    y_lim_memPot = None
    n_params = None
    n = num_neurons
    n_params = None
    if n_model == "LIF":
        if y_lim_ind_plot:
            y_lim_memPot = [-70, -50]
            if tau_m == 1 and num_syn == 100: y_lim_memPot = [-70, -43]
            if tau_m == 30 and num_syn == 100: y_lim_memPot = [-65.7, -52.5]
            if tau_m == 1 and num_syn == 1: y_lim_memPot = [-70.05, -67.4]
            if tau_m == 10 and num_syn == 1: y_lim_memPot = [-70.05, -69]
            if tau_m == 30 and num_syn == 1: y_lim_memPot = [-70.05, -69.5]

        n_params = {'V_threshold': np.array([1000 for _ in range(n)]), 'V_reset': np.array([-70 for _ in range(n)]),
                    'tau_m': np.array([tau_m * 1e-3 for _ in range(n)]), 'g_L': np.array([2.7e-2 for _ in range(n)]),
                    'V_init': np.array([-70 for _ in range(n)]), 'V_equilibrium': np.array([-70 for _ in range(n)]),
                    't_refractory': np.array([0.01 for _ in range(n)]), 'y_lim_plot': y_lim_memPot}
    if n_model == "HH":
        y_lim_memPot = [-40e-3, 0]
        n_params = {'Cm': np.array([6e-12 for _ in range(n)]),  # pF (from area=300 um², Cm=2 uF/cm²)
                    'g_na': np.array([240e-9 for _ in range(n)]),  # nS -> (1.6*50 mS/cm² * 300 um²)
                    'g_kd': np.array([19.5e-9 for _ in range(n)]),  # nS -> (1.3*5 mS/cm² * 300 um²)
                    'g_l': np.array([0.9e-9 for _ in range(n)]),  # nS (0.3 mS/cm² * 300 um²)
                    'El': np.array([-39.2e-3 for _ in range(n)]),  # -39.2,  # mV
                    'EK': np.array([-80.0e-3 for _ in range(n)]),  # mV
                    'ENa': np.array([70.0e-3 for _ in range(n)]),  # mV
                    'VT': np.array([-30.4e-3 for _ in range(n)]),  # mV
                    'sigma': np.array([1e-4 for _ in range(n)]),  # 6.0e-3,  # mV (noise std dev)
                    # 'sigma': np.array([8e-3 for _ in range(n)]),  # mV (noise std dev)
                    'g_AHP': np.array([5.0e-9 for _ in range(n)]),  # nS
                    'E_AHP': np.array([-80.0e-3 for _ in range(n)]),  # mV (= EK)
                    'g_ampa': np.array([1.6e-9 for _ in range(n)]),  # nS
                    'g_nmda': np.array([0.4e-9 for _ in range(n)]),  # nS
                    'E_ampa': np.array([0e-3 for _ in range(n)]),  # mV
                    'E_nmda': np.array([0e-3 for _ in range(n)]),  # mV
                    'tau_Ca': np.array([8.0e-3 for _ in range(n)]),  # ms  8000e-3
                    'alpha_Ca': np.array([0.00035 for _ in range(n)]),  # per spike
                    'V_init': np.array([-39.0e-3 for _ in range(n)]),  # mV
                    'V_reset': np.array([-39.0e-3 for _ in range(n)]),  # mV (no voltage reset, just refractory)
                    'V_threshold': np.array([0.0e-3 for _ in range(n)]),  # mV (from Brian2: V>0*mV)
                    't_refractory': np.array([2.0e-3 for _ in range(n)]),  # ms
                    'tau_m': np.array([tau_m * 1e-3 for _ in range(n)]),  # NOT USED IN THIS MODEL
                    'y_lim_plot': y_lim_memPot,  # NOT USED IN THIS MODEL
                    }
        # Missing to tune sigma  ********
        # For strong STD
        if ind == 2 or ind == 3:
            n_params['g_AHP'] = np.array([4.0e-9 for _ in range(n)])
            # n_params['sigma'] = np.array([6e-3 for _ in range(n)])
        # For low and high Asynchronous release
        if ind == 4 or ind == 5:
            n_params['g_AHP'] = np.array([8.0e-9 for _ in range(n)])
            # n_params['sigma'] = np.array([5.5e-3 for _ in range(n)])
        # For Strong NMDA currents
        if ind == 6:
            n_params['Cm'] = np.array([3e-12 for _ in range(n)]),  # pF (from area=300 um², Cm=1 uF/cm²)
            n_params['g_na'] = np.array([150e-9 for _ in range(n)]),  # nS -> (50 mS/cm² * 300 um²)
            n_params['g_kd'] = np.array([15e-9 for _ in range(n)]),  # nS -> (5 mS/cm² * 300 um²)
            # For delta = 0.5 -> g_ampa = 1 + delta, g_nmda = 1 - delta
            n_params['g_ampa'] = np.array([1.5e-9 for _ in range(n)]),  # nS
            n_params['g_nmda'] = np.array([0.5e-9 for _ in range(n)]),  # nS
            # n_params['sigma'] = np.array([3.5e-3 for _ in range(n)])
        # For STF
        if ind == 7:
            pass
            # n_params['sigma'] = np.array([5.5e-3 for _ in range(n)])
    assert n_params is not None, 'parameters for neuron model %s and index %d not found' % (n_model, ind)
    return n_params


def get_params_stp(name_model, ind):
    syn_params = None
    description = ""
    name_params = None

    if name_model == "MSSM":
        name_params = params_name_mssm
    elif name_model == "TM":
        name_params = params_name_tm
    elif name_model == "DoornSTD" or name_model == "DoornAsyn" or name_model == "DoornSTF":
        name_params = params_name_doorn

    # (Experiment 2) freq. response decay around 100Hz {5: 0.99, 10: 0.9, 20: 0.8, 50: 0.7, 100: 0.5, 200: 0.2}
    if name_model == "MSSM" and ind == 2:
        description = "MSSM " + str(ind) + " Experiment, decay around 100Hz"
        syn_params = [5.42451963e-02, 2.92925980e+00, 6.67821125e+01, 1.80143000e-02, 7.54167519e-01,
                      5.99119322e+01, 9.94215228e-01, 1.03825167e-03,
                      (4.52507712e-01 * 0.075) / 779.1984,
                      1.00243185e-03]  # 4.52507712e-01, g_L = 5e-6 homogeneous / 4.6e-3 poisson
    if name_model == "TM" and ind == 2:
        description = "TM " + str(ind) + " Experiment, decay around 100Hz"
        syn_params = [1.32748878e-01, 2.19116160e-02, 1.32465006e-01, 2.16882855e-00 * 2.60498474e-1, 1.00766466e-03]  #
    # (Experiment 3) freq. response decay around 10Hz {2: 0.889, 5: 0.7016, 10: 0.576, 15: 0.2, 20: 0.1, 50: 0.07}
    if name_model == "MSSM" and ind == 3:
        description = "MSSM " + str(ind) + " Experiment, decay around 10Hz"
        syn_params = [6.53659368e-03, 1.75660742e-01, 3.17123087e+01, 1.78659320e-01, 2.50362727e-01,
                      9.12004545e+01, 9.13420672e-01, 2.14204288e-03, 5.20907890e-03 / 4.29,
                      4.32890680e-03]  # 5.20907890e-01, g_L = 2.6e-2 homogeneous / 4.6e-2 poisson
    if name_model == "TM" and ind == 3:
        description = "TM " + str(ind) + " Experiment, decay around 10Hz"
        syn_params = [2.37698417e-01, 3.30564024e-01, 8.51177265e-01, 3.67454564e-01 * 8.58679963e-1, 3.04982285e-03]  #
    # (Experiment 4) freq. response from Gain Control paper {5: 0.785, 10: 0.6, 20: 0.415, 50: 0.205, 100: 0.115}
    if name_model == "MSSM" and ind == 4:
        description = "MSSM " + str(ind) + " Experiment, Gain-Control paper"
        syn_params = [7.85735182e-02, 4.56599128e-01, 1.46835212e+00, 1.63998958e-01, 2.41885797e-04,
                      5.84619146e+01, 8.00871281e-01, 1.50280526e-03, (5.94890729e-02 * 0.075) / 1.0217,
                      1.75609424e-03]  # 5.94890729e-01, g_L = 5.4e-4 homogeneous / 3.21e-3 poisson [1.36e-3 for subthreshold]  # g_L = 5.4e-5 for static synapse
    if name_model == "TM" and ind == 4:
        description = "TM " + str(ind) + " Experiment, Gain-Control paper"
        syn_params = [3.79643805e-01, 3.71724581e-02, 2.31713484e-01, 4.71504487e-01 * 4.18985618e-1, 3.55003518e-03]  #
    # (Experiment 5) freq. response decay around 100Hz {5: 0.99, 10: 0.9, 20: 0.8, 50: 0.7, 100: 0.5, 200: 0.2}
    if name_model == "MSSM" and ind == 5:
        description = "MSSM " + str(ind) + " Experiment, decay around 100Hz"
        syn_params = [3.03503488e-02, 1.49534243e-01, 1.27807004e+00, 8.32749189e-02,
                      1.03652329e-03, 7.06246475e+01, 9.84186889e-01, 1.00258903e-03,
                      (4.42926340e-01 * 0.075) / 2.975, 1.00046291e-03]  # g_L = 2.46e-1 poisson
    if name_model == "TM" and ind == 5:
        description = "TM " + str(ind) + " Experiment, decay around 100Hz"
        syn_params = [1.31247486e-01, 2.18003024e-02, 1.33872486e-01, 3.17757124e-00 * 1.79835313e-1, 1.01522251e-03]  #
    # (Experiment 6) freq. response decay around 10Hz {[2:0.889, 4:0.7016, 5.4:0.576, 10:0.2, 15:0.1, 30:0.06, 50:0.035]
    if name_model == "MSSM" and ind == 6:
        description = "MSSM " + str(ind) + " Experiment, decay around 10Hz"
        syn_params = [3.61148253e-03, 9.98782883e-02, 9.99236857e+00, 2.81436921e-01,
                      1.97666651e-02, 1.00657445e+01, 7.39059950e-01, 1.07519099e-03,
                      3.11356220e-01 * 1e-2, 1.74116605e-03]  # g_L = 1.4e-1 poisson [6.4e-2 subthreshold]
    if name_model == "TM" and ind == 6:
        description = "TM " + str(ind) + " Experiment, decay around 100Hz"
        syn_params = [3.94924669e-01, 8.23537249e-01, 7.81110270e-01, 8.16033593e+03 * 2.32722819e-05, 1.77108850e-03]
    # (Experiment 7) freq. response facilitation {5:2.65, 8:3.72, 9:4.05, 10:4.34, 23:6.44, 50:5.88, 100:4.05, 149:3.08}
    if name_model == "MSSM" and ind == 7:
        description = "MSSM " + str(ind) + " Experiment, facilitation"
        syn_params = [4.99904393e-02, 1.28833999e-02, 2.92508311e+00, 4.88095651e-02,
                      2.11579945e-04, 6.46772602e+01, 7.71595702e-01, 1.52095675e-03,
                      1.76132558e-01 * 9e-2, 4.36917566e-03]
    if name_model == "TM" and ind == 7:
        description = "TM " + str(ind) + " Experiment, decay around 100Hz"
        syn_params = []
    # (Experiment 8) differential signaling
    if name_model == "MSSM" and ind == 8:
        description = "MSSM " + str(ind) + " Experiment, facilitation diff. signaling"
        syn_params = []
    if name_model == "TM" and ind == 8:
        description = "TM " + str(ind) + " Experiment, facilitation diff. signaling"
        # params_name_tm = ['U0', 'tau_f', 'tau_d', 'Ase', 'tau_syn']
        syn_params = [0.03, 530e-3, 130e-3, 1540 * 2.9090352375e-04, 2.5e-3]  # For having Ase = 0.075
    # params_s_dep = {'tau_g': 2e-3, 'tau_alpha': 300e-3, 'g0': 0.075, 'f': 0.75}

    if name_model == "DoornSTD" and ind == 1:
        description = "DoornSTD " + str(ind) + ", Control net, (Breaking the burst)"
        name_params = ['E_ampa', 'E_nmda', 'tau_ampa', 'tau_nmda_rise', 'tau_nmda_decay', 'alpha_nmda', 'tau_d', 'U',
                       'S']
        syn_params = [0.0e-3,  # mV
                      0.0e-3,  # mV
                      2e-3,  # 0.002,  # s (2 ms)
                      2e-3,  # s (2 ms)
                      100e-3,  # s (100 ms)
                      0.5,  # Hz (0.5 kHz)
                      200e-3,  # s (200 ms) - STD recovery
                      0.2,  # unitless - STD release probability
                      0.4  # unitless - overall strength
                      ]
    if name_model == "DoornSTD" and ind == 2:
        description = "DoornSTD " + str(ind) + ", Strong STD (a), (Breaking the burst)"
        name_params = ['E_ampa', 'E_nmda', 'tau_ampa', 'tau_nmda_rise', 'tau_nmda_decay', 'alpha_nmda', 'tau_d', 'U',
                       'S']
        syn_params = [0.0e-3, 0.0e-3, 2e-3, 2e-3, 100e-3, 0.5, 250e-3, 6e-3, 0.25]
    if name_model == "DoornSTD" and ind == 3:
        description = "DoornSTD " + str(ind) + ", Strong STD (b), (Breaking the burst)"
        name_params = ['E_ampa', 'E_nmda', 'tau_ampa', 'tau_nmda_rise', 'tau_nmda_decay', 'alpha_nmda', 'tau_d', 'U',
                       'S']
        syn_params = [0.0e-3, 0.0e-3, 2e-3, 2e-3, 100e-3, 0.5, 250e-3, 35e-3, 0.25]
    if name_model == "DoornAsyn" and ind == 4:
        description = "DoornAsyn " + str(ind) + ", low Asynchronous rel, (Breaking the burst)"
        name_params = ['E_ampa', 'E_nmda', 'tau_ampa', 'tau_nmda_rise', 'tau_nmda_decay', 'alpha_nmda', 'tau_d', 'U',
                       'S',
                       # Asynchronous parameters
                       'x0', 'tau_ar', 'Uar', 'Umax']
        syn_params = [0.0e-3, 0.0e-3, 2e-3, 2e-3, 100e-3, 0.5, 200e-3, 0.01, 0.4, 5, 700e-3, 0.5, 0.5 * 1e3]  # 5e-4*1e3
    if name_model == "DoornAsyn" and ind == 5:
        description = "DoornAsyn " + str(ind) + ", high Asynchronous rel, (Breaking the burst)"
        name_params = ['E_ampa', 'E_nmda', 'tau_ampa', 'tau_nmda_rise', 'tau_nmda_decay', 'alpha_nmda', 'tau_d', 'U',
                       'S',
                       # Asynchronous parameters
                       'x0', 'tau_ar', 'Uar', 'Umax']
        syn_params = [0.0e-3, 0.0e-3, 2e-3, 2e-3, 100e-3, 0.5, 200e-3, 0.01, 0.4, 5, 700e-3, 0.5,
                      4.5 * 1e3]  # 45e-4*1e3
    if name_model == "DoornSTD" and ind == 6:
        description = "DoornSTD " + str(ind) + ", Strong NMDA current, (Breaking the burst)"
        name_params = ['E_ampa', 'E_nmda', 'tau_ampa', 'tau_nmda_rise', 'tau_nmda_decay', 'alpha_nmda', 'tau_d', 'U',
                       'S']
        syn_params = [0.0e-3, 0.0e-3, 2e-3, 2e-3, 100e-3, 0.5, 800e-3, 0.2, 1.5]
    if name_model == "DoornSTF" and ind == 7:
        description = "DoornSTF " + str(ind) + ", STF, (Breaking the burst)"
        name_params = ['tau_ampa', 'tau_nmda_rise', 'tau_nmda_decay', 'alpha_nmda', 'tau_d', 'tau_f', 'U', 'S']
        syn_params = [2e-3, 2e-3, 100e-3, 0.5, 1000e-3, 1000e-3, 0.005, 13]

    assert syn_params is not None, "Not parameters for model %s and index %d" % (name_model, ind)

    return syn_params, description, name_params


def get_name_file(sfreq, s_model, n_model, ind, num_syn, lif_output, tau_n, stoch_inp, imputations, gain, n_noise):
    aux_name = "_ind_" + str(ind) + "_gain_" + str(int(gain * 100)) + "_sf_" + str(int(sfreq / 1000)) + "k_syn_" + str(
        num_syn)
    if lif_output and n_model == 'LIF': aux_name += "_tau" + n_model + "_" + str(tau_n) + "ms"
    file_name = s_model + aux_name
    if not stoch_inp:
        file_name = s_model + '_det' + aux_name
    else:
        if n_noise: self.file_name += '_noise'
    return file_name


def static_synapse(lif, Input, g):
    # Number of samples
    L = Input.shape[1]

    # Running model
    for it in range(L):
        # Evaluating change in LIF neuron - membrane potential
        I_args = [Input[:, it] * g]
        lif.update_state(it, I_args)
        # Detecting spike events and storing model output
        # mssm.detect_spike_event(it, mssm.get_output())

    # Computing output spike event in the last ISI
    it = L
    lif.membrane_potential[0, -1] = lif.membrane_potential[0, -2]
    # spike_range = (mssm.time_spike_events[-1], it)
    # mssm.compute_output_spike_event(spike_range, mssm.get_output())


def model_stp(mssm, lif, params, Input, lif_n=None):
    # Update parameters and initial conditions
    mssm.set_model_params(params)

    # Number of samples
    L = Input.shape[1]

    # Num neurons and synapses
    num_syn = mssm.n_syn
    num_neu = lif.n_neurons

    # Running model
    for it in range(L):
        # Evaluating TM model
        mssm.evaluate_model_euler(Input[:, it], it)
        # Evaluating change in LIF neuron - membrane potential
        I_args = [mssm.get_output()[:, it]]
        lif.update_state(it, None, False, I_args)

        if lif_n is not None:
            I_args = [mssm.N[:, it]]
            lif_n.update_state(it, None, False, I_args)

    # Computing output spike event in the last ISI
    it = L
    lif.membrane_potential[0, -1] = lif.membrane_potential[0, -2]
    if lif_n is not None:
        lif_n.membrane_potential[0, -1] = lif_n.membrane_potential[0, -2]
    # spike_range = (mssm.time_spike_events[-1], it)
    # mssm.compute_output_spike_event(spike_range, mssm.get_output())


def sliding_window_indices(t_ms, win_len_ms, step_ms):
    """
    Return list of (start_idx, end_idx) for sliding windows.
    """
    windows = []
    t0 = t_ms[0]
    t_end = t_ms[-1]
    start = t0
    while start + win_len_ms <= t_end:
        i0 = np.searchsorted(t_ms, start)
        i1 = np.searchsorted(t_ms, start + win_len_ms)
        windows.append((i0, i1))
        start += step_ms
    return windows


def compute_time_tr_st(window_length, num_slid_wins, sliding_step, dt, t):
    # Computing time of end of transition
    is_tr_st_time = (window_length + (num_slid_wins * sliding_step)) * 1e-3  # in ms
    ind_tr_st = t - int(is_tr_st_time / dt)
    tr_st_time = ind_tr_st * dt
    return tr_st_time


def update_tr_st_trackers(conds, tr_st_array, window_length, num_slid_wins, sliding_step, dt, t):
    # Number of neurons
    num_neu = conds.shape[1]
    # copy of tr_st_array
    aux_array = np.copy(tr_st_array)
    # If the statistical descriptors of some neurons reach steady-state, update steady-state trackers
    mask_neurons = np.all(conds, axis=0)
    if np.sum(mask_neurons) > 0:
        # Computing tr_st_time
        tr_st_time = compute_time_tr_st(window_length, num_slid_wins, sliding_step, dt, t)

        # Storing tr_st_time for all neurons that reach steady-state for the first time
        if np.any(aux_array[0, mask_neurons]) == 0:
            aux_array[0, mask_neurons] = \
                np.array([[tr_st_time] for _ in range(num_neu)])[mask_neurons, 0]

        # updating last time that tr_st_time is computed
        aux_array[1, mask_neurons] = tr_st_time
        # Counting times that neurons reach steady-state
        aux_array[2, mask_neurons] += 1

    return aux_array


def model_stp_parallel(stp_model, n_model, params, Input, seeds=None, use_noise=False, lif_n=None, I_ext=0,
                       len_windows=None, epsilon=None, rate_input=None, st_prior=None):
    # Update parameters and initial conditions
    stp_model.set_model_params(params)

    # Number of samples
    L = Input.shape[1]
    num_neu = n_model.n_neurons
    num_syn = int(stp_model.n_syn / num_neu)
    if len_windows is None: len_windows = L  # int(L / 3)
    if epsilon is None: epsilon = 1e-3  # 0.1% of maximum

    # Creating connectiviy matrix
    id_mat = np.eye(num_neu)
    id_rep = np.tile(np.eye(num_syn), num_neu).T
    connectivity = np.repeat(id_mat, num_syn, axis=0)

    # Check of correct creation of connectivity matrix
    for n in range(num_neu):
        if np.sum(connectivity[n * num_syn: (n + 1) * num_syn, :]) > num_syn:
            print("error in files from %d to %d" % (n * num_syn, (n + 1) * num_syn))
        if np.sum(connectivity[:, n]) > num_syn:
            print("error in column %d" % n)

    # ******************************************************************************************************************
    # Sliding windows
    finish_sliding_window = False
    Input_copy = np.copy(Input)
    window_length = 200  # ms
    sliding_step = 20  # ms
    if rate_input is not None:
        window_length = np.ceil(2e3 / rate_input)
        sliding_step = 0.1 * window_length
    # variables related to sliding windows computations
    slid_win_start = 0
    num_tail_wins = 20
    num_slid_wins = 5

    # Output variables
    stat_descriptors_tr_st, time_series_tr, tr_st_time, th_tr_an, th_tr_ab = None, None, None, None, None

    # Extra variables
    num_stat_des = 9
    num_stat_des_supra = 3
    num_stat_des_sub = num_stat_des - num_stat_des_supra
    colors = ['black', 'yellow', 'green', 'tab:orange', 'tab:blue', 'tab:green', 'tab:green', 'tab:red', 'tab:red']
    titles = ['rate', r'ISI($\mu$)', r'ISI($\sigma$)', r'V($\mu$)', r'V(med)', r'V($q_{10}$)', r'V($q_{90}$)',
              r'V(min)', r'V(max)']
    labels = ['rate', 'μISI', 'sISI', 'V_mu', 'V_me', 'V_10', 'V_90', 'Vmin', 'Vmax']
    c_tol_dev = [1, 1, 1, 1, 1, 1, 1, 1, 1.5, 1, 1.5]
    c_tol_dev = [1, 1, 1, 1.5, 1.5, 1, 1, 1, 1, 1, 1]
    bias_plot = [5, 0.04, 0.012, 0.004, 0.004, 0.004, 0.01, 0.004, 0.04]
    epsilon = [1e-1, 1e-2, 1e-2, 2e-3, 2e-3, 3e-3, 4e-3, 3e-3, 4e-3]
    # epsilon_min_max = [1e1, 1e0, 1e0, 2e-1, 2e-1, 2e-1, 3e-1, 2e-1, 4e-1]
    epsilon_min_max = [1e1, 1e0, 1e0, 5e-2, 5e-2, 1e-1, 1e-1, 2e-1, 2e-1]

    # ******************************************************************************************************************
    # slid_win = SlidingWindowTransitoryAnalyser(n_model, Input, seeds, L, window_length, sliding_step, num_tail_wins,
    #                                            num_slid_wins, epsilon, epsilon_min_max, c_tol_dev, plot=False,
    #                                            verbose=True)
    # ******************************************************************************************************************
    """
    # Sliding windows
    t_vec = n_model.time_vector[:len_windows + 1] * 1e3  # to ms
    windows = sliding_window_indices(t_ms=t_vec, win_len_ms=window_length, step_ms=sliding_step)
    num_windows = len(windows)
    counter_slid_win = 0
    # suprathreshold statistics
    supra_rate = np.zeros((num_neu, num_windows))
    supra_mean_ISI = np.zeros((num_neu, num_windows))
    supra_std_ISI = np.zeros((num_neu, num_windows))
    # Subthreshold statistics
    sub_mean_v = np.zeros((num_neu, num_windows))
    sub_median_v = np.zeros((num_neu, num_windows))
    sub_q5_v = np.zeros((num_neu, num_windows))
    sub_q10_v = np.zeros((num_neu, num_windows))
    sub_q90_v = np.zeros((num_neu, num_windows))
    sub_q95_v = np.zeros((num_neu, num_windows))
    sub_min_v = np.zeros((num_neu, num_windows))
    sub_max_v = np.zeros((num_neu, num_windows))

    # Auxiliar figure to plot dynamic of statistical descriptors
    # num_stat_des = 11
    # colors = ['black', 'yellow', 'green', 'tab:orange', 'tab:blue', 'tab:olive', 'tab:green', 'tab:green'
    #           , 'tab:olive', 'tab:red', 'tab:red']
    # titles = ['rate', r'ISI($\mu$)', r'ISI($\sigma$)', r'V($\mu$)', r'V(med)', r'V($q_5$)', r'V($q_{10}$)',
    #           r'V($q_{90}$)', r'V($q_{95}$)', r'V(min)', r'V(max)']
    # bias_plot = [5, 0.04, 0.012, 0.004, 0.004, 0.004, 0.004, 0.01, 0.01, 0.004, 0.04]
    # c_tol_dev = [1, 1, 1, 1, 1, 1, 1, 1.5, 1.5, 1, 1.5]
    # epsilon = [1e-1, 1e-2, 1e-2, 2e-3, 2e-3, 2e-3, 2e-3, 2e-3, 4e-3, 2e-3, 4e-3]
    # epsilon_min_max = [1e1, 1e0, 1e0, 2e-1, 2e-1, 2e-1, 2e-1, 3e-1, 4e-1, 2e-1, 4e-1]

    # Auxiliar variables to compute maxi and mini in sliding windows
    maxi_v_supra = np.array([[-np.inf for _ in range(num_neu)] for _ in range(num_stat_des_supra)])
    mini_v_supra = np.array([[np.inf for _ in range(num_neu)] for _ in range(num_stat_des_supra)])
    maxi_v_sub = np.array([[-np.inf for _ in range(num_neu)] for _ in range(num_stat_des_sub)])
    mini_v_sub = np.array([[np.inf for _ in range(num_neu)] for _ in range(num_stat_des_sub)])

    # Transition times
    # Saving first time tr_st_time, how many times and last time tr_st_time
    supra_mini_maxi = np.zeros((3, 3, num_neu))
    sub_mini_maxi = np.zeros((3, 3, num_neu))
    supra_rel_cha = np.zeros((3, 3, num_neu))
    sub_rel_cha = np.zeros((3, 3, num_neu))
    supra_tol_dev = np.zeros((3, 3, num_neu))
    sub_tol_dev = np.zeros((3, 3, num_neu))
    counter_stim_win = 0
    # len_stimuli_windows = np.array([[[0, int(L / 3), 0, len_windows],
    #                                 [int(L / 3), int(2 * L / 3), 0, len_windows],
    #                                 [int(2 * L / 3), L, 0, len_windows]] for _ in range(num_neu)])
    len_stimuli_windows = np.array([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] for _ in range(num_neu)])
    tr_st_3_cond = np.zeros((2, num_neu))

    # Auxiliar plot
    fig_stat = plt.figure(figsize=[12, 4])
    ax = []
    for i in range(num_stat_des):
        ax.append(plt.subplot(int(np.ceil(num_stat_des / 5)), 5, i + 1))
        ax[i].set_title(titles[i], c='gray')
        ax[i].grid()
        ax[i].set_xlim(0, 100)  # num_windows)
    fig_stat.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
    # ******************************************************************************************************************
    # """

    # Running model
    flex_t = 0
    it = 0
    while it < L:  # for it in range(L):
        # Evaluating Synaptic model
        stp_model.evaluate_model_euler(Input[:, flex_t], flex_t)

        # Converting model output into matrix alike
        stp_output = stp_model.get_output()
        I_args = []

        if stp_output.ndim == 3:
            # stp_output shape: (K, n_syn, L)
            for k in range(stp_output.shape[0]):
                # aux_input = np.resize(np.repeat(stp_output[k, :, it], num_neu), (num_syn * num_neu, num_neu))
                aux_input = np.resize(np.repeat(stp_output[k, :, flex_t], num_neu), (num_syn * num_neu, num_neu))
                c = aux_input * connectivity
                aux_2 = np.matmul(c.T, id_rep).T
                aux_3 = np.sum(aux_2.reshape(num_neu, -1), axis=1)  # [n_neu]
                I_args.append(aux_3)  # shape: (n_syn, L)
            I_args.append(I_ext)

        elif stp_output.ndim == 2:
            # stp_output shape: (n_syn, L)
            # aux_input = np.resize(np.repeat(stp_output[:, it], num_neu), (num_syn * num_neu, num_neu))
            aux_input = np.resize(np.repeat(stp_output[:, flex_t], num_neu), (num_syn * num_neu, num_neu))
            c = aux_input * connectivity
            aux_2 = np.matmul(c.T, id_rep).T
            I_args.append(aux_2)

        # Evaluating change in LIF neuron - membrane potential
        seed = None
        # if seeds is not None: seed = seeds[it]
        # n_model.update_state(it, seed, use_noise, I_args)
        if seeds is not None: seed = seeds[flex_t]
        n_model.update_state(flex_t, seed, use_noise, I_args)

        # Detecting spike events and storing model output
        n_model.detect_spike_event(flex_t, Input, n_model.membrane_potential)

        """
        # **************************************************************************************************************
        # Sliding windows approach
        window = windows[counter_slid_win]
        t0, t1 = window
        win_duration_s = (t1 - t0) * n_model.dt
        # Suprathreshold regime
        spike_in_window = [[] for _ in range(num_neu)]
        rate_in_window = np.zeros(num_neu)
        ISI_in_window = [[] for _ in range(num_neu)]
        mean_ISI_in_window = np.zeros(num_neu)
        std_ISI_in_window = np.zeros(num_neu)
        # subthreshold regime
        v_in_window = [[] for _ in range(num_neu)]
        mean_v_in_window = np.zeros(num_neu)
        median_v_in_window = np.zeros(num_neu)
        # q5_v_in_window = np.zeros(num_neu)
        q10_v_in_window = np.zeros(num_neu)
        q90_v_in_window = np.zeros(num_neu)
        # q95_v_in_window = np.zeros(num_neu)
        min_v_in_window = np.zeros(num_neu)
        max_v_in_window = np.zeros(num_neu)

        # If time is longer than first time window
        if flex_t >= t1:
            # Compute statistics of membrane potential to get transition and steady-state
            # Suprathreshold: spikes in window
            for n in range(num_neu):
                spk_mask = (n_model.time_spikes_generated[n] >= t0) & (n_model.time_spikes_generated[n] < t1)
                spike_in_window[n] = np.array(n_model.time_spikes_generated[n])[spk_mask] * n_model.dt

                # rate computation
                # If there are not spikes in the window, assign a neg value to the rate
                if len(spike_in_window[n]) == 0:
                    if np.sum(supra_rate[n, :counter_slid_win]) > 0:
                        rate_in_window[n] = len(spike_in_window[n]) / win_duration_s
                    # else:
                    #     rate_in_window[n] = np.nan  # -10
                else:
                    rate_in_window[n] = len(spike_in_window[n]) / win_duration_s

                # ISI computation
                if len(spike_in_window[n]) >= 2:
                    ISI_in_window[n] = np.diff(spike_in_window[n])
                    mean_ISI_in_window[n] = np.mean(ISI_in_window[n])
                    std_ISI_in_window[n] = np.std(ISI_in_window[n])

                # For subthreshold regime
                mask_v_aux = n_model.membrane_potential[n, t0:t1] < n_model.V_threshold[n]
                v_aux = n_model.membrane_potential[n, t0:t1][mask_v_aux]
                # Getting statistical descriptors
                v_in_window[n] = v_aux
                # a, b, c, d = np.mean(v_aux), np.median(v_aux), np.quantile(v_aux, 0.05), np.quantile(v_aux, 0.1)
                # e, f, g, h = np.quantile(v_aux, 0.9), np.quantile(v_aux, 0.95), np.min(v_aux), np.max(v_aux)
                # mean_v_in_window[n], median_v_in_window[n], q5_v_in_window[n], q10_v_in_window[n] = a, b, c, d
                # q90_v_in_window[n], q95_v_in_window[n], min_v_in_window[n], max_v_in_window[n] = e, f, g, h
                a, b, d = np.mean(v_aux), np.median(v_aux), np.quantile(v_aux, 0.1)
                e, g, h = np.quantile(v_aux, 0.9), np.min(v_aux), np.max(v_aux)
                mean_v_in_window[n], median_v_in_window[n], q10_v_in_window[n] = a, b, d
                q90_v_in_window[n], min_v_in_window[n], max_v_in_window[n] = e, g, h

            # Suprathreshold: spikes in windows
            supra_rate[:, counter_slid_win] = rate_in_window
            supra_mean_ISI[:, counter_slid_win] = mean_ISI_in_window
            supra_std_ISI[:, counter_slid_win] = std_ISI_in_window
            # Subthreshold: V(t) in windows
            sub_mean_v[:, counter_slid_win] = mean_v_in_window
            sub_median_v[:, counter_slid_win] = median_v_in_window
            # sub_q5_v[:, counter_slid_win] = q5_v_in_window
            sub_q10_v[:, counter_slid_win] = q10_v_in_window
            sub_q90_v[:, counter_slid_win] = q90_v_in_window
            # sub_q95_v[:, counter_slid_win] = q95_v_in_window
            sub_min_v[:, counter_slid_win] = min_v_in_window
            sub_max_v[:, counter_slid_win] = max_v_in_window

            # Checking if statistical descriptors don't change more than the epsilon value in the last 10 windows
            if counter_slid_win >= slid_win_start + num_tail_wins + num_slid_wins:
                # Updating statistical descriptors in list
                stat_des_supra = [supra_rate, supra_mean_ISI, supra_std_ISI]
                stat_des_sub = [sub_mean_v, sub_median_v, sub_q10_v, sub_q90_v, sub_min_v, sub_max_v]
                if st_prior is not None:
                    stat_des_sub = [sub_mean_v]
                # Steady-state condition-array for neurons and statistical descriptors
                conds_supra_max_min = np.array([[False for _ in range(num_neu)] for _ in range(len(stat_des_supra))])
                conds_supra_rel_cha = np.array([[False for _ in range(num_neu)] for _ in range(len(stat_des_supra))])
                conds_supra_tol_dev = np.array([[False for _ in range(num_neu)] for _ in range(len(stat_des_supra))])
                conds_sub_max_min = np.array([[False for _ in range(num_neu)] for _ in range(len(stat_des_sub))])
                conds_sub_rel_cha = np.array([[False for _ in range(num_neu)] for _ in range(len(stat_des_sub))])
                conds_sub_tol_dev = np.array([[False for _ in range(num_neu)] for _ in range(len(stat_des_sub))])

                # Tail for computing stationary mean and std
                min_tail = counter_slid_win - num_tail_wins - num_slid_wins
                if counter_stim_win > 0:
                    min_tail = np.max(len_stimuli_windows[:, counter_stim_win, 2])  # ...[:, counter_stim_win - 1, 3])
                max_tail = counter_slid_win - num_slid_wins

                # Counter statistical descriptors
                c_sd = 0
                # For suprathreshold regime. Looping through statistical descriptors
                for i in range(num_stat_des_supra):
                    # Selecting values of current statistical descriptor in the last num_slid_wins windows
                    tail_stat_des = stat_des_supra[i][:, min_tail:max_tail]
                    # Selecting values of current statistical descriptor in the last num_slid_wins windows
                    aux_stat_des = stat_des_supra[i][:, counter_slid_win - num_slid_wins:counter_slid_win]

                    # Computing mean and standard deviation in the provitional stationary tail
                    aux_mean = np.repeat(np.reshape(np.mean(tail_stat_des, axis=1), (num_neu, 1)), num_slid_wins,
                                         axis=1)
                    aux_std = np.repeat(np.reshape(np.std(tail_stat_des, axis=1), (num_neu, 1)), num_slid_wins, axis=1)

                    # Updating maxi and mini masks for a specific statistica descriptor
                    aux_mask_ma = np.max(aux_stat_des, axis=1) > maxi_v_supra[i, :]
                    if np.any(aux_mask_ma):
                        maxi_v_supra[i, aux_mask_ma] = np.max(aux_stat_des, axis=1)[aux_mask_ma]
                    aux_mask_mi = np.min(aux_stat_des, axis=1) < mini_v_supra[i, :]
                    if np.any(aux_mask_mi):
                        mini_v_supra[i, aux_mask_mi] = np.min(aux_stat_des, axis=1)[aux_mask_mi]
                    # Stationary state by checking epsilon times difference between maxi and mini
                    diff_max_min_local = np.max(aux_stat_des, axis=1) - np.min(aux_stat_des, axis=1)
                    ep = (maxi_v_supra[i, :] - mini_v_supra[i, :]) * epsilon_min_max[c_sd]
                    aux = diff_max_min_local <= ep

                    # Stationary state by checking relative change of stat_des (∣stat_des−μ∣/(μ+1e-9)) <= epsilon
                    delta_k = np.abs(aux_stat_des - aux_mean) / (np.abs(aux_mean) + 1e-9)
                    aux_per_win = delta_k <= epsilon[c_sd]
                    aux_rel = np.sum(aux_per_win, axis=1) == num_slid_wins

                    # stationary state by checking relative tolerance
                    delta_k2 = np.abs(aux_stat_des - aux_mean) / (aux_std + 1e-9)
                    aux2_per_win = delta_k2 <= 1  # c[i] * epsilon[c_sd]
                    aux_rel_tol = np.sum(aux2_per_win, axis=1) == num_slid_wins

                    # Updating arrays that check conditions
                    conds_supra_max_min[i, :] = aux
                    conds_supra_rel_cha[i, :] = aux_rel
                    conds_supra_tol_dev[i, :] = aux_rel_tol
                    print("%s, maxi-mini: %s, max-min is %.2E and ep %.2E. "
                          "Cond rel. change: %s, d_k %.2E, epsilon %.1E. "
                          "Cond rel. tolerance: %s, d_k %.2E, epsilon 1" %
                          (labels[i], aux[0], diff_max_min_local[0], ep[0], aux_rel[0], np.max(delta_k, axis=1)[0],
                           epsilon[i], aux_rel_tol[0], np.max(delta_k2, axis=1)[0]))

                    # Auxiliar plotting of statistical descriptors
                    for n in range(num_neu):
                        # aux_thr = mini_v[i, :] + ep[n] + n * bias_plot[i]
                        # ax[i].plot([0, counter_slid_win], [aux_thr, aux_thr], color='gray')
                        ax[c_sd].plot([ki for ki in range(counter_slid_win)], stat_des_supra[i][n, :counter_slid_win] +
                                      n * bias_plot[c_sd], color=colors[c_sd])
                    # Increasing counter of statistical descriptors
                    c_sd += 1

                # For subthreshold regime. Looping through statistical descriptors
                for i in range(len(stat_des_sub)):
                    # EVALUATING ONLY FOR NEURONS THAT DOES NOT REACH SUPRATHRESHOLD IN THE TIME OF TH SLIDING WINDOW
                    neurons_not_spiking = supra_rate[:, min_tail:counter_slid_win] == 0
                    mask_in_sub = np.sum(neurons_not_spiking, axis=1) == counter_slid_win - min_tail

                    # If there is at least one neuron in subthreshold regime
                    if np.any(mask_in_sub):
                        # Selecting values of current statistical descriptor in the last num_slid_wins windows
                        # tail_stat_des = stat_des_sub[i][:, min_tail:max_tail]
                        tail_stat_des = stat_des_sub[i][mask_in_sub, min_tail:max_tail]
                        # Selecting values of current statistical descriptor in the last num_slid_wins windows
                        # aux_stat_des = stat_des_sub[i][:, counter_slid_win - num_slid_wins:counter_slid_win]
                        aux_stat_des = stat_des_sub[i][mask_in_sub, counter_slid_win - num_slid_wins:counter_slid_win]

                        # Default setup of aux_mean and std_mean in case all neurons are in suprathreshold during the
                        # evaluation windows
                        aux_mean = np.zeros((0, num_slid_wins))
                        aux_std = np.zeros((0, num_slid_wins))
                        if tail_stat_des.shape[0] > 0:
                            if st_prior is None:
                                # Computing mean and standard deviation in the provitional stationary tail
                                aux1 = np.reshape(np.mean(tail_stat_des, axis=1), (tail_stat_des.shape[0], 1))
                                aux_mean = np.repeat(aux1, num_slid_wins, axis=1)
                                aux1 = np.reshape(np.std(tail_stat_des, axis=1), (tail_stat_des.shape[0], 1))
                                aux_std = np.repeat(aux1, num_slid_wins, axis=1)
                            else:
                                # Using mean value from deterministic steady-state priors
                                # aux_vals = [st_prior[counter_stim_win, 6] for _ in range(tail_stat_des.shape[0])]
                                # aux1 = np.reshape(aux_vals, (tail_stat_des.shape[0], 1))
                                # aux_mean = np.repeat(aux1, num_slid_wins, axis=1)
                                # Computing mean in the provitional stationary tail
                                aux1 = np.reshape(np.mean(tail_stat_des, axis=1), (tail_stat_des.shape[0], 1))
                                aux_mean = np.repeat(aux1, num_slid_wins, axis=1)
                                # Using variance value as diff between q95 and 95 from deterministic steady-state priors
                                # ['q5', 'q10', 'q90', 'q95', 'min', 'max', 'mean', 'median']
                                aux_vals = [[np.array([st_prior[counter_stim_win, 5] - st_prior[counter_stim_win, 4]])] for _ in range(tail_stat_des.shape[0])]
                                aux1 = np.reshape(aux_vals, (tail_stat_des.shape[0], 1))
                                aux_std = np.repeat(aux1, num_slid_wins, axis=1)

                        # Updating maxi and mini masks for a specific statistica descriptor
                        # aux_mask_ma = np.max(aux_stat_des, axis=1) > maxi_v_sub[i, :]
                        aux_mask_ma = np.max(aux_stat_des, axis=1) > maxi_v_sub[i, mask_in_sub]
                        if np.any(aux_mask_ma):
                            aux = maxi_v_sub[i, mask_in_sub]
                            aux[aux_mask_ma] = np.max(aux_stat_des, axis=1)[aux_mask_ma]
                            maxi_v_sub[i, mask_in_sub] = aux
                        # aux_mask_mi = np.min(aux_stat_des, axis=1) < mini_v_sub[i, :]
                        aux_mask_mi = np.min(aux_stat_des, axis=1) < mini_v_sub[i, mask_in_sub]
                        if np.any(aux_mask_mi):
                            aux = mini_v_sub[i, mask_in_sub]
                            aux[aux_mask_mi] = np.min(aux_stat_des, axis=1)[aux_mask_mi]
                            mini_v_sub[i, mask_in_sub] = aux
                        # Stationary state by checking epsilon times difference between maxi and mini
                        diff_max_min_local = np.max(aux_stat_des, axis=1) - np.min(aux_stat_des, axis=1)
                        # ep = (maxi_v_sub[i, :] - mini_v_sub[i, :]) * epsilon_min_max[c_sd]
                        ep = (maxi_v_sub[i, mask_in_sub] - mini_v_sub[i, mask_in_sub]) * epsilon_min_max[c_sd]
                        aux = diff_max_min_local <= ep

                        # Stationary state by checking relative change of stat_des (∣stat_des−μ∣/(μ+1e-9)) <= epsilon
                        delta_k = np.abs(aux_stat_des - aux_mean) / (np.abs(aux_mean) + 1e-9)
                        aux_per_win = delta_k <= epsilon[c_sd]
                        aux_rel = np.sum(aux_per_win, axis=1) == num_slid_wins

                        # stationary state by checking relative tolerance
                        delta_k2 = np.abs(aux_stat_des - aux_mean) / (aux_std + 1e-9)
                        aux2_per_win = delta_k2 <= c_tol_dev[c_sd]  # 1.5  # c[i] * epsilon[c_sd]
                        aux_rel_tol = np.sum(aux2_per_win, axis=1) == num_slid_wins

                        # Updating arrays that check conditions
                        # conds_sub_max_min[i, :] = aux
                        # conds_sub_rel_cha[i, :] = aux_rel
                        # conds_sub_tol_dev[i, :] = aux_rel_tol
                        conds_sub_max_min[i, mask_in_sub] = aux
                        conds_sub_rel_cha[i, mask_in_sub] = aux_rel
                        conds_sub_tol_dev[i, mask_in_sub] = aux_rel_tol
                        if tail_stat_des.shape[0] > 0:
                            print("%s, maxi-mini: %s, max-min is %.2E and ep %.2E. "
                                  "Cond rel. change: %s, d_k %.2E, epsilon %.1E. "
                                  "Cond rel. tolerance: %s, d_k %.2E, epsilon %.2E" %
                                  (labels[c_sd], aux[0], diff_max_min_local[0], ep[0], aux_rel[0],
                                   np.max(delta_k, axis=1)[0], epsilon[c_sd], aux_rel_tol[0],
                                   np.max(delta_k2, axis=1)[0], c_tol_dev[c_sd]))
                            # Auxiliar plotting of statistical descriptors
                            for n in range(num_neu):
                                ax[c_sd].plot([ki for ki in range(counter_slid_win)],
                                              stat_des_sub[i][n, :counter_slid_win] +
                                              n * bias_plot[c_sd], color=colors[c_sd])
                                # ['q5', 'q10', 'q90', 'q95', 'min', 'max', 'mean', 'median']
                                lims = [0, counter_slid_win]
                                if st_prior is not None:
                                    aux_a = st_prior[counter_stim_win, 6]
                                    ax[c_sd].plot(lims, np.array([aux_a, aux_a]) + n * bias_plot[c_sd], c="tab:orange")
                                    aux_a = st_prior[counter_stim_win, 5]
                                    ax[c_sd].plot(lims, np.array([aux_a, aux_a]) + n * bias_plot[c_sd], c="tab:olive")
                                    aux_a = st_prior[counter_stim_win, 4]
                                    ax[c_sd].plot(lims, np.array([aux_a, aux_a]) + n * bias_plot[c_sd], c="tab:olive")
                        else:
                            print("%s, Neuron in suprathreshold regime" % labels[c_sd])

                    # Increasing counter of statistical descriptors
                    c_sd += 1

                # If all signals reach the steady-state value, then stop the loop
                # compute time of possible steady-state for suprathreshold statistical descriptors:
                # ind_ = 3
                a = counter_stim_win
                supra_mini_maxi[a, :] = update_tr_st_trackers(conds_supra_max_min, supra_mini_maxi[a, :],
                                                              window_length, num_slid_wins, sliding_step, n_model.dt,
                                                              flex_t)
                sub_mini_maxi[a, :] = update_tr_st_trackers(conds_sub_max_min, sub_mini_maxi[a, :], window_length,
                                                            num_slid_wins, sliding_step, n_model.dt, flex_t)
                supra_rel_cha[a, :] = update_tr_st_trackers(conds_supra_rel_cha, supra_rel_cha[a, :], window_length,
                                                            num_slid_wins, sliding_step, n_model.dt, flex_t)
                sub_rel_cha[a, :] = update_tr_st_trackers(conds_sub_rel_cha, sub_rel_cha[a, :], window_length,
                                                          num_slid_wins, sliding_step, n_model.dt, flex_t)
                supra_tol_dev[a, :] = update_tr_st_trackers(conds_supra_tol_dev, supra_tol_dev[a, :], window_length,
                                                            num_slid_wins, sliding_step, n_model.dt, flex_t)
                sub_tol_dev[a, :] = update_tr_st_trackers(conds_sub_tol_dev, sub_tol_dev[a, :], window_length,
                                                          num_slid_wins, sliding_step, n_model.dt, flex_t)
                print("Stim win: " + str(a) + ", supra(min-max): " + str(supra_mini_maxi[a, 0, :]) +
                      ", supra(rel-cha): " + str(supra_rel_cha[a, 0, :]) + ", supra(tol_dev): " +
                      str(supra_tol_dev[a, 0, :]))
                print("Stim win: " + str(a) + ", sub(min-max): " + str(sub_mini_maxi[a, 0, :]) + ", sub(rel-cha): " +
                      str(sub_rel_cha[a, 0, :]) + ", sub(tol_dev): " + str(sub_tol_dev[a, 0, :]))

                mask_supr_mm = np.where(supra_mini_maxi[a, 0, :] > 0)[0]
                mask_sub_mm = np.where(sub_mini_maxi[a, 0, :] > 0)[0]
                mask_supr_rc = np.where(supra_rel_cha[a, 0, :] > 0)[0]
                mask_sub_rc = np.where(sub_rel_cha[a, 0, :] > 0)[0]
                mask_supr_td = np.where(supra_tol_dev[a, 0, :] > 0)[0]
                mask_sub_td = np.where(sub_tol_dev[a, 0, :] > 0)[0]
                # Updating time of reaching st based on each approach (mini-maxi, relative change, tolarance deviation)
                mask_to_update = tr_st_3_cond[0, mask_sub_mm] == 0
                tr_update = [flex_t for i in range(int(np.sum(mask_to_update)))]
                if np.any(mask_to_update):
                    # tr_st_3_cond[0, mask_sub_mm[mask_to_update]] = tr_update
                    pass
                mask_to_update = tr_st_3_cond[1, mask_sub_td] == 0
                tr_update = [flex_t for i in range(int(np.sum(mask_to_update)))]
                if np.any(mask_to_update):
                    tr_st_3_cond[1, mask_sub_td[mask_to_update]] = tr_update
                # if tr_st_3_cond[2, mask_sub_rc] == 0: tr_st_3_cond[2, mask_sub_rc] = flex_t
                # Which neurons reach steady-state with all methods?
                mask_sub = np.sum(tr_st_3_cond > 0, axis=0) == 1 # tr_st_3_cond.shape[0]  # num_neu
                # updating index of reaching steady-state for each neuron
                if np.any(mask_sub):
                    ind_to_update = len_stimuli_windows[:, counter_stim_win, 1] == 0
                    if np.any(ind_to_update):
                        mask_to_update = np.logical_and(mask_sub, ind_to_update)
                        tr_update = [flex_t for _ in range(int(np.sum(mask_to_update)))]
                        win_update = [counter_slid_win for _ in range(int(np.sum(mask_to_update)))]
                        len_stimuli_windows[mask_to_update, counter_stim_win, 1] = tr_update
                        len_stimuli_windows[mask_to_update, counter_stim_win, 3] = win_update

                # counting of neurons reaching steady-state
                cond_supr_mm, cond_sub_mm = mask_supr_mm.shape[0], mask_sub_mm.shape[0]
                cond_supr_rc, cond_sub_rc = mask_supr_rc.shape[0], mask_sub_rc.shape[0]
                cond_supr_td, cond_sub_td = mask_supr_td.shape[0], mask_sub_td.shape[0]
                # if cond_sub_mm == num_neu and cond_sub_rc == num_neu and cond_sub_td == num_neu:
                # if cond_sub_mm == num_neu and cond_sub_td == num_neu:
                if cond_sub_td == num_neu:
                    # if cond_sub_td == num_neu:
                    print("All neurons reach steady-state")
                    fig_mem = plt.figure()
                    axfm = fig_mem.add_subplot(111)
                    # axfm.plot(n_model.time_vector[:flex_t], n_model.membrane_potential[0, :flex_t], c="gray")

                    min_m = np.min(n_model.membrane_potential[:, :flex_t])
                    max_m = np.max(n_model.membrane_potential[:, :flex_t])
                    for n in range(num_neu):
                        # Plotting membrane potential
                        aux_t_vec = n_model.time_vector[:flex_t]
                        axfm.plot(aux_t_vec, n_model.membrane_potential[n, :flex_t] + n * .01,
                                  c="gray")
                        # Through stimuli-windows
                        for b in range(a + 1):
                            # Plotting tr_st_time for each method
                            lim_a, lim_b = min_m + 0.01 * n, max_m + 0.01 * n
                            axfm.plot([sub_mini_maxi[b, 0, n], sub_mini_maxi[b, 0, n]], [lim_a, lim_b], c="tab:red")
                            axfm.plot([sub_rel_cha[b, 0, n], sub_rel_cha[b, 0, n]], [lim_a, lim_b], c="tab:green")
                            axfm.plot([sub_tol_dev[b, 0, n], sub_tol_dev[b, 0, n]], [lim_a, lim_b], c="tab:blue")
                            aux_p = len_stimuli_windows[n, b, 0] * n_model.dt
                            axfm.plot([aux_p, aux_p], [min_m, max_m + 0.01 * n], c="black")
                            if st_prior is not None:
                                # ['q5', 'q10', 'q90', 'q95', 'min', 'max', 'mean', 'median']
                                lims = len_stimuli_windows[n, b, :2] * n_model.dt
                                axfm.plot(lims, [st_prior[b, 0] + 0.01 * n, st_prior[b, 0] + 0.01 * n], c="tab:olive")
                                axfm.plot(lims, [st_prior[b, 1] + 0.01 * n, st_prior[b, 1] + 0.01 * n], c="tab:green")
                                axfm.plot(lims, [st_prior[b, 2] + 0.01 * n, st_prior[b, 2] + 0.01 * n], c="tab:green")
                                axfm.plot(lims, [st_prior[b, 3] + 0.01 * n, st_prior[b, 3] + 0.01 * n], c="tab:olive")
                                axfm.plot(lims, [st_prior[b, 4] + 0.01 * n, st_prior[b, 4] + 0.01 * n], c="tab:red")
                                axfm.plot(lims, [st_prior[b, 5] + 0.01 * n, st_prior[b, 5] + 0.01 * n], c="tab:red")
                                axfm.plot(lims, [st_prior[b, 6] + 0.01 * n, st_prior[b, 6] + 0.01 * n], c="tab:orange")
                                axfm.plot(lims, [st_prior[b, 7] + 0.01 * n, st_prior[b, 7] + 0.01 * n], c="tab:blue")
                    axfm.grid()

                    # updating it
                    if 0 < it <= int(L / 3):
                        # ini window is computed
                        ind_max_tr_st = [np.max(len_stimuli_windows[:, counter_stim_win, 1]) for _ in range(num_neu)]
                        # len_stimuli_windows[:, counter_stim_win + 1, 0] = len_stimuli_windows[:, counter_stim_win, 1]
                        len_stimuli_windows[:, counter_stim_win, 1] = ind_max_tr_st
                        len_stimuli_windows[:, counter_stim_win + 1, 0] = ind_max_tr_st
                        len_stimuli_windows[:, counter_stim_win + 1, 2] = (len_stimuli_windows[:, counter_stim_win, 3] +
                                                                           int(window_length / sliding_step))
                        # len_stimuli_windows[0][1] = flex_t
                        # len_stimuli_windows[1][0] = flex_t
                        # len_stimuli_windows[0][3] = counter_slid_win
                        # len_stimuli_windows[1][2] = counter_slid_win + int(window_length / sliding_step)
                        it = int(L / 3) - 1
                    elif int(L / 3) < it <= int(2 * L / 3):
                        # mid window is computed
                        ind_max_tr_st = [np.max(len_stimuli_windows[:, counter_stim_win, 1]) for _ in range(num_neu)]
                        len_stimuli_windows[:, counter_stim_win, 1] = ind_max_tr_st
                        len_stimuli_windows[:, counter_stim_win + 1, 0] = ind_max_tr_st
                        len_stimuli_windows[:, counter_stim_win + 1, 2] = (len_stimuli_windows[:, counter_stim_win, 3] +
                                                                           int(window_length / sliding_step))
                        # len_stimuli_windows[1][1] = flex_t
                        # len_stimuli_windows[2][0] = flex_t
                        # len_stimuli_windows[1][3] = counter_slid_win
                        # len_stimuli_windows[2][2] = counter_slid_win + int(window_length / sliding_step)
                        it = int(2 * L / 3) - 1
                    else:
                        # end window is computed
                        # len_stimuli_windows[2][1] = flex_t
                        # len_stimuli_windows[2][3] = counter_slid_win
                        ind_max_tr_st = [np.max(len_stimuli_windows[:, counter_stim_win, 1]) for _ in range(num_neu)]
                        len_stimuli_windows[:, counter_stim_win, 1] = ind_max_tr_st
                        it = L - 1
                        finish_sliding_window = True

                    print(tr_st_3_cond)
                    print(len_stimuli_windows)
                    # Updating window
                    counter_slid_win += int(window_length / sliding_step) - 1
                    # updating when to start analysing sliding windows
                    slid_win_start = counter_slid_win + 1
                    # Updating input
                    Input = np.copy(np.concatenate((Input[:, :flex_t], Input_copy[:, it + 1:]), axis=1))
                    # Initialising mini and maxi arrays to begin computing new window
                    # Auxiliar variables to compute maxi and mini in sliding windows
                    maxi_v_supra = np.array([[-np.inf for _ in range(num_neu)] for _ in range(num_stat_des_supra)])
                    mini_v_supra = np.array([[np.inf for _ in range(num_neu)] for _ in range(num_stat_des_supra)])
                    maxi_v_sub = np.array([[-np.inf for _ in range(num_neu)] for _ in range(num_stat_des_sub)])
                    mini_v_sub = np.array([[np.inf for _ in range(num_neu)] for _ in range(num_stat_des_sub)])
                    # Resetting matrix with indices where each neuron reaches transition-state
                    tr_st_3_cond = np.zeros((2, num_neu))
                    # updating counter of stimuli windows
                    counter_stim_win += 1

            # increasing counter of windows
            counter_slid_win += 1
            print("Computing window %d/%d while t is %.3f" % (counter_slid_win, num_windows, flex_t * n_model.dt))

        if finish_sliding_window:
            # Computing transition and steady-state values
            # sliding windows of ini, mid and end stimuli-windows
            i_iw = (len_stimuli_windows[0, 0, :2] * n_model.dt * 1e3) / sliding_step  # len_stimuli_windows[0, 2:4]
            i_mw = (len_stimuli_windows[0, 1, :2] * n_model.dt * 1e3) / sliding_step  # len_stimuli_windows[1, 2:4]
            i_ew = (len_stimuli_windows[0, 2, :2] * n_model.dt * 1e3) / sliding_step  # len_stimuli_windows[2, 2:4]
            l_win = window_length / sliding_step
            i_iw[1], i_mw[1], i_ew[1] = i_iw[1] - l_win, i_mw[1] - l_win, i_ew[1] - l_win
            i_iw = i_iw.astype(int)
            i_mw = i_mw.astype(int)
            i_ew = i_ew.astype(int)
            # Computing steady-state statistics for the last 25 windows
            st_win_l = num_tail_wins + num_slid_wins
            mem_pot_iw_st = n_model.membrane_potential[:, windows[i_iw[1] - st_win_l][0]: windows[i_iw[1]][1]]
            mem_pot_mw_st = n_model.membrane_potential[:, windows[i_mw[1] - st_win_l][0]: windows[i_mw[1]][1]]
            mem_pot_ew_st = n_model.membrane_potential[:, windows[i_ew[1] - st_win_l][0]: windows[i_ew[1]][1]]
            # Auxiliar function to compute: mean, median, q5, q10, q90, q95, min, max
            iw_st = statistics_signal(mem_pot_iw_st, axis=1)
            mw_st = statistics_signal(mem_pot_mw_st, axis=1)
            ew_st = statistics_signal(mem_pot_ew_st, axis=1)
            tr_signals = [sub_mean_v, sub_median_v, sub_q5_v, sub_q10_v, sub_q90_v, sub_q95_v, sub_min_v, sub_max_v]
            iw_tr = stat_tr_slid(tr_signals, i_iw, st_win_l)
            mw_tr = stat_tr_slid(tr_signals, i_mw, st_win_l)
            ew_tr = stat_tr_slid(tr_signals, i_ew, st_win_l)
            # time to reach steady-state for each stimuli-window
            windows_a = np.array(windows)
            tr_st_time = np.array([windows_a[np.diff(len_stimuli_windows[:, :, 2:4])[:, i, 0], 0] for i in range(3)])
            tr_st_time = tr_st_time * n_model.dt
            # Final summary of sliding window:
            # steady-state mean, median, q5, q10, q90, q95, min, max of ini, mid and end stimuli-windows, then
            # transition-series mean, median, q5, q10, q90, q95, min, max of ini, mid and end stimuli-windows, then
            # time for reaching steady-state of ini, mid and end stimuli-windows
            a = np.array([iw_st[0], iw_st[1], iw_st[2], iw_st[3], iw_st[4], iw_st[5], iw_st[6], iw_st[7],
                          mw_st[0], mw_st[1], mw_st[2], mw_st[3], mw_st[4], mw_st[5], mw_st[6], mw_st[7],
                          ew_st[0], ew_st[1], ew_st[2], ew_st[3], ew_st[4], ew_st[5], ew_st[6], ew_st[7]])
            b = [iw_tr[0], iw_tr[1], iw_tr[2], iw_tr[3], iw_tr[4], iw_tr[5], iw_tr[6], iw_tr[7],
                 mw_tr[0], mw_tr[1], mw_tr[2], mw_tr[3], mw_tr[4], mw_tr[5], mw_tr[6], mw_tr[7],
                 ew_tr[0], ew_tr[1], ew_tr[2], ew_tr[3], ew_tr[4], ew_tr[5], ew_tr[6], ew_tr[7]]
            stat_descriptors_tr_st = a
            time_series_tr = b
            # Old method to get beginning of steady-state
            # a = np.max([mem_pot_iw_st.shape[1], mem_pot_ew_st.shape[1]])
            # piw, pew = mem_pot_iw_st[:, :a], mem_pot_ew_st[:, :a]
            piw = n_model.membrane_potential[:, :len_stimuli_windows[0, 0, 1]]
            lim_a, lim_b = len_stimuli_windows[0, 2, :2]
            pew = n_model.membrane_potential[:, lim_a:lim_b]
            # auxiliar plots
            plt.figure()
            for n in range(num_neu):
                aux = piw[n, :] + n * 0.005
                plt.plot(n_model.time_vector[:aux.shape[0]], aux)
                aux = pew[n, :] + n * 0.005
                plt.plot(n_model.time_vector[:aux.shape[0]], aux)
            plt.grid()
            plt.figure()
            for n in range(num_neu):
                plt.plot(sub_mean_v[n, i_iw[0]:i_iw[1]] + n * 0.005)
                plt.plot(sub_mean_v[n, i_ew[0]:i_ew[1]] + n * 0.005)
            plt.grid()
            a = np.min([piw.shape[1], pew.shape[1]])
            th_tr_a = get_transition_time_from_2_signals(piw[:, :a], pew[:, :a], n_model.dt,
                                                         th_percentage=1e-2) * n_model.dt
            th_tr_ab = get_transition_time_from_2_signals(piw[:, :a], pew[:, :a], n_model.dt, th_percentage=1e-1,
                                                          filtering=True) * n_model.dt
        # End sliding windows
        # **************************************************************************************************************
        # """

        # Sliding window analysis
        # it, finish_sliding_window = slid_win.update(flex_t, it)
        if finish_sliding_window:
            print("finished sliding window")
            # aa, bb, cc, dd, ee = slid_win.analyse()

        # Forcing spike event when changing from ini-to-mid and mid-to-end windows
        if it == int(L / 3) - 1 or it == int(2 * L / 3) - 1:
            count_spikes_in_t = 0
            new_spikes = [True for _ in range(num_neu)]
            for n in range(num_neu):
                # if it in n_model.time_spike_events[n]: count_spikes_in_t += 1
                if flex_t in n_model.time_spike_events[n]: new_spikes[n] = False
            # if count_spikes_in_t != num_neu:
            if np.sum(new_spikes) == num_neu:
                # n_model.append_spike_event(it, [True for _ in range(num_neu)], n_model.membrane_potential)
                n_model.append_spike_event(flex_t, new_spikes, n_model.membrane_potential)
        else:
            pass

        if lif_n is not None:
            # Converting model output into matrix alike
            aux_input_n = np.resize(np.repeat(stp_model.N[:, it], num_neu), (num_syn * num_neu, num_neu))
            c = aux_input_n * connectivity
            input_lif_n = np.matmul(c.T, id_rep).T * stp_model.params[
                'k_EPSP'] / 2  # k_EPSP/2, factor to transform N(t) into the small range as Epsp(t)

            # Evaluating change in LIF neuron - membrane potential
            # lif.update_state(mssm.get_output()[:, it], it)
            I_args = [input_lif_n]
            lif_n.update_state(it, I_args)

        # Increasing time steps
        flex_t += 1
        it += 1

    # Computing output spike event in the last ISI
    it = L
    n_model.membrane_potential[0, -1] = n_model.membrane_potential[0, -2]
    if lif_n is not None:
        lif_n.membrane_potential[0, -1] = lif_n.membrane_potential[0, -2]

    # Detecting spike events and storing model output
    spike_range = (n_model.time_spike_events[-1], it)
    n_model.append_spike_event(it, [True for _ in range(num_neu)], n_model.membrane_potential, append_time=False)

    # return stat_descriptors_tr_st, tr_st_time


def model_simple_dep(s_dep, lif, params, Input):
    # Update parameters and initial conditions
    s_dep.set_model_params(params)

    # Number of samples
    L = Input.shape[1]

    # Running model
    for it in range(L):
        # Evaluating TM model
        s_dep.evaluate_model_euler(Input[:, it], it)
        # Evaluating change in LIF neuron - membrane potential
        I_args = [s_dep.get_output()[:, it]]
        lif.update_state(it, I_args)
        # Detecting spike events and storing model output
        # mssm.detect_spike_event(it, mssm.get_output())

    # Computing output spike event in the last ISI
    it = L
    lif.membrane_potential[0, -1] = lif.membrane_potential[0, -2]
    # spike_range = (mssm.time_spike_events[-1], it)
    # mssm.compute_output_spike_event(spike_range, mssm.get_output())


def compute_new_r(percentage, r, increase):
    if percentage:
        return (increase + 1) * r
    else:
        return increase


def rolling_window(arr, window):
    """
    Transforms an array arr into a separate matrix dividing the array in sliding windows of shape window
    Parameters
    ----------
    arr
    window

    Returns
    -------

    """
    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def correct_poisson_spike_trains(Input_aux, num_realizations, seed=None, imputation=False):
    """

    Parameters
    ----------
    Input_aux
    num_realizations
    seed
    imputation

    Returns
    -------

    """
    L = Input_aux.shape[1]
    aux1 = np.zeros((num_realizations, L))
    aux1[:, 1:] = np.diff(Input_aux, 1, axis=1)
    Input_aux2 = np.where(aux1 == 1, aux1, 0)
    Input_test = np.copy(Input_aux2)

    if imputation:
        #
        kernel = np.array([0, 0, 0])

        # Number of missing events
        miss_events = np.sum(Input_aux, axis=1) - np.sum(Input_aux2, axis=1)
        realization_with_missing_events = list(np.where(miss_events > 0)[0])
        cond = []
        if len(realization_with_missing_events) > 0:
            # Organising array adding a new dimension with the shape of the kernel
            arr = rolling_window(Input_aux2, kernel.shape[0])
            arr = np.reshape(Input_aux2[:, :int(L / 3) * 3], (num_realizations, int(L / 3), 3))
            # Finding indices in array where the kernel is found
            cond = (arr == kernel).all(axis=2)

        for col in realization_with_missing_events:
            # Available spaces to fill with a spike
            ind_kernel_no_spikes = np.where(cond[col, :] == True)[0] * 3
            # to select the indices of the kernel corresponding to the middle (where a new spike is set)
            mid_index = ind_kernel_no_spikes + 1

            # Sampling integers uniformly, the number of samples corresponds to the double of the number of missing
            # events for each realization
            rng = np.random.default_rng(seed)
            aux = rng.choice(np.arange(0, mid_index.shape[0]), size=int(miss_events[col]), replace=False)
            new_ind = mid_index[aux.tolist()]

            # Updating the corresponding input spike train
            Input_test[col, list(new_ind)] = 1

    return Input_test


def simple_spike_train(sfreq, rate, L, num_realizations=1, poisson=False, seeds=None, correction=True,
                       imputation=True):
    seed = None
    if seeds is not None:
        assert isinstance(seeds, list), "seeds must be a list"
        seed = np.random.choice(seeds)
    seed_print = []
    # L = len(modulation_signal)

    if poisson:
        Input_aux = poisson_generator2(1 / sfreq, L, rate, num_realizations, myseed=seed)
        Input_test = np.copy(Input_aux)

        if correction:
            Input_test = correct_poisson_spike_trains(Input_aux, num_realizations, seed=seed, imputation=imputation)

    else:
        aux_s = input_spike_train(sfreq, rate, L / sfreq)
        Input_test = np.repeat(np.expand_dims(aux_s, axis=0), num_realizations, axis=0)

    # print(seed_print)
    return Input_test[:, :L]


def inter_spike_intervals(spike_trains, dt, bin_size, max_time_hist=None):
    """
    Computing the ISI of a spike train
    Parameters
    ----------
    spike_trains
    dt
    bin_size
    max_time_hist
    Returns
    -------

    """
    all_isis = []
    isis = []
    for spike_train in spike_trains:
        spike_indices = np.where(spike_train == 1)[0]
        if len(spike_indices) < 2:
            all_isis.append([])
        else:
            isi = np.diff(spike_indices) * dt
            all_isis.append(isi.tolist())
            isis = isis + isi.tolist()

    histograms = []
    # for isi_list in all_isis:
    # if len(isi_list) == 0:
    if len(isis) == 0:
        histograms.append([[], []])  # empty histogram and bins
    else:
        # isi_array = np.array(isi_list)
        isi_array = np.array(isis)
        max_val = max_time_hist if max_time_hist is not None else isi_array.max()
        bins = np.arange(0, max_val + bin_size, bin_size)
        hist, bin_edges = np.histogram(isi_array, bins=bins)
        histograms.append([hist.tolist(), bin_edges.tolist()])

    return all_isis, histograms


def highpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, cutoff, 'highpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data


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


def get_time_series_statistics_of_transitions(time_series, f_vector, prop_rates, dt, th_percentage=1e-2):
    res_iniw = [[] for _ in range(len(time_series[0]))]  # [num frequencies, num time-steps, num statistics]
    res_midw = [[] for _ in range(len(time_series[0]))]  # [num frequencies, num time-steps, num statistics]
    res_endw = [[] for _ in range(len(time_series[0]))]  # [num frequencies, num time-steps, num statistics]
    res = [res_iniw, res_midw, res_endw]
    th_tr_a = [[] for _ in range(len(time_series[0]))]

    # Iterating through windows (ini, mid, end)
    for window in range(3):

        # Iterating through frequencies
        for freq in range(len(time_series[0]))[::-1]:

            # Computing statistical descriptors
            aux_array = np.array(time_series[window][freq])
            a = [np.mean(aux_array, axis=0), np.median(aux_array, axis=0), np.quantile(aux_array, 0.05, axis=0),
                 np.quantile(aux_array, 0.1, axis=0), np.quantile(aux_array, 0.9, axis=0),
                 np.quantile(aux_array, 0.95, axis=0), np.min(aux_array, axis=0), np.max(aux_array, axis=0)]
            res[window][freq] = np.array(a)

            # Once statistical descriptors for end window are computed, get transition times for freq
            if window == 2:
                th_tr_a[freq] = get_transition_time_from_2_signals(res[0][freq], res[2][freq], dt, th_percentage)
                max_th_tr_st = np.max(th_tr_a[freq])
                # Updating transition components of initial and ending windows
                res[0][freq] = res[0][freq][:, :max_th_tr_st]
                res[2][freq] = res[2][freq][:, :max_th_tr_st]

                # For middle window
                aux_cond = np.where(prop_rates[freq] <= f_vector)
                # If the transition step of a higher rate was already computed (in comparison to the proportional
                # increase of rate), then take the transition step of the closest higher rate for the corresponding
                # prop. rate.
                t_tra_mid_win = max_th_tr_st
                if len(aux_cond[0]) > 0:
                    aux_i = aux_cond[0][0]
                    t_tra_mid_win = np.max(th_tr_a[aux_i])

                # Updating transition components of initial and ending windows
                res[1][freq] = res[1][freq][:, :t_tra_mid_win]

    return th_tr_a, res


def get_transition_time_from_2_signals(signal1, signal2, dt, th_percentage=1e-5, filtering=False, cutoff=5):
    # Filtering signals if flag is true
    if filtering:
        signal1 = lowpass(signal1, cutoff, 1 / dt)
        signal2 = lowpass(signal2, cutoff, 1 / dt)
    # Substract ini and end window to define the transition period (exclude lasst 10 samples to avoid errors)
    ini_minus_end_windows = np.abs(signal1[:, :-10] - signal2[:, :-10])
    # Find the 0.1% of the maximum for each realization (the threshold to define that the difference between ini and end
    # windows is sufficiently low to be considered zero)
    thresholds = np.max(ini_minus_end_windows, axis=1) * th_percentage
    shapes_diff = ini_minus_end_windows.shape
    # Create the mask to compare each realization with the 0.1% of their maximums
    mask_thr = np.repeat(np.reshape(thresholds, (shapes_diff[0], 1)), shapes_diff[1], axis=1)
    # If filtering is activated, apply the thresholds on the filtered signals
    # if filtering:
        # assert sfreq is not None, "sfreq must be given"
    #     ini_minus_end_windows = lowpass(ini_minus_end_windows, cutoff, 1 / dt)
    # find indices where the difference is bigger than 0.1% of maximum
    ind_tr = np.where(ini_minus_end_windows > mask_thr)
    # Getting indices of unique values (i.e. realizations)
    val_unique, ind_unique = np.unique(ind_tr[0], return_index=True)
    # getting last index (indicating that after that, diff. is lower than 1e-6)
    first_indtr = np.roll(ind_tr[1][list(np.array(ind_unique) - 1)], -1)
    # Answering if first_indtr is lower than the index of the last 500ms of the window
    ind_500ms = int(0.5 / dt)
    mask_change_indtr = [False for _ in range(signal1.shape[0])]
    if ind_500ms < signal1.shape[1]:
        mask_change_indtr = first_indtr > (signal1.shape[1] - ind_500ms)
    # if first_indtr of a neuron is greater than len_window minus 500ms, then set first_indtr to len_window minus 500ms
    first_indtr[mask_change_indtr] = [signal1.shape[1] - ind_500ms for _ in range(np.sum(mask_change_indtr))]
    """
    fig = plt.figure(figsize=(8, 3))
    time_vec = np.arange(0, int(signal1.shape[1] * dt), dt)
    for n in range(signal1.shape[0]):
        aux = signal1[n, :] + n * 0.005
        plt.plot(time_vec, aux)
        aux = signal2[n, :] + n * 0.005
        plt.plot(time_vec, aux)
        aux = np.array([np.min(signal1[n, :]), np.min(signal1[n, :]) + 0.01]) + (n * 0.005)
        plt.plot([first_indtr[n] * dt, first_indtr[n] * dt], aux, c='black')
    plt.grid()
    plt.title(str(first_indtr))
    # """
    return first_indtr


def aux_statistics_prop_cons(sig_prop, sig_cons, Le_time_win, threshold_transition, sim_params, t_transitions, dt,
                             th_percentage=1e-2, filtering=False, cutoff=5):
    """

    Parameters
    ----------
    sig_prop
    sig_cons
    Le_time_win
    threshold_transition
    sim_params
    t_transitions
    dt
    th_percentage
    filtering
    cutoff
    Returns
    statistical descriptors array
    transition_time_array
    list of transition time arrays
    -------

    """
    max_t = sim_params['max_t']
    dt = 1 / sim_params['sfreq']

    # List of transition times if there exist
    t_transition_ini_win, t_transition_mid_win, t_transition_end_win = t_transitions
    # Lists for transition arrays
    tr_p_iw_timeSeries, tr_p_mw_timeSeries, tr_p_ew_timeSeries = [], [], []
    tr_c_iw_timeSeries, tr_c_mw_timeSeries, tr_c_ew_timeSeries = [], [], []

    # Extracting stimuli windows
    piw = sig_prop[:, 0:int(Le_time_win / dt)]  # [, 0s:2s]
    pmw = sig_prop[:, int(Le_time_win / dt):int(2 * Le_time_win / dt)]  # [, 2s:4s]
    pew = sig_prop[:, int(2 * Le_time_win / dt):int(max_t / dt)]  # [, 4s:6s]
    ciw = sig_cons[:, 0:int(Le_time_win / dt)]  # [, 0s:2s]
    cmw = sig_cons[:, int(Le_time_win / dt):int(2 * Le_time_win / dt)]  # [, 2s:4s]
    cew = sig_cons[:, int(2 * Le_time_win / dt):int(max_t / dt)]  # [, 4s:6s]

    # Initial values of time of transition vars
    th_tr = threshold_transition
    th_tr_a = [threshold_transition for _ in range(sig_prop.shape[0])]
    th_tr_a_filt = [0.0 for _ in range(sig_prop.shape[0])]
    # Getting time range of transition period
    if threshold_transition is None:
        th_tr_a = get_transition_time_from_2_signals(piw, pew, dt, th_percentage=th_percentage) * dt
        if filtering:
            th_tr_a_filt = get_transition_time_from_2_signals(piw, pew, dt, th_percentage=5e-2, filtering=filtering,
                                                              cutoff=cutoff) * dt
        """
        # Substract ini and end window to define the transition period (exclude lasst 10 samples to avoid errors)
        ini_minus_end_windows = np.abs(piw[:, :-10] - pew[:, :-10])
        # find indices where the difference is bigger than 1e-6
        ind_tr = np.where(ini_minus_end_windows > 1e-6)
        # Getting indices of unique values (i.e. realizations)
        val_unique, ind_unique = np.unique(ind_tr[0], return_index=True)
        # getting last index (indicating that after that, diff. is lower than 1e-6)
        first_indtr = np.roll(ind_tr[1][list(np.array(ind_unique) - 1)], -1)
        # getting time of transition period
        th_tr_a = first_indtr * dt
        # To extract steady-state and transition signals, use the max. transition time
        th_tr = np.max(th_tr_a)
        # In case a vector in aa and cc are equal, then the unique function ommits the position of that vector,
        # therefore the threshold of transition should be zero.
        # This code calculates which indices in val_unique are not in the list of all signals in sig_prop
        diff = [i for i in np.array(range(sig_prop.shape[0])) if i not in val_unique]
        # if diff has elements, then insert in th_tr_a zeros in the positions associated to the values of these elements
        if len(diff) > 0: np.insert(th_tr_a, diff, 0)
        # Not knowing yet why the size of th_tr_a is still lower than the size of sig_prop, then create a new array of
        # size sig_prop (first dim) populated by th_tr (max. of the found transition times)
        if len(th_tr_a) < sig_prop.shape[0]: th_tr_a = [th_tr for _ in range(sig_prop.shape[0])]
        # """

    # Extracting steady-state parts of stimuli windows: prop and cons ini, mid, and end = pi, pm, pe, ci, cm, ce
    mean_st_pi, median_st_pi, q5_st_pi, q10_st_pi, q90_st_pi, q95_st_pi, min_st_pi, max_st_pi = [[] for _ in range(8)]
    mean_st_pm, median_st_pm, q5_st_pm, q10_st_pm, q90_st_pm, q95_st_pm, min_st_pm, max_st_pm = [[] for _ in range(8)]
    mean_st_pe, median_st_pe, q5_st_pe, q10_st_pe, q90_st_pe, q95_st_pe, min_st_pe, max_st_pe = [[] for _ in range(8)]
    mean_st_ci, median_st_ci, q5_st_ci, q10_st_ci, q90_st_ci, q95_st_ci, min_st_ci, max_st_ci = [[] for _ in range(8)]
    mean_st_cm, median_st_cm, q5_st_cm, q10_st_cm, q90_st_cm, q95_st_cm, min_st_cm, max_st_cm = [[] for _ in range(8)]
    mean_st_ce, median_st_ce, q5_st_ce, q10_st_ce, q90_st_ce, q95_st_ce, min_st_ce, max_st_ce = [[] for _ in range(8)]
    mean_tr_pi, median_tr_pi, q5_tr_pi, q10_tr_pi, q90_tr_pi, q95_tr_pi, min_tr_pi, max_tr_pi = [[] for _ in range(8)]
    mean_tr_pm, median_tr_pm, q5_tr_pm, q10_tr_pm, q90_tr_pm, q95_tr_pm, min_tr_pm, max_tr_pm = [[] for _ in range(8)]
    mean_tr_pe, median_tr_pe, q5_tr_pe, q10_tr_pe, q90_tr_pe, q95_tr_pe, min_tr_pe, max_tr_pe = [[] for _ in range(8)]
    mean_tr_ci, median_tr_ci, q5_tr_ci, q10_tr_ci, q90_tr_ci, q95_tr_ci, min_tr_ci, max_tr_ci = [[] for _ in range(8)]
    mean_tr_cm, median_tr_cm, q5_tr_cm, q10_tr_cm, q90_tr_cm, q95_tr_cm, min_tr_cm, max_tr_cm = [[] for _ in range(8)]
    mean_tr_ce, median_tr_ce, q5_tr_ce, q10_tr_ce, q90_tr_ce, q95_tr_ce, min_tr_ce, max_tr_ce = [[] for _ in range(8)]

    # Getting statistics for transition and steady-state components of mid window if time_transition is provided
    if t_transition_mid_win is not None:
        # Extracting steady-state statistical descriptors of middle window
        st_pm = sig_prop[:, int((Le_time_win + t_transition_mid_win) / dt):int(2 * Le_time_win / dt)]  # [, t_tr:4.0s]
        st_cm = sig_cons[:, int((Le_time_win + t_transition_mid_win) / dt):int(2 * Le_time_win / dt)]  # [, t_tr:4.0s]
        # Getting statistics
        bu, eu = statistics_signal(st_pm, axis=1), statistics_signal(st_cm, axis=1)
        mean_st_pm, median_st_pm, q5_st_pm, q10_st_pm, q90_st_pm, q95_st_pm, min_st_pm, max_st_pm = bu
        mean_st_cm, median_st_cm, q5_st_cm, q10_st_cm, q90_st_cm, q95_st_cm, min_st_cm, max_st_cm = eu

        # Extracting transition statistical descriptors of middle windows
        tr_pm = sig_prop[:, int(Le_time_win / dt):int((Le_time_win + t_transition_mid_win) / dt)]  # [, 2s:t_tr]
        tr_cm = sig_cons[:, int(Le_time_win / dt):int((Le_time_win + t_transition_mid_win) / dt)]  # [, 2s:t_tr]
        # Getting statistics
        hu, nu = statistics_signal(tr_pm, axis=1), statistics_signal(tr_cm, axis=1)
        mean_tr_pm, median_tr_pm, q5_tr_pm, q10_tr_pm, q90_tr_pm, q95_tr_pm, min_tr_pm, max_tr_pm = hu
        mean_tr_cm, median_tr_cm, q5_tr_cm, q10_tr_cm, q90_tr_cm, q95_tr_cm, min_tr_cm, max_tr_cm = nu

        # Setting time-series of transition periods for mid windows
        tr_p_mw_timeSeries = list(tr_pm)
        tr_c_mw_timeSeries = list(tr_cm)

    # Getting statistical descriptors for transition and steady-state components of signals
    for r in range(sig_prop.shape[0]):
        # Extracting steady-state statistical descriptors of initial and ending windows
        st_pi = sig_prop[r, int(th_tr_a[r] / dt):int(Le_time_win / dt)]  # [, t_tr:2s]
        st_pe = sig_prop[r, int((2 * Le_time_win + th_tr_a[r]) / dt):int(max_t / dt)]  # [, t_tr:6.0s]
        st_ci = sig_cons[r, int(th_tr_a[r] / dt):int(Le_time_win / dt)]  # [, t_tr:2s]
        st_ce = sig_cons[r, int((2 * Le_time_win + th_tr_a[r]) / dt):int(max_t / dt)]  # [, t_tr:6.0s]
        # Getting statistics
        au, cu = statistics_signal(st_pi), statistics_signal(st_pe)
        du, fu = statistics_signal(st_ci), statistics_signal(st_ce)
        # Updating final variables
        mean_st_pi.append(au[0]), median_st_pi.append(au[1]), q5_st_pi.append(au[2]), q10_st_pi.append(au[3])
        q90_st_pi.append(au[4]), q95_st_pi.append(au[5]), min_st_pi.append(au[6]), max_st_pi.append(au[7])
        mean_st_pe.append(cu[0]), median_st_pe.append(cu[1]), q5_st_pe.append(cu[2]), q10_st_pe.append(cu[3])
        q90_st_pe.append(cu[4]), q95_st_pe.append(cu[5]), min_st_pe.append(cu[6]), max_st_pe.append(cu[7])
        mean_st_ci.append(du[0]), median_st_ci.append(du[1]), q5_st_ci.append(du[2]), q10_st_ci.append(du[3])
        q90_st_ci.append(du[4]), q95_st_ci.append(du[5]), min_st_ci.append(du[6]), max_st_ci.append(du[7])
        mean_st_ce.append(fu[0]), median_st_ce.append(fu[1]), q5_st_ce.append(fu[2]), q10_st_ce.append(fu[3])
        q90_st_ce.append(fu[4]), q95_st_ce.append(fu[5]), min_st_ce.append(fu[6]), max_st_ce.append(fu[7])

        # Extracting transition parts of statistical descriptors of initial and ending windows
        tr_pi = sig_prop[r, 0:int(th_tr_a[r] / dt)]  # [, 0s:t_tr]
        tr_pe = sig_prop[r, int(2 * Le_time_win / dt):int((2 * Le_time_win + th_tr_a[r]) / dt)]  # [, 4s:t_tr]
        tr_ci = sig_cons[r, 0:int(th_tr_a[r] / dt)]  # [, 0s:t_tr]
        tr_ce = sig_cons[r, int(2 * Le_time_win / dt):int((2 * Le_time_win + th_tr_a[r]) / dt)]  # [, 4s:t_tr]
        # Getting statistics
        gu, ku = statistics_signal(tr_pi), statistics_signal(tr_pe)
        mu, nu = statistics_signal(tr_ci), statistics_signal(tr_ce)
        # Updating final variables
        mean_tr_pi.append(gu[0]), median_tr_pi.append(gu[1]), q5_tr_pi.append(gu[2]), q10_tr_pi.append(gu[3])
        q90_tr_pi.append(gu[4]), q95_tr_pi.append(gu[5]), min_tr_pi.append(gu[6]), max_tr_pi.append(gu[7])
        mean_tr_pe.append(ku[0]), median_tr_pe.append(ku[1]), q5_tr_pe.append(ku[2]), q10_tr_pe.append(ku[3])
        q90_tr_pe.append(ku[4]), q95_tr_pe.append(ku[5]), min_tr_pe.append(ku[6]), max_tr_pe.append(ku[7])
        mean_tr_ci.append(mu[0]), median_tr_ci.append(mu[1]), q5_tr_ci.append(mu[2]), q10_tr_ci.append(mu[3])
        q90_tr_ci.append(mu[4]), q95_tr_ci.append(mu[5]), min_tr_ci.append(mu[6]), max_tr_ci.append(mu[7])
        mean_tr_ce.append(nu[0]), median_tr_ce.append(nu[1]), q5_tr_ce.append(nu[2]), q10_tr_ce.append(nu[3])
        q90_tr_ce.append(nu[4]), q95_tr_ce.append(nu[5]), min_tr_ce.append(nu[6]), max_tr_ce.append(nu[7])

        # Updating time-series of transition periods for ini and end windows
        tr_p_iw_timeSeries.append(tr_pi), tr_p_ew_timeSeries.append(tr_pe)
        tr_c_iw_timeSeries.append(tr_ci), tr_c_ew_timeSeries.append(tr_ce)

        # Getting statistics for transition and steady-state components of middle windows
        if t_transition_mid_win is None:
            # Extracting steady-state statistical descriptors of middle windows
            st_pm = sig_prop[r, int((Le_time_win + th_tr_a[r]) / dt):int(2 * Le_time_win / dt)]  # [, t_tr:4.0s]
            st_cm = sig_cons[r, int((Le_time_win + th_tr_a[r]) / dt):int(2 * Le_time_win / dt)]  # [, t_tr:4.0s]
            bu, eu = statistics_signal(st_pm), statistics_signal(st_cm)
            # Getting statistics
            mean_st_pm.append(bu[0]), median_st_pm.append(bu[1]), q5_st_pm.append(bu[2]), q10_st_pm.append(bu[3])
            q90_st_pm.append(bu[4]), q95_st_pm.append(bu[5]), min_st_pm.append(bu[6]), max_st_pm.append(bu[7])
            mean_st_cm.append(eu[0]), median_st_cm.append(eu[1]), q5_st_cm.append(eu[2]), q10_st_cm.append(eu[3])
            q90_st_cm.append(eu[4]), q95_st_cm.append(eu[5]), min_st_cm.append(eu[6]), max_st_cm.append(eu[7])

            # Extracting transition statistical descriptors of middle windows
            tr_pm = sig_prop[r, int(Le_time_win / dt):int((Le_time_win + th_tr_a[r]) / dt)]  # [, 2s:t_tr]
            tr_cm = sig_cons[r, int(Le_time_win / dt):int((Le_time_win + th_tr_a[r]) / dt)]  # [, 2s:t_tr]
            # Getting statistics
            hu, nu = statistics_signal(tr_pm), statistics_signal(tr_cm)
            # Updating final variables
            mean_tr_pm.append(hu[0]), median_tr_pm.append(hu[1]), q5_tr_pm.append(hu[2]), q10_tr_pm.append(hu[3])
            q90_tr_pm.append(hu[4]), q95_tr_pm.append(hu[5]), min_tr_pm.append(hu[6]), max_tr_pm.append(hu[7])
            mean_tr_cm.append(nu[0]), median_tr_cm.append(nu[1]), q5_tr_cm.append(nu[2]), q10_tr_cm.append(nu[3])
            q90_tr_cm.append(nu[4]), q95_tr_cm.append(nu[5]), min_tr_cm.append(nu[6]), max_tr_cm.append(nu[7])

            # Updating time-series of transition periods for mid windows
            tr_p_mw_timeSeries.append(tr_pm)
            tr_c_mw_timeSeries.append(tr_cm)

    tr_timeSeries = [tr_p_iw_timeSeries, tr_p_mw_timeSeries, tr_p_ew_timeSeries,
                     tr_c_iw_timeSeries, tr_c_mw_timeSeries, tr_c_ew_timeSeries]
    tr_timeSeries = [list(piw), list(pmw), list(pew), list(ciw), list(cmw), list(cew)]

    return np.array([  # For steady-state
        np.array(mean_st_pi), np.array(median_st_pi), np.array(q5_st_pi), np.array(q10_st_pi),
        np.array(q90_st_pi), np.array(q95_st_pi), np.array(min_st_pi), np.array(max_st_pi),  # 7
        np.array(mean_st_pm), np.array(median_st_pm), np.array(q5_st_pm), np.array(q10_st_pm),
        np.array(q90_st_pm), np.array(q95_st_pm), np.array(min_st_pm), np.array(max_st_pm),  # 15
        np.array(mean_st_pe), np.array(median_st_pe), np.array(q5_st_pe), np.array(q10_st_pe),
        np.array(q90_st_pe), np.array(q95_st_pe), np.array(min_st_pe), np.array(max_st_pe),  # 23
        np.array(mean_st_ci), np.array(median_st_ci), np.array(q5_st_ci), np.array(q10_st_ci),
        np.array(q90_st_ci), np.array(q95_st_ci), np.array(min_st_ci), np.array(max_st_ci),  # 31
        np.array(mean_st_cm), np.array(median_st_cm), np.array(q5_st_cm), np.array(q10_st_cm),
        np.array(q90_st_cm), np.array(q95_st_cm), np.array(min_st_cm), np.array(max_st_cm),  # 39
        np.array(mean_st_ce), np.array(median_st_ce), np.array(q5_st_ce), np.array(q10_st_ce),
        np.array(q90_st_ce), np.array(q95_st_ce), np.array(min_st_ce), np.array(max_st_ce),  # 47
        # For transition-state
        np.array(max_tr_pi), np.array(max_tr_pm), np.array(max_tr_pe),
        np.array(min_tr_pi), np.array(min_tr_pm), np.array(min_tr_pe)]  # 50
    ), th_tr_a, tr_timeSeries, piw, pmw, pew, th_tr_a_filt


def stat_tr_slid(signals, win_r, win_l):
    res = []
    for signal in signals:
        res.append(signal[:, win_r[0]:win_r[1] - win_l])
    return res

    
def statistics_signal(signal, axis=0):
    # mean, median, q5, q10, 190, 195, min, max
    return (np.mean(signal, axis=axis), np.median(signal, axis=axis), np.quantile(signal, 0.05, axis=axis),
            np.quantile(signal, 0.1, axis=axis), np.quantile(signal, 0.9, axis=axis),
            np.quantile(signal, 0.95, axis=axis), np.min(signal, axis=axis), np.max(signal, axis=axis))


def aux_statistics_sin(mp_signal, coff, sfreq):
    low_pass_mempot = lowpass(mp_signal, coff, sfreq)
    high_pass_mempot = highpass(mp_signal, coff, sfreq)
    dt = 1 / sfreq
    pos_mempot = None
    mem_pot_low_filt = None
    mem_pot_high_filt = None
    neg_mempot = mp_signal[int(5 / dt): int(10 / dt)]
    pos_mempot_low_filt = None
    neg_mempot_low_filt = low_pass_mempot[int(5 / dt): int(10 / dt)]
    pos_mempot_high_filt = None
    neg_mempot_high_filt = high_pass_mempot[int(5 / dt): int(10 / dt)]

    pos_mempot = mp_signal[int(10 / dt): int(15 / dt)]
    pos_mempot_low_filt = low_pass_mempot[int(10 / dt): int(15 / dt)]
    pos_mempot_high_filt = high_pass_mempot[int(10 / dt): int(15 / dt)]
    mem_pot_low_filt = low_pass_mempot[int(5 / dt): int(15 / dt)]
    mem_pot_high_filt = high_pass_mempot[int(5 / dt): int(15 / dt)]

    st = np.array([np.max(pos_mempot_low_filt), np.min(neg_mempot_low_filt),
                   np.quantile(pos_mempot_high_filt, 0.1), np.quantile(pos_mempot_high_filt, 0.9),
                   np.quantile(neg_mempot_high_filt, 0.1), np.quantile(neg_mempot_high_filt, 0.9),
                   np.max(mem_pot_low_filt), np.min(mem_pot_low_filt),
                   np.quantile(mem_pot_high_filt, 0.1), np.quantile(mem_pot_high_filt, 0.9)])

    return st, low_pass_mempot, high_pass_mempot


def sec2hour(secs, exp, realizations):
    rep = int(np.ceil(100 / realizations))
    return secs * exp * rep / 3600

# ini_high_rate = 50  # 50
# step_high_rate = 50  # 10 # 50
# ini_low_rate = 8  # 8
# step_low_rate = 2  # 1  # 2
# proportion = 0.5  # 0.5
# iterator = 10  # 56  # 12
# # auxiliars
# ihr = ini_high_rate
# shr = step_high_rate
# ilr = ini_low_rate
# slr = step_low_rate
# pr = proportion
# mean_rates = [[(shr * i) + ihr, (slr * i) + ilr, (shr * i) + ihr] for i in range(iterator)]
# max_oscils = [[((shr * i) + ihr) * pr, ((slr * i) + ilr) * pr, ((slr * i) + ilr) * pr] for i in range(iterator)]
# fix_rates = [[(slr * i) + ilr, (shr * i) + ihr, (slr * i) + ilr] for i in range(iterator)]
