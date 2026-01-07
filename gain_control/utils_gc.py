import os.path
import matplotlib.pyplot as plt
import scipy.signal
import numpy as np
from synaptic_dynamic_models.TM import TM_model
from synaptic_dynamic_models.MSSM import MSSM_model
from spiking_neuron_models.LIF import LIF_model
from synaptic_dynamic_models.simple_depression import Simple_Depression
from libraries.frequency_analysis import Freq_analysis
from utils import *


# ******************************************************************************************************************
# Local variables
stat_list = ['st_ini_prop_mean', 'st_ini_prop_med', 'st_ini_prop_q5', 'st_ini_prop_q10', 'st_ini_prop_q90', 'st_ini_prop_q95', 'st_ini_prop_min', 'st_ini_prop_max',
             'st_mid_prop_mean', 'st_mid_prop_med', 'st_mid_prop_q5', 'st_mid_prop_q10', 'st_mid_prop_q90', 'st_mid_prop_q95', 'st_mid_prop_min', 'st_mid_prop_max',
             'st_end_prop_mean', 'st_end_prop_med', 'st_end_prop_q5', 'st_end_prop_q10', 'st_end_prop_q90', 'st_end_prop_q95', 'st_end_prop_min', 'st_end_prop_max',
             'st_ini_fix_mean', 'st_ini_fix_med', 'st_ini_fix_q5', 'st_ini_fix_q10', 'st_ini_fix_q90', 'st_ini_fix_q95', 'st_ini_fix_min', 'st_ini_fix_max',
             'st_mid_fix_mean', 'st_mid_fix_med', 'st_mid_fix_q5', 'st_mid_fix_q10', 'st_mid_fix_q90', 'st_mid_fix_q95', 'st_mid_fix_min', 'st_mid_fix_max',
             'st_end_fix_mean', 'st_end_fix_med', 'st_end_fix_q5', 'st_end_fix_q10', 'st_end_fix_q90', 'st_end_fix_q95', 'st_end_fix_min', 'st_end_fix_max',
             'w_ini_prop_mean', 'w_ini_prop_med', 'w_ini_prop_q5', 'w_ini_prop_q10', 'w_ini_prop_q90', 'w_ini_prop_q95', 'w_ini_prop_min', 'w_ini_prop_max',
             'w_mid_prop_mean', 'w_mid_prop_med', 'w_mid_prop_q5', 'w_mid_prop_q10', 'w_mid_prop_q90', 'w_mid_prop_q95', 'w_mid_prop_min', 'w_mid_prop_max',
             'w_end_prop_mean', 'w_end_prop_med', 'w_end_prop_q5', 'w_end_prop_q10', 'w_end_prop_q90', 'w_end_prop_q95', 'w_end_prop_min', 'w_end_prop_max',
             'w_ini_fix_mean', 'w_ini_fix_med', 'w_ini_fix_q5', 'w_ini_fix_q10', 'w_ini_fix_q90', 'w_ini_fix_q95', 'w_ini_fix_min', 'w_ini_fix_max',
             'w_mid_fix_mean', 'w_mid_fix_med', 'w_mid_fix_q5', 'w_mid_fix_q10', 'w_mid_fix_q90', 'w_mid_fix_q95', 'w_mid_fix_min', 'w_mid_fix_max',
             'w_end_fix_mean', 'w_end_fix_med', 'w_end_fix_q5', 'w_end_fix_q10', 'w_end_fix_q90', 'w_end_fix_q95', 'w_end_fix_min', 'w_end_fix_max',
             'mtr_ini_prop_mean', 'mtr_ini_prop_med', 'mtr_ini_prop_q5', 'mtr_ini_prop_q10', 'mtr_ini_prop_q90', 'mtr_ini_prop_q95', 'mtr_ini_prop_min', 'mtr_ini_prop_max',
             'mtr_mid_prop_mean', 'mtr_mid_prop_med', 'mtr_mid_prop_q5', 'mtr_mid_prop_q10', 'mtr_mid_prop_q90', 'mtr_mid_prop_q95', 'mtr_mid_prop_min', 'mtr_mid_prop_max',
             'mtr_end_prop_mean', 'mtr_end_prop_med', 'mtr_end_prop_q5', 'mtr_end_prop_q10', 'mtr_end_prop_q90', 'mtr_end_prop_q95', 'mtr_end_prop_min', 'mtr_end_prop_max',
             'mtr_ini_fix_mean', 'mtr_ini_fix_med', 'mtr_ini_fix_q5', 'mtr_ini_fix_q10', 'mtr_ini_fix_q90', 'mtr_ini_fix_q95', 'mtr_ini_fix_min', 'mtr_ini_fix_max',
             'mtr_mid_fix_mean', 'mtr_mid_fix_med', 'mtr_mid_fix_q5', 'mtr_mid_fix_q10', 'mtr_mid_fix_q90', 'mtr_mid_fix_q95', 'mtr_mid_fix_min', 'mtr_mid_fix_max',
             'mtr_end_fix_mean', 'mtr_end_fix_med', 'mtr_end_fix_q5', 'mtr_end_fix_q10', 'mtr_end_fix_q90', 'mtr_end_fix_q95', 'mtr_end_fix_min', 'mtr_end_fix_max',
             'initial_frequencies', 'stp_model', 'name_params', 'dyn_synapse', 'num_synapses', 'syn_params',
             'sim_params', 'lif_params', 'lif_params2', 'prop_rate_change_a', 'fix_rate_change_a', 'num_changes_rate',
             'description', 'seeds', 'realizations', 't_realizations', 'time_transition']
stat_list_sin = ['vec_max_mp_pos', 'vec_min_mp_neg', 'vec_q1_mp_pos', 'vec_q90_mp_pos',
                 'vec_q1_mp_neg', 'vec_q90_mp_neg', 'vec_max_mp', 'vec_min_mp', 'vec_q1_mp', 'vec_q90_mp']

# Parameters for LiF neuron
lif_params = {'V_threshold': np.array([50 for _ in range(1)]), 'V_reset': np.array([-70 for _ in range(1)]),
              'tau_m': np.array([30e-3 for _ in range(1)]),
              'g_L': np.array([7.5e-2 for _ in range(1)]),
              'V_init': np.array([-70 for _ in range(1)]), 'V_equilibrium': np.array([-70 for _ in range(1)]),
              't_refractory': np.array([0.01 for _ in range(1)])}


def get_params_stp(name_model, ind):
    syn_params = None
    description = ""
    name_params = None

    if name_model == "MSSM":
        name_params = params_name_mssm
    elif name_model == "TM":
        name_params = params_name_tm

    # (Experiment 2) freq. response decay around 100Hz {5: 0.99, 10: 0.9, 20: 0.8, 50: 0.7, 100: 0.5, 200: 0.2}
    if name_model == "MSSM" and ind == 2:
        description = "MSSM " + str(ind) + " Experiment, decay around 100Hz"
        syn_params = [5.42451963e-02, 2.92925980e+00, 6.67821125e+01, 1.80143000e-02, 7.54167519e-01,
                      5.99119322e+01, 9.94215228e-01, 1.03825167e-03,
                      (4.52507712e-01 * 0.075) / 779.1984, 1.00243185e-03]  # 4.52507712e-01, g_L = 5e-6 homogeneous / 4.6e-3 poisson
    if name_model == "TM" and ind == 2:
        description = "TM " + str(ind) + " Experiment, decay around 100Hz"
        syn_params = [1.32748878e-01, 2.19116160e-02, 1.32465006e-01, 2.16882855e-00 * 2.60498474e-1, 1.00766466e-03]  #
    # (Experiment 3) freq. response decay around 10Hz {2: 0.88946682, 5: 0.70155924, 10: 0.57552428, 15: 0.2, 20: 0.1, 50: 0.07}
    if name_model == "MSSM" and ind == 3:
        description = "MSSM " + str(ind) + " Experiment, decay around 10Hz"
        syn_params = [6.53659368e-03, 1.75660742e-01, 3.17123087e+01, 1.78659320e-01, 2.50362727e-01,
                      9.12004545e+01, 9.13420672e-01, 2.14204288e-03, 5.20907890e-03 / 4.29, 4.32890680e-03]  # 5.20907890e-01, g_L = 2.6e-2 homogeneous / 4.6e-2 poisson
    if name_model == "TM" and ind == 3:
        description = "TM " + str(ind) + " Experiment, decay around 10Hz"
        syn_params = [2.37698417e-01, 3.30564024e-01, 8.51177265e-01, 3.67454564e-01 * 8.58679963e-1, 3.04982285e-03]  #
    # (Experiment 4) freq. response from Gain Control paper {5: 0.785, 10: 0.6, 20: 0.415, 50: 0.205, 100: 0.115}
    if name_model == "MSSM" and ind == 4:
        description = "MSSM " + str(ind) + " Experiment, Gain-Control paper"
        syn_params = [7.85735182e-02, 4.56599128e-01, 1.46835212e+00, 1.63998958e-01, 2.41885797e-04,
                      5.84619146e+01, 8.00871281e-01, 1.50280526e-03,
                      (5.94890729e-02 * 0.075) / 1.0217, 1.75609424e-03]  # 5.94890729e-01, g_L = 5.4e-4 homogeneous / 3.21e-3 poisson [1.36e-3 for subthreshold]  # g_L = 5.4e-5 for static synapse
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
    # (Experiment 6) freq. response decay around 10Hz {[2:0.88946682, 4:0.70155924, 5.4:0.57552428, 10:0.2, 15:0.1, 30:0.06, 50:0.035]
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
        syn_params = [0.03, 530e-3, 130e-3, 1540 * 2.32722819e-05, 2.5e-3]
    # params_s_dep = {'tau_g': 2e-3, 'tau_alpha': 300e-3, 'g0': 0.075, 'f': 0.75}
    assert syn_params is not None, "Not parameters for model %s and index %d" % (name_model, ind)

    return syn_params, description, name_params


def static_synapse(lif, Input, g):
    # Number of samples
    L = Input.shape[1]

    # Running model
    for it in range(L):
        # Evaluating change in LIF neuron - membrane potential
        lif.update_state(Input[:, it] * g, it)
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
        lif.update_state(mssm.get_output()[:, it], it)

        if lif_n is not None:
            lif_n.update_state(mssm.N[:, it], it)

    # Computing output spike event in the last ISI
    it = L
    lif.membrane_potential[0, -1] = lif.membrane_potential[0, -2]
    if lif_n is not None:
        lif_n.membrane_potential[0, -1] = lif_n.membrane_potential[0, -2]
    # spike_range = (mssm.time_spike_events[-1], it)
    # mssm.compute_output_spike_event(spike_range, mssm.get_output())


def model_stp_parallel(stp_model, lif, params, Input, lif_n=None):
    # Update parameters and initial conditions
    stp_model.set_model_params(params)

    # Number of samples
    L = Input.shape[1]
    num_neu = lif.n_neurons
    num_syn = int(stp_model.n_syn / num_neu)

    # Creating connectiviy matrix
    id_mat = np.eye(num_neu)
    id_rep = np.tile(np.eye(num_syn), num_neu).T
    # connectivity = np.zeros((Input.shape[0], num_neu))
    # for n in range(num_neu):
    #     connectivity[:, n] = np.repeat(aux_matrix[:, n], num_syn, axis=0)
    connectivity = np.repeat(id_mat, num_syn, axis=0)

    # Cech of correct creation of connectivity matrix
    for n in range(num_neu):
        if np.sum(connectivity[n * num_syn: (n + 1) * num_syn, :]) > num_syn:
            print("error in files from %d to %d" % (n * num_syn, (n + 1) * num_syn))
        if np.sum(connectivity[:, n]) > num_syn:
            print("error in column %d" % n)

    # Running model
    for it in range(L):
        # Evaluating TM model
        stp_model.evaluate_model_euler(Input[:, it], it)

        # Converting model output into matrix alike
        aux_input = np.resize(np.repeat(stp_model.get_output()[:, it], num_neu), (num_syn * num_neu, num_neu))
        c = aux_input * connectivity
        input_lif = np.matmul(c.T, id_rep).T

        # Evaluating change in LIF neuron - membrane potential
        # lif.update_state(mssm.get_output()[:, it], it)
        lif.update_state(input_lif, it)

        if lif_n is not None:
            # Converting model output into matrix alike
            aux_input_n = np.resize(np.repeat(stp_model.N[:, it], num_neu), (num_syn * num_neu, num_neu))
            c = aux_input_n * connectivity
            input_lif_n = np.matmul(c.T, id_rep).T * stp_model.params[
                'k_EPSP'] / 2  # k_EPSP/2, factor to transform N(t) into the small range as Epsp(t)

            # Evaluating change in LIF neuron - membrane potential
            # lif.update_state(mssm.get_output()[:, it], it)
            lif_n.update_state(input_lif_n, it)

    # Computing output spike event in the last ISI
    it = L
    lif.membrane_potential[0, -1] = lif.membrane_potential[0, -2]
    if lif_n is not None:
        lif_n.membrane_potential[0, -1] = lif_n.membrane_potential[0, -2]
    # spike_range = (mssm.time_spike_events[-1], it)
    # mssm.compute_output_spike_event(spike_range, mssm.get_output())


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
        lif.update_state(s_dep.get_output()[:, it], it)
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
            mid_index = ind_kernel_no_spikes + 1  # to select the indices of the kernel corresponding to the middle (where a new spike is set)

            # Sampling integers uniformly, the number of samples corresponds to the double of the number of missing
            # events for each realization
            rng = np.random.default_rng(seed)
            aux = rng.choice(np.arange(0, mid_index.shape[0]), size=int(miss_events[col]), replace=False)
            new_ind = mid_index[aux.tolist()]

            # Updating the corresponding input spike train
            Input_test[col, list(new_ind)] = 1

    return Input_test


def simple_spike_train(sfreq, rate, L, num_realizations=1, poisson=False, seeds=None, correction=False,
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

            """
            # Deleting consecutive events
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
                    arr = np.reshape(Input_aux2[:, :int(L/3) * 3], (num_realizations, int(L/3), 3))
                    # Finding indices in array where the kernel is found
                    cond = (arr == kernel).all(axis=2)

                for col in realization_with_missing_events:
                    # Available spaces to fill with a spike
                    b = np.where(cond[col, :] == True)[0] * 3
                    c = b + 1  # to select the indices of the kernel corresponding to the middle (where a new spike is set)

                    # Sampling integers uniformly, the number of samples corresponds to the double of the number of missing
                    # events for each realization
                    rng = np.random.default_rng(seed)
                    aux = rng.choice(np.arange(0, c.shape[0]), size=int(miss_events[col]), replace=False)
                    new_ind = c[aux.tolist()]

                    # Updating the corresponding input spike train
                    Input_test[col, list(new_ind)] = 1
            # """

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
        histograms.append(([], []))  # empty histogram and bins
    else:
        # isi_array = np.array(isi_list)
        isi_array = np.array(isis)
        max_val = max_time_hist if max_time_hist is not None else isi_array.max()
        bins = np.arange(0, max_val + bin_size, bin_size)
        hist, bin_edges = np.histogram(isi_array, bins=bins)
        histograms.append((hist.tolist(), bin_edges.tolist()))

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


def aux_statistics_prop_cons(sig_prop, sig_cons, Le_time_win, threshold_transition, sim_params, t_transition_mid_win):
    """

    Parameters
    ----------
    sig_prop
    sig_cons
    Le_time_win
    threshold_transition
    sim_params
    t_transition_mid_win
    Returns
    -------

    """
    max_t = sim_params['max_t']
    dt = 1 / sim_params['sfreq']

    # Extracting stimuli windows
    aa = sig_prop[:, 0:int(Le_time_win / dt)]  # [, 0s:2s]
    bb = sig_prop[:, int(Le_time_win / dt):int(2 * Le_time_win / dt)]  # [, 2s:4s]
    cc = sig_prop[:, int(2 * Le_time_win / dt):int(max_t / dt)]  # [, 4s:6s]
    dd = sig_cons[:, 0:int(Le_time_win / dt)]  # [, 0s:2s]
    ee = sig_cons[:, int(Le_time_win / dt):int(2 * Le_time_win / dt)]  # [, 2s:4s]
    ff = sig_cons[:, int(2 * Le_time_win / dt):int(max_t / dt)]  # [, 4s:6s]

    # Initial values of time of transition vars
    th_tr = threshold_transition
    th_tr_a = [threshold_transition for _ in range(sig_prop.shape[0])]

    # Getting time range of transition period
    if threshold_transition is None:
        ini_minus_end_windows = np.abs(aa[:, :-10] - cc[:, :-10])
        ind_tr = np.where(ini_minus_end_windows > 1e-4)  # find indices where the difference is bigger than 1e-6
        val_unique, ind_unique = np.unique(ind_tr[0], return_index=True)  # Getting indices of unique values (i.e. realizations)
        first_indtr = np.roll(ind_tr[1][list(np.array(ind_unique) - 1)], -1)  # getting last index (indicating that after that, diff. is lower than 1e-6)
        th_tr_a = first_indtr * dt  # getting time of transition period
        th_tr = np.max(th_tr_a)  # To extract steady-state and transition signals, use the max. transition time
        # In case a vector in aa and cc are equal, then the unique function ommits the position of that vector,
        # therefore the threshold of transition should be zero.
        # This code calculates which indices in val_unique are not in the list of all signals in sig_prop
        diff = [i for i in np.array(range(sig_prop.shape[0])) if i not in val_unique]
        # if the diff has elements, then insert in th_tr_a zeros in the positions associated to the values of these elements
        if len(diff) > 0: np.insert(th_tr_a, diff, 0)

    # Extracting steady-state parts of stimuli windows
    mean_a, median_a, q5_a, q10_a, q90_a, q95_a, min_a, max_a = [], [], [], [], [], [], [], []
    mean_b, median_b, q5_b, q10_b, q90_b, q95_b, min_b, max_b = [], [], [], [], [], [], [], []
    mean_c, median_c, q5_c, q10_c, q90_c, q95_c, min_c, max_c = [], [], [], [], [], [], [], []
    mean_d, median_d, q5_d, q10_d, q90_d, q95_d, min_d, max_d = [], [], [], [], [], [], [], []
    mean_e, median_e, q5_e, q10_e, q90_e, q95_e, min_e, max_e = [], [], [], [], [], [], [], []
    mean_f, median_f, q5_f, q10_f, q90_f, q95_f, min_f, max_f = [], [], [], [], [], [], [], []
    mean_g, median_g, q5_g, q10_g, q90_g, q95_g, min_g, max_g = [], [], [], [], [], [], [], []
    mean_h, median_h, q5_h, q10_h, q90_h, q95_h, min_h, max_h = [], [], [], [], [], [], [], []
    mean_k, median_k, q5_k, q10_k, q90_k, q95_k, min_k, max_k = [], [], [], [], [], [], [], []
    mean_m, median_m, q5_m, q10_m, q90_m, q95_m, min_m, max_m = [], [], [], [], [], [], [], []
    mean_n, median_n, q5_n, q10_n, q90_n, q95_n, min_n, max_n = [], [], [], [], [], [], [], []
    mean_o, median_o, q5_o, q10_o, q90_o, q95_o, min_o, max_o = [], [], [], [], [], [], [], []
    
    # Getting statistics for transition and steady-state components of mid window if time_transition is provided
    if t_transition_mid_win is not None:
        b = sig_prop[:, int((Le_time_win + t_transition_mid_win) / dt):int(2 * Le_time_win / dt)]  # [, t_tr:4.0s]
        e = sig_cons[:, int((Le_time_win + t_transition_mid_win) / dt):int(2 * Le_time_win / dt)]  # [, t_tr:4.0s]
        # Getting statistics
        bu, eu = statistics_signal(b, axis=1), statistics_signal(e, axis=1)
        mean_b, median_b, q5_b, q10_b, q90_b, q95_b, min_b, max_b = bu
        mean_e, median_e, q5_e, q10_e, q90_e, q95_e, min_e, max_e = eu

        # Extracting transition parts of stimuli windows
        h = sig_prop[:, int(Le_time_win / dt):int((Le_time_win + t_transition_mid_win) / dt)]  # [, 2s:t_tr]
        n = sig_cons[:, int(Le_time_win / dt):int((Le_time_win + t_transition_mid_win) / dt)]  # [, 2s:t_tr]
        # Getting statistics
        hu, nu = statistics_signal(h, axis=1), statistics_signal(n, axis=1)
        mean_h, median_h, q5_h, q10_h, q90_h, q95_h, min_h, max_h = hu
        mean_n, median_n, q5_n, q10_n, q90_n, q95_n, min_n, max_n = nu

    # Getting statistics for transition and steady-state components of signals
    for r in range(sig_prop.shape[0]):
        # Getting statistics for transition and steady-state components of initial and ending windows
        # Extracting steady-state parts of stimuli windows
        a = sig_prop[r, int(th_tr_a[r] / dt):int(Le_time_win / dt)]  # [, t_tr:2s]
        c = sig_prop[r, int((2 * Le_time_win + th_tr_a[r]) / dt):int(max_t / dt)]  # [, t_tr:6.0s]
        d = sig_cons[r, int(th_tr_a[r] / dt):int(Le_time_win / dt)]  # [, t_tr:2s]
        f = sig_cons[r, int((2 * Le_time_win + th_tr_a[r]) / dt):int(max_t / dt)]  # [, t_tr:6.0s]
        # Getting statistics
        au, cu, du, fu = statistics_signal(a), statistics_signal(c), statistics_signal(d), statistics_signal(f)
        # Updating final variables
        mean_a.append(au[0]), median_a.append(au[1]), q5_a.append(au[2]), q10_a.append(au[3]), q90_a.append(au[4]), q95_a.append(au[5]), min_a.append(au[6]), max_a.append(au[7])
        mean_c.append(cu[0]), median_c.append(cu[1]), q5_c.append(cu[2]), q10_c.append(cu[3]), q90_c.append(cu[4]), q95_c.append(cu[5]), min_c.append(cu[6]), max_c.append(cu[7])
        mean_d.append(du[0]), median_d.append(du[1]), q5_d.append(du[2]), q10_d.append(du[3]), q90_d.append(du[4]), q95_d.append(du[5]), min_d.append(du[6]), max_d.append(du[7])
        mean_f.append(fu[0]), median_f.append(fu[1]), q5_f.append(fu[2]), q10_f.append(fu[3]), q90_f.append(fu[4]), q95_f.append(fu[5]), min_f.append(fu[6]), max_f.append(fu[7])

        # Extracting transition parts of stimuli windows
        g = sig_prop[r, 0:int(th_tr_a[r] / dt)]  # [, 0s:t_tr]
        k = sig_prop[r, int(2 * Le_time_win / dt):int((2 * Le_time_win + th_tr_a[r]) / dt)]  # [, 4s:t_tr]
        m = sig_cons[r, 0:int(th_tr_a[r] / dt)]  # [, 0s:t_tr]
        o = sig_cons[r, int(2 * Le_time_win / dt):int((2 * Le_time_win + th_tr_a[r]) / dt)]  # [, 4s:t_tr]
        # Getting statistics
        gu, ku, mu, ou = statistics_signal(g), statistics_signal(k), statistics_signal(m), statistics_signal(o)
        # Updating final variables
        mean_g.append(gu[0]), median_g.append(gu[1]), q5_g.append(gu[2]), q10_g.append(gu[3]), q90_g.append(gu[4]), q95_g.append(gu[5]), min_g.append(gu[6]), max_g.append(gu[7])
        mean_k.append(ku[0]), median_k.append(ku[1]), q5_k.append(ku[2]), q10_k.append(ku[3]), q90_k.append(ku[4]), q95_k.append(ku[5]), min_k.append(ku[6]), max_k.append(ku[7])
        mean_m.append(mu[0]), median_m.append(mu[1]), q5_m.append(mu[2]), q10_m.append(mu[3]), q90_m.append(mu[4]), q95_m.append(mu[5]), min_m.append(mu[6]), max_m.append(mu[7])
        mean_o.append(ou[0]), median_o.append(ou[1]), q5_o.append(ou[2]), q10_o.append(ou[3]), q90_o.append(ou[4]), q95_o.append(ou[5]), min_o.append(ou[6]), max_o.append(ou[7])

        # Getting statistics for transition and steady-state components of middle windows
        if t_transition_mid_win is None:
            # Extracting steady-state parts of stimuli windows
            b = sig_prop[r, int((Le_time_win + th_tr_a[r]) / dt):int(2 * Le_time_win / dt)]  # [, t_tr:4.0s]
            e = sig_cons[r, int((Le_time_win + th_tr_a[r]) / dt):int(2 * Le_time_win / dt)]  # [, t_tr:4.0s]
            bu, eu = statistics_signal(b), statistics_signal(e)
            # Getting statistics
            mean_b.append(bu[0]), median_b.append(bu[1]), q5_b.append(bu[2]), q10_b.append(bu[3]), q90_b.append(bu[4]), q95_b.append(bu[5]), min_b.append(bu[6]), max_b.append(bu[7])
            mean_e.append(eu[0]), median_e.append(eu[1]), q5_e.append(eu[2]), q10_e.append(eu[3]), q90_e.append(eu[4]), q95_e.append(eu[5]), min_e.append(eu[6]), max_e.append(eu[7])

            # Extracting transition parts of stimuli windows
            h = sig_prop[r, int(Le_time_win / dt):int((Le_time_win + th_tr_a[r]) / dt)]  # [, 2s:t_tr]
            n = sig_cons[r, int(Le_time_win / dt):int((Le_time_win + th_tr_a[r]) / dt)]  # [, 2s:t_tr]
            # Getting statistics
            hu, nu = statistics_signal(h), statistics_signal(n)
            # Updating final variables
            mean_h.append(hu[0]), median_h.append(hu[1]), q5_h.append(hu[2]), q10_h.append(hu[3]), q90_h.append(hu[4]), q95_h.append(hu[5]), min_h.append(hu[6]), max_h.append(hu[7])
            mean_n.append(nu[0]), median_n.append(nu[1]), q5_n.append(nu[2]), q10_n.append(nu[3]), q90_n.append(nu[4]), q95_n.append(nu[5]), min_n.append(nu[6]), max_n.append(nu[7])

    return np.array([# For steady-state
                     np.array(mean_a), np.array(median_a), np.array(q5_a), np.array(q10_a), np.array(q90_a), np.array(q95_a), np.array(min_a), np.array(max_a),  # 7
                     np.array(mean_b), np.array(median_b), np.array(q5_b), np.array(q10_b), np.array(q90_b), np.array(q95_b), np.array(min_b), np.array(max_b),  # 15
                     np.array(mean_c), np.array(median_c), np.array(q5_c), np.array(q10_c), np.array(q90_c), np.array(q95_c), np.array(min_c), np.array(max_c),  # 23
                     np.array(mean_d), np.array(median_d), np.array(q5_d), np.array(q10_d), np.array(q90_d), np.array(q95_d), np.array(min_d), np.array(max_d),  # 31
                     np.array(mean_e), np.array(median_e), np.array(q5_e), np.array(q10_e), np.array(q90_e), np.array(q95_e), np.array(min_e), np.array(max_e),  # 39
                     np.array(mean_f), np.array(median_f), np.array(q5_f), np.array(q10_f), np.array(q90_f), np.array(q95_f), np.array(min_f), np.array(max_f),  # 47
                     # For all window
                     np.mean(aa, axis=1), np.median(aa, axis=1), np.quantile(aa, 0.05, axis=1), np.quantile(aa, 0.1, axis=1),  # 51
                     np.quantile(aa, 0.9, axis=1), np.quantile(aa, 0.95, axis=1), np.min(aa, axis=1), np.max(aa, axis=1),  # 55
                     np.mean(bb, axis=1), np.median(bb, axis=1), np.quantile(bb, 0.05, axis=1), np.quantile(bb, 0.1, axis=1),  # 59
                     np.quantile(bb, 0.9, axis=1), np.quantile(bb, 0.95, axis=1), np.min(bb, axis=1), np.max(bb, axis=1),  # 63
                     np.mean(cc, axis=1), np.median(cc, axis=1), np.quantile(cc, 0.05, axis=1), np.quantile(cc, 0.1, axis=1),  # 67
                     np.quantile(cc, 0.9, axis=1), np.quantile(cc, 0.95, axis=1), np.min(cc, axis=1), np.max(cc, axis=1),  # 71
                     np.mean(dd, axis=1), np.median(dd, axis=1), np.quantile(dd, 0.05, axis=1), np.quantile(dd, 0.1, axis=1),  # 75
                     np.quantile(dd, 0.9, axis=1), np.quantile(dd, 0.95, axis=1), np.min(dd, axis=1), np.max(dd, axis=1),  # 79
                     np.mean(ee, axis=1), np.median(ee, axis=1), np.quantile(ee, 0.05, axis=1), np.quantile(ee, 0.1, axis=1),  # 83
                     np.quantile(ee, 0.9, axis=1), np.quantile(ee, 0.95, axis=1), np.min(ee, axis=1), np.max(ee, axis=1),  # 87
                     np.mean(ff, axis=1), np.median(ff, axis=1), np.quantile(ff, 0.05, axis=1), np.quantile(ff, 0.1, axis=1),  # 91
                     np.quantile(ff, 0.9, axis=1), np.quantile(ff, 0.95, axis=1), np.min(ff, axis=1), np.max(ff, axis=1),  # 95
                     # For transition-state
                     np.array(mean_g), np.array(median_g), np.array(q5_g), np.array(q10_g), np.array(q90_g), np.array(q95_g), np.array(min_g), np.array(max_g),  # 103
                     np.array(mean_h), np.array(median_h), np.array(q5_h), np.array(q10_h), np.array(q90_h), np.array(q95_h), np.array(min_h), np.array(max_h),  # 111
                     np.array(mean_k), np.array(median_k), np.array(q5_k), np.array(q10_k), np.array(q90_k), np.array(q95_k), np.array(min_k), np.array(max_k),  # 119
                     np.array(mean_m), np.array(median_m), np.array(q5_m), np.array(q10_m), np.array(q90_m), np.array(q95_m), np.array(min_m), np.array(max_m),  # 127
                     np.array(mean_n), np.array(median_n), np.array(q5_n), np.array(q10_n), np.array(q90_n), np.array(q95_n), np.array(min_n), np.array(max_n),  # 135
                     np.array(mean_o), np.array(median_o), np.array(q5_o), np.array(q10_o), np.array(q90_o), np.array(q95_o), np.array(min_o), np.array(max_o)]  # 143
                    ), th_tr_a


def statistics_signal(signal, axis=0):
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
