import os.path
import matplotlib.pyplot as plt
import scipy.signal
import numpy as np
from synaptic_dynamic_models.TM import TM_model
from synaptic_dynamic_models.MSSM import MSSM_model
from spiking_neuron_models.LIF import LIF_model
# from synaptic_dynamic_models.simple_depression import Simple_Depression
from libraries.frequency_analysis import Freq_analysis
from utils import *


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
        description = str(ind) + " Experiment, decay around 100Hz"
        syn_params = [5.42451963e-02, 2.92925980e+00, 6.67821125e+01, 1.80143000e-02, 7.54167519e-01,
                      5.99119322e+01, 9.94215228e-01, 1.03825167e-03,
                      (4.52507712e-01 * 0.075) / 779.1984, 1.00243185e-03]  # 4.52507712e-01, g_L = 5e-6 homogeneous / 4.6e-3 poisson
    if name_model == "TM" and ind == 2:
        description = str(ind) + " Experiment, decay around 100Hz"
        syn_params = [1.32748878e-01, 2.19116160e-02, 1.32465006e-01, 2.16882855e-00, 1.00766466e-03]  #
    # (Experiment 3) freq. response decay around 10Hz {2: 0.88946682, 5: 0.70155924, 10: 0.57552428, 15: 0.2, 20: 0.1, 50: 0.07}
    if name_model == "MSSM" and ind == 3:
        description = str(ind) + " Experiment, decay around 10Hz"
        syn_params = [6.53659368e-03, 1.75660742e-01, 3.17123087e+01, 1.78659320e-01, 2.50362727e-01,
                      9.12004545e+01, 9.13420672e-01, 2.14204288e-03,
                      5.20907890e-03 / 4.29, 4.32890680e-03]  # 5.20907890e-01, g_L = 2.6e-2 homogeneous / 4.6e-2 poisson
    if name_model == "TM" and ind == 3:
        description = str(ind) + " Experiment, decay around 10Hz"
        syn_params = [2.37698417e-01, 3.30564024e-01, 8.51177265e-01, 3.67454564e-01, 3.04982285e-03]  #
    # (Experiment 4) freq. response from Gain Control paper {5: 0.785, 10: 0.6, 20: 0.415, 50: 0.205, 100: 0.115}
    if name_model == "MSSM" and ind == 4:
        description = str(ind) + " Experiment, Gain-Control paper"
        syn_params = [7.85735182e-02, 4.56599128e-01, 1.46835212e+00, 1.63998958e-01, 2.41885797e-04,
                      5.84619146e+01, 8.00871281e-01, 1.50280526e-03,
                      (5.94890729e-02 * 0.075) / 1.0217, 1.75609424e-03]  # 5.94890729e-01, g_L = 5.4e-4 homogeneous / 3.21e-3 poisson [1.36e-3 for subthreshold]  # g_L = 5.4e-5 for static synapse
    if name_model == "TM" and ind == 4:
        description = str(ind) + " Experiment, Gain-Control paper"
        syn_params = [3.79643805e-01, 3.71724581e-02, 2.31713484e-01, 4.71504487e-01, 3.55003518e-03]  #
    # (Experiment 5) freq. response decay around 100Hz {5: 0.99, 10: 0.9, 20: 0.8, 50: 0.7, 100: 0.5, 200: 0.2}
    if name_model == "MSSM" and ind == 5:
        description = str(ind) + " Experiment, decay around 100Hz"
        syn_params = [3.03503488e-02, 1.49534243e-01, 1.27807004e+00, 8.32749189e-02,
                      1.03652329e-03, 7.06246475e+01, 9.84186889e-01, 1.00258903e-03,
                      (4.42926340e-01 * 0.075) / 2.975, 1.00046291e-03]  # g_L = 2.46e-1 poisson
    if name_model == "TM" and ind == 5:
        description = str(ind) + " Experiment, decay around 100Hz"
        syn_params = [1.31247486e-01, 2.18003024e-02, 1.33872486e-01, 3.17757124e-00,  1.01522251e-03]  #
    # (Experiment 6) freq. response decay around 10Hz {[2:0.88946682, 4:0.70155924, 5.4:0.57552428, 10:0.2, 15:0.1, 30:0.06, 50:0.035]
    if name_model == "MSSM" and ind == 6:
        description = str(ind) + " Experiment, decay around 10Hz"
        syn_params = [3.61148253e-03, 9.98782883e-02, 9.99236857e+00, 2.81436921e-01,
                      1.97666651e-02, 1.00657445e+01, 7.39059950e-01, 1.07519099e-03,
                      3.11356220e-01 * 1e-2, 1.74116605e-03]  # g_L = 1.4e-1 poisson [6.4e-2 subthreshold]

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
            input_lif_n = np.matmul(c.T, id_rep).T * stp_model.params['k_EPSP'] / 2  # k_EPSP/2, factor to transform N(t) into the small range as Epsp(t)

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


def plot_isi_histogram(histograms, ind):
    """
    Plot histograms of ISI from function inter_spike_intervals()
    Parameters
    ----------
    histograms
    ind

    Returns
    -------

    """
    counts, bin_edges = histograms[ind]
    if len(counts) == 0:
        print(f"No data for histogram index {ind}")
        return
    width = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + width / 2
    plt.figure()
    plt.bar(bin_centers, counts, width=width, edgecolor='black')
    plt.xlabel('Inter-spike interval (seconds)')
    plt.ylabel('Count')
    plt.title(f'ISI Histogram {ind}')
    plt.grid()
    plt.show()


def lowpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data


def highpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, cutoff, 'highpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data


def aux_statistics_prop_cons(sig_prop, sig_cons, Le_time_win, threshold_transition, sim_params):
    """

    Parameters
    ----------
    sig_prop
    sig_cons
    Le_time_win
    threshold_transition
    sim_params

    Returns
    -------

    """
    th_tr = threshold_transition
    max_t = sim_params['max_t']
    dt = 1 / sim_params['sfreq']

    # Extracting stimuli windows
    aa = sig_prop[:, 0:int(Le_time_win / dt)]  # [, 0s:2s]
    bb = sig_prop[:, int(Le_time_win / dt):int(2 * Le_time_win / dt)]  # [, 2s:4s]
    cc = sig_prop[:, int(2 * Le_time_win / dt):int(max_t / dt)]  # [, 4s:6s]
    dd = sig_cons[:, 0:int(Le_time_win / dt)]  # [, 0bs:2s]
    ee = sig_cons[:, int(Le_time_win / dt):int(2 * Le_time_win / dt)]  # [, 2s:4s]
    ff = sig_cons[:, int(2 * Le_time_win / dt):int(max_t / dt)]  # [, 4s:6s]

    # Extracting steady-state parts of stimuli windows
    a = sig_prop[:, int(th_tr / dt):int(Le_time_win / dt)]  # [, 0.5s:2s]
    b = sig_prop[:, int((Le_time_win + th_tr) / dt):int(2 * Le_time_win / dt)]  # [, 2.5s:4.0s]
    c = sig_prop[:, int((2 * Le_time_win + th_tr) / dt):int(max_t / dt)]  # [, 4.5s:6.0s]
    d = sig_cons[:, int(th_tr / dt):int(Le_time_win / dt)]  # [, 0.5s:2s]
    e = sig_cons[:, int((Le_time_win + th_tr) / dt):int(2 * Le_time_win / dt)]  # [, 2.5s:4.0s]
    f = sig_cons[:, int((2 * Le_time_win + th_tr) / dt):int(max_t / dt)]  # [, 4.5s:6.0s]

    # Extracting transition parts of stimuli windows
    g = sig_prop[:, 0:int(th_tr / dt)]  # [, 0s:0.5s]
    h = sig_prop[:, int(Le_time_win / dt):int((Le_time_win + th_tr) / dt)]  # [, 2.5s:3.0s]
    k = sig_prop[:, int(2 * Le_time_win / dt):int((2 * Le_time_win + th_tr) / dt)]  # [, 2.5s:3.0s]
    m = sig_cons[:, 0:int(th_tr / dt)]  # [, 0s:0.5s]
    n = sig_cons[:, int(Le_time_win / dt):int((Le_time_win + th_tr) / dt)]  # [, 2s:2.5s]
    o = sig_cons[:, int(2 * Le_time_win / dt):int((2 * Le_time_win + th_tr) / dt)]  # [, 4s:4.5s]

    return np.array([np.median(a, axis=1), np.median(b, axis=1), np.median(c, axis=1),
            np.quantile(a, 0.1, axis=1), np.quantile(b, 0.1, axis=1), np.quantile(c, 0.1, axis=1),
            np.quantile(a, 0.9, axis=1), np.quantile(b, 0.9, axis=1), np.quantile(c, 0.9, axis=1),
            np.min(a, axis=1), np.min(b, axis=1), np.min(c, axis=1),
            np.max(a, axis=1), np.max(b, axis=1), np.max(c, axis=1),
            np.median(d, axis=1), np.median(e, axis=1), np.median(f, axis=1),
            np.quantile(d, 0.1, axis=1), np.quantile(e, 0.1, axis=1), np.quantile(f, 0.1, axis=1),
            np.quantile(d, 0.9, axis=1), np.quantile(e, 0.9, axis=1), np.quantile(f, 0.9, axis=1),
            np.min(d, axis=1), np.min(e, axis=1), np.min(f, axis=1),
            np.max(d, axis=1), np.max(e, axis=1), np.max(f, axis=1),
            np.max(g, axis=1), np.max(h, axis=1), np.min(k, axis=1),
            np.max(m, axis=1), np.max(n, axis=1), np.min(o, axis=1),
            np.median(aa, axis=1), np.quantile(aa, 0.1, axis=1), np.quantile(aa, 0.9, axis=1),
                     np.min(aa, axis=1), np.max(aa, axis=1),
            np.median(bb, axis=1), np.quantile(bb, 0.1, axis=1), np.quantile(bb, 0.9, axis=1),
                     np.min(bb, axis=1), np.max(bb, axis=1),
            np.median(cc, axis=1), np.quantile(cc, 0.1, axis=1), np.quantile(cc, 0.9, axis=1),
                     np.min(cc, axis=1), np.max(cc, axis=1),
            np.median(dd, axis=1), np.quantile(dd, 0.1, axis=1), np.quantile(dd, 0.9, axis=1),
                     np.min(dd, axis=1), np.max(dd, axis=1),
            np.median(ee, axis=1), np.quantile(ee, 0.1, axis=1), np.quantile(ee, 0.9, axis=1),
                     np.min(ee, axis=1), np.max(ee, axis=1),
            np.median(ff, axis=1), np.quantile(ff, 0.1, axis=1), np.quantile(ff, 0.9, axis=1),
                     np.min(ff, axis=1), np.max(ff, axis=1)]
                    )


def sec2hour(secs, exp, realizations):
    rep = int(np.ceil(100 / realizations))
    return secs * exp * rep / 3600