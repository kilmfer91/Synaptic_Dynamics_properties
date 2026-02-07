import os.path
# import matplotlib.pyplot as plt
import scipy.signal
# import numpy as np

from synaptic_dynamic_models.TM import TM_model
from synaptic_dynamic_models.MSSM import MSSM_model
from spiking_neuron_models.LIF import LIF_model
from synaptic_dynamic_models.simple_depression import Simple_Depression
from libraries.frequency_analysis import Freq_analysis
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
             'w_ini_prop_mean', 'w_ini_prop_med', 'w_ini_prop_q5', 'w_ini_prop_q10',
             'w_ini_prop_q90', 'w_ini_prop_q95', 'w_ini_prop_min', 'w_ini_prop_max',
             'w_mid_prop_mean', 'w_mid_prop_med', 'w_mid_prop_q5', 'w_mid_prop_q10',
             'w_mid_prop_q90', 'w_mid_prop_q95', 'w_mid_prop_min', 'w_mid_prop_max',
             'w_end_prop_mean', 'w_end_prop_med', 'w_end_prop_q5', 'w_end_prop_q10',
             'w_end_prop_q90', 'w_end_prop_q95', 'w_end_prop_min', 'w_end_prop_max',
             'w_ini_fix_mean', 'w_ini_fix_med', 'w_ini_fix_q5', 'w_ini_fix_q10',
             'w_ini_fix_q90', 'w_ini_fix_q95', 'w_ini_fix_min', 'w_ini_fix_max',
             'w_mid_fix_mean', 'w_mid_fix_med', 'w_mid_fix_q5', 'w_mid_fix_q10',
             'w_mid_fix_q90', 'w_mid_fix_q95', 'w_mid_fix_min', 'w_mid_fix_max',
             'w_end_fix_mean', 'w_end_fix_med', 'w_end_fix_q5', 'w_end_fix_q10',
             'w_end_fix_q90', 'w_end_fix_q95', 'w_end_fix_min', 'w_end_fix_max',
             'mtr_ini_prop_mean', 'mtr_ini_prop_med', 'mtr_ini_prop_q5', 'mtr_ini_prop_q10',
             'mtr_ini_prop_q90', 'mtr_ini_prop_q95', 'mtr_ini_prop_min', 'mtr_ini_prop_max',
             'mtr_mid_prop_mean', 'mtr_mid_prop_med', 'mtr_mid_prop_q5', 'mtr_mid_prop_q10',
             'mtr_mid_prop_q90', 'mtr_mid_prop_q95', 'mtr_mid_prop_min', 'mtr_mid_prop_max',
             'mtr_end_prop_mean', 'mtr_end_prop_med', 'mtr_end_prop_q5', 'mtr_end_prop_q10',
             'mtr_end_prop_q90', 'mtr_end_prop_q95', 'mtr_end_prop_min', 'mtr_end_prop_max',
             'mtr_ini_fix_mean', 'mtr_ini_fix_med', 'mtr_ini_fix_q5', 'mtr_ini_fix_q10',
             'mtr_ini_fix_q90', 'mtr_ini_fix_q95', 'mtr_ini_fix_min', 'mtr_ini_fix_max',
             'mtr_mid_fix_mean', 'mtr_mid_fix_med', 'mtr_mid_fix_q5', 'mtr_mid_fix_q10',
             'mtr_mid_fix_q90', 'mtr_mid_fix_q95', 'mtr_mid_fix_min', 'mtr_mid_fix_max',
             'mtr_end_fix_mean', 'mtr_end_fix_med', 'mtr_end_fix_q5', 'mtr_end_fix_q10',
             'mtr_end_fix_q90', 'mtr_end_fix_q95', 'mtr_end_fix_min', 'mtr_end_fix_max',
             'initial_frequencies', 'stp_model', 'name_params', 'dyn_synapse', 'num_synapses', 'syn_params',
             'sim_params', 'lif_params', 'lif_params2', 'prop_rate_change_a', 'fix_rate_change_a', 'num_changes_rate',
             'description', 'seeds', 'realizations', 't_realizations', 'time_transition']
stat_list_sin = ['vec_max_mp_pos', 'vec_min_mp_neg', 'vec_q1_mp_pos', 'vec_q90_mp_pos',
                 'vec_q1_mp_neg', 'vec_q90_mp_neg', 'vec_max_mp', 'vec_min_mp', 'vec_q1_mp', 'vec_q90_mp']


# Parameters for LiF neuron
def get_neuron_params(tau_m, y_lim_ind_plot=False, num_syn=1):
    y_lim_memPot = None
    if y_lim_ind_plot:
        y_lim_memPot = [-70, -50]
        if tau_m == 1 and num_syn == 100: y_lim_memPot = [-70, -43]
        if tau_m == 30 and num_syn == 100: y_lim_memPot = [-65.7, -52.5]
        if tau_m == 1 and num_syn == 1: y_lim_memPot = [-70.05, -67.4]
        if tau_m == 10 and num_syn == 1: y_lim_memPot = [-70.05, -69]
        if tau_m == 30 and num_syn == 1: y_lim_memPot = [-70.05, -69.5]

    return {'V_threshold': np.array([1000 for _ in range(1)]), 'V_reset': np.array([-70 for _ in range(1)]),
            'tau_m': np.array([tau_m * 1e-3 for _ in range(1)]), 'g_L': np.array([2.7e-2 for _ in range(1)]),
            'V_init': np.array([-70 for _ in range(1)]), 'V_equilibrium': np.array([-70 for _ in range(1)]),
            't_refractory': np.array([0.01 for _ in range(1)]), 'y_lim_plot': y_lim_memPot}


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


def load_set_simulation_params(dr_ini, path_vars, file_name, run_experiment=False):
    file_loaded = False
    if os.path.isfile(path_vars + file_name) and not run_experiment:
        file_loaded = True
        dr = loadObject(file_name, path_vars)
    else:
        # ******************************************************************************************************************
        # Running freq. response of Gain Control
        # For gain control, 100 inputs to a single LIF neuron
        dyn_synapse = True
        dr = dr_ini.copy()

        # Model parameters
        syn_params, description, name_params = get_params_stp(dr['stp_model'], dr['ind'])

        if not dyn_synapse:
            description = "0_th Static synapse"

        description += ", " + str(dr['num_synapses']) + " synapses"

        # time conditions
        max_t = 6
        dt = 1 / dr['sfreq']
        time_vector = np.arange(0, max_t, dt)
        L = time_vector.shape[0]

        # Parameters definition
        params = dict(zip(name_params, syn_params))
        sim_params = {'sfreq': dr['sfreq'], 'max_t': max_t, 'L': L, 'time_vector': time_vector}

        # PARAMS FOR LIF MODEL
        lif_params = {'V_threshold': np.array([1000 for _ in range(1)]), 'V_reset': np.array([-70 for _ in range(1)]),
                      'tau_m': np.array([dr['tau_lif'] * 1e-3 for _ in range(1)]),
                      'g_L': np.array([2.7e-2 for _ in range(1)]),
                      'V_init': np.array([-70 for _ in range(1)]), 'V_equilibrium': np.array([-70 for _ in range(1)]),
                      't_refractory': np.array([0.01 for _ in range(1)])}

        # Time conditions
        num_changes_rate = 3
        Le_time_win = int(max_t / num_changes_rate)
        prop_rate_change_a = dr['gain_v']  # [0.5, 1, 2]
        fix_rate_change_a = [5]  # [5, 10, 20]

        num_experiments = dr['initial_frequencies'].shape[0]

        # array for time of transition-states
        t_tra = [[] for _ in range(num_experiments)]

        # For poisson or deterministic inputs
        seeds = []
        if not dr['Stoch_input']:
            total_realizations = 1
            num_realizations = 1
            dr['t_realizations'] = total_realizations
            dr['realizations'] = num_realizations

        dr['name_params'], dr['dyn_synapse'], dr['syn_params'] = name_params, dyn_synapse, syn_params
        dr['sim_params'], dr['lif_params'], dr['time_transition'] = sim_params, lif_params, t_tra
        dr['prop_rate_change_a'], dr['fix_rate_change_a'] = prop_rate_change_a, fix_rate_change_a
        dr['num_changes_rate'], dr['description'], dr['seeds'] = num_changes_rate, description, seeds
    return file_loaded, dr


def models_creation(model, aux_num_r, sim_params, params, num_realizations, lif_params):
    # Creating STP models for proportional rate change
    stp_prop, stp_fix = None, None
    if model == "MSSM": stp_prop = MSSM_model(n_syn=aux_num_r)
    if model == "MSSM": stp_fix = MSSM_model(n_syn=aux_num_r)
    if model == "TM": stp_prop = TM_model(n_syn=aux_num_r)
    if model == "TM": stp_fix = TM_model(n_syn=aux_num_r)
    assert stp_prop is not None, "Cannot set stp_model"

    # Setting initial conditions
    stp_prop.set_model_params(params)
    stp_prop.set_simulation_params(sim_params)
    stp_fix.set_model_params(params)
    stp_fix.set_simulation_params(sim_params)

    # Creating LIF models for proportional rate change
    lif_prop = LIF_model(n_neu=num_realizations)
    lif_prop.set_model_params(lif_params)
    lif_fix = LIF_model(n_neu=num_realizations)
    lif_fix.set_model_params(lif_params)

    return stp_prop, stp_fix, lif_prop, lif_fix


def gc_prop_fix_gain(arguments):
    # [total_realizations, plus_cond, Stoch_input, seeds, num_realizations, num_experiments, sfreq,
    #  initial_frequencies, num_changes_rate, aux_num_r, imputations, dyn_synapse, stp_prop, stp_fix, lif_prop,
    #  lif_fix, sim_params, params, t_tra, Le_time_win, lif_output, file_name, prop_rate_change_a, fix_rate_change_a,
    #  gain, fixed_rate_change, path_vars, title_graph] = arguments
    [total_realizations, plus_cond, Stoch_input, seeds, num_realizations, num_experiments, sfreq,
     initial_frequencies, num_changes_rate, aux_num_r, dyn_synapse, stp_prop, stp_fix, lif_prop,
     lif_fix, sim_params, params, t_tra, Le_time_win, lif_output, file_name, prop_rate_change_a, fix_rate_change_a,
     gain, fixed_rate_change, path_vars, title_graph] = arguments

    # Sim params
    L = sim_params['L']
    time_vector = sim_params['time_vector']

    # Getting num of realizations
    num_loop_realizations = int(total_realizations / num_realizations)

    # Auxiliar variables for statistics
    res_per_reali = np.zeros((144, num_experiments, num_realizations))
    res_real = np.zeros((144, total_realizations, num_experiments))

    # Setting proportional and fixed rates of change
    proportional_rate_change = gain
    proportional_changes = proportional_rate_change * initial_frequencies + initial_frequencies
    constant_changes = fixed_rate_change + initial_frequencies

    ini_loop_time = m_time()
    print("Ini big loop")
    realization = 0
    while realization < num_loop_realizations and plus_cond:
        loop_time = m_time()
        t_tra_mid_win = None

        # Building reference signal for constant and fixed rate changes
        i = num_experiments - 1
        while i >= 0:  # while i < num_experiments:
            loop_experiments = m_time()

            # For poisson or deterministic inputs
            seeds1, seeds2, seeds3 = [0], [0], [0]
            if Stoch_input:
                se = int(time.time())
                seeds.append(se)
                seeds1 = [j + se for j in range(num_realizations)]
                seeds2 = [j + se + 2 for j in range(num_realizations)]
                seeds3 = [j + se + 3 for j in range(num_realizations)]

            ref_signals = simple_spike_train(sfreq, initial_frequencies[i], int(L / num_changes_rate),
                                             num_realizations=aux_num_r, poisson=Stoch_input, seeds=seeds1)
            # ISIs, histograms = inter_spike_intervals(ref_signals, dt, 1e-3)
            # plot_isi_histogram(histograms, 0)
            cons_aux = simple_spike_train(sfreq, proportional_changes[i], int(L / num_changes_rate),
                                          num_realizations=aux_num_r, poisson=Stoch_input, seeds=seeds2)
            fix_aux = simple_spike_train(sfreq, constant_changes[i], int(L / num_changes_rate),
                                         num_realizations=aux_num_r, poisson=Stoch_input, seeds=seeds3)

            cons_input = np.concatenate((ref_signals, cons_aux, ref_signals), axis=1)
            fix_input = np.concatenate((ref_signals, fix_aux, ref_signals), axis=1)

            # Avoiding spikes in t==0
            cons_input[:, 0], fix_input[:, 0] = 0, 0
            if not Stoch_input: cons_input[:, 1], fix_input[:, 1] = 1, 1

            # Running STP model
            if dyn_synapse:
                # Reseting initial conditions
                stp_prop.set_initial_conditions()
                lif_prop.set_simulation_params(sim_params)
                stp_fix.set_initial_conditions()
                lif_fix.set_simulation_params(sim_params)
                # Running the models
                model_stp_parallel(stp_prop, lif_prop, params, cons_input)
                model_stp_parallel(stp_fix, lif_fix, params, fix_input)
            else:
                # Reseting initial conditions
                lif_prop.set_simulation_params(sim_params)
                # lif_fix.set_simulation_params(sim_params)
                # Running the models
                static_synapse(lif_prop, cons_input, 9e0)  # , 0.0125e-6)
                # static_synapse(lif_fix, fix_input, 9e0)  # , 0.0125e-6)

            # Defining output of the model in order to compute statistics
            signal_prop, signal_fix = stp_prop.get_output(), stp_fix.get_output()
            if lif_output:
                signal_prop, signal_fix = lif_prop.membrane_potential, lif_fix.membrane_potential

            # getting transition time for rate of proportional  change if possible
            aux_cond = np.where(proportional_changes[i] <= initial_frequencies)
            if len(aux_cond[0]) > 0:
                aux_i = aux_cond[0][0]
                t_tra_mid_win = np.max(t_tra[aux_i])

            # Computing statistics of each window, either for the whole window or for the transition- and steady-states
            res_per_reali[:, i, :], t_tr_ = aux_statistics_prop_cons(signal_prop, signal_fix, Le_time_win,
                                                                     None, sim_params, t_tra_mid_win)

            # Updating array of time_transitions
            t_tra[i].append(t_tr_)

            # Final print of the loop
            print_time(m_time() - loop_experiments, file_name + ", Realisation " + str(realization) +
                       ", frequency " + str(initial_frequencies[i]))

            # """
            # path_save = folder_plots + file_name + '_' + str(initial_frequencies[i]) + '_.png'
            title_graph_ = title_graph + ", freq. %dHz" % initial_frequencies[i]
            t_tr = t_tr_[0]
            plot_gc_mem_potential_prop_fix(time_vector, i, signal_prop, signal_fix, t_tr, res_per_reali, title_graph_,
                                           path_save="", save_figs=False)
            # """
            i -= 1

        # steady-state part
        for res_i in range(res_real.shape[0]):
            r = realization
            res_real[res_i, r * num_realizations:(r + 1) * num_realizations] = res_per_reali[res_i, :].T

        print_time(m_time() - loop_time, file_name + ", Realisation " + str(realization))

        realization += 1

    # transition-state
    for i in range(num_experiments):
        t_tra[i] = np.ravel(t_tra[i])
    t_tra = np.array(t_tra).T

    if not os.path.isfile(path_vars + file_name):
        dr = {'initial_frequencies': initial_frequencies,
              'stp_model': model, 'name_params': name_params, 'dyn_synapse': dyn_synapse,
              'num_synapses': num_syn, 'syn_params': syn_params, 'sim_params': sim_params,
              'lif_params': lif_params, 'lif_params2': lif_params2, 'prop_rate_change_a': prop_rate_change_a,
              'fix_rate_change_a': prop_rate_change_a, 'num_changes_rate': num_changes_rate,
              'description': description, 'seeds': seeds,
              'realizations': num_realizations, 't_realizations': total_realizations, 'time_transition': t_tra}
        for nam in range(res_real.shape[0]):
            dr[stat_list[nam]] = res_real[nam, :]

        if save_vars:
            saveObject(dr, file_name, path_vars)

    print_time(m_time() - ini_loop_time, "Total big loop")


def get_name_file(sfreq, model, ind, num_syn, lif_output, tau_lif, stoch_inp, imputations, gain):
    aux_name = "_ind_" + str(ind) + "_gain_" + str(int(gain * 100)) + "_sf_" + str(int(sfreq / 1000)) + "k_syn_" + str(
        num_syn)
    if lif_output: aux_name += "_tauLiF_" + str(tau_lif) + "ms"
    file_name = (model + aux_name)
    if not stoch_inp:
        file_name = (model + '_det' + aux_name)
        file_name += "_cwi"
    else:
        if imputations:
            file_name += "_cwi"
        else:
            file_name += "_cni"

    return file_name


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


def get_time_series_statistics_of_transitions(time_series, f_vector, prop_rates, th_percentage=1e-2):

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
                th_tr_a[freq] = get_transition_time_from_2_signals(res[0][freq], res[2][freq], th_percentage)
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


def get_transition_time_from_2_signals(signal1, signal2, th_percentage=1e-2):
    # Substract ini and end window to define the transition period (exclude lasst 10 samples to avoid errors)
    ini_minus_end_windows = np.abs(signal1[:, :-10] - signal2[:, :-10])
    # Find the 0.1% of the maximum for each realization (the threshold to define that the difference between ini and end
    # windows is sufficiently low to be considered zero)
    thresholds = np.max(ini_minus_end_windows, axis=1) * th_percentage
    shapes_diff = ini_minus_end_windows.shape
    # Create the mask to compare each realization with the 0.1% of their maximums
    mask_thr = np.repeat(np.reshape(thresholds, (shapes_diff[0], 1)), shapes_diff[1], axis=1)
    # find indices where the difference is bigger than 0.1% of maximum
    ind_tr = np.where(ini_minus_end_windows > mask_thr)
    # Getting indices of unique values (i.e. realizations)
    val_unique, ind_unique = np.unique(ind_tr[0], return_index=True)
    # getting last index (indicating that after that, diff. is lower than 1e-6)
    first_indtr = np.roll(ind_tr[1][list(np.array(ind_unique) - 1)], -1)

    return first_indtr


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
    statistical descriptors array
    transition_time_array
    list of transition time arrays
    -------

    """
    max_t = sim_params['max_t']
    dt = 1 / sim_params['sfreq']

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

    # Getting time range of transition period
    if threshold_transition is None:
        th_tr_a = get_transition_time_from_2_signals(piw, pew, th_percentage=1e-2) * dt
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

    return np.array([# For steady-state
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
                     # For all window
                     np.mean(piw, axis=1), np.median(piw, axis=1), np.quantile(piw, 0.05, axis=1), np.quantile(piw, 0.1, axis=1),  # 51
                     np.quantile(piw, 0.9, axis=1), np.quantile(piw, 0.95, axis=1), np.min(piw, axis=1), np.max(piw, axis=1),  # 55
                     np.mean(pmw, axis=1), np.median(pmw, axis=1), np.quantile(pmw, 0.05, axis=1), np.quantile(pmw, 0.1, axis=1),  # 59
                     np.quantile(pmw, 0.9, axis=1), np.quantile(pmw, 0.95, axis=1), np.min(pmw, axis=1), np.max(pmw, axis=1),  # 63
                     np.mean(pew, axis=1), np.median(pew, axis=1), np.quantile(pew, 0.05, axis=1), np.quantile(pew, 0.1, axis=1),  # 67
                     np.quantile(pew, 0.9, axis=1), np.quantile(pew, 0.95, axis=1), np.min(pew, axis=1), np.max(pew, axis=1),  # 71
                     np.mean(ciw, axis=1), np.median(ciw, axis=1), np.quantile(ciw, 0.05, axis=1), np.quantile(ciw, 0.1, axis=1),  # 75
                     np.quantile(ciw, 0.9, axis=1), np.quantile(ciw, 0.95, axis=1), np.min(ciw, axis=1), np.max(ciw, axis=1),  # 79
                     np.mean(cmw, axis=1), np.median(cmw, axis=1), np.quantile(cmw, 0.05, axis=1), np.quantile(cmw, 0.1, axis=1),  # 83
                     np.quantile(cmw, 0.9, axis=1), np.quantile(cmw, 0.95, axis=1), np.min(cmw, axis=1), np.max(cmw, axis=1),  # 87
                     np.mean(cew, axis=1), np.median(cew, axis=1), np.quantile(cew, 0.05, axis=1), np.quantile(cew, 0.1, axis=1),  # 91
                     np.quantile(cew, 0.9, axis=1), np.quantile(cew, 0.95, axis=1), np.min(cew, axis=1), np.max(cew, axis=1),  # 95
                     # For transition-state
                     np.array(mean_tr_pi), np.array(median_tr_pi), np.array(q5_tr_pi), np.array(q10_tr_pi),
                     np.array(q90_tr_pi), np.array(q95_tr_pi), np.array(min_tr_pi), np.array(max_tr_pi),  # 103
                     np.array(mean_tr_pm), np.array(median_tr_pm), np.array(q5_tr_pm), np.array(q10_tr_pm),
                     np.array(q90_tr_pm), np.array(q95_tr_pm), np.array(min_tr_pm), np.array(max_tr_pm),  # 111
                     np.array(mean_tr_pe), np.array(median_tr_pe), np.array(q5_tr_pe), np.array(q10_tr_pe),
                     np.array(q90_tr_pe), np.array(q95_tr_pe), np.array(min_tr_pe), np.array(max_tr_pe),  # 119
                     np.array(mean_tr_ci), np.array(median_tr_ci), np.array(q5_tr_ci), np.array(q10_tr_ci),
                     np.array(q90_tr_ci), np.array(q95_tr_ci), np.array(min_tr_ci), np.array(max_tr_ci),  # 127
                     np.array(mean_tr_cm), np.array(median_tr_cm), np.array(q5_tr_cm), np.array(q10_tr_cm),
                     np.array(q90_tr_cm), np.array(q95_tr_cm), np.array(min_tr_cm), np.array(max_tr_cm),  # 135
                     np.array(mean_tr_ce), np.array(median_tr_ce), np.array(q5_tr_ce), np.array(q10_tr_ce),
                     np.array(q90_tr_ce), np.array(q95_tr_ce), np.array(min_tr_ce), np.array(max_tr_ce)]  # 143
                    ), th_tr_a, tr_timeSeries


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
