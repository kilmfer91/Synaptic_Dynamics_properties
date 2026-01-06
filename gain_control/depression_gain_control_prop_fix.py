# from synaptic_dynamic_models.simple_depression import Simple_Depression
import numpy as np

from gain_control.utils_gc import *

# ******************************************************************************************************************
# STP model and extra global variables
model = 'MSSM'
# (Experiment 2) freq. response decay around 100Hz
# (Experiment 3) freq. response decay around 10Hz
# (Experiment 4) freq. response from Gain Control paper
# (Experiment 5) freq. response decay around 100Hz
# (Experiment 6) freq. response decay around 10Hz
ind = 4
save_vars = False
run_experiment = True
save_figs = False
imputations = True
Stoch_input = True
lif_output = True
num_syn = 1

# Sampling frequency and conditions for running parallel or single LIF neurons
sfreq = 5e3
tau_lif = 1  # ms
total_realizations = 10  # 100
num_realizations = 5  # 8 for server, 4 for macbook air
t_tra = None  # None  # 0.25
t_tra_mid_win = None

# Input modulations
range_f = [10, 20, 30, 40]  # [i for i in range(10, 100, 5)]
range_f2 = [100, 150, 200, 300]  # [i for i in range(100, 500, 10)]  # # sfreq>3kHz:501, 2kHz:321
range_f3 = [500]  # [i for i in range(500, 801, 50)]  # Max prop freq. must be less than sfreq/4  # 16kHz:2501, 5kHz:801
initial_frequencies = np.array(range_f + range_f2 + range_f3)

# Path variables
path_vars = "../gain_control/variables/"
check_create_folder(path_vars)
folder_plots = '../gain_control/plots/'
check_create_folder(folder_plots)
aux_name = "_gain_control_" + str(int(sfreq / 1000)) + "k_ind_" + str(ind) + "_syn_" + str(num_syn)
if lif_output: aux_name += "_tauLiF_" + str(tau_lif) + "ms"
file_name = (model + aux_name)
if not Stoch_input: file_name = (model + '_det' + aux_name)

if imputations: file_name += "_cwi"
else: file_name += "_cni"

print("For file %s and index %d" % (file_name, ind))

# ******************************************************************************************************************
# Trying to load freq. response of Gain Control
file_loaded = False
if os.path.isfile(path_vars + file_name) and not run_experiment:
    file_loaded = True
    dr = loadObject(file_name, path_vars)

    # Auxiliar variables
    initial_frequencies, model, name_params = dr['initial_frequencies'], dr['stp_model'], dr['name_params']
    dyn_synapse, num_synapses, syn_params = dr['dyn_synapse'], dr['num_synapses'], dr['syn_params']
    num_realizations, sim_params, lif_params = dr['realizations'], dr['sim_params'], dr['lif_params']
    lif_params2, prop_rate_change_a = dr['lif_params2'], dr['prop_rate_change_a']
    fix_rate_change_a, num_changes_rate, description = dr['fix_rate_change_a'], dr['num_changes_rate'], dr[
        'description']
    seeds = dr['seeds']
    total_realizations = dr['t_realizations']

    # Time conditions
    max_t, sfreq, time_vector, L = sim_params['max_t'], sim_params['sfreq'], sim_params['time_vector'], sim_params['L']
    dt = 1 / sfreq
    Le_time_win = int(max_t / num_changes_rate)

    # time transition
    if 'time_transition' in dr:
        t_tra = dr['time_transition']
    else:
        t_tra = Le_time_win * 0.25

    # Parameters in dict format
    params = dict(zip(name_params, syn_params))

    # Number of experiments
    num_experiments = initial_frequencies.shape[0]

    # Number of iterations while looping the realizations
    num_loop_realizations = int(total_realizations / num_realizations)

    # If poisson input
    Stoch_input = False
    if isinstance(seeds, list):
        Stoch_input = True
        if len(seeds) > 0:
            seeds1 = [j + seeds[0] for j in range(num_realizations)]
            seeds2 = [j + seeds[0] + 2 for j in range(num_realizations)]
            seeds3 = [j + seeds[0] + 3 for j in range(num_realizations)]
else:
    # ******************************************************************************************************************
    # Running freq. response of Gain Control
    # For gain control, 100 inputs to a single LIF neuron
    dyn_synapse = True
    num_synapses = num_syn  # 100
    dr = {}

    # Model parameters
    syn_params, description, name_params = get_params_stp(model, ind)

    if not dyn_synapse:
        description = "0_th Static synapse"

    description += ", " + str(num_synapses) + " synapses"

    # time conditions
    max_t = 6
    dt = 1 / sfreq
    time_vector = np.arange(0, max_t, dt)
    L = time_vector.shape[0]

    # Parameters definition
    params = dict(zip(name_params, syn_params))
    sim_params = {'sfreq': sfreq, 'max_t': max_t, 'L': L, 'time_vector': time_vector}

    # ******************************************************************************************************************
    # PARAMS FOR LIF MODEL
    lif_params = {'V_threshold': np.array([1000 for _ in range(1)]), 'V_reset': np.array([-70 for _ in range(1)]),
                  'tau_m': np.array([tau_lif * 1e-3 for _ in range(1)]),
                  'g_L': np.array([2.7e-2 for _ in range(1)]),
                  'V_init': np.array([-70 for _ in range(1)]), 'V_equilibrium': np.array([-70 for _ in range(1)]),
                  't_refractory': np.array([0.01 for _ in range(1)])}

    lif_params2 = {'V_threshold': np.array([1000 for _ in range(1)]), 'V_reset': np.array([-70 for _ in range(1)]),
                   'tau_m': np.array([tau_lif * 1e-3 for _ in range(1)]),
                   'g_L': np.array([2.7e-2 for _ in range(1)]),  # 3.21e-3
                   'V_init': np.array([-70 for _ in range(1)]), 'V_equilibrium': np.array([-70 for _ in range(1)]),
                   't_refractory': np.array([0.001 for _ in range(1)])}

    # ******************************************************************************************************************
    # SIMULATION GAIN CONTROL (1 SYNAPSE WITH CONSTANT OR FIXED RATE CHANGES)
    # """
    # Time conditions
    num_changes_rate = 3
    Le_time_win = int(max_t / num_changes_rate)
    prop_rate_change_a = [0.5]  # [0.5, 1, 2]
    fix_rate_change_a = [5]  # [5, 10, 20]

    num_experiments = initial_frequencies.shape[0]

    # array for time of transition-states
    t_tra = [[] for _ in range(num_experiments)]

    # For poisson or deterministic inputs
    # if not lif_parallel: num_realizations = 1
    seeds, seeds1, seeds2, seeds3 = [], None, None, None
    if not Stoch_input:
        total_realizations = 1
        num_realizations = 1

    num_loop_realizations = int(total_realizations / num_realizations)
#
# aux_num_r = num_synapses
aux_num_r = int(num_realizations * num_synapses)

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
# lif_prop = LIF_model(n_neu=1)
lif_prop = LIF_model(n_neu=num_realizations)
lif_prop.set_model_params(lif_params)
lif_prop_n = None
# if model == "MSSM":
#     lif_prop_n = LIF_model(n_neu=1)
#     if lif_parallel: lif_prop_n = LIF_model(n_neu=num_realizations)
#     lif_prop_n.set_model_params(lif_params)

# Creating LIF models for constant rate change
# lif_fix = LIF_model(n_neu=1)
lif_fix = LIF_model(n_neu=num_realizations)
lif_fix.set_model_params(lif_params2)
lif_fix_n = None
# if model == "MSSM":
#     lif_fix_n = LIF_model(n_neu=1)
#     if lif_parallel: lif_fix_n = LIF_model(n_neu=num_realizations)
#     lif_fix_n.set_model_params(lif_params2)

# Setting proportional and fixed rates of change
proportional_rate_change = prop_rate_change_a[0]  # [k]
fixed_rate_change = fix_rate_change_a[0]  # [k]
proportional_changes = proportional_rate_change * initial_frequencies + initial_frequencies
constant_changes = fixed_rate_change + initial_frequencies

# Auxiliar variables for statistics
res_per_reali = np.zeros((144, num_experiments, num_realizations))
res_real = np.zeros((144, total_realizations, num_experiments))

ini_loop_time = m_time()
print("Ini big loop")
realization = 0
while realization < num_loop_realizations and (not file_loaded or run_experiment):
    loop_time = m_time()
    t_tra_mid_win = None

    # Building reference signal for constant and fixed rate changes
    i = num_experiments - 1
    # for i in range(num_experiments):
    while i >= 0:  # while i < num_experiments:
        loop_experiments = m_time()

        # # Updating maximum time of simulation
        # max_t = 6
        # # if 30 < initial_frequencies[i] <= 100: max_t = 3
        # # if initial_frequencies[i] > 100: max_t = 1.5
        # # Updating temporal variables
        # time_vector = np.arange(0, max_t, dt)
        # L = time_vector.shape[0]
        # Le_time_win = max_t / num_changes_rate
        # sim_params = {'sfreq': sfreq, 'max_t': max_t, 'L': L, 'time_vector': time_vector}

        # For poisson or deterministic inputs
        if Stoch_input:
            se = int(time.time())
            seeds.append(se)
            seeds1 = [j + se for j in range(num_realizations)]
            seeds2 = [j + se + 2 for j in range(num_realizations)]
            seeds3 = [j + se + 3 for j in range(num_realizations)]
            # seeds = [j + 1761027202 for j in range(num_realizations)]

        ref_signals = simple_spike_train(sfreq, initial_frequencies[i], int(L / num_changes_rate),
                                         num_realizations=aux_num_r, poisson=Stoch_input, seeds=seeds1, correction=True,
                                         imputation=imputations)
        # ISIs, histograms = inter_spike_intervals(ref_signals, dt, 1e-3)
        # plot_isi_histogram(histograms, 0)
        cons_aux = simple_spike_train(sfreq, proportional_changes[i], int(L / num_changes_rate),
                                      num_realizations=aux_num_r, poisson=Stoch_input, seeds=seeds2, correction=True,
                                      imputation=imputations)
        fix_aux = simple_spike_train(sfreq, constant_changes[i], int(L / num_changes_rate),
                                     num_realizations=aux_num_r, poisson=Stoch_input, seeds=seeds3, correction=True,
                                     imputation=imputations)

        cons_input = np.concatenate((ref_signals, cons_aux, ref_signals), axis=1)
        fix_input = np.concatenate((ref_signals, fix_aux, ref_signals), axis=1)

        cons_input[:, 0] = 0
        fix_input[:, 0] = 0
        if not Stoch_input:
            cons_input[:, 1] = 1
            fix_input[:, 1] = 1

        # Running STP model
        if dyn_synapse:
            # Reseting initial conditions
            stp_prop.set_initial_conditions()
            lif_prop.set_simulation_params(sim_params)
            stp_fix.set_initial_conditions()
            lif_fix.set_simulation_params(sim_params)
            # if model == "MSSM":
                # lif_prop_n.set_simulation_params(sim_params)
                # lif_fix_n.set_simulation_params(sim_params)
            # Running the models
            # if lif_parallel:
            model_stp_parallel(stp_prop, lif_prop, params, cons_input, lif_prop_n)
            model_stp_parallel(stp_fix, lif_fix, params, fix_input, lif_fix_n)
            # else:
            #     model_stp(stp_prop, lif_prop, params, cons_input, lif_prop_n)
            #     model_stp(stp_fix, lif_fix, params, fix_input, lif_fix_n)
        else:
            # Reseting initial conditions
            lif_prop.set_simulation_params(sim_params)
            lif_fix.set_simulation_params(sim_params)
            # Running the models
            static_synapse(lif_prop, cons_input, 9e0)  # , 0.0125e-6)
            static_synapse(lif_fix, fix_input, 9e0)  # , 0.0125e-6)

        # Defining output of the model in order to compute statistics
        signal_prop = stp_prop.get_output()
        signal_fix = stp_fix.get_output()
        if lif_output:
            signal_prop = lif_prop.membrane_potential
            signal_fix = lif_fix.membrane_potential

        # getting transition time for rate of proportional  change if possible
        aux_cond = np.where(proportional_changes[i] <= initial_frequencies)
        if len(aux_cond[0]) > 0:
            aux_i = aux_cond[0][0]
            t_tra_mid_win = np.max(t_tra[aux_i])

        # Computing statistics of each window, either for the whole window or for the transition- and steady-states
        res_per_reali[:, i, :], t_tr_ = aux_statistics_prop_cons(signal_prop, signal_fix, Le_time_win,
                                                                 None, sim_params, t_tra_mid_win)
        if lif_prop_n is not None:
            res_per_reali_n[:, i, :], t_tr_ = aux_statistics_prop_cons(lif_prop_n.membrane_potential,
                                                                       lif_fix_n.membrane_potential, Le_time_win,
                                                                       None, sim_params, t_tra_mid_win)

        # Updating array of time_transitions
        t_tra[i].append(t_tr_)

        # Final print of the loop
        print_time(m_time() - loop_experiments,
                   file_name + ", Realisation " + str(realization) + ", frequency " + str(initial_frequencies[i]))

        """
        t_tr = t_tr_[0]
        figc = plt.figure(figsize=(10, 3))
        plt.suptitle("For model %s, index %d, frequency %dHz, for %d synapses" % (
            model, ind, initial_frequencies[i], num_synapses))
        ylims = [-70, -67]  # tm=30/syn=100 [-65.7,-52.5], tm=1/syn=100[-70,-35], tm=1/syn=1 [-70.05,-67.4]
        ax1 = figc.add_subplot(1, 2, 1)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Mem. potential (mV)")
        ax1.plot(time_vector, signal_prop[0, :], c="black", alpha=0.3)
        ax1.plot([0 + t_tr, 2], [res_per_reali[0, i, 0], res_per_reali[0, i, 0]], c="tab:orange")  # mean ini window
        ax1.plot([2 + t_tr, 4], [res_per_reali[8, i, 0], res_per_reali[8, i, 0]], c="tab:orange")  # mean mid window
        ax1.plot([4 + t_tr, 6], [res_per_reali[16, i, 0], res_per_reali[16, i, 0]], c="tab:orange")  # mean end window
        ax1.plot([0, 2], [res_per_reali[54, i, 0], res_per_reali[54, i, 0]], c="red", alpha=0.5)  # min w ini window
        ax1.plot([2, 4], [res_per_reali[62, i, 0], res_per_reali[62, i, 0]], c="red", alpha=0.5)  # min w mid window
        ax1.plot([4, 6], [res_per_reali[70, i, 0], res_per_reali[70, i, 0]], c="red", alpha=0.5)  # min w end window
        ax1.plot([0 + t_tr, 2], [res_per_reali[6, i, 0], res_per_reali[6, i, 0]], c="tab:red")  # min ini window
        ax1.plot([2 + t_tr, 4], [res_per_reali[14, i, 0], res_per_reali[14, i, 0]], c="tab:red")  # min mid window
        ax1.plot([4 + t_tr, 6], [res_per_reali[22, i, 0], res_per_reali[22, i, 0]], c="tab:red")  # min end window
        ax1.plot([0, 2], [res_per_reali[55, i, 0], res_per_reali[55, i, 0]], c="red", alpha=0.5)  # max w ini window
        ax1.plot([2, 4], [res_per_reali[63, i, 0], res_per_reali[63, i, 0]], c="red", alpha=0.5)  # max w mid window
        ax1.plot([4, 6], [res_per_reali[71, i, 0], res_per_reali[71, i, 0]], c="red", alpha=0.5)  # max w end window
        ax1.plot([0 + t_tr, 2], [res_per_reali[7, i, 0], res_per_reali[7, i, 0]], c="tab:red")  # max ini window
        ax1.plot([2 + t_tr, 4], [res_per_reali[15, i, 0], res_per_reali[15, i, 0]], c="tab:red")  # max mid window
        ax1.plot([4 + t_tr, 6], [res_per_reali[23, i, 0], res_per_reali[23, i, 0]], c="tab:red")  # max end window
        ax1.plot([0, 2], [res_per_reali[51, i, 0], res_per_reali[51, i, 0]], c="green", alpha=0.5)  # q1 w ini window
        ax1.plot([2, 4], [res_per_reali[59, i, 0], res_per_reali[59, i, 0]], c="green", alpha=0.5)  # q1 w mid window
        ax1.plot([4, 6], [res_per_reali[67, i, 0], res_per_reali[67, i, 0]], c="green", alpha=0.5)  # q1 w end window
        ax1.plot([0 + t_tr, 2], [res_per_reali[3, i, 0], res_per_reali[3, i, 0]], c="tab:green")  # q1 ini window
        ax1.plot([2 + t_tr, 4], [res_per_reali[11, i, 0], res_per_reali[11, i, 0]], c="tab:green")  # q1 mid window
        ax1.plot([4 + t_tr, 6], [res_per_reali[19, i, 0], res_per_reali[19, i, 0]], c="tab:green")  # q1 end window
        ax1.plot([0, 2], [res_per_reali[52, i, 0], res_per_reali[52, i, 0]], c="green", alpha=0.5)  # q90 w ini window
        ax1.plot([2, 4], [res_per_reali[60, i, 0], res_per_reali[60, i, 0]], c="green", alpha=0.5)  # q90 w mid window
        ax1.plot([4, 6], [res_per_reali[68, i, 0], res_per_reali[68, i, 0]], c="green", alpha=0.5)  # q90 w end window
        ax1.plot([0 + t_tr, 2], [res_per_reali[4, i, 0], res_per_reali[4, i, 0]], c="tab:green")  # q90 ini window
        ax1.plot([2 + t_tr, 4], [res_per_reali[12, i, 0], res_per_reali[12, i, 0]], c="tab:green")  # q90 mid window
        ax1.plot([4 + t_tr, 6], [res_per_reali[20, i, 0], res_per_reali[20, i, 0]], c="tab:green")  # q90 end window
        ax1.plot([0 + t_tr, 2], [res_per_reali[1, i, 0], res_per_reali[1, i, 0]], c="tab:blue")  # median ini window
        ax1.plot([2 + t_tr, 4], [res_per_reali[9, i, 0], res_per_reali[9, i, 0]], c="tab:blue")  # median mid window
        ax1.plot([4 + t_tr, 6], [res_per_reali[17, i, 0], res_per_reali[17, i, 0]], c="tab:blue")  # median end window
        # ax1.grid()
        ax1.set_title("Proportional changes", color="gray")
        ax1.set_ylim(ylims)
        ax2 = figc.add_subplot(1, 2, 2)
        ax2.plot(time_vector, signal_fix[0, :], c="black", alpha=0.3)
        ax2.plot([0 + t_tr, 2], [res_per_reali[18, i, 0], res_per_reali[18, i, 0]], c="tab:orange", label=r'$\mu$')  # mean ini window
        ax2.plot([2 + t_tr, 4], [res_per_reali[19, i, 0], res_per_reali[19, i, 0]], c="tab:orange")  # mean mid window
        ax2.plot([4 + t_tr, 6], [res_per_reali[20, i, 0], res_per_reali[20, i, 0]], c="tab:orange")  # mean end window
        ax2.plot([0, 2], [res_per_reali[58, i, 0], res_per_reali[58, i, 0]], c="red", alpha=0.5)  # min w ini window
        ax2.plot([2, 4], [res_per_reali[64, i, 0], res_per_reali[64, i, 0]], c="red", alpha=0.5)  # min w mid window
        ax2.plot([4, 6], [res_per_reali[70, i, 0], res_per_reali[70, i, 0]], c="red", alpha=0.5)  # min w end window
        ax2.plot([0 + t_tr, 2], [res_per_reali[30, i, 0], res_per_reali[30, i, 0]], c="tab:red")  # min ini window
        ax2.plot([2 + t_tr, 4], [res_per_reali[31, i, 0], res_per_reali[31, i, 0]], c="tab:red")  # min mid window
        ax2.plot([4 + t_tr, 6], [res_per_reali[32, i, 0], res_per_reali[32, i, 0]], c="tab:red")  # min end window
        ax2.plot([0, 2], [res_per_reali[59, i, 0], res_per_reali[59, i, 0]], c="red", alpha=0.5)  # max w ini window
        ax2.plot([2, 4], [res_per_reali[65, i, 0], res_per_reali[65, i, 0]], c="red", alpha=0.5)  # max w mid window
        ax2.plot([4, 6], [res_per_reali[71, i, 0], res_per_reali[71, i, 0]], c="red", alpha=0.5)  # max w end window
        ax2.plot([0 + t_tr, 2], [res_per_reali[33, i, 0], res_per_reali[33, i, 0]], c="tab:red", label='max')  # max ini window
        ax2.plot([2 + t_tr, 4], [res_per_reali[34, i, 0], res_per_reali[34, i, 0]], c="tab:red")  # max mid window
        ax2.plot([4 + t_tr, 6], [res_per_reali[35, i, 0], res_per_reali[35, i, 0]], c="tab:red")  # max end window
        ax2.plot([0, 2], [res_per_reali[56, i, 0], res_per_reali[56, i, 0]], c="green", alpha=0.5)  # q1 w ini window
        ax2.plot([2, 4], [res_per_reali[62, i, 0], res_per_reali[62, i, 0]], c="green", alpha=0.5)  # q1 w mid window
        ax2.plot([4, 6], [res_per_reali[68, i, 0], res_per_reali[68, i, 0]], c="green", alpha=0.5)  # q1 w end window
        ax2.plot([0 + t_tr, 2], [res_per_reali[24, i, 0], res_per_reali[24, i, 0]], c="tab:green", label=r'$q_1$')  # q1 ini window
        ax2.plot([2 + t_tr, 4], [res_per_reali[25, i, 0], res_per_reali[25, i, 0]], c="tab:green")  # q1 mid window
        ax2.plot([4 + t_tr, 6], [res_per_reali[26, i, 0], res_per_reali[26, i, 0]], c="tab:green")  # q1 end window
        ax2.plot([0, 2], [res_per_reali[57, i, 0], res_per_reali[57, i, 0]], c="green", alpha=0.5)  # q90 w ini window
        ax2.plot([2, 4], [res_per_reali[63, i, 0], res_per_reali[63, i, 0]], c="green", alpha=0.5)  # q90 w mid window
        ax2.plot([4, 6], [res_per_reali[69, i, 0], res_per_reali[69, i, 0]], c="green", alpha=0.5)  # q90 w end window
        ax2.plot([0 + t_tr, 2], [res_per_reali[27, i, 0], res_per_reali[27, i, 0]], c="tab:green", label=r'$q_{90}$')  # q90 ini window
        ax2.plot([2 + t_tr, 4], [res_per_reali[28, i, 0], res_per_reali[28, i, 0]], c="tab:green")  # q90 mid window
        ax2.plot([4 + t_tr, 6], [res_per_reali[29, i, 0], res_per_reali[29, i, 0]], c="tab:green")  # q90 end window
        ax2.plot([0 + t_tr, 2], [res_per_reali[21, i, 0], res_per_reali[21, i, 0]], c="tab:blue", label='med')  # median ini window
        ax2.plot([2 + t_tr, 4], [res_per_reali[22, i, 0], res_per_reali[22, i, 0]], c="tab:blue")  # median mid window
        ax2.plot([4 + t_tr, 6], [res_per_reali[23, i, 0], res_per_reali[23, i, 0]], c="tab:blue")  # median end window
        ax2.set_title("Constant changes", color="gray")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Mem. potential (mV)")
        # ax2.grid()
        ax2.set_ylim(ylims)
        ax2.legend(loc="upper right")
        figc.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
        if save_figs: figc.savefig(folder_plots + file_name + '_' + str(initial_frequencies[i]) + '_.png', format='png')
        # """
        i -= 1  # i += 1

    # steady-state part
    for res_i in range(res_real.shape[0]):
        r = realization
        res_real[res_i, r * num_realizations:(r + 1) * num_realizations] = res_per_reali[res_i, :].T
        if lif_prop_n is not None:
            res_real_n[res_i, r * num_realizations:(r + 1) * num_realizations] = res_per_reali_n[res_i, :].T

    print_time(m_time() - loop_time, file_name + ", Realisation " + str(realization))

    realization += 1

# transition-state
for i in range(num_experiments):
    t_tra[i] = np.ravel(t_tra[i])
t_tra = np.array(t_tra).T

if not os.path.isfile(path_vars + file_name):
    dr = {'initial_frequencies': initial_frequencies,
          'stp_model': model, 'name_params': name_params, 'dyn_synapse': dyn_synapse,
          'num_synapses': num_synapses, 'syn_params': syn_params, 'sim_params': sim_params,
          'lif_params': lif_params, 'lif_params2': lif_params2, 'prop_rate_change_a': prop_rate_change_a,
          'fix_rate_change_a': prop_rate_change_a, 'num_changes_rate': num_changes_rate,
          'description': description, 'seeds': seeds,
          'realizations': num_realizations, 't_realizations': total_realizations, 'time_transition': t_tra}
    num_stat = res_real.shape[0]
    max_iter = res_real.shape[0]
    if lif_prop_n is not None: max_iter = res_real.shape[0] + res_real_n.shape[0]
    for nam in range(max_iter):
        if nam < num_stat:
            dr[stat_list[nam]] = res_real[nam, :]
        elif lif_prop_n is not None:
            dr[stat_list[nam]] = res_real_n[nam - num_stat, :]

    if save_vars:
        saveObject(dr, file_name, path_vars)

print_time(m_time() - ini_loop_time, "Total big loop")

# **********************************************************************************************************************
# PLOTS
# **********************************************************************************************************************

# PLOT OF MSSM OUTPUT (EPSP) FOR BOTH PROPORTIONAL AND CONSTANT RATE CHANGES, CHOOSING THE FREQUENCY WITH i_interest
"""
i_interest = 17
fig = plt.figure()
plt.suptitle("Proportional (" + str(proportional_rate_change) + "Hz) vs. constant rate changes (" + 
             str(fixed_rate_change) + "Hz), initial freq. " + str(initial_frequencies[i_interest]) + "Hz")

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(time_vector, np.mean(mssm_prop.EPSP[i_interest * num_realizations:
                                                 (i_interest + 1) * num_realizations, :], axis=0), alpha=0.8)
ax1.set_title("proportional, 2nd freq " + str(constant_changes[i_interest]) + "Hz", c="gray")
ax1.grid()

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(time_vector, np.mean(mssm_fix.EPSP[i_interest * num_realizations:
                                                 (i_interest + 1) * num_realizations, :], axis=0), alpha=0.8)
ax2.set_title("Fixed, 2nd freq "  + str(fixed_changes[i_interest]) + "Hz", c="gray")
ax2.grid()

fig.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
# """

# PLOT OF MEMBRANE POTENTIAL FOR BOTH PROPORTIONAL AND CONSTANT RATE CHANGES AT THE LAST FREQUENCY
"""
i_interest = -1
fig = plt.figure()
plt.suptitle("Proportional (" + str(proportional_rate_change) + "Hz) vs. constant rate changes (" + 
             str(fixed_rate_change) + "Hz), initial freq. " + str(initial_frequencies[i_interest]) + "Hz")

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(time_vector, lif.membrane_potential[0, :], alpha=0.8)
ax1.set_title("Proportional, 2nd freq " + str(constant_changes[i_interest]) + "Hz", c="gray")
ax1.grid()

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(time_vector, lif.membrane_potential[0, :], alpha=0.8)
ax2.set_title("Fixed, 2nd freq "  + str(fixed_changes[i_interest]) + "Hz", c="gray")
ax2.grid()

fig.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
# """

# PLOT OF CHARACTERISTICS FOR INI, MID, AND END WINDOWS. SPLITTED BY PROPORTIONAL AND CONSTANT INPUT RATE CHANGES
# """
fig_st = plt.figure(figsize=(10, 6))
plt.suptitle(description + " " + str(tau_lif) + "ms")
lbl = ['st_ini_prop', 'st_mid_prop', 'st_end_prop', 'st_ini_fix', 'st_mid_fix', 'st_end_fix']
st_lbl = ['_mean', '_med', '_max', '_min', '_q1', '_q90']
# st_lbl = ['', '', '_max', '_min', '_q1', '_q90']
cols_ = ['tab:orange', 'tab:blue', 'tab:red', 'tab:red', 'tab:green', 'tab:green']
ylims = [-70.15, -67.3]  # [-70.05, -52]
for i in range(6):
    ax_st = fig_st.add_subplot(2, 3, i + 1)
    for j in range(6):
        ax_st.plot(initial_frequencies, np.median(dr[lbl[i] + st_lbl[j]], axis=0), c=cols_[j], label=st_lbl[j][1:])
        ax_st.fill_between(initial_frequencies, np.quantile(dr[lbl[i] + st_lbl[j]], 0.1, axis=0),
                            np.quantile(dr[lbl[i] + st_lbl[j]], 0.9, axis=0), color=cols_[j], alpha=0.3)
    ax_st.set_xlabel("Frequency (Hz)")
    ax_st.set_ylabel("mem. pot. (mV)")
    # ax_st.set_ylim(ylims)
    ax_st.set_title(lbl[i].split("_")[1] + " win. (" + lbl[i].split("_")[2] + ")", color='gray')
    # ax_st.set_title("Frequency response", color='gray')
    ax_st.grid()
    ax_st.set_xscale('log')
    if (i + 1) % 3 == 0: ax_st.legend(loc='upper right')
    # ax_st.set_ylim(ylims)
fig_st.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
if save_figs: fig_st.savefig(folder_plots + file_name + '_windows_statistics.png', format='png')

fig_st2 = plt.figure(figsize=(5, 6))
plt.suptitle(description + " " + str(tau_lif) + "ms")
lbl = ['st_ini_prop', 'st_mid_prop', 'st_end_prop', 'st_ini_fix', 'st_mid_fix', 'st_end_fix']
ax_st2 = fig_st2.add_subplot(2, 1, 1)
for j in range(6):
    ax_st2.plot(initial_frequencies, np.median(dr[lbl[0] + st_lbl[j]], axis=0), c='gray')
    ax_st2.fill_between(initial_frequencies, np.quantile(dr[lbl[0] + st_lbl[j]], 0.1, axis=0),
                     np.quantile(dr[lbl[0] + st_lbl[j]], 0.9, axis=0), color='gray', alpha=0.3)
for j in range(6):
    ax_st2.plot(initial_frequencies, np.median(dr[lbl[1] + st_lbl[j]], axis=0), c=cols_[j])
    ax_st2.fill_between(initial_frequencies, np.quantile(dr[lbl[1] + st_lbl[j]], 0.1, axis=0),
                     np.quantile(dr[lbl[1] + st_lbl[j]], 0.9, axis=0), color=cols_[j], alpha=0.3)
ax_st2.set_title("Windows ini and mid (" + lbl[0].split("_")[2] + ")", color='gray')
ax_st2.set_xlabel("Frequency (Hz)")
ax_st2.grid()
ax_st2.set_xscale('log')
# ax_st2.set_ylim(ylims)
ax_st2 = fig_st2.add_subplot(2, 1, 2)
for j in range(6):
    ax_st2.plot(initial_frequencies, np.median(dr[lbl[3] + st_lbl[j]], axis=0), c='gray')
    ax_st2.fill_between(initial_frequencies, np.quantile(dr[lbl[3] + st_lbl[j]], 0.1, axis=0),
                     np.quantile(dr[lbl[3] + st_lbl[j]], 0.9, axis=0), color='gray', alpha=0.3)
for j in range(6):
    ax_st2.plot(initial_frequencies, np.median(dr[lbl[4] + st_lbl[j]], axis=0), c=cols_[j], label=st_lbl[j][1:])
    ax_st2.fill_between(initial_frequencies, np.quantile(dr[lbl[4] + st_lbl[j]], 0.1, axis=0),
                     np.quantile(dr[lbl[4] + st_lbl[j]], 0.9, axis=0), color=cols_[j], alpha=0.3)
ax_st2.set_xlabel("Frequency (Hz)")
ax_st2.set_ylabel("mem. pot. (mV)")
# ax_st2.set_ylim(ylims)
ax_st2.set_title("Windows ini and mid (" + lbl[i].split("_")[2] + ")", color='gray')
ax_st2.grid()
ax_st2.set_xscale('log')
ax_st2.legend(loc='upper right')
# ax_st2.set_ylim(ylims)
fig_st2.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
if save_figs: fig_st2.savefig(folder_plots + file_name + '_windows_statistics3.png', format='png')
# PLOT OF DIFFERENCES OF STEADY-STATE BETWEEN MID AND INI WINDOWS FOR PROPORTIONAL AND CONSTANT CHANGE OF RATES
"""
dims = total_realizations
factor = 1  # initial_frequencies  # 1
fig = plt.figure(figsize=(9, 7))
plt.suptitle(description)

aux_ = [[dr['st_mid_prop_mean'] - dr['st_ini_prop_mean'], dr['st_mid_prop_min'] - dr['st_ini_prop_min'], dr['st_mid_prop_max'] -
         dr['st_ini_prop_max'], dr['st_mid_prop_q1'] - dr['st_ini_prop_q1'],
         dr['st_mid_prop_q90'] - dr['st_ini_prop_q90']],
        [dr['st_mid_fix_mean'] - dr['st_ini_fix_mean'], dr['st_mid_fix_min'] - dr['st_ini_fix_min'], dr['st_mid_fix_max'] -
         dr['st_ini_fix_max'], dr['st_mid_fix_q1'] - dr['st_ini_fix_q1'], dr['st_mid_fix_q90'] - dr['st_ini_fix_q90']],
        [dr['mtr_mid_prop'] - dr['st_ini_prop_mean'], dr['mtr_mid_prop'] - dr['st_ini_prop_max'], dr['mtr_mid_prop'] -
         dr['w_mid_prop_max'], dr['mtr_mid_prop'] - dr['st_mid_prop_mean'], dr['mtr_mid_prop'] - dr['st_mid_prop_q90']],
        [dr['mtr_mid_fix'] - dr['st_ini_fix_mean'], dr['mtr_mid_fix'] - dr['st_ini_fix_max'], dr['mtr_mid_fix'] -
         dr['w_mid_fix_max'], dr['mtr_mid_fix'] - dr['st_mid_fix_mean'], dr['mtr_mid_fix'] - dr['st_mid_fix_q90']]
        ]
lbl_aux = [[r'$mean_{mid}$ - $mean_{ini}$', r'$min_{mid}$ - $min_{ini}$', r'$max_{mid}$ - $max_{ini}$',
            r'$q1_{mid}$ - $q1_{ini}$', r'$q90_{mid}$ - $q90_{ini}$'],
           [r'$mean_{mid}$ - $mean_{ini}$', r'$min_{mid}$ - $min_{ini}$', r'$max_{mid}$ - $max_{ini}$',
            r'$q1_{mid}$ - $q1_{ini}$', r'$q90_{mid}$ - $q90_{ini}$'],
           [r'$max_{tr-mid}$ - $mean_{ini}$', r'$max_{tr-mid}$ - $max_{ini}$', r'$max_{tr-mid}$ - $max_{mid}$',
            r'$max_{tr-mid}$ - $mean_{mid}$', r'$max_{tr-max}$ - $q90_{mid}$'],
           [r'$max_{tr-mid}$ - $mean_{ini}$', r'$max_{tr-mid}$ - $max_{ini}$', r'$max_{tr-mid}$ - $max_{mid}$',
            r'$max_{tr-mid}$ - $mean_{mid}$', r'$max_{tr-max}$ - $q90_{mid}$']]

aux_titles = [r"$mid_{st}-ini_{st}$ (Prop.)", r"$mid_{st}-ini_{st}$ (Cons.)", r"Transient (Prop.)", r"Transient (Cons)"]
for graph in range(1, 5):
    i_a = graph - 1
    aux_mean, aux_min, aux_max, aux_1, aux_90 = aux_[i_a][0], aux_[i_a][1], aux_[i_a][2], aux_[i_a][3], aux_[i_a][4]
    ax = fig.add_subplot(2, 2, graph)
    ax.plot(initial_frequencies, np.mean(aux_mean, axis=0) * factor, alpha=0.8, c="black", label=lbl_aux[i_a][0])
    ax.fill_between(initial_frequencies, np.quantile(aux_mean, 0.1, axis=0) * factor,
                    np.quantile(aux_mean, 0.9, axis=0) * factor, color="gray", alpha=0.3)
    ax.plot(initial_frequencies, np.mean(aux_min, axis=0) * factor, alpha=0.8, c="tab:red", label=lbl_aux[i_a][1])
    ax.fill_between(initial_frequencies, np.quantile(aux_min, 0.1, axis=0) * factor,
                    np.quantile(aux_min, 0.9, axis=0) * factor, color="tab:red", alpha=0.3)
    ax.plot(initial_frequencies, np.mean(aux_max, axis=0) * factor, alpha=0.8, c="tab:orange", label=lbl_aux[i_a][2])
    ax.fill_between(initial_frequencies, np.quantile(aux_max, 0.1, axis=0) * factor,
                    np.quantile(aux_max, 0.9, axis=0) * factor, color="tab:orange", alpha=0.3)
    ax.plot(initial_frequencies, np.mean(aux_1, axis=0) * factor, alpha=0.8, c="tab:blue", label=lbl_aux[i_a][3])
    ax.fill_between(initial_frequencies, np.quantile(aux_1, 0.1, axis=0) * factor,
                    np.quantile(aux_1, 0.9, axis=0) * factor, color="tab:blue", alpha=0.3)
    ax.plot(initial_frequencies, np.mean(aux_90, axis=0) * factor, alpha=0.8, c="tab:green", label=lbl_aux[i_a][4])
    ax.fill_between(initial_frequencies, np.quantile(aux_90, 0.1, axis=0) * factor,
                    np.quantile(aux_90, 0.9, axis=0) * factor, color="tab:green", alpha=0.3)
    ax.set_xscale('log')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mem. pot. diff. (mV)")
    ax.set_title(aux_titles[i_a], c="gray")
    ax.grid()
    ax.legend()

fig.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
plt.savefig(folder_plots + 'test.png', format='png')
# """

# SIMPLE PLOT OF DIFFERENCES OF STEADY-STATE BETWEEN MID AND INI WINDOWS FOR PROPORTIONAL AND CONSTANT CHANGE OF RATES
# AND THE DIFFERENCES BETWEEN MAX OF MID WINDOW AND MEDIAN OF INI WINDOW
# """
dims = total_realizations
factor = 1  # initial_frequencies  # 1
fig2 = plt.figure(figsize=(6.5, 2.5))  # 6.5, 5
plt.suptitle(description + " " + str(tau_lif) + "ms")


lbl = ['st_mid_prop', 'st_mid_fix']
lbl2 = ['st_ini_prop', 'st_ini_fix']
st_lbl = ['_max', '_min', '_q1', '_q90', '_mean', '_med']
cols_ = ['tab:red', 'tab:red', 'tab:green', 'tab:green', 'tab:orange', 'tab:blue']
t_ = [r"$mid_{st} - ini_{st}$ (Prop)", r"$mid_{st} - ini_{st}$ (Cons)",
      "Max Transient (Proportional)", "Max Transient (Constant)"]
ylims = [-62.5, -54.0]  # [-70.05, -52]
alpha = 0.1
for i in range(len(lbl)):
    ax_3 = fig2.add_subplot(1, 2, i + 1)
    for j in range(len(st_lbl)):
        if j > 3: alpha = 0.3
        else: alpha = 0.1
        aux = dr[lbl[i] + st_lbl[j]] - dr[lbl2[i] + st_lbl[j]]
        ax_3.plot(initial_frequencies, np.median(aux, axis=0), c=cols_[j], label=st_lbl[j][1:])
        ax_3.fill_between(initial_frequencies, np.quantile(aux, 0.1, axis=0),
                            np.quantile(aux, 0.9, axis=0), color=cols_[j], alpha=alpha)
    ax_3.set_xlabel("Frequency (Hz)")
    ax_3.set_ylabel("mem. pot. (mV)")
    # ax_3.set_ylim(ylims)
    ax_3.set_title(t_[i], c="gray")
    ax_3.grid()
    ax_3.set_xscale('log')
    if (i + 1) % 3 == 0: ax_3.legend(loc='upper right')
    # ax_st.set_ylim(ylims)
fig2.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
if save_figs: fig2.savefig(folder_plots + file_name + '_' + 'diff_st_log.png', format='png')

"""
dims = total_realizations
factor = 1  # initial_frequencies  # 1
fig2 = plt.figure(figsize=(6.5, 2.5))  # 6.5, 5
plt.suptitle(description + " " + str(tau_lif) + "ms")
s_a = [dr['st_mid_prop_med'] - dr['st_ini_prop_med'], dr['st_mid_fix_med'] - dr['st_ini_fix_med'],
       dr['mtr_mid_prop'] - dr['st_ini_prop_med'], dr['mtr_mid_fix'] - dr['st_ini_fix_med']]
s_b = [dr['st_mid_prop_mean'] - dr['st_ini_prop_mean'], dr['st_mid_fix_mean'] - dr['st_ini_fix_mean'],
       dr['mtr_mid_prop'] - dr['st_ini_prop_mean'], dr['mtr_mid_fix'] - dr['st_ini_fix_mean']]
t_ = [r"$mid_{st} - ini_{st}$ (Prop)", r"$mid_{st} - ini_{st}$ (Cons)",
      "Max Transient (Proportional)", "Max Transient (Constant)"]
ylims = [-0.005, 0.08]  # 100 syn [-0.005, 1.9], 1 syn [-0.005,1.9] [-0.005,0.08]
for i in range(2):
    ax3 = fig2.add_subplot(1, 2, i + 1)
    aux = s_a[i]
    ax3.plot(initial_frequencies, np.mean(aux, axis=0) * factor, c="tab:blue", label='med')
    ax3.fill_between(initial_frequencies, np.quantile(aux, 0.1, axis=0) * factor, np.quantile(aux, 0.9, axis=0) * factor,
                     color="tab:blue", alpha=0.2)
    aux2 = s_b[i]
    ax3.plot(initial_frequencies, np.mean(aux2, axis=0) * factor, c="tab:orange", label=r'$\mu$')
    ax3.fill_between(initial_frequencies, np.quantile(aux2, 0.1, axis=0) * factor, np.quantile(aux2, 0.9, axis=0) * factor,
                     color="tab:orange", alpha=0.2)
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Mem. pot. diff. (mV)")
    ax3.set_title(t_[i], c="gray")
    ax3.set_xscale('log')
    ax3.legend()
    # ax3.set_ylim(ylims)
    ax3.grid()
fig2.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
if save_figs: fig2.savefig(folder_plots + file_name + '_' + 'diff_st_log.png', format='png')
# """

# PLOT OF MEMBRANE POTENTIAL AND SOME OF THE CHARACTERISTICS (MEDIAN, MAX, MIN) IN INI AND MID WINDOWS
"""
i = 1
# signal = mssm_prop.EPSP[i * num_realizations:(i + 1) * num_realizations, :]
signal = lif_prop.membrane_potential[0, :]
a = np.mean(signal, axis=0)
b = np.quantile(signal, 0.1, axis=0)
c = np.quantile(signal, 0.9, axis=0)
aa = lif_prop.membrane_potential[0, int(0.5 / dt):int(Le_time_win / dt)]  # [, 0.5s:2s]
bb = lif_prop.membrane_potential[0, int((Le_time_win + 0.5) / dt):int(2 * Le_time_win / dt)]  # [, 2.5s:4.0s]
cc = lif_prop.membrane_potential[0, int((2 * Le_time_win + 0.5) / dt):int(max_t / dt)]  # [, 4.5s:6.0s]

plt.figure()
plt.fill_between(time_vector, b, c, color="darkgrey", alpha=0.3)
plt.plot(time_vector, a, alpha=0.8)
plt.grid()
d = int(Le_time_win / dt)
plt.plot(time_vector[:d], [np.mean(aa) for _ in range(d)], color="red", alpha=0.8)
plt.plot(time_vector[d: 2 * d], [np.mean(bb) for _ in range(d)], color="red", alpha=0.8)
plt.plot(time_vector[2 * d:], [np.mean(cc) for _ in range(d)], color="red", alpha=0.8)
plt.fill_between(time_vector[:d], np.quantile(aa, 0.1), np.quantile(aa, 0.9), color="red", alpha=0.1)
plt.fill_between(time_vector[d: 2 * d], np.quantile(bb, 0.1), np.quantile(bb, 0.9), color="red", alpha=0.1)
plt.fill_between(time_vector[2 * d:], np.quantile(cc, 0.1), np.quantile(cc, 0.9), color="red", alpha=0.1)
# """

# EXAMPLE OF HOW INPUT FIRING RATES CHANGE EITHER PROPORTIONALLY OR CONSTANTLY
"""
f_ref = 50
sf = 1000
dt = 1 / sfreq
t_vec = np.arange(0, 6, dt)
prop_freq = []
cons_freq = []


fig3 = plt.figure(figsize=(7.5, 3.5))
plt.suptitle("Configuration of changing input firing rates")
ax6 = fig3.add_subplot(1, 2, 1)
ax6.set_ylim(0, 350)
ax7 = fig3.add_subplot(1, 2, 2)
ax7.set_ylim(0, 350)

for f_ref in [50, 100, 200]:
    aux_r = range(int(2 / dt))
    prop_freq = [f_ref for _ in aux_r] + [f_ref * 1.5 for _ in aux_r] + [f_ref for _ in aux_r]
    cons_freq = [f_ref for _ in aux_r] + [f_ref + 10 for _ in aux_r] + [f_ref for _ in aux_r]
    ax6.plot(t_vec, prop_freq, label=str(f_ref) + "Hz")
    ax7.plot(t_vec, cons_freq, label=str(f_ref) + "Hz")


ax6.grid()
ax7.grid()
ax6.legend()
ax7.legend()
ax6.set_xlabel("time (s)")
ax7.set_xlabel("time (s)")
ax6.set_ylabel("firing rate (Hz)")
ax6.set_title("Proportional", color="gray")
ax7.set_title("Constant", color="gray")
fig3.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)


fig4 = plt.figure(figsize=(7.5, 3.5))
plt.suptitle("Example of input patterns")
ax6 = fig4.add_subplot(1, 2, 1)
ax7 = fig4.add_subplot(1, 2, 2)
ax6.plot(time_vector, cons_input[0, :])
ax7.plot(time_vector, fix_input[0, :])
ax6.grid()
ax7.grid()
ax6.set_xlabel("time (s)")
ax7.set_xlabel("time (s)")
ax6.set_ylabel("Action potentials")
ax6.set_title("Proportional", color="gray")
ax7.set_title("Constant", color="gray")
fig4.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
# """

# PLOT OF MEMBRANE POTENTIAL WHEN OUTPUT OF MSSM IS EITHER EPSP(T) OR N(T)
"""
i = 0
fig5 = plt.figure(figsize=(10, 5))
# plt.suptitle("Membrane potential response for ref rate " + str(initial_frequencies[i - 1]) + "Hz")
ax8 = fig5.add_subplot(2, 2, 1)
ax9 = fig5.add_subplot(2, 2, 2)
ax8.plot(time_vector, lif_prop.membrane_potential[0, :])
ax9.plot(time_vector, lif_fix.membrane_potential[0, :])
ax8.grid()
ax9.grid()
ax8.set_xlabel("time (s)")
ax9.set_xlabel("time (s)")
ax8.set_ylabel("membrane potential (mV)")
ax8.set_title("Epsp Proportional", color="gray")
ax9.set_title("Epsp Constant", color="gray")
th_tr = Le_time_win * 0.25
ax8.plot(time_vector[int(Le_time_win / dt):int((Le_time_win + th_tr) / dt)],
         [res_per_reali[31, 0, 0] for _ in range(int(th_tr / dt))], c='red', label='w_mid_prop_max')
ax8.plot(time_vector[int(Le_time_win / dt):int(2 * Le_time_win / dt)],
         [res_per_reali[45, 0, 0] for _ in range(int(Le_time_win / dt))], c='orange', label='w_mid_prop_max')
ax10 = fig5.add_subplot(2, 2, 3)
ax11 = fig5.add_subplot(2, 2, 4)
ax10.plot(time_vector, lif_prop_n.membrane_potential[0, :])
ax11.plot(time_vector, lif_fix_n.membrane_potential[0, :])
ax10.grid()
ax11.grid()
ax10.set_xlabel("time (s)")
ax11.set_xlabel("time (s)")
ax10.set_ylabel("membrane potential (mV)")
ax10.set_title("N(t) Proportional", color="gray")
ax11.set_title("N(t) Constant", color="gray")
fig5.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
th_tr = Le_time_win * 0.25
ax10.plot(time_vector[int(Le_time_win / dt):int((Le_time_win + th_tr) / dt)], 
          [res_per_reali_n[31, 0, 0] for _ in range(int(th_tr / dt))], c='red', label='w_mid_prop_max')
ax10.plot(time_vector[int(Le_time_win / dt):int(2 * Le_time_win / dt)], 
          [res_per_reali_n[45, 0, 0] for _ in range(int(Le_time_win / dt))], c='orange', label='w_mid_prop_max')

# """

# CHECKING IF mtr_mid_prop INDEED CORRESPONDS TO THE MAXIMUM OF MID WINDOW
"""
# Plot of steady-state of experiments
# plt.suptitle(description)
fig6 = plt.figure(figsize=(4.5, 4.5))
ax1 = fig6.add_subplot(1, 1, 1)
a = dr['mtr_mid_prop']
b = dr['w_mid_prop_max']
aux = a - b
ax1.plot(initial_frequencies, np.mean(aux, axis=0), alpha=0.8, c="tab:blue", label='mtr_mid_prop')
# ax1.plot(initial_frequencies, np.mean(b, axis=0), alpha=0.8, c="tab:red", label='w_mid_prop_max')
ax1.fill_between(initial_frequencies, np.quantile(aux, 0.1, axis=0), np.quantile(aux, 0.9, axis=0), 
                 color="tab:blue", alpha=0.2)
ax1.set_xlabel("Frequency (Hz)")
ax1.set_title("Max mid (Proportional) vs. max. mid transition", c="gray")
ax1.grid()
ax1.legend()
# """
