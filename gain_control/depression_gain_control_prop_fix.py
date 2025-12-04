# from synaptic_dynamic_models.simple_depression import Simple_Depression
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
save_vars = True
run_experiment = False
lif_parallel = True
imputations = False
Stoch_input = True
num_syn = 100

# Sampling frequency and conditions for running parallel or single LIF neurons
sfreq = 5e3
total_realizations = 100  # 100
num_realizations = 1  # 4

# Input modulations
range_f = [i for i in range(10, 100, 5)]
range_f2 = [i for i in range(100, 500, 10)] # [i for i in range(100, 500, 10)] [i for i in range(100, 321, 10)]
range_f3 = [i for i in range(500, 2501, 50)]  # Max prop. freq. must be less than sfreq/4
initial_frequencies = np.array(range_f + range_f2 + range_f3)

# Path variables
path_vars = "../gain_control/variables/"
check_create_folder(path_vars)
folder_plots = '../gain_control/plots/'
check_create_folder(folder_plots)
file_name = model + "_gain_control_" + str(int(sfreq / 1000)) + "k_ind_" + str(ind) + "_syn_" + str(num_syn)
if lif_parallel:
    file_name += "_p"
else:
    file_name += "_i"
if imputations:
    file_name += "_cwi"
else:
    file_name += "_cni"

print("For file %s and index %d" % (file_name, ind))
# 2KHz from 10 to 200, 100 syn, 120 exp. old-mac / new-mac / server
# 2 rea, 1 exp 0:01.30, 100 rea 2.16 hours / 1 exp. 0:01.34, 100 rea 2.23 hours / 1 exp 0:01.87, 100 rea 3.12 hours
# 3 rea, 1 exp 0:01.60, 102 rea 1.81 hours / 1 exp. 0:01.72, 102 rea 1.95 hours / 1 exp 0:02.20, 102 rea 2.5 hours
# 4 rea, 1 exp 0:01.70, 100 rea 1.42 hours / 1 exp. 0:02.0, 100 rea 1.66 hours / 1 exp 0:02.47, 100 rea 2.06 hours
# 5 rea, 1 exp 0:02.15, 100 rea 1.46 hours / 1 exp. 0:02.46, 100 rea 1.64 hours / 1 exp 0:02.75, 100 rea 1.84 hours
# 6 rea, 1 exp 0:0x.xx, 100 rea x.x hours / 1 exp. 0:06.78, 102 rea 3.84 hours / 1 exp 0:03.13, 102 rea 1.77 hours
# 7 rea, 1 exp 0:0x.xx, 100 rea x.x hours / 1 exp. 0:09.38, 102 rea 4.69 hours / 1 exp 0:03.4, 105 rea 1.7 hours
# 8 rea, 1 exp 0:06.00, 100 rea 2.5 hours / 1 exp. 0:10.5.2, 104 rea 4.55 hours / 1 exp 0:03.70, 104 rea 1.51 hours
# 9 rea, 1 exp 0:06.00, 100 rea 2.5 hours / 1 exp. 0:11.50, 104 rea 4.6 hours / 1 exp 0:05.22, 108 rea 2.08 hours
# 10 rea, 1 exp 0:10.00, 100 rea 3.3 hours / 1 exp. 0:14.5, 100 rea 4.84 hours / 1 exp 0:05.64, 100 rea 1.88 hours
# 20 rea, 1 exp 0:20.00, / 1 exp. 0:32.40, 100 rea 3.12 hours / 1 exp 0:30.60, 100 rea 5.1 hours
# 30 rea, 1 exp 0:30.00, / 1 exp. 0:46.20, 100 rea 3.12 hours / 1 exp 0:44.50, 100 rea 5.94 hours
# 35 rea, 1 exp 0:42.00,
# 40 rea, 1 exp 0:55.00,
# 50 rea, 1 exp 1:35.00 / 1 exp. 2:08.60 / 1 exp. 2:02.20, 100 rea 8.15 hours
# 100 rea, 1 exp. 5:35.00,/ one exp. 9:00.00 / 1 exp. 6:36.30, 100 rea 13.21 hours
# nohub &
# 120 experiments, one rea 1:30.00, 100 rea 2.5 hours
# ******************************************************************************************************************
# Local variables
stat_list = ['st_ini_prop', 'st_mid_prop', 'st_end_prop', 'st_ini_prop_q1', 'st_mid_prop_q1', 'st_end_prop_q1',
             'st_ini_prop_q90', 'st_mid_prop_q90', 'st_end_prop_q90', 'st_ini_prop_min', 'st_mid_prop_min',
             'st_end_prop_min', 'st_ini_prop_max', 'st_mid_prop_max', 'st_end_prop_max',
             'st_ini_fix', 'st_mid_fix', 'st_end_fix', 'st_ini_fix_q1', 'st_mid_fix_q1', 'st_end_fix_q1',
             'st_ini_fix_q90', 'st_mid_fix_q90', 'st_end_fix_q90', 'st_ini_fix_min', 'st_mid_fix_min',
             'st_end_fix_min', 'st_ini_fix_max', 'st_mid_fix_max', 'st_end_fix_max',
             'mtr_ini_prop', 'mtr_mid_prop', 'mtr_end_prop', 'mtr_ini_fix', 'mtr_mid_fix', 'mtr_end_fix',
             'w_ini_prop', 'w_ini_prop_q1', 'w_ini_prop_q90', 'w_ini_prop_min', 'w_ini_prop_max',
             'w_mid_prop', 'w_mid_prop_q1', 'w_mid_prop_q90', 'w_mid_prop_min', 'w_mid_prop_max',
             'w_end_prop', 'w_end_prop_q1', 'w_end_prop_q90', 'w_end_prop_min', 'w_end_prop_max',
             'w_ini_fix', 'w_ini_fix_q1', 'w_ini_fix_q90', 'w_ini_fix_min', 'w_ini_fix_max',
             'w_mid_fix', 'w_mid_fix_q1', 'w_mid_fix_q90', 'w_mid_fix_min', 'w_mid_fix_max',
             'w_end_fix', 'w_end_fix_q1', 'w_end_fix_q90', 'w_end_fix_min', 'w_end_fix_max',
             'st_n_ini_prop', 'st_n_mid_prop', 'st_n_end_prop', 'st_n_ini_prop_q1', 'st_n_mid_prop_q1',
             'st_n_end_prop_q1',
             'st_n_ini_prop_q90', 'st_n_mid_prop_q90', 'st_n_end_prop_q90', 'st_n_ini_prop_min', 'st_n_mid_prop_min',
             'st_n_end_prop_min', 'st_n_ini_prop_max', 'st_n_mid_prop_max', 'st_n_end_prop_max',
             'st_n_ini_fix', 'st_n_mid_fix', 'st_n_end_fix', 'st_n_ini_fix_q1', 'st_n_mid_fix_q1', 'st_n_end_fix_q1',
             'st_n_ini_fix_q90', 'st_n_mid_fix_q90', 'st_n_end_fix_q90', 'st_n_ini_fix_min', 'st_n_mid_fix_min',
             'st_n_end_fix_min', 'st_n_ini_fix_max', 'st_n_mid_fix_max', 'st_n_end_fix_max',
             'mtr_n_ini_prop', 'mtr_n_mid_prop', 'mtr_n_end_prop', 'mtr_n_ini_fix', 'mtr_n_mid_fix', 'mtr_n_end_fix',
             'w_n_ini_prop', 'w_n_ini_prop_q1', 'w_n_ini_prop_q90', 'w_n_ini_prop_min', 'w_n_ini_prop_max',
             'w_n_mid_prop', 'w_n_mid_prop_q1', 'w_n_mid_prop_q90', 'w_n_mid_prop_min', 'w_n_mid_prop_max',
             'w_n_end_prop', 'w_n_end_prop_q1', 'w_n_end_prop_q90', 'w_n_end_prop_min', 'w_n_end_prop_max',
             'w_n_ini_fix', 'w_n_ini_fix_q1', 'w_n_ini_fix_q90', 'w_n_ini_fix_min', 'w_n_ini_fix_max',
             'w_n_mid_fix', 'w_n_mid_fix_q1', 'w_n_mid_fix_q90', 'w_n_mid_fix_min', 'w_n_mid_fix_max',
             'w_n_end_fix', 'w_n_end_fix_q1', 'w_n_end_fix_q90', 'w_n_end_fix_min', 'w_n_end_fix_max',
             'initial_frequencies', 'stp_model', 'name_params', 'dyn_synapse', 'num_synapses', 'syn_params',
             'sim_params', 'lif_params', 'lif_params2', 'prop_rate_change_a', 'fix_rate_change_a', 'num_changes_rate',
             'description', 'seeds', 'realizations', 't_realizations']

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
    # If poisson input
    Stoch_input = False
    if seeds is not None: Stoch_input = True

    # Time conditions
    max_t, sfreq, time_vector, L = sim_params['max_t'], sim_params['sfreq'], sim_params['time_vector'], sim_params['L']
    dt = 1 / sfreq
    Le_time_win = int(max_t / num_changes_rate)

    # Parameters in dict format
    params = dict(zip(name_params, syn_params))

    # Number of experiments
    num_experiments = initial_frequencies.shape[0]

    # Number of iterations while looping the realizations
    num_loop_realizations = int(total_realizations / num_realizations)

    # seeds (shape (num_realizations * num_experiments))
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
                  'tau_m': np.array([30e-3 for _ in range(1)]),
                  'g_L': np.array([2.7e-2 for _ in range(1)]),
                  'V_init': np.array([-70 for _ in range(1)]), 'V_equilibrium': np.array([-70 for _ in range(1)]),
                  't_refractory': np.array([0.01 for _ in range(1)])}

    lif_params2 = {'V_threshold': np.array([1000 for _ in range(1)]), 'V_reset': np.array([-70 for _ in range(1)]),
                   'tau_m': np.array([30e-3 for _ in range(1)]),
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

    # For poisson or deterministic inputs
    if not lif_parallel: num_realizations = 1
    seeds, seeds1, seeds2, seeds3 = [], None, None, None
    if not Stoch_input:
        total_realizations = 1
        num_realizations = 1

    num_loop_realizations = int(total_realizations / num_realizations)
#
aux_num_r = num_synapses
if lif_parallel: aux_num_r = int(num_realizations * num_synapses)

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
lif_prop = LIF_model(n_neu=1)
if lif_parallel: lif_prop = LIF_model(n_neu=num_realizations)
lif_prop.set_model_params(lif_params)
lif_prop_n = None
if model == "MSSM":
    lif_prop_n = LIF_model(n_neu=1)
    if lif_parallel: lif_prop_n = LIF_model(n_neu=num_realizations)
    lif_prop_n.set_model_params(lif_params)

# Creating LIF models for constant rate change
lif_fix = LIF_model(n_neu=1)
if lif_parallel: lif_fix = LIF_model(n_neu=num_realizations)
lif_fix.set_model_params(lif_params2)
lif_fix_n = None
if model == "MSSM":
    lif_fix_n = LIF_model(n_neu=1)
    if lif_parallel: lif_fix_n = LIF_model(n_neu=num_realizations)
    lif_fix_n.set_model_params(lif_params2)

# Setting proportional and fixed rates of change
proportional_rate_change = prop_rate_change_a[0]  # [k]
fixed_rate_change = fix_rate_change_a[0]  # [k]
constant_changes = proportional_rate_change * initial_frequencies + initial_frequencies
fixed_changes = fixed_rate_change + initial_frequencies

# Auxiliar variables for statistics
res_per_reali = np.zeros((66, num_experiments, num_realizations))
res_real = np.zeros((66, total_realizations, num_experiments))
res_per_reali_n = np.zeros((66, num_experiments, num_realizations))
res_real_n = np.zeros((66, total_realizations, num_experiments))

ini_loop_time = m_time()
print("Ini big loop")
realization = 0
# for realization in range(num_realizations):  # range(len(prop_rate_change_a)):
while realization < num_loop_realizations and (not file_loaded or run_experiment):
    loop_time = m_time()

    # Building reference signal for constant and fixed rate changes
    i = 0
    # for i in range(num_experiments):
    while i < num_experiments:
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
        cons_aux = simple_spike_train(sfreq, constant_changes[i], int(L / num_changes_rate),
                                      num_realizations=aux_num_r, poisson=Stoch_input, seeds=seeds2, correction=True,
                                      imputation=imputations)
        fix_aux = simple_spike_train(sfreq, fixed_changes[i], int(L / num_changes_rate),
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
            if model == "MSSM":
                lif_prop_n.set_simulation_params(sim_params)
                lif_fix_n.set_simulation_params(sim_params)
            # Running the models
            if lif_parallel:
                model_stp_parallel(stp_prop, lif_prop, params, cons_input, lif_prop_n)
                model_stp_parallel(stp_fix, lif_fix, params, fix_input, lif_fix_n)
            else:
                model_stp(stp_prop, lif_prop, params, cons_input, lif_prop_n)
                model_stp(stp_fix, lif_fix, params, fix_input, lif_fix_n)
        else:
            # Reseting initial conditions
            lif_prop.set_simulation_params(sim_params)
            lif_fix.set_simulation_params(sim_params)
            # Running the models
            static_synapse(lif_prop, cons_input, 9e0)  # , 0.0125e-6)
            static_synapse(lif_fix, fix_input, 9e0)  # , 0.0125e-6)

        res_per_reali[:, i, :] = aux_statistics_prop_cons(lif_prop.membrane_potential, lif_fix.membrane_potential,
                                                          Le_time_win, Le_time_win * 0.25, sim_params)
        if lif_prop_n is not None:
            res_per_reali_n[:, i, :] = aux_statistics_prop_cons(lif_prop_n.membrane_potential,
                                                                lif_fix_n.membrane_potential, Le_time_win,
                                                                Le_time_win * 0.25, sim_params)

        print_time(m_time() - loop_experiments,
                   "Realisation " + str(realization) + ", frequency " + str(initial_frequencies[i]))

        """
        fig = plt.figure(figsize=(10, 4))
        plt.suptitle("For model %s, index %d, frequency %dHz, for %d synapses" % (
            model, ind, initial_frequencies[i], num_synapses))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Mem. potential (mV)")
        plt.plot(time_vector, lif_prop.membrane_potential[0, :], c="tab:blue")
        plt.grid()
        ax1.set_title("Proportional changes", color="gray")
        ax2 = fig.add_subplot(1, 2, 2)
        plt.plot(time_vector, lif_fix.membrane_potential[0, :], c="tab:blue")
        ax2.set_title("Constant changes", color="gray")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Mem. potential (mV)")
        plt.grid()
        # """
        i += 1

    # steady-state part
    for res_i in range(res_real.shape[0]):
        r = realization
        res_real[res_i, r * num_realizations:(r + 1) * num_realizations] = res_per_reali[res_i, :].T
        if lif_prop_n is not None:
            res_real_n[res_i, r * num_realizations:(r + 1) * num_realizations] = res_per_reali_n[res_i, :].T

    print_time(m_time() - loop_time, "Realisation " + str(realization))

    realization += 1

if not os.path.isfile(path_vars + file_name):
    dr = {'initial_frequencies': initial_frequencies,
          'stp_model': model, 'name_params': name_params, 'dyn_synapse': dyn_synapse,
          'num_synapses': num_synapses, 'syn_params': syn_params, 'sim_params': sim_params,
          'lif_params': lif_params, 'lif_params2': lif_params2, 'prop_rate_change_a': prop_rate_change_a,
          'fix_rate_change_a': prop_rate_change_a, 'num_changes_rate': num_changes_rate,
          'description': description, 'seeds': seeds,
          'realizations': num_realizations, 't_realizations': total_realizations}
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
"""
ylims = (-70, -69.5)
fig_st = plt.figure(figsize=(6, 6))
plt.suptitle(description)

ax1 = fig_st.add_subplot(2, 3, 1)
ax1.plot(initial_frequencies, np.mean(dr['st_ini_prop'], axis=0), c='black')
ax1.fill_between(initial_frequencies, np.quantile(dr['st_ini_prop'], 0.1, axis=0),
                 np.quantile(dr['st_ini_prop'], 0.9, axis=0), color="black", alpha=0.3)
ax1.plot(initial_frequencies, np.mean(dr['st_ini_prop_max'], axis=0), c='red')
ax1.fill_between(initial_frequencies, np.quantile(dr['st_ini_prop_max'], 0.1, axis=0),
                 np.quantile(dr['st_ini_prop_max'], 0.9, axis=0), color="red", alpha=0.3)
ax1.plot(initial_frequencies, np.mean(dr['st_ini_prop_min'], axis=0), c='red')
ax1.fill_between(initial_frequencies, np.quantile(dr['st_ini_prop_min'], 0.1, axis=0),
                 np.quantile(dr['st_ini_prop_min'], 0.9, axis=0), color="red", alpha=0.3)
ax1.plot(initial_frequencies, np.mean(dr['st_ini_prop_q1'], axis=0), c='tab:blue')
ax1.fill_between(initial_frequencies, np.quantile(dr['st_ini_prop_q1'], 0.1, axis=0),
                 np.quantile(dr['st_ini_prop_q1'], 0.9, axis=0), color="tab:blue", alpha=0.3)
ax1.plot(initial_frequencies, np.mean(dr['st_ini_prop_q90'], axis=0), c='tab:blue')
ax1.fill_between(initial_frequencies, np.quantile(dr['st_ini_prop_q90'], 0.1, axis=0),
                 np.quantile(dr['st_ini_prop_q90'], 0.9, axis=0), color="tab:blue", alpha=0.3)
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("mem. pot. (mV)")
ax1.set_ylim(ylims)
ax1.set_title("Prop. ini win.", color='gray')
ax1.grid()

ax2 = fig_st.add_subplot(2, 3, 2)
ax2.plot(initial_frequencies, np.mean(dr['st_mid_prop'], axis=0), c='black')
ax2.fill_between(initial_frequencies, np.quantile(dr['st_mid_prop'], 0.1, axis=0),
                 np.quantile(dr['st_mid_prop'], 0.9, axis=0), color="black", alpha=0.3)
ax2.plot(initial_frequencies, np.mean(dr['st_mid_prop_max'], axis=0), c='red')
ax2.fill_between(initial_frequencies, np.quantile(dr['st_mid_prop_max'], 0.1, axis=0),
                 np.quantile(dr['st_mid_prop_max'], 0.9, axis=0), color="red", alpha=0.3)
ax2.plot(initial_frequencies, np.mean(dr['st_mid_prop_min'], axis=0), c='red')
ax2.fill_between(initial_frequencies, np.quantile(dr['st_mid_prop_min'], 0.1, axis=0),
                 np.quantile(dr['st_mid_prop_min'], 0.9, axis=0), color="red", alpha=0.3)
ax2.plot(initial_frequencies, np.mean(dr['st_mid_prop_q1'], axis=0), c='tab:blue')
ax2.fill_between(initial_frequencies, np.quantile(dr['st_mid_prop_q1'], 0.1, axis=0),
                 np.quantile(dr['st_mid_prop_q1'], 0.9, axis=0), color="tab:blue", alpha=0.3)
ax2.plot(initial_frequencies, np.mean(dr['st_mid_prop_q90'], axis=0), c='tab:blue')
ax2.fill_between(initial_frequencies, np.quantile(dr['st_mid_prop_q90'], 0.1, axis=0),
                 np.quantile(dr['st_mid_prop_q90'], 0.9, axis=0), color="tab:blue", alpha=0.3)
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("mem. pot. (mV)")
ax2.set_title("Prop. mid win.", color='gray')
ax2.set_ylim(ylims)
ax2.grid()

ax3 = fig_st.add_subplot(2, 3, 3)
ax3.plot(initial_frequencies, np.mean(dr['st_end_prop'], axis=0), c='black')
ax3.fill_between(initial_frequencies, np.quantile(dr['st_end_prop'], 0.1, axis=0),
                 np.quantile(dr['st_end_prop'], 0.9, axis=0), color="black", alpha=0.3)
ax3.plot(initial_frequencies, np.mean(dr['st_end_prop_max'], axis=0), c='red')
ax3.fill_between(initial_frequencies, np.quantile(dr['st_end_prop_max'], 0.1, axis=0),
                 np.quantile(dr['st_end_prop_max'], 0.9, axis=0), color="red", alpha=0.3)
ax3.plot(initial_frequencies, np.mean(dr['st_end_prop_min'], axis=0), c='red')
ax3.fill_between(initial_frequencies, np.quantile(dr['st_end_prop_min'], 0.1, axis=0),
                 np.quantile(dr['st_end_prop_min'], 0.9, axis=0), color="red", alpha=0.3)
ax3.plot(initial_frequencies, np.mean(dr['st_end_prop_q1'], axis=0), c='tab:blue')
ax3.fill_between(initial_frequencies, np.quantile(dr['st_end_prop_q1'], 0.1, axis=0),
                 np.quantile(dr['st_end_prop_q1'], 0.9, axis=0), color="tab:blue", alpha=0.3)
ax3.plot(initial_frequencies, np.mean(dr['st_end_prop_q90'], axis=0), c='tab:blue')
ax3.fill_between(initial_frequencies, np.quantile(dr['st_end_prop_q90'], 0.1, axis=0),
                 np.quantile(dr['st_end_prop_q90'], 0.9, axis=0), color="tab:blue", alpha=0.3)
ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("mem. pot. (mV)")
ax3.set_title("Prop. end win.", color='gray')
ax3.set_ylim(ylims)
ax3.grid()

ax4 = fig_st.add_subplot(2, 3, 4)
ax4.plot(initial_frequencies, np.mean(dr['st_ini_fix'], axis=0), c='black')
ax4.fill_between(initial_frequencies, np.quantile(dr['st_ini_fix'], 0.1, axis=0),
                 np.quantile(dr['st_ini_fix'], 0.9, axis=0), color="black", alpha=0.3)
ax4.plot(initial_frequencies, np.mean(dr['st_ini_fix_max'], axis=0), c='red')
ax4.fill_between(initial_frequencies, np.quantile(dr['st_ini_fix_max'], 0.1, axis=0),
                 np.quantile(dr['st_ini_fix_max'], 0.9, axis=0), color="red", alpha=0.3)
ax4.plot(initial_frequencies, np.mean(dr['st_ini_fix_min'], axis=0), c='red')
ax4.fill_between(initial_frequencies, np.quantile(dr['st_ini_fix_min'], 0.1, axis=0),
                 np.quantile(dr['st_ini_fix_min'], 0.9, axis=0), color="red", alpha=0.3)
ax4.plot(initial_frequencies, np.mean(dr['st_ini_fix_q1'], axis=0), c='tab:blue')
ax4.fill_between(initial_frequencies, np.quantile(dr['st_ini_fix_q1'], 0.1, axis=0),
                 np.quantile(dr['st_ini_fix_q1'], 0.9, axis=0), color="tab:blue", alpha=0.3)
ax4.plot(initial_frequencies, np.mean(dr['st_ini_fix_q90'], axis=0), c='tab:blue')
ax4.fill_between(initial_frequencies, np.quantile(dr['st_ini_fix_q90'], 0.1, axis=0),
                 np.quantile(dr['st_ini_fix_q90'], 0.9, axis=0), color="tab:blue", alpha=0.3)
ax4.set_xlabel("Frequency (Hz)")
ax4.set_ylabel("mem. pot. (mV)")
ax4.set_title("Cons. ini win.", color='gray')
ax4.set_ylim(ylims)
ax4.grid()

ax5 = fig_st.add_subplot(2, 3, 5)
ax5.plot(initial_frequencies, np.mean(dr['st_mid_fix'], axis=0), c='black')
ax5.fill_between(initial_frequencies, np.quantile(dr['st_mid_fix'], 0.1, axis=0),
                 np.quantile(dr['st_mid_fix'], 0.9, axis=0), color="black", alpha=0.3)
ax5.plot(initial_frequencies, np.mean(dr['st_mid_fix_max'], axis=0), c='red')
ax5.fill_between(initial_frequencies, np.quantile(dr['st_mid_fix_max'], 0.1, axis=0),
                 np.quantile(dr['st_mid_fix_max'], 0.9, axis=0), color="red", alpha=0.3)
ax5.plot(initial_frequencies, np.mean(dr['st_mid_fix_min'], axis=0), c='red')
ax5.fill_between(initial_frequencies, np.quantile(dr['st_mid_fix_min'], 0.1, axis=0),
                 np.quantile(dr['st_mid_fix_min'], 0.9, axis=0), color="red", alpha=0.3)
ax5.plot(initial_frequencies, np.mean(dr['st_mid_fix_q1'], axis=0), c='tab:blue')
ax5.fill_between(initial_frequencies, np.quantile(dr['st_mid_fix_q1'], 0.1, axis=0),
                 np.quantile(dr['st_mid_fix_q1'], 0.9, axis=0), color="tab:blue", alpha=0.3)
ax5.plot(initial_frequencies, np.mean(dr['st_mid_fix_q90'], axis=0), c='tab:blue')
ax5.fill_between(initial_frequencies, np.quantile(dr['st_mid_fix_q90'], 0.1, axis=0),
                 np.quantile(dr['st_mid_fix_q90'], 0.9, axis=0), color="tab:blue", alpha=0.3)
ax5.set_xlabel("Frequency (Hz)")
ax5.set_ylabel("mem. pot. (mV)")
ax5.set_title("Cons. mid win.", color='gray')
ax5.set_ylim(ylims)
ax5.grid()

ax7 = fig_st.add_subplot(2, 3, 6)
ax7.plot(initial_frequencies, np.mean(dr['st_end_fix'], axis=0), c='black')
ax7.fill_between(initial_frequencies, np.quantile(dr['st_end_fix'], 0.1, axis=0),
                 np.quantile(dr['st_end_fix'], 0.9, axis=0), color="black", alpha=0.3)
ax7.plot(initial_frequencies, np.mean(dr['st_end_fix_max'], axis=0), c='red')
ax7.fill_between(initial_frequencies, np.quantile(dr['st_end_fix_max'], 0.1, axis=0),
                 np.quantile(dr['st_end_fix_max'], 0.9, axis=0), color="red", alpha=0.3)
ax7.plot(initial_frequencies, np.mean(dr['st_end_fix_min'], axis=0), c='red')
ax7.fill_between(initial_frequencies, np.quantile(dr['st_end_fix_min'], 0.1, axis=0),
                 np.quantile(dr['st_end_fix_min'], 0.9, axis=0), color="red", alpha=0.3)
ax7.plot(initial_frequencies, np.mean(dr['st_end_fix_q1'], axis=0), c='tab:blue')
ax7.fill_between(initial_frequencies, np.quantile(dr['st_end_fix_q1'], 0.1, axis=0),
                 np.quantile(dr['st_end_fix_q1'], 0.9, axis=0), color="tab:blue", alpha=0.3)
ax7.plot(initial_frequencies, np.mean(dr['st_end_fix_q90'], axis=0), c='tab:blue')
ax7.fill_between(initial_frequencies, np.quantile(dr['st_end_fix_q90'], 0.1, axis=0),
                 np.quantile(dr['st_end_fix_q90'], 0.9, axis=0), color="tab:blue", alpha=0.3)
ax7.set_xlabel("Frequency (Hz)")
ax7.set_ylabel("mem. pot. (mV)")
ax7.set_title("Cons. end win.", color='gray')
ax7.set_ylim(ylims)
ax7.grid()

fig_st.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
# """

# PLOT OF DIFFERENCES OF STEADY-STATE BETWEEN MID AND INI WINDOWS FOR PROPORTIONAL AND CONSTANT CHANGE OF RATES
# """
dims = total_realizations
factor = 1  # initial_frequencies  # 1
fig2 = plt.figure(figsize=(7, 7))
plt.suptitle(description)

aux_ = [[dr['st_mid_prop'] - dr['st_ini_prop'], dr['st_mid_prop_min'] - dr['st_ini_prop_min'], dr['st_mid_prop_max'] -
         dr['st_ini_prop_max'], dr['st_mid_prop_q1'] - dr['st_ini_prop_q1'],
         dr['st_mid_prop_q90'] - dr['st_ini_prop_q90']],
        [dr['st_mid_fix'] - dr['st_ini_fix'], dr['st_mid_fix_min'] - dr['st_ini_fix_min'], dr['st_mid_fix_max'] -
         dr['st_ini_fix_max'], dr['st_mid_fix_q1'] - dr['st_ini_fix_q1'], dr['st_mid_fix_q90'] - dr['st_ini_fix_q90']],
        [dr['mtr_mid_prop'] - dr['st_ini_prop'], dr['mtr_mid_prop'] - dr['st_ini_prop_max'], dr['mtr_mid_prop'] -
         dr['w_mid_prop_max'], dr['mtr_mid_prop'] - dr['st_mid_prop'], dr['mtr_mid_prop'] - dr['st_mid_prop_q90']],
        [dr['mtr_mid_fix'] - dr['st_ini_fix'], dr['mtr_mid_fix'] - dr['st_ini_fix_max'], dr['mtr_mid_fix'] -
         dr['w_mid_fix_max'], dr['mtr_mid_fix'] - dr['st_mid_fix'], dr['mtr_mid_fix'] - dr['st_mid_fix_q90']]
        ]
aux_ = [[dr['st_mid_prop'] - dr['st_ini_prop'], dr['st_mid_prop_q90'] - dr['st_mid_prop_q1'], dr['st_ini_prop_q90'] -
         dr['st_ini_prop_q1'], dr['mtr_mid_prop'] - dr['st_ini_prop'], dr['mtr_mid_prop'] - dr['st_mid_prop']],
        [dr['st_mid_fix'] - dr['st_ini_fix'], dr['st_mid_fix_q90'] - dr['st_mid_fix_q1'], dr['st_ini_fix_q90'] -
         dr['st_ini_fix_q1'], dr['mtr_mid_fix'] - dr['st_ini_fix'], dr['mtr_mid_fix'] - dr['st_mid_fix']],
        [dr['mtr_mid_prop'] - dr['st_ini_prop'], dr['w_mid_prop_max'] - dr['st_ini_prop'], dr['w_mid_prop_max'] -
         dr['st_ini_prop_max'], dr['mtr_mid_prop'] - dr['st_ini_prop_max'], dr['w_mid_prop_max'] - dr['mtr_mid_prop']],
        [dr['mtr_mid_fix'] - dr['st_ini_fix'], dr['w_mid_fix_max'] - dr['st_ini_fix'], dr['w_mid_fix_max'] -
         dr['st_ini_fix_max'], dr['mtr_mid_fix'] - dr['st_ini_fix_max'], dr['w_mid_fix_max'] - dr['mtr_mid_fix']]
        ]
aux_titles = [r"$mid_{st}-ini_{st}$ (Prop.)", r"$mid_{st}-ini_{st}$ (Cons.)", r"Transient (Prop.)", r"Transient (Cons)"]
for graph in range(1, 5):
    i_a = graph - 1
    aux, aux_min, aux_max, aux_1, aux_90 = aux_[i_a][0], aux_[i_a][1], aux_[i_a][2], aux_[i_a][3], aux_[i_a][4]
    ax = fig2.add_subplot(2, 2, graph)
    ax.plot(initial_frequencies, np.mean(aux, axis=0) * factor, alpha=0.8, c="black")
    ax.fill_between(initial_frequencies, np.quantile(aux, 0.1, axis=0) * factor,
                    np.quantile(aux, 0.9, axis=0) * factor, color="gray", alpha=0.3)
    ax.plot(initial_frequencies, np.mean(aux_min, axis=0) * factor, alpha=0.8, c="tab:red")
    ax.fill_between(initial_frequencies, np.quantile(aux_min, 0.1, axis=0) * factor,
                    np.quantile(aux_min, 0.9, axis=0) * factor, color="tab:red", alpha=0.3)
    ax.plot(initial_frequencies, np.mean(aux_max, axis=0) * factor, alpha=0.8, c="tab:orange")
    ax.fill_between(initial_frequencies, np.quantile(aux_max, 0.1, axis=0) * factor,
                    np.quantile(aux_max, 0.9, axis=0) * factor, color="tab:orange", alpha=0.3)
    ax.plot(initial_frequencies, np.mean(aux_1, axis=0) * factor, alpha=0.8, c="tab:blue")
    ax.fill_between(initial_frequencies, np.quantile(aux_1, 0.1, axis=0) * factor,
                    np.quantile(aux_1, 0.9, axis=0) * factor, color="tab:blue", alpha=0.3)
    ax.plot(initial_frequencies, np.mean(aux_90, axis=0) * factor, alpha=0.8, c="tab:green")
    ax.fill_between(initial_frequencies, np.quantile(aux_90, 0.1, axis=0) * factor,
                    np.quantile(aux_90, 0.9, axis=0) * factor, color="tab:green", alpha=0.3)
    ax.set_xscale('log')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mem. pot. diff. (mV)")
    ax.set_title(aux_titles[i_a], c="gray")
    ax.grid()

fig2.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
plt.savefig(folder_plots + 'test.png', format='png')
# """

# SIMPLE PLOT OF DIFFERENCES OF STEADY-STATE BETWEEN MID AND INI WINDOWS FOR PROPORTIONAL AND CONSTANT CHANGE OF RATES
# AND THE DIFFERENCES BETWEEN MAX OF MID WINDOW AND MEDIAN OF INI WINDOW
"""
dims = total_realizations
factor = 1  # initial_frequencies  # 1
fig2 = plt.figure(figsize=(4.5, 4.5))
plt.suptitle(description)

ax3 = fig2.add_subplot(2, 2, 1)
a = dr['st_mid_prop']
b = dr['st_ini_prop']
aux = a - b
ax3.plot(initial_frequencies, np.mean(aux, axis=0) * factor, alpha=0.8, c="black")
ax3.fill_between(initial_frequencies, np.quantile(aux, 0.1, axis=0) * factor, np.quantile(aux, 0.9, axis=0) * factor,
                 color="lightgray")
ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("Mem. pot. diff. (mV)")
ax3.set_title("Steady-state ini (Proportional)", c="gray")
ax3.set_xscale('log')
ax3.grid()

ax4 = fig2.add_subplot(2, 2, 2)
a = dr['st_mid_fix']
b = dr['st_ini_fix']
aux = a - b
ax4.plot(initial_frequencies, np.mean(aux, axis=0) * factor, alpha=0.8, c="black")
ax4.fill_between(initial_frequencies, np.quantile(aux, 0.1, axis=0) * factor, np.quantile(aux, 0.9, axis=0) * factor,
                 color="lightgray")
ax4.set_xlabel("Frequency (Hz)")
ax4.set_ylabel("Mem. pot. diff. (mV)")
ax4.set_title("Steady-state (Constant)", c="gray")
ax4.set_xscale('log')
ax4.grid()

ax5 = fig2.add_subplot(2, 2, 3)
a = dr['mtr_mid_prop']
b = dr['st_ini_prop']
aux = a - b
ax5.plot(initial_frequencies, np.mean(aux, axis=0) * factor, alpha=0.8, c="black")
ax5.fill_between(initial_frequencies, np.quantile(aux, 0.1, axis=0) * factor, np.quantile(aux, 0.9, axis=0) * factor,
                 color="lightgray")
ax5.set_xlabel("Frequency (Hz)")
ax5.set_title("Max Transient (Proportional)", c="gray")
ax5.set_xscale('log')
ax5.grid()

ax6 = fig2.add_subplot(2, 2, 4)
a = dr['mtr_mid_fix']
b = dr['st_ini_fix']
aux = a - b
ax6.plot(initial_frequencies, np.mean(aux, axis=0) * factor, alpha=0.8, c="black")
ax6.fill_between(initial_frequencies, np.quantile(aux, 0.1, axis=0) * factor, np.quantile(aux, 0.9, axis=0) * factor,
                 color="lightgray")
ax6.set_xlabel("Frequency (Hz)")
ax6.set_title("Max Transient (Constant)", c="gray")
ax6.set_xscale('log')
ax6.grid()

fig2.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
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
