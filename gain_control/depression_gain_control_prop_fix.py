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
ind = 7
save_vars = False
run_experiment = False
save_figs = False
# imputations = True
Stoch_input = True
lif_output = True
num_syn = 1
# gain = 0.5

# Sampling frequency and conditions for running parallel or single LIF neurons
sfreq = 6e3
tau_lif = 10  # ms
total_realizations = 100  # 100
num_realizations = 8  # 8 for server, 4 for macbook air
gain_v = [0.1, 0.2]

# Input modulations
range_f = [10]  # [i for i in range(10, 100, 5)]
range_f2 = [100]  # [i for i in range(100, 500, 10)]  # # sfreq>3kHz:501, 2kHz:321
range_f3 = [900]  # [i for i in range(500, 951, 50)]  # Max prop freq. must be less than sfreq/4,
# so max. ini freq sfreq/12 | 16kHz:2501, 5kHz:801, 6KHz: 951
initial_frequencies = np.array(range_f + range_f2 + range_f3)

# Path variables
path_vars = "../gain_control/variables/"
check_create_folder(path_vars)
folder_plots = '../gain_control/plots/'
check_create_folder(folder_plots)
aux_name = "_ind_" + str(ind) + "_gain_" + str(int(gain * 100)) + "_sf_" + str(int(sfreq / 1000)) + "k_syn_" + str(
    num_syn)
if lif_output: aux_name += "_tauLiF_" + str(tau_lif) + "ms"
file_name = (model + aux_name)
if not Stoch_input: file_name = (model + '_det' + aux_name)
# if imputations: file_name += "_cwi" else: file_name += "_cni"
print("For file %s and index %d" % (file_name, ind))

# ******************************************************************************************************************
# SETTING/LOADING GLOBAL VARIABLES
# defining initial dictionary with global variables
dr_ini = {'initial_frequencies': initial_frequencies, 'stp_model': model, 'num_synapses': num_syn,
          't_realizations': total_realizations, 'realizations': num_realizations, 'ind': ind, 'sfreq': sfreq,
          'tau_lif': tau_lif, 'gain_v': gain_v, 'stoch_input': Stoch_input}

# Loading or setting dictionary with global variables
file_loaded, dr = load_set_simulation_params(dr_ini, path_vars, file_name, run_experiment)

# Setting rest of global variables based on the loaded dictionary
initial_frequencies, model, name_params = dr['initial_frequencies'], dr['stp_model'], dr['name_params']
dyn_synapse, num_syn, syn_params, sim_params = dr['dyn_synapse'], dr['num_synapses'], dr['syn_params'], dr['sim_params']
num_realizations, lif_params, prop_rate_change_a = dr['realizations'], dr['lif_params'], dr['prop_rate_change_a']
fix_rate_change_a, num_changes_rate, t_tra = dr['fix_rate_change_a'], dr['num_changes_rate'], dr['time_transition']
description, seeds, total_realizations = dr['description'], dr['seeds'], dr['t_realizations']

# Time conditions
max_t, sfreq, time_vector, L = sim_params['max_t'], sim_params['sfreq'], sim_params['time_vector'], sim_params['L']
dt = 1 / sfreq
Le_time_win = int(max_t / num_changes_rate)

# Parameters in dict format
params = dict(zip(name_params, syn_params))

# Number of experiments
num_experiments = initial_frequencies.shape[0]

# Setting seeds
seeds1, seeds2, seeds3 = None, None, None
if len(seeds) > 0:
    seeds1 = [j + seeds[0] for j in range(num_realizations)]
    seeds2 = [j + seeds[0] + 2 for j in range(num_realizations)]
    seeds3 = [j + seeds[0] + 3 for j in range(num_realizations)]

# Creating stp models
aux_num_r = int(num_realizations * num_syn)
stp_prop, stp_fix, lif_prop, lif_fix = models_creation(model, aux_num_r, sim_params, params,
                                                       num_realizations, lif_params)

plus_cond = (not file_loaded or run_experiment)
title_graph = "Model %s, index %d, for %d synapses" % (model, ind, num_syn)

# arguments = [total_realizations, plus_cond, Stoch_input, seeds, num_realizations, num_experiments, sfreq,
#              initial_frequencies, num_changes_rate, aux_num_r, imputations, dyn_synapse, stp_prop, stp_fix, lif_prop,
#              lif_fix, sim_params, params, t_tra, Le_time_win, lif_output, file_name, prop_rate_change_a,
#              fix_rate_change_a, gain, fix_change, path_vars, title_graph]
arguments = [total_realizations, plus_cond, Stoch_input, seeds, num_realizations, num_experiments, sfreq,
             initial_frequencies, num_changes_rate, aux_num_r, dyn_synapse, stp_prop, stp_fix, lif_prop,
             lif_fix, sim_params, params, t_tra, Le_time_win, lif_output, file_name, prop_rate_change_a,
             fix_rate_change_a, None, None, path_vars, title_graph]

# **********************************************************************************************************************
# RUNNING CYCLE TO COMPUTE PROPORTIONAL/CONSTANT RATE CHANGE EXPERIMENT AT A GIVEN GAIN
for k in range(len(gain_v)):
    arguments[23] = prop_rate_change_a[k]  # gain
    arguments[24] = fix_rate_change_a[k]  # fixed change
    gc_prop_fix_gain(arguments)

# **********************************************************************************************************************
# PLOTS
# PLOT OF CHARACTERISTICS FOR INI, MID, AND END WINDOWS. SPLITTED BY PROPORTIONAL AND CONSTANT INPUT RATE CHANGES
# """
# Steady-state
lbl = ['st_ini_prop', 'st_mid_prop', 'st_end_prop', 'st_ini_fix', 'st_mid_fix', 'st_end_fix']
st_lbl = ['_mean', '_med', '_max', '_min', '_q10', '_q90', '_q5', '_q95']
cols = ['tab:orange', 'tab:blue', 'tab:red', 'tab:red', 'tab:green', 'tab:green', 'tab:olive', 'tab:olive']
title = description + " " + str(tau_lif) + "ms for steady-state"
path_save = folder_plots + file_name + '_windows_statistics.png'
plot_features_windows_prop_fix(initial_frequencies, dr, lbl, st_lbl, cols, title, path_save, save_figs)

# PLOT CHARACTERISTICS OF MID AND INI WINDOWS IN THE SAME PLOT, FOR PROPORTIONAL AND CONSTANT INPUT RATE CHANGES
lbl = ['st_ini_prop', 'mtr_ini_prop', 'st_ini_fix', 'mtr_ini_fix']
lbl2 = ['st_mid_prop', 'mtr_mid_prop', 'st_mid_fix', 'mtr_mid_fix']
st_lbl = ['_mean', '_med', '_max', '_min', '_q10', '_q90']  # , '_q5', '_q95']
t_ = ['Steady-state, ini/mid windows (prop)', 'Transition-state, ini/mid windows (prop)',
      'Steady-state, ini/mid windows (cons)', 'Transition-state, ini/mid windows (cons)']
cols = ['tab:orange', 'tab:blue', 'tab:red', 'tab:red', 'tab:green', 'tab:green']  # , 'tab:olive', 'tab:olive']
name_save = folder_plots + file_name + '_windows_statistics3.png'
title = description + " " + str(tau_lif) + "ms"
plot_features_2windows_prop_fix(initial_frequencies, dr, lbl, lbl2, st_lbl, cols, t_, title, path_save, save_figs)
# """

# SIMPLE PLOT OF DIFFERENCES OF STEADY-STATE BETWEEN MID AND INI WINDOWS FOR PROPORTIONAL AND CONSTANT CHANGE OF RATES
# AND THE DIFFERENCES BETWEEN MAX OF MID WINDOW AND MEDIAN OF INI WINDOW
# """
lbl = ['st_mid_prop', 'st_mid_fix']
lbl2 = ['st_ini_prop', 'st_ini_fix']
st_lbl = ['_max', '_min', '_q10', '_q90', '_mean', '_med']
cols_ = ['tab:red', 'tab:red', 'tab:green', 'tab:green', 'tab:orange', 'tab:blue']
t_ = [r"$mid_{st} - ini_{st}$ (Prop)", r"$mid_{st} - ini_{st}$ (Cons)",
      "Max Transient (Proportional)", "Max Transient (Constant)"]
# ylims = [-62.5, -54.0]  # [-70.05, -52]
name_save = folder_plots + file_name + '_' + 'diff_st_log.png'
title = description + " " + str(tau_lif) + "ms"
plot_diff_windows(initial_frequencies, dr, lbl, lbl2, st_lbl, cols_, t_, title_graph=title, name_save=name_save,
                  save_figs=save_figs)

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
