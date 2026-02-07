from gain_control.utils_gc import *
from libraries.proportional_constant_rate_change import GC_prop_cons

# ******************************************************************************************************************
# Global variables
# (Experiment 2) freq. response decay around 100Hz
# (Experiment 3) freq. response decay around 10Hz
# (Experiment 4) freq. response from Gain Control paper
# (Experiment 5) freq. response decay around 100Hz
# (Experiment 6) freq. response decay around 10Hz
model = 'MSSM'
ind = 4
save_vars = True
force_experiment = False
stoch_input = True

plot_ind_memPot = False
save_figs = False

imputations = True
lif_output = True
dyn_synapse = True

tau_m_lif = 2  # ms
total_realizations = 104  # 100
num_realizations = 8
gain_v = [.5]  # [0.1, 0.2, 0.5]
folder_vars = "../gain_control/variables/"
folder_plots = '../gain_control/plots/'
# ******************************************************************************************************************
# Time conditions
sfreq = 6e3
max_t = 6
dt = 1 / sfreq
time_vector = np.arange(0, max_t, dt)
L = time_vector.shape[0]
sim_params = {'sfreq': sfreq, 'max_t': max_t, 'L': L, 'time_vector': time_vector}

# ******************************************************************************************************************
# STP model
num_syn = 1
syn_params, description, name_params = get_params_stp(model, ind)

# Neuron model
n_model = "LIF"
lif_params = get_neuron_params(tau_m=tau_m_lif, y_lim_ind_plot=True, num_syn=num_syn)

dict_params = {'stp_model': model, 'stp_name_params': name_params, 'stp_value_params': syn_params, 'num_syn': num_syn,
               'neuron_model': n_model, 'neuron_params': lif_params, 'sim_params': sim_params, 'gain_vector': gain_v,
               'folder_vars': folder_vars, 'folder_plots': folder_plots, 'save_vars': save_vars, 'save_figs': save_figs,
               'force_experiment': force_experiment, 'imputations': imputations, 'stoch_input': stoch_input,
               'lif_output': lif_output, 'dynamic_synapse': dyn_synapse, 'description': description,
               'num_realizations': num_realizations, 'total_realizations': total_realizations}

# Instance of Gain-Control class
initial_frequencies = np.array([10, 100, 200]) if force_experiment else None
gc_prop_cons = GC_prop_cons(dict_params)
_ = gc_prop_cons.set_experiment_vars(gain_v, f_vec=initial_frequencies)

# Running Gain-Control process for different gains
file_name = ""
dr = {}
for gain in gain_v:
    file_name = gc_prop_cons.get_folder_file_name(model, gain, ind)
    file_loaded, dr_aux = gc_prop_cons.load_set_simulation_params()
    gc_prop_cons.models_creation()
    dr = gc_prop_cons.run(gain=gain, fixed_rate_change=5, soft_stop_cond=(not file_loaded or force_experiment),
                          plot_ind_figs=plot_ind_memPot, y_lims_ind_plot=lif_params['y_lim_plot'])

# **********************************************************************************************************************
# FOR TRANSITION CALCULATIONS
lbl = ['st_ini_prop', 'st_mid_prop', 'st_end_prop']
st_lbl = ['_mean', '_med', '_max', '_min', '_q10', '_q90', '_q5', '_q95']
cols = ['tab:orange', 'tab:blue', 'tab:red', 'tab:red', 'tab:green', 'tab:green', 'tab:olive', 'tab:olive']

res = dr['stat_tSeries_transition']
st_tr_a = dr['stat_time_transition']

fig = plt.figure(figsize=[14, 5])
f_int = 10
ax = [fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)]
for i in range(3):
    ax[i].plot(res[i][f_int][6, :], c="tab:red", label='min')
    ax[i].plot(res[i][f_int][2, :], c="tab:olive", label='q5%')
    ax[i].plot(res[i][f_int][3, :], c="tab:green", label='q10%')
    ax[i].plot(res[i][f_int][0, :], c="tab:orange", label=r'$\mu$')
    ax[i].plot(res[i][f_int][1, :], c="tab:blue", label='median')
    ax[i].plot(res[i][f_int][4, :], c="tab:green", label='q90%')
    ax[i].plot(res[i][f_int][5, :], c="tab:olive", label='q95%')
    ax[i].plot(res[i][f_int][7, :], c="tab:red", label='max')
    ax[i].grid()
    ax[i].set_xlim(-0.5, np.max(st_tr_a[f_int]))
    ax[i].set_ylim(-70.05, -67)

    for j in range(len(st_lbl)):
        a = dr[lbl[i] + st_lbl[j]]
        ax[i].plot([-0.5, np.max(st_tr_a[f_int])], [np.mean(a[:, f_int]), np.mean(a[:, f_int])], c=cols[j],
                   label=st_lbl[j][1:], alpha=0.4)
        ax[i].fill_between([-0.5, np.max(st_tr_a[f_int])], np.quantile(a[:, 10], 0.05), np.quantile(a[:, 10], 0.95),
                           color=cols[j], alpha=0.3)

ax[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.suptitle("Transition-states statistics for rate " + str(gc_prop_cons.f_vector[f_int]) + "Hz")

# **********************************************************************************************************************
# PLOTS
# PLOT OF CHARACTERISTICS FOR INI, MID, AND END WINDOWS. SPLITTED BY PROPORTIONAL AND CONSTANT INPUT RATE CHANGES
# """
initial_frequencies = dr['initial_frequencies']
# Steady-state
lbl = ['st_ini_prop', 'st_mid_prop', 'st_end_prop', 'st_ini_fix', 'st_mid_fix', 'st_end_fix']
st_lbl = ['_mean', '_med', '_max', '_min', '_q10', '_q90', '_q5', '_q95']
cols = ['tab:orange', 'tab:blue', 'tab:red', 'tab:red', 'tab:green', 'tab:green', 'tab:olive', 'tab:olive']
title = description + " " + str(tau_m_lif) + "ms for steady-state"
path_save = folder_plots + file_name + '_windows_statistics.png'
cond_save = save_figs and not force_experiment
plot_features_windows_prop_fix(initial_frequencies, dr, lbl, st_lbl, cols, title, path_save, cond_save,
                               y_lims_ind_plot=lif_params['y_lim_plot'])

# PLOT CHARACTERISTICS OF MID AND INI WINDOWS IN THE SAME PLOT, FOR PROPORTIONAL AND CONSTANT INPUT RATE CHANGES
# lbl = ['st_ini_prop', 'mtr_ini_prop', 'st_ini_fix', 'mtr_ini_fix']
# lbl2 = ['st_mid_prop', 'mtr_mid_prop', 'st_mid_fix', 'mtr_mid_fix']
lbl = ['st_ini_prop', 'st_ini_fix']
lbl2 = ['st_mid_prop', 'st_mid_fix']
st_lbl = ['_mean', '_med', '_max', '_min', '_q1', '_q90']  # , '_q5', '_q95']
# t_ = ['Steady-state, ini/mid windows (prop)', 'Transition-state, ini/mid windows (prop)',
#       'Steady-state, ini/mid windows (cons)', 'Transition-state, ini/mid windows (cons)']
t_ = ['Steady-state, ini/mid windows (prop)', 'Steady-state, ini/mid windows (cons)']
cols = ['tab:orange', 'tab:blue', 'tab:red', 'tab:red', 'tab:green', 'tab:green']  # , 'tab:olive', 'tab:olive']
path_save = folder_plots + file_name + '_windows_statistics3.png'
title = description + " " + str(tau_m_lif) + "ms"
plot_features_2windows_prop_fix(initial_frequencies, dr, lbl, lbl2, st_lbl, cols, t_, title, path_save, cond_save,
                                y_lims_ind_plot=lif_params['y_lim_plot'])
# """

# SIMPLE PLOT OF DIFFERENCES OF STEADY-STATE BETWEEN MID AND INI WINDOWS FOR PROPORTIONAL AND CONSTANT CHANGE OF RATES
# AND THE DIFFERENCES BETWEEN MAX OF MID WINDOW AND MEDIAN OF INI WINDOW
# """
# lbl = ['st_mid_prop', 'st_mid_fix']
# lbl2 = ['st_ini_prop', 'st_ini_fix']
lbl = ['st_mid_prop']
lbl2 = ['st_ini_prop']
st_lbl = ['_max', '_min', '_q1', '_q90', '_mean', '_med']
cols_ = ['tab:red', 'tab:red', 'tab:green', 'tab:green', 'tab:orange', 'tab:blue']
t_ = [r"$mid_{st} - ini_{st}$ (Prop)", r"$mid_{st} - ini_{st}$ (Cons)",
      "Max Transient (Proportional)", "Max Transient (Constant)"]
# ylims = [-62.5, -54.0]  # [-70.05, -52]
name_save = folder_plots + file_name + '_' + 'diff_st_log.png'
title = description + " " + str(tau_m_lif) + "ms"
y_lims = [-1.5, 3.5]
if num_syn == 100 and tau_m_lif == 1: y_lims = [-3.4, 4.4]
if num_syn == 100 and tau_m_lif == 30: y_lims = [-1.5, 3.5]
if num_syn == 1 and tau_m_lif == 1: y_lims = [-.22, .16]
if num_syn == 1 and tau_m_lif == 10: y_lims = [-.09, .08]
if num_syn == 1 and tau_m_lif == 30: y_lims = [-.042, .07]
y_lims = [-.22, .16]
# 1syn/1msTau_m:[-.22,.16],1syn/10msTau_m:[-.09,.08],1syn/30msTau_m:[-.042,.07]
plot_diff_windows(initial_frequencies, dr, lbl, lbl2, st_lbl, cols_, t_, title_graph=title, name_save=name_save,
                  save_figs=cond_save, y_lims_ind_plot=y_lims)
# """
