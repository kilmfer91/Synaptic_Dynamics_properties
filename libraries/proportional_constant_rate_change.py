from gain_control.utils_gc import *


class GC_prop_cons:
    def __init__(self, dict_params):

        self.validate_args(dict_params)
        self.model = dict_params['stp_model']
        self.stp_name_params = dict_params['stp_name_params']
        self.stp_value_params = dict_params['stp_value_params']
        self.stp_params = dict(zip(self.stp_name_params, self.stp_value_params))
        self.num_syn = dict_params['num_syn']
        self.neuron_model = dict_params['neuron_model']
        self.neuron_params = dict_params['neuron_params']
        self.sim_params = dict_params['sim_params']
        self.gain_vector = dict_params['gain_vector']

        self.folder_vars = dict_params['folder_vars']
        self.folder_plots = dict_params['folder_plots']
        self.description = dict_params['description']

        self.save_vars = dict_params['save_vars']
        self.force_experiment = dict_params['force_experiment']
        self.save_figs = dict_params['save_figs']
        self.imputations = dict_params['imputations']
        self.stoch_input = dict_params['stoch_input']
        self.lif_output = dict_params['lif_output']
        self.dynamic_synapse = dict_params['dynamic_synapse']

        self.file_loaded = None
        self.file_name = None

        # models variables
        self.stp_model = None
        self.stp_prop = None
        self.stp_fix = None
        self.num_instance_model = None
        self.lif_prop = None
        self.lif_fix = None

        # Experiment variables
        self.sfreq = None
        self.tau_lif = dict_params['neuron_params']['tau_m'][0]
        self.total_realizations = dict_params['total_realizations'] if not dict_params['force_experiment'] else 1
        self.num_realizations = dict_params['num_realizations'] if not dict_params['force_experiment'] else 1
        self.f_vector = None
        self.tr_time_series = None
        self.dict_results = None

    """
    def __init__(self, num_syn, ind, sfreq, stoch_input=True, force_experiment=False, lif_output=True, save_vars=False,
                 save_figs=False):
        self.num_syn = num_syn
        self.ind = ind
        self.save_vars = save_vars
        self.force_experiment = force_experiment
        self.save_figs = save_figs
        self.stoch_input = stoch_input
        self.lif_output = lif_output

        self.sim_params = sim_params
        self.file_loaded = None
        self.file_name = None

        # models variables
        self.stp_model = stp_model
        self.model = None
        self.stp_prop = None
        self.stp_fix = None

        # Experiment variables
        # Sampling frequency and conditions for running parallel or single LIF neurons
        self.sfreq = sfreq
        self.tau_lif = 10  # ms
        self.total_realizations = None
        self.num_realizations = None
        self.f_vector = None
        gain_v = [0.1, 0.2]

        self.set_experiment_vars(sfreq)

    # """
    def validate_args(self, dict_params):
        pass

    def get_folder_file_name(self, model, gain, ind, sfreq=None, num_syn=None, tau_lif=None, folder_vars=None,
                             folder_plots=None):
        if ind is None: ind = 0
        if sfreq is None: sfreq = self.sim_params['sfreq']
        if num_syn is None: num_syn = self.num_syn
        if tau_lif is None: tau_lif = self.tau_lif
        if folder_vars is None: folder_vars = self.folder_vars
        if folder_plots is None: folder_plots = self.folder_plots

        check_create_folder(folder_vars)
        check_create_folder(folder_plots)

        aux_name = "_ind_" + str(ind) + "_gain_" + str(int(gain * 100)) + "_sf_" + str(
            int(sfreq / 1000)) + "k_syn_" + str(num_syn)
        if self.lif_output: aux_name += "_tauLiF_" + str(int(tau_lif * 1e3)) + "ms"
        self.file_name = model + aux_name
        if not self.stoch_input: self.file_name = model + '_det' + aux_name
        if self.imputations: self.file_name += "_cwi"
        else: self.file_name += "_cni"
        print("For file %s and index %d" % (self.file_name, ind))
        return self.file_name

    def set_experiment_vars(self, gain_vector, sfreq=None, total_realizations=None, num_realizations=None, f_vec=None):
        # Setting variables for the experiment
        assert isinstance(gain_vector, list), "gain_vector must be a list"

        sfreq = self.sim_params['sfreq'] if sfreq is None else sfreq
        total_realizations = self.total_realizations if total_realizations is None else total_realizations
        if num_realizations is None: num_realizations = self.num_realizations

        # Setting sampling frequency
        self.sfreq = sfreq

        # Input modulations
        # Max prop freq. must be less than sfreq/4, therefore the max_freq is sfreq/6 minus a small value
        max_freq = int(sfreq / 6) - 10
        # so max. ini freq sfreq/12 | 16kHz:2501, 5kHz:801, 6KHz: 951
        range_f = [i for i in range(10, 100, 5)]
        range_f2 = [i for i in range(100, 500, 10)] if 500 < max_freq else [i for i in range(100, max_freq, 10)]
        range_f3 = [i for i in range(500, max_freq, 50)] if 500 < max_freq else []

        self.f_vector = np.array(range_f + range_f2 + range_f3) if f_vec is None else f_vec

        # Setting gain vector
        self.gain_vector = gain_vector

        return self.f_vector

    def load_set_simulation_params(self, folder_vars=None, file_name=None, total_realizations=None,
                                   num_realizations=None):
        if folder_vars is None: folder_vars = self.folder_vars
        if file_name is None: file_name = self.file_name
        if total_realizations is None: total_realizations = self.total_realizations
        if num_realizations is None: num_realizations = self.num_realizations

        self.file_loaded = False
        if os.path.isfile(folder_vars + file_name) and not self.force_experiment:
            # Loading saved dictionary
            self.file_loaded = True
            dr = loadObject(file_name, folder_vars)
            # number of instance of stp model
            self.num_instance_model = int(dr['realizations'] * dr['num_synapses'])
            self.description = dr['description']
            if 'num_experiments' not in dr: dr['num_experiments'] = dr['initial_frequencies'].shape[0]
            if 'sfreq' not in dr: dr['sfreq'] = self.sim_params['sfreq']
            if 'num_instance_model' not in dr: dr['num_instance_model'] = self.num_instance_model
            if 'time_transition' not in dr: dr['time_transition'] = [[] for _ in range(dr['num_experiments'])]
            if 'stat_tSeries_transition' not in dr: dr['stat_tSeries_transition'] = [[np.zeros((1, 1))]]
            if 'stat_time_transition' not in dr: dr['stat_time_transition'] = [np.zeros(1)]
        else:
            # number of instance of stp model
            self.num_instance_model = int(num_realizations * self.num_syn)
            # Creating dictionary to be saved
            dr = {'initial_frequencies': self.f_vector, 'stp_model': self.model, 'num_synapses': self.num_syn,
                  't_realizations': total_realizations, 'realizations': num_realizations, 'sfreq': self.sfreq,
                  'tau_lif': self.tau_lif, 'gain_v': self.gain_vector, 'Stoch_input': self.stoch_input,
                  'stp_name_params': self.stp_name_params, 'stp_value_params': self.stp_value_params,
                  'sim_params': self.sim_params, 'n_params': self.neuron_params, 'dyn_synapse': self.dynamic_synapse,
                  'num_instance_model': self.num_instance_model}

            # Model parameters
            # syn_params, description, name_params = get_params_stp(dr['stp_model'], dr['ind'])

            self.description += ", " + str(dr['num_synapses']) + " synapses"
            dr['description'] = self.description

            # Time conditions
            dr['num_changes_rate'] = 3
            Le_time_win = int(self.sim_params['max_t'] / dr['num_changes_rate'])
            dr['fix_rate_change_a'] = [5 + (5 * i) for i in range(len(self.gain_vector))]  # [5, 10, 15]

            dr['num_experiments'] = self.f_vector.shape[0]

            # array for time of transition-states
            dr['time_transition'] = [[] for _ in range(dr['num_experiments'])]

            # For poisson or deterministic inputs
            dr['seeds'] = []
            if not self.stoch_input:
                total_realizations = 1
                num_realizations = 1
                dr['t_realizations'] = total_realizations
                dr['realizations'] = num_realizations

        self.dict_results = dr
        return self.file_loaded, self.dict_results

    def models_creation(self, model=None, sim_params=None, params=None, num_realizations=None, lif_params=None,
                        num_instance_model=None):
        if model is None: model = self.model
        if sim_params is None: sim_params = self.sim_params
        if params is None: params = self.stp_params
        if num_realizations is None: num_realizations = self.num_realizations
        if lif_params is None: lif_params = self.neuron_params
        if num_instance_model is None: num_instance_model = self.num_instance_model

        # Creating STP models for proportional rate change
        if self.model == "MSSM": self.stp_prop = MSSM_model(n_syn=num_instance_model)
        if self.model == "MSSM": self.stp_fix = MSSM_model(n_syn=num_instance_model)
        if self.model == "TM": self.stp_prop = TM_model(n_syn=num_instance_model)
        if self.model == "TM": self.stp_fix = TM_model(n_syn=num_instance_model)
        assert self.stp_prop is not None, "Cannot set stp_model"

        # Setting initial conditions
        self.stp_prop.set_model_params(params)
        self.stp_prop.set_simulation_params(sim_params)
        self.stp_fix.set_model_params(params)
        self.stp_fix.set_simulation_params(sim_params)

        # Creating LIF models for proportional rate change
        self.lif_prop = LIF_model(n_neu=num_realizations)
        self.lif_prop.set_model_params(lif_params)
        self.lif_fix = LIF_model(n_neu=num_realizations)
        self.lif_fix.set_model_params(lif_params)

    def validate_dict_params(self, dr):
        pass

    def run(self, gain, fixed_rate_change=None, dr=None, soft_stop_cond=True, plot_ind_figs=False, th_percentage=1e-2,
            y_lims_ind_plot=None):
        if dr is None: dr = self.dict_results
        self.validate_dict_params(dr)

        [total_realizations, stoch_input, seeds, num_realizations, num_experiments, sfreq, f_vector, num_changes_rate,
         aux_num_r, dyn_synapse, sim_params, t_tra, lif_output, file_name, gain_v, fix_rate_change_a, folder_vars,
         stp_params] = [dr['t_realizations'], self.stoch_input, dr['seeds'], dr['realizations'], dr['num_experiments'],
                        dr['sfreq'], dr['initial_frequencies'], dr['num_changes_rate'], dr['num_instance_model'],
                        dr['dyn_synapse'], dr['sim_params'], dr['time_transition'], self.lif_output, self.file_name,
                        self.gain_vector, dr['fix_rate_change_a'], self.folder_vars, self.stp_params]

        title_graph = self.description.split(",")[0] + ", gain " + str(gain)
        stp_params = dict(zip(self.stp_name_params, self.stp_value_params))

        # Sim params
        L = sim_params['L']
        time_vector = sim_params['time_vector']
        max_t = sim_params['max_t']

        Le_time_win = int(max_t / num_changes_rate)

        # Getting num of realizations
        num_loop_realizations = int(total_realizations / num_realizations)

        # Auxiliar variables for statistics
        res_per_reali = np.zeros((144, num_experiments, num_realizations))
        res_real = np.zeros((144, total_realizations, num_experiments))

        # Setting proportional and fixed rates of change
        proportional_changes = gain * f_vector + f_vector
        constant_changes = fixed_rate_change + f_vector

        # time-series for transition times: piw, pmw, pew, ciw, cmw, cew
        tr_time_series = [[[] for _ in range(num_experiments)] for _ in range(6)]

        ini_loop_time = m_time()
        print("Ini big loop")
        realization = 0
        while realization < num_loop_realizations and soft_stop_cond:
            loop_time = m_time()
            t_tra_mid_win = None

            # Building reference signal for constant and fixed rate changes
            i = num_experiments - 1
            while i >= 0:  # while i < num_experiments:
                loop_experiments = m_time()

                seeds1, seeds2, seeds3 = [0], [0], [0]
                # For poisson or deterministic inputs
                if self.stoch_input:
                    se = int(time.time())
                    seeds.append(se)
                    seeds1 = [j + se for j in range(num_realizations)]
                    seeds2 = [j + se + 2 for j in range(num_realizations)]
                    seeds3 = [j + se + 3 for j in range(num_realizations)]

                ref_signals = simple_spike_train(sfreq, f_vector[i], int(L / num_changes_rate),
                                                 num_realizations=aux_num_r, poisson=self.stoch_input, seeds=seeds1)
                # ISIs, histograms = inter_spike_intervals(ref_signals, dt, 1e-3)
                # plot_isi_histogram(histograms, 0)
                cons_aux = simple_spike_train(sfreq, proportional_changes[i], int(L / num_changes_rate),
                                              num_realizations=aux_num_r, poisson=stoch_input, seeds=seeds2)
                fix_aux = simple_spike_train(sfreq, constant_changes[i], int(L / num_changes_rate),
                                             num_realizations=aux_num_r, poisson=stoch_input, seeds=seeds3)

                cons_input = np.concatenate((ref_signals, cons_aux, ref_signals), axis=1)
                fix_input = np.concatenate((ref_signals, fix_aux, ref_signals), axis=1)

                # Avoiding spikes in t==0
                cons_input[:, 0], fix_input[:, 0] = 0, 0
                if not stoch_input: cons_input[:, 1], fix_input[:, 1] = 1, 1

                # Running STP model
                if dyn_synapse:
                    # Reseting initial conditions
                    self.stp_prop.set_initial_conditions()
                    self.lif_prop.set_simulation_params(sim_params)
                    self.stp_fix.set_initial_conditions()
                    self.lif_fix.set_simulation_params(sim_params)
                    # Running the models
                    model_stp_parallel(self.stp_prop, self.lif_prop, stp_params, cons_input)
                    # model_stp_parallel(self.stp_fix, self.lif_fix, stp_params, fix_input)
                else:
                    # Reseting initial conditions
                    self.lif_prop.set_simulation_params(sim_params)
                    # lif_fix.set_simulation_params(sim_params)
                    # Running the models
                    static_synapse(self.lif_prop, cons_input, 9e0)  # , 0.0125e-6)
                    # static_synapse(lif_fix, fix_input, 9e0)  # , 0.0125e-6)

                # Defining output of the model in order to compute statistics
                signal_prop, signal_fix = self.stp_prop.get_output(), self.stp_fix.get_output()
                if lif_output:
                    signal_prop, signal_fix = self.lif_prop.membrane_potential, self.lif_fix.membrane_potential

                # getting transition time for rate of proportional change if possible
                aux_cond = np.where(proportional_changes[i] <= f_vector)
                if len(aux_cond[0]) > 0:
                    aux_i = aux_cond[0][0]
                    t_tra_mid_win = np.max(t_tra[aux_i])

                # Computing statistics of each window, for the whole window and for the transition- and steady-states
                a, b, c = aux_statistics_prop_cons(signal_prop, signal_fix, Le_time_win, None,
                                                   sim_params, t_tra_mid_win)
                res_per_reali[:, i, :], t_tr_, tr_time_series_i = a, b, c

                # Updating array of time_transitions
                t_tra[i].append(t_tr_)
                for tr_i in range(6):
                    if len(tr_time_series[tr_i][i]) == 0:
                        tr_time_series[tr_i][i] = tr_time_series_i[tr_i]
                    else:
                        tr_time_series[tr_i][i] = tr_time_series[tr_i][i] + tr_time_series_i[tr_i]

                # Plotting individual figures if indicated
                if plot_ind_figs:
                    path_save = self.folder_plots + file_name + '_' + str(f_vector[i]) + '.png'
                    title_graph_ = title_graph + ", freq. %dHz" % f_vector[i]
                    t_tr = t_tr_[0]
                    plot_gc_mem_potential_prop_fix(time_vector, i, signal_prop, signal_fix, t_tr, res_per_reali,
                                                   title_graph_, path_save=path_save, save_figs=self.save_figs,
                                                   y_lims_ind_plot=y_lims_ind_plot, plot_stats=False, plt_grid=True)
                    path_save = self.folder_plots + file_name + '_' + str(f_vector[i]) + '_stat.png'
                    plot_gc_mem_potential_prop_fix(time_vector, i, signal_prop, signal_fix, t_tr, res_per_reali,
                                                   title_graph_, path_save=path_save, save_figs=self.save_figs,
                                                   y_lims_ind_plot=y_lims_ind_plot, plot_stats=True, plt_grid=False)

                # Final print of the loop
                print_time(m_time() - loop_experiments, file_name + ", Realisation " + str(realization) +
                           ", frequency " + str(f_vector[i]))
                i -= 1

            # steady-state part
            for res_i in range(res_real.shape[0]):
                r = realization
                res_real[res_i, r * num_realizations:(r + 1) * num_realizations] = res_per_reali[res_i, :].T

            print_time(m_time() - loop_time, file_name + ", Realisation " + str(realization))

            realization += 1

        #
        if soft_stop_cond:
            # Obtaining time series of statistical descriptors for transition component of signals (ini, mid, end win)
            self.tr_time_series = tr_time_series
            st_tr_a, res = get_time_series_statistics_of_transitions(tr_time_series, f_vector, proportional_changes,
                                                                     th_percentage)
            dr['stat_tSeries_transition'] = res
            dr['stat_time_transition'] = st_tr_a

            # transition-state
            for i in range(num_experiments):
                t_tra[i] = np.ravel(t_tra[i])
            t_tra = np.array(t_tra).T

            # Saving transition times of each window
            dr['time_transition'] = t_tra

        # Saving final dictionary if file does not exist
        if not os.path.isfile(folder_vars + file_name):
            """
            dr = {'initial_frequencies': f_vector,
                  'stp_model': self.model, 'name_params': self.stp_name_params, 'dyn_synapse': dyn_synapse,
                  'num_synapses': self.num_syn, 'syn_params': syn_params, 'sim_params': sim_params,
                  'lif_params': lif_params, 'lif_params2': lif_params2, 'gain_v': gain_v,
                  'fix_rate_change_a': fix_rate_change_a, 'num_changes_rate': num_changes_rate,
                  'description': description, 'seeds': seeds,
                  'realizations': num_realizations, 't_realizations': self.total_realizations, 'time_transition': t_tra}
            # """
            # Updating final dictionary
            for nam in range(res_real.shape[0]):
                dr[stat_list[nam]] = res_real[nam, :]

            if self.save_vars:
                saveObject(dr, file_name, folder_vars)

        # if experiment is forced, then update final dictionary to be returned
        if soft_stop_cond:
            # Updating final dictionary
            for nam in range(res_real.shape[0]):
                dr[stat_list[nam]] = res_real[nam, :]

        print_time(m_time() - ini_loop_time, "Total big loop")

        return dr
