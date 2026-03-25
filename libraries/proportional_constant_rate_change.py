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
        self.neuron_noise = dict_params['neuron_noise']

        self.file_loaded = None
        self.file_name = None

        # models variables
        self.stp_model = None
        self.stp_prop = None
        self.stp_fix = None
        self.num_instance_model = None
        self.neuron_prop = None
        self.neuron_fix = None

        # Experiment variables
        self.sfreq = None
        self.tau_neu = dict_params['neuron_params']['tau_m'][0]
        self.total_realizations = dict_params['total_realizations'] if not dict_params['force_experiment'] else 1
        self.num_realizations = dict_params['num_realizations'] if not dict_params['force_experiment'] else 1
        self.f_vector = None
        self.tr_time_series = None
        self.dict_results = None

    def validate_args(self, dict_params):
        pass

    def get_folder_file_name(self, model, n_model, gain, ind, sfreq=None, num_syn=None, tau_n=None, folder_vars=None,
                             folder_plots=None):
        if ind is None: ind = 0
        if sfreq is None: sfreq = self.sim_params['sfreq']
        if num_syn is None: num_syn = self.num_syn
        if tau_n is None: tau_n = self.tau_neu
        if folder_vars is None: folder_vars = self.folder_vars
        if folder_plots is None: folder_plots = self.folder_plots

        check_create_folder(folder_vars)
        check_create_folder(folder_plots)

        aux_name = "_ind_" + str(ind) + "_gain_" + str(int(gain * 100)) + "_sf_" + str(
            int(sfreq / 1000)) + "k_syn_" + str(num_syn)
        if self.lif_output and self.neuron_model == 'LIF': aux_name += "_tau" + n_model + "_" + str(tau_n) + "ms"
        self.file_name = model + aux_name
        if not self.stoch_input: self.file_name = model + '_det' + aux_name
        else:
            if self.neuron_noise: self.file_name += '_noise'
        print("For file %s and index %d" % (self.file_name, ind))
        return self.file_name

    def set_experiment_vars(self, gain_vector, sfreq=None, total_realizations=None, num_realizations=None, f_vec=None,
                            max_freq=None):
        # Setting variables for the experiment
        assert isinstance(gain_vector, list), "gain_vector must be a list"

        sfreq = self.sim_params['sfreq'] if sfreq is None else sfreq
        total_realizations = self.total_realizations if total_realizations is None else total_realizations
        if num_realizations is None: num_realizations = self.num_realizations

        # Setting sampling frequency
        self.sfreq = sfreq

        # Input modulations
        # Max prop freq. must be less than sfreq/4, therefore the max_freq is sfreq/6 minus a small value
        aux_max_freq = int(sfreq / 6) - 10
        if max_freq is None:
            max_freq = aux_max_freq
        else:
            if max_freq > aux_max_freq:
                max_freq = aux_max_freq
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
            if 'num_freq_exp' not in dr: dr['num_freq_exp'] = dr['initial_frequencies'].shape[0]
            if 'sfreq' not in dr: dr['sfreq'] = self.sim_params['sfreq']
            if 'num_instance_model' not in dr: dr['num_instance_model'] = self.num_instance_model
            if 'time_transition' not in dr: dr['time_transition'] = [[] for _ in range(dr['num_freq_exp'])]
            if 'time_transition_syn' not in dr: dr['time_transition_syn'] = [[] for _ in range(dr['num_freq_exp'])]
            if 'time_transition_syn_b' not in dr: dr['time_transition_syn_b'] = [[] for _ in range(dr['num_freq_exp'])]
            # if 'stat_tSeries_transition' not in dr: dr['stat_tSeries_transition'] = [[np.zeros((1, 1))]]
            # if 'stat_time_transition' not in dr: dr['stat_time_transition'] = [np.zeros(1)]
            if 'neuron_noise' not in dr: dr['neuron_noise'] = False
            if 'PSR_events' not in dr: dr['PSR_events'] = [[] for _ in range(dr['num_freq_exp'])]
            if 'spike_events' not in dr: dr['spike_events'] = [[] for _ in range(dr['num_freq_exp'])]
            if 'PSR_events_wind' not in dr: dr['PSR_events_wind'] = [[], [], []]
            if 'spike_events_wind' not in dr: dr['spike_events'] = [[], [], []]

        else:
            # Creating dictionary to be saved
            dr = {'initial_frequencies': self.f_vector, 'stp_model': self.model, 'num_synapses': self.num_syn,
                  't_realizations': total_realizations, 'realizations': num_realizations, 'sfreq': self.sfreq,
                  'tau_lif': self.tau_neu, 'gain_v': self.gain_vector, 'Stoch_input': self.stoch_input,
                  'stp_name_params': self.stp_name_params, 'stp_value_params': self.stp_value_params,
                  'sim_params': self.sim_params, 'n_params': self.neuron_params, 'dyn_synapse': self.dynamic_synapse}

            # Model parameters
            # syn_params, description, name_params = get_params_stp(dr['stp_model'], dr['ind'])

            self.description += ", " + str(dr['num_synapses']) + " synapses"
            dr['description'] = self.description

            # Time conditions
            dr['num_changes_rate'] = 3
            Le_time_win = int(self.sim_params['max_t'] / dr['num_changes_rate'])
            dr['fix_rate_change_a'] = [5 + (5 * i) for i in range(len(self.gain_vector))]  # [5, 10, 15]

            dr['num_freq_exp'] = self.f_vector.shape[0]

            # array for time of transition-states
            dr['time_transition'] = [[] for _ in range(dr['num_freq_exp'])]
            dr['time_transition_syn'] = [[] for _ in range(dr['num_freq_exp'])]
            dr['time_transition_syn_b'] = [[] for _ in range(dr['num_freq_exp'])]

            # For poisson or deterministic inputs
            dr['seeds'] = []
            if not self.stoch_input:
                self.total_realizations = 1
                self.num_realizations = 1
                dr['t_realizations'] = self.total_realizations
                dr['realizations'] = self.num_realizations

            # number of instance of stp model
            self.num_instance_model = int(self.num_realizations * self.num_syn)
            dr['num_instance_model'] = self.num_instance_model

        self.dict_results = dr
        return self.file_loaded, self.dict_results

    def models_creation(self, model=None, sim_params=None, params=None, num_realizations=None, neuron_params=None,
                        num_instance_model=None):
        if model is None: model = self.model
        if sim_params is None: sim_params = self.sim_params
        if params is None: params = self.stp_params
        if num_realizations is None: num_realizations = self.num_realizations
        if neuron_params is None: neuron_params = self.neuron_params
        if num_instance_model is None: num_instance_model = self.num_instance_model

        # Creating STP models for proportional rate change
        if self.model == "MSSM": self.stp_prop = MSSM_model(n_syn=num_instance_model)
        if self.model == "MSSM": self.stp_fix = MSSM_model(n_syn=num_instance_model)
        if self.model == "TM": self.stp_prop = TM_model(n_syn=num_instance_model)
        if self.model == "TM": self.stp_fix = TM_model(n_syn=num_instance_model)
        if self.model == "DoornSTD": self.stp_prop = DoornSTD_model(n_syn=num_instance_model)
        if self.model == "DoornSTD": self.stp_fix = DoornSTD_model(n_syn=num_instance_model)
        if self.model == "DoornSTF": self.stp_prop = DoornSTF_model(n_syn=num_instance_model)
        if self.model == "DoornSTF": self.stp_fix = DoornSTF_model(n_syn=num_instance_model)
        if self.model == "DoornAsyn": self.stp_prop = DoornAsyn_model(n_syn=num_instance_model)
        if self.model == "DoornAsyn": self.stp_fix = DoornAsyn_model(n_syn=num_instance_model)

        assert self.stp_prop is not None, "Cannot set stp_model"

        # Setting initial conditions
        self.stp_prop.set_model_params(params)
        self.stp_prop.set_simulation_params(sim_params)
        self.stp_fix.set_model_params(params)
        self.stp_fix.set_simulation_params(sim_params)

        # Creating Neuron models for proportional rate change
        if self.neuron_model == "LIF":
            self.neuron_prop = LIF_model(n_neu=num_realizations)
            self.neuron_fix = LIF_model(n_neu=num_realizations)
        elif self.neuron_model == "HH":
            self.neuron_prop = HH_AHP_model(n_neu=num_realizations)
            self.neuron_fix = HH_AHP_model(n_neu=num_realizations)
        assert self.neuron_prop is not None, "Cannot set neuron model"

        self.neuron_prop.set_model_params(neuron_params)
        self.neuron_fix.set_model_params(neuron_params)

    def validate_dict_params(self, dr):
        pass

    def run(self, gain, fixed_rate_change=None, dr=None, soft_stop_cond=True, plot_ind_figs=False, th_percentage=1e-2,
            y_lims_ind_plot=None, st_prior=None, filtering=False, cutoff=5):
        if dr is None: dr = self.dict_results
        self.validate_dict_params(dr)

        [total_realizations, stoch_input, seeds, num_realizations, num_freq_exp, sfreq, f_vector, num_changes_rate,
         aux_num_r, dyn_synapse, sim_params, t_tra, t_tra_syn, t_tra_syn_b, lif_output, file_name, gain_v,
         fix_rate_change_a, folder_vars, stp_params, n_noise] = [dr['t_realizations'], self.stoch_input, dr['seeds'],
         dr['realizations'], dr['num_freq_exp'], dr['sfreq'], dr['initial_frequencies'], dr['num_changes_rate'],
         dr['num_instance_model'], dr['dyn_synapse'], dr['sim_params'], dr['time_transition'],
         dr['time_transition_syn'], dr['time_transition_syn_b'], self.lif_output, self.file_name, self.gain_vector,
         dr['fix_rate_change_a'], self.folder_vars, self.stp_params, self.neuron_noise]

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
        shape_stat = 54
        res_per_reali = np.zeros((shape_stat, num_freq_exp, num_realizations))
        res_per_reali_syn = np.zeros((shape_stat, num_freq_exp, num_realizations))
        res_per_reali_syn_b = np.zeros((shape_stat, num_freq_exp, num_realizations))
        res_real = np.zeros((shape_stat, total_realizations, num_freq_exp))
        res_real_syn = np.zeros((shape_stat, total_realizations, num_freq_exp))
        res_real_syn_b = np.zeros((shape_stat, total_realizations, num_freq_exp))

        # Auxiliar variables for Information theory
        PSR_per_freq = [[] for _ in range(num_freq_exp)]
        spike_event_per_freq = [[] for _ in range(num_freq_exp)]
        PSR_per_freq_syn = [[] for _ in range(num_freq_exp)]
        spike_event_per_freq_syn = [[] for _ in range(num_freq_exp)]
        PSR_per_freq_syn_b = [[] for _ in range(num_freq_exp)]
        spike_event_per_freq_syn_b = [[] for _ in range(num_freq_exp)]

        # Setting proportional and fixed rates of change
        proportional_changes = gain * f_vector + f_vector
        constant_changes = fixed_rate_change + f_vector

        # time-series for transition times: piw, pmw, pew, ciw, cmw, cew
        tr_time_series = [[[] for _ in range(num_freq_exp)] for _ in range(6)]

        ini_loop_time = m_time()
        print("Ini big loop")
        realization = 0
        while realization < num_loop_realizations and soft_stop_cond:
            loop_time = m_time()
            t_tra_mid_win, t_tra_mid_win_syn = None, None

            # Building reference signal for constant and fixed rate changes
            i = num_freq_exp - 1
            while i >= 0:  # while i < num_freq_exp:
                loop_experiments = m_time()

                se = int(time.time())
                seeds1, seeds2, seeds3, n_seeds = [1984], [1884], [1848], None
                # For poisson or deterministic inputs
                if self.stoch_input:
                    seeds.append(se)
                    seeds1 = [j + se for j in range(num_realizations)]
                    seeds2 = [j + se + 2 for j in range(num_realizations)]
                    seeds3 = [j + se + 3 for j in range(num_realizations)]

                # For neuron noise
                if n_noise:
                    if not self.stoch_input: seeds.append(se)
                    n_seeds = [j + se for j in range(int(2 * L / 3))] + [j + se for j in range(int(L / 3))]

                ref_signals = simple_spike_train(sfreq, f_vector[i], int(L / num_changes_rate),
                                                 num_realizations=aux_num_r, poisson=self.stoch_input, seeds=seeds1)
                # ISIs, histograms = inter_spike_intervals(ref_signals, 1 / sfreq, 1e-3)
                # histograms[0][1] *= 1000
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
                else:
                    for neuron in range(cons_input.shape[0]):
                        cons_input[neuron, neuron + 1] = 1
                        fix_input[neuron, neuron + 1] = 1

                # Running STP model
                if dyn_synapse:
                    # Reseting initial conditions
                    self.stp_prop.set_initial_conditions()
                    self.neuron_prop.set_simulation_params(sim_params, seeds1[0])
                    self.stp_fix.set_initial_conditions()
                    self.neuron_fix.set_simulation_params(sim_params, seeds1[0])
                    # Running the models
                    st_prior_ = None
                    # if st_prior is not None:
                    #     st_i = np.where(f_vector[i] <= st_prior[0, :])[0][0]
                    #     st_ip = np.where(proportional_changes[i] <= st_prior[0, :])[0][0]
                    #     st_prior_ = np.array([st_prior[1:, st_i], st_prior[1:, st_ip], st_prior[1:, st_i]])
                    model_stp_parallel(self.stp_prop, self.neuron_prop, stp_params, cons_input, n_seeds, n_noise,
                                       rate_input=f_vector[i], st_prior=st_prior_)
                    # model_stp_parallel(self.stp_fix, self.lif_fix, stp_params, fix_input)
                else:
                    # Reseting initial conditions
                    self.neuron_prop.set_simulation_params(sim_params)
                    # lif_fix.set_simulation_params(sim_params)
                    # Running the models
                    static_synapse(self.neuron_prop, cons_input, 9e0)  # , 0.0125e-6)
                    # static_synapse(lif_fix, fix_input, 9e0)  # , 0.0125e-6)

                # MEMBRANE POTENTIAL
                # Compute statistical descriptors
                signal_prop, signal_fix = self.neuron_prop.membrane_potential, self.neuron_fix.membrane_potential
                # getting transition time for rate of proportional change if possible
                aux_cond = np.where(proportional_changes[i] <= f_vector)
                if len(aux_cond[0]) > 0:
                    aux_i = aux_cond[0][0]
                    t_tra_mid_win = np.max(t_tra[aux_i])
                # Clipping signals to avoid suprathreshold maxima
                signal_prop = np.clip(signal_prop, None, 0.0)
                signal_fix = np.clip(signal_fix, None, 0.0)
                # Computing statistics of each window, for the whole window and for the transition- and steady-states
                a, b, c, d, e, f, h = aux_statistics_prop_cons(signal_prop, signal_fix, Le_time_win, None,
                                                   sim_params, [None, t_tra_mid_win, None], 1 / sfreq,
                                                   th_percentage=th_percentage, filtering=filtering, cutoff=cutoff)
                if a.shape[1] != res_per_reali.shape[2]:
                    assert a.shape[1] == res_per_reali.shape[2], "not same shape"
                res_per_reali[:, i, :], t_tr_, tr_time_series_i, piw, pmw, pew, t_tr_filt = a, b, c, d, e, f, h
                t_tra[i].append(t_tr_)

                # Plotting individual figures if indicated
                if plot_ind_figs:
                    title_graph_ = title_graph + ", freq. %dHz" % f_vector[i]
                    t_tr = t_tr_[0]
                    path_save = self.folder_plots + file_name + '_' + str(f_vector[i]) + '_stat.png'
                    plot_gc_mem_potential_prop_fix(time_vector, i, signal_prop, signal_fix, t_tr, res_per_reali,
                                                   title_graph_, max_t, path_save=path_save, save_figs=self.save_figs,
                                                   y_lims_ind_plot=y_lims_ind_plot, plot_stats=True, plt_grid=False)

                    """
                    fig = plt.figure(figsize=(8, 3))
                    dt = 1 / self.sfreq
                    time_vec = np.arange(0, int(piw.shape[1] * dt), dt)
                    for n in range(piw.shape[0]):
                        aux = piw[n, :] + n * 0.005
                        plt.plot(time_vec, aux)
                        aux = pew[n, :] + n * 0.005
                        plt.plot(time_vec, aux)
                        aux = np.array([np.min(piw[n, :]), np.min(piw[n, :]) + 0.01]) + (n * 0.005)
                        plt.plot([t_tr_[n], t_tr_[n]], aux, c='black')
                    plt.grid()
                    plt.title(title_graph_ + ". tr_st_time " + str(t_tr_) + ". No filter")
                    path_save = self.folder_plots + file_name + '_' + str(f_vector[i]) + '_ini_end_windows.png'
                    if self.save_figs: fig.savefig(path_save, format='png')
                    if filtering:
                        fig = plt.figure(figsize=(8, 3))
                        piw = lowpass(piw, cutoff, 1 / dt)
                        pew = lowpass(pew, cutoff, 1 / dt)
                        for n in range(piw.shape[0]):
                            aux = piw[n, :] + n * 0.005
                            plt.plot(time_vec, aux)
                            aux = pew[n, :] + n * 0.005
                            plt.plot(time_vec, aux)
                            aux = np.array([np.min(piw[n, :]), np.min(piw[n, :]) + 0.01]) + (n * 0.005)
                            plt.plot([t_tr_filt[n], t_tr_filt[n]], aux, c='black')
                        plt.grid()
                        plt.title(title_graph_ + ". tr_st_time " + str(t_tr_filt) + ". Filtered")
                        path_save = self.folder_plots + file_name + '_' + str(f_vector[i]) + '_ini_end_windows_filt.png'
                        if self.save_figs: fig.savefig(path_save, format='png')
                    # """

                # SYNAPTIC CONTRIBUTION
                # Compute statistical descriptors
                signal_props, signal_fixes = self.stp_prop.get_output(), self.stp_fix.get_output()
                signal_prop = None
                if signal_props.ndim == 2:
                    signal_prop, signal_fix = signal_props, signal_fixes
                else:
                    signal_prop, signal_fix = signal_props[0, :], signal_fixes[0, :]
                # getting transition time for rate of proportional change if possible
                aux_cond = np.where(proportional_changes[i] <= f_vector)
                if len(aux_cond[0]) > 0:
                    aux_i = aux_cond[0][0]
                    t_tra_mid_win_syn = np.max(t_tra_syn[aux_i])
                # Computing statistics of each window for transition- and steady-states
                a, b, c, d, e, f, h = aux_statistics_prop_cons(signal_prop, signal_fix, Le_time_win, None,
                                                               sim_params, [None, t_tra_mid_win_syn, None], 1 / sfreq,
                                                               th_percentage=th_percentage, filtering=filtering,
                                                               cutoff=cutoff)
                if a.shape[1] != res_per_reali.shape[2]:
                    assert a.shape[1] == res_per_reali.shape[2], "not same shape"
                res_per_reali_syn[:, i, :], t_tr_syn, tr_time_series_i, piw, pmw, pew, t_tr_filt = a, b, c, d, e, f, h
                t_tra_syn[i].append(t_tr_syn)
                # For ampa if synapse is Doorn
                if signal_props.ndim == 3:
                    signal_prop, signal_fix = signal_props[1, :], signal_fixes[1, :]
                    # getting transition time for rate of proportional change if possible
                    aux_cond = np.where(proportional_changes[i] <= f_vector)
                    if len(aux_cond[0]) > 0:
                        aux_i = aux_cond[0][0]
                        t_tra_mid_win_syn = np.max(t_tra_syn[aux_i])
                    # Computing statistics of each window for transition- and steady-states
                    a, b, c, d, e, f, h = aux_statistics_prop_cons(signal_prop, signal_fix, Le_time_win, None,
                                                                   sim_params, [None, t_tra_mid_win_syn, None],
                                                                   1 / sfreq, th_percentage=th_percentage,
                                                                   filtering=filtering, cutoff=cutoff)
                    if a.shape[1] != res_per_reali.shape[2]:
                        assert a.shape[1] == res_per_reali.shape[2], "not same shape"
                    res_per_reali_syn_b[:, i, :], t_tr_syn_b, tr_time_series_i,  = a, b, c
                    piw, pmw, pew, t_tr_filt = d, e, f, h
                    t_tra_syn_b[i].append(t_tr_syn)
                # Updating array of time_transitions
                """
                t_tra[i].append(t_tr_)
                for tr_i in range(6):
                    if len(tr_time_series[tr_i][i]) == 0:
                        tr_time_series[tr_i][i] = tr_time_series_i[tr_i]
                    else:
                        tr_time_series[tr_i][i] = tr_time_series[tr_i][i] + tr_time_series_i[tr_i]
                # """

                # Computing Postsynaptic response (PSR) to compute H(PSR) -unconditional entropy-
                # Neuron contribution
                PSR_per_freq[i].append(self.neuron_prop.output_spike_events)
                spike_event_per_freq[i].append(self.neuron_prop.time_spike_events)
                # Synaptic contribution
                PSR_per_freq_syn[i].append(self.stp_prop.output_spike_events)

                # Final print of the loop
                print_time(m_time() - loop_experiments, file_name + ", Realisation " + str(realization) +
                           ", frequency " + str(f_vector[i]))
                i -= 1

            # steady-state part
            for res_i in range(res_real.shape[0]):
                r = realization
                res_real[res_i, r * num_realizations:(r + 1) * num_realizations] = res_per_reali[res_i, :].T
                res_real_syn[res_i, r * num_realizations:(r + 1) * num_realizations] = res_per_reali_syn[res_i, :].T
                # For ampa if synapse is Doorn
                if self.stp_prop.get_output().ndim == 3:
                    res_real_syn_b[res_i, r * num_realizations:(r + 1) * num_realizations] = res_per_reali_syn_b[res_i, :].T

            print_time(m_time() - loop_time, file_name + ", Realisation " + str(realization))

            realization += 1

        #
        if soft_stop_cond:
            # Obtaining time series of statistical descriptors for transition component of signals (ini, mid, end win)
            # self.tr_time_series = tr_time_series
            # st_tr_a, res = get_time_series_statistics_of_transitions(tr_time_series, f_vector, proportional_changes,
            #                                                          1/sfreq, th_percentage)
            # Saving extra variables
            # dr['stat_tSeries_transition'] = res
            # dr['stat_time_transition'] = st_tr_a

            # transition-state
            for i in range(num_freq_exp):
                t_tra[i] = np.ravel(t_tra[i])
                t_tra_syn[i] = np.ravel(t_tra_syn[i])
                # For ampa if synapse is Doorn
                if self.stp_prop.get_output().ndim == 3: t_tra_syn_b[i] = np.ravel(t_tra_syn_b[i])
            t_tra = np.array(t_tra).T
            t_tra_syn = np.array(t_tra_syn).T
            if self.stp_prop.get_output().ndim == 3: t_tra_syn_b = np.array(t_tra_syn_b).T

            # Saving transition times of each window
            dr['time_transition'] = t_tra
            dr['time_transition_syn'] = t_tra_syn
            if self.stp_prop.get_output().ndim == 3: dr['time_transition_syn_b'] = t_tra_syn_b
            # ##########################################################################################################
            # """
            # Getting information theory analysis
            # Iterating through general realizations
            a = [[] for _ in range(num_freq_exp)]
            ISI_per_freq_iw_tr, ISI_per_freq_mw_tr, ISI_per_freq_ew_tr = a.copy(), a.copy(), a.copy()
            ISI_per_freq_iw_st, ISI_per_freq_mw_st, ISI_per_freq_ew_st = a.copy(), a.copy(), a.copy()
            PSR_per_freq_iw_tr, PSR_per_freq_mw_tr, PSR_per_freq_ew_tr = a.copy(), a.copy(), a.copy()
            PSR_per_freq_iw_st, PSR_per_freq_mw_st, PSR_per_freq_ew_st = a.copy(), a.copy(), a.copy()
            PSR_syn_per_freq_iw_tr, PSR_syn_per_freq_mw_tr, PSR_syn_per_freq_ew_tr = a.copy(), a.copy(), a.copy()
            PSR_syn_per_freq_iw_st, PSR_syn_per_freq_mw_st, PSR_syn_per_freq_ew_st = a.copy(), a.copy(), a.copy()
            PSR_syn_b_per_freq_iw_tr, PSR_syn_b_per_freq_mw_tr, PSR_syn_b_per_freq_ew_tr = a.copy(), a.copy(), a.copy()
            PSR_syn_b_per_freq_iw_st, PSR_syn_b_per_freq_mw_st, PSR_syn_b_per_freq_ew_st = a.copy(), a.copy(), a.copy()

            # min-max of synaptic contributions
            min_syn, max_syn, min_syn_b, max_syn_b = np.inf, -np.inf, np.inf, -np.inf
            realization = 0
            for realization in range(num_loop_realizations):
                # Iterating through frequencies
                i = num_freq_exp - 1
                for i in range(num_freq_exp):
                    f_ = f_vector[i]
                    # Iterating through specific realizations
                    for neuron_realization in range(num_realizations):
                        ta = np.array(spike_event_per_freq[i][realization][neuron_realization]) / sfreq
                        PSR_aux = np.array(PSR_per_freq[i][realization][neuron_realization])
                        PSR_aux_syn = np.array(PSR_per_freq_syn[i][realization][neuron_realization])
                        # Getting time of reaching steady-state for ini and end windows
                        tr_st_time = dr['time_transition'][num_realizations * realization + neuron_realization, i]
                        # Getting time of reaching steady-state for mid window
                        tr_st_time_mw = tr_st_time  # In case there is no tr_st_time for rate of mid window
                        aux_cond = np.where(proportional_changes[i] <= f_vector)
                        if len(aux_cond[0]) > 0:
                            aux_i = aux_cond[0][0]
                            tr_st_time_mw = dr['time_transition'][num_realizations * realization + neuron_realization, aux_i]
                        # Getting masks to separate transitory and stationary states for ini, mid and end windows
                        mask_iw_tr = (ta >= 0) & (ta < tr_st_time)
                        mask_iw_st = (ta >= tr_st_time) & (ta < 2)
                        mask_mw_tr = (ta >= 2) & (ta < 2 + tr_st_time_mw)
                        mask_mw_st = (ta >= 2 + tr_st_time_mw) & (ta < 4)
                        mask_ew_tr = (ta >= 4) & (ta < 4 + tr_st_time)
                        mask_ew_st = (ta >= 4 + tr_st_time)
                        
                        # Getting masks to separate ini, mid and end windows
                        mask_iniw = ta < 2
                        mask_endw = ta >= 4
                        mask_midw = np.logical_not(np.logical_xor(mask_iniw, mask_endw))
                        
                        # Separating ini, mid and end windows of spike events
                        spike_ev_iw_tr = np.append(ta[mask_iw_tr], tr_st_time)
                        spike_ev_iw_st = np.append(ta[mask_iw_st], 2.)
                        spike_ev_mw_tr = np.append(ta[mask_mw_tr], 2. + tr_st_time_mw)
                        spike_ev_mw_st = np.append(ta[mask_mw_st], 4.)
                        spike_ev_ew_tr = np.append(ta[mask_ew_tr], 4. + tr_st_time)
                        spike_ev_ew_st = np.append(ta[mask_ew_st], 6.)
                        # Separating ini, mid and end windows of ISI
                        ISI_iw_tr = np.diff(spike_ev_iw_tr)
                        ISI_iw_st = np.diff(spike_ev_iw_st)
                        ISI_mw_tr = np.diff(spike_ev_mw_tr)
                        ISI_mw_st = np.diff(spike_ev_mw_st)
                        ISI_ew_tr = np.diff(spike_ev_ew_tr)
                        ISI_ew_st = np.diff(spike_ev_ew_st)
                        # Updating general varibles of ISI
                        ISI_per_freq_iw_tr[i] = ISI_per_freq_iw_tr[i] + list(ISI_iw_tr)
                        ISI_per_freq_iw_st[i] = ISI_per_freq_iw_st[i] + list(ISI_iw_st)
                        ISI_per_freq_mw_tr[i] = ISI_per_freq_mw_tr[i] + list(ISI_mw_tr)
                        ISI_per_freq_mw_st[i] = ISI_per_freq_mw_st[i] + list(ISI_mw_st)
                        ISI_per_freq_ew_tr[i] = ISI_per_freq_ew_tr[i] + list(ISI_ew_tr)
                        ISI_per_freq_ew_st[i] = ISI_per_freq_ew_st[i] + list(ISI_ew_st)
                        
                        # Separating ini, mid and end windows of neuron PSR
                        PSR_iw_tr = PSR_aux[mask_iw_tr]
                        PSR_iw_st = PSR_aux[mask_iw_st]
                        PSR_mw_tr = PSR_aux[mask_mw_tr]
                        PSR_mw_st = PSR_aux[mask_mw_st]
                        PSR_ew_tr = PSR_aux[mask_ew_tr]
                        PSR_ew_st = PSR_aux[mask_ew_st]
                        # Updating general varibles of neuron PSR
                        PSR_per_freq_iw_tr[i] = PSR_per_freq_iw_tr[i] + list(PSR_iw_tr)
                        PSR_per_freq_iw_st[i] = PSR_per_freq_iw_st[i] + list(PSR_iw_st)
                        PSR_per_freq_mw_tr[i] = PSR_per_freq_mw_tr[i] + list(PSR_mw_tr)
                        PSR_per_freq_mw_st[i] = PSR_per_freq_mw_st[i] + list(PSR_mw_st)
                        PSR_per_freq_ew_tr[i] = PSR_per_freq_ew_tr[i] + list(PSR_ew_tr)
                        PSR_per_freq_ew_st[i] = PSR_per_freq_ew_st[i] + list(PSR_ew_st)
                        
                        # Separating ini, mid and end windows of synapse PSR
                        PSR_syn_iw_tr = PSR_aux_syn[mask_iw_tr]
                        PSR_syn_iw_st = PSR_aux_syn[mask_iw_st]
                        PSR_syn_mw_tr = PSR_aux_syn[mask_mw_tr]
                        PSR_syn_mw_st = PSR_aux_syn[mask_mw_st]
                        PSR_syn_ew_tr = PSR_aux_syn[mask_ew_tr]
                        PSR_syn_ew_st = PSR_aux_syn[mask_ew_st]
                        if PSR_syn_iw_tr.ndim == 2:
                            # Updating general varibles of synapse PSR for AMPA
                            PSR_syn_per_freq_iw_tr[i] = PSR_syn_per_freq_iw_tr[i] + list(PSR_syn_iw_tr[:, 0])
                            PSR_syn_per_freq_iw_st[i] = PSR_syn_per_freq_iw_st[i] + list(PSR_syn_iw_st[:, 0])
                            PSR_syn_per_freq_mw_tr[i] = PSR_syn_per_freq_mw_tr[i] + list(PSR_syn_mw_tr[:, 0])
                            PSR_syn_per_freq_mw_st[i] = PSR_syn_per_freq_mw_st[i] + list(PSR_syn_mw_st[:, 0])
                            PSR_syn_per_freq_ew_tr[i] = PSR_syn_per_freq_ew_tr[i] + list(PSR_syn_ew_tr[:, 0])
                            PSR_syn_per_freq_ew_st[i] = PSR_syn_per_freq_ew_st[i] + list(PSR_syn_ew_st[:, 0])
                            # Updating general varibles of synapse PSR for NMDA
                            PSR_syn_b_per_freq_iw_tr[i] = PSR_syn_b_per_freq_iw_tr[i] + list(PSR_syn_iw_tr[:, 1])
                            PSR_syn_b_per_freq_iw_st[i] = PSR_syn_b_per_freq_iw_st[i] + list(PSR_syn_iw_st[:, 1])
                            PSR_syn_b_per_freq_mw_tr[i] = PSR_syn_b_per_freq_mw_tr[i] + list(PSR_syn_mw_tr[:, 1])
                            PSR_syn_b_per_freq_mw_st[i] = PSR_syn_b_per_freq_mw_st[i] + list(PSR_syn_mw_st[:, 1])
                            PSR_syn_b_per_freq_ew_tr[i] = PSR_syn_b_per_freq_ew_tr[i] + list(PSR_syn_ew_tr[:, 1])
                            PSR_syn_b_per_freq_ew_st[i] = PSR_syn_b_per_freq_ew_st[i] + list(PSR_syn_ew_st[:, 1])
                            # Updating min-max of synaptic contributions
                            max_ = np.max(PSR_aux_syn, axis=0)
                            if max_[0] > max_syn: max_syn = max_[0]
                            if max_[1] > max_syn_b: max_syn_b = max_[1]
                            min_ = np.min(PSR_aux_syn, axis=0)
                            if min_[0] < min_syn: min_syn = min_[0]
                            if min_[1] < min_syn_b: min_syn_b = min_[1]
                        else:
                            # Updating general varibles of synapse PSR
                            PSR_syn_per_freq_iw_tr[i] = PSR_syn_per_freq_iw_tr[i] + list(PSR_syn_iw_tr)
                            PSR_syn_per_freq_iw_st[i] = PSR_syn_per_freq_iw_st[i] + list(PSR_syn_iw_st)
                            PSR_syn_per_freq_mw_tr[i] = PSR_syn_per_freq_mw_tr[i] + list(PSR_syn_mw_tr)
                            PSR_syn_per_freq_mw_st[i] = PSR_syn_per_freq_mw_st[i] + list(PSR_syn_mw_st)
                            PSR_syn_per_freq_ew_tr[i] = PSR_syn_per_freq_ew_tr[i] + list(PSR_syn_ew_tr)
                            PSR_syn_per_freq_ew_st[i] = PSR_syn_per_freq_ew_st[i] + list(PSR_syn_ew_st)
                            max_ = np.max(PSR_aux_syn)
                            if max_ > max_syn: max_syn = max_
                            min_ = np.min(PSR_aux_syn)
                            if min_ < min_syn: min_syn = min_

                        # Updating min-max of synaptic contributions

            # Getting information theory analysis
            H_ISI_iw_tr, H_ISI_mw_tr, H_ISI_ew_tr = [], [], []
            H_ISI_iw_st, H_ISI_mw_st, H_ISI_ew_st = [], [], []
            # Entropy for neuronal output
            H_PSR_iw_tr, H_PSR_mw_tr, H_PSR_ew_tr = [], [], []
            H_PSR_iw_st, H_PSR_mw_st, H_PSR_ew_st = [], [], []
            # Entropy for synaptic output(s)
            H_PSR_syn_iw_tr, H_PSR_syn_mw_tr, H_PSR_syn_ew_tr = [], [], []
            H_PSR_syn_iw_st, H_PSR_syn_mw_st, H_PSR_syn_ew_st = [], [], []
            H_PSR_syn_b_iw_tr, H_PSR_syn_b_mw_tr, H_PSR_syn_b_ew_tr = [], [], []
            H_PSR_syn_b_iw_st, H_PSR_syn_b_mw_st, H_PSR_syn_b_ew_st = [], [], []
            # Getting information theory analysis
            H_ISI_iw_tr_100bins, H_ISI_mw_tr_100bins, H_ISI_ew_tr_100bins = [], [], []
            H_ISI_iw_st_100bins, H_ISI_mw_st_100bins, H_ISI_ew_st_100bins = [], [], []
            # Entropy for neuronal output
            H_PSR_iw_tr_100bins, H_PSR_mw_tr_100bins, H_PSR_ew_tr_100bins = [], [], []
            H_PSR_iw_st_100bins, H_PSR_mw_st_100bins, H_PSR_ew_st_100bins = [], [], []
            # Entropy for synaptic output(s)
            H_PSR_syn_iw_tr_100bins, H_PSR_syn_mw_tr_100bins, H_PSR_syn_ew_tr_100bins = [], [], []
            H_PSR_syn_iw_st_100bins, H_PSR_syn_mw_st_100bins, H_PSR_syn_ew_st_100bins = [], [], []
            H_PSR_syn_b_iw_tr_100bins, H_PSR_syn_b_mw_tr_100bins, H_PSR_syn_b_ew_tr_100bins = [], [], []
            H_PSR_syn_b_iw_st_100bins, H_PSR_syn_b_mw_st_100bins, H_PSR_syn_b_ew_st_100bins = [], [], []

            # conductance of synaptic contributions to neuron model
            max_syn_contr = self.neuron_prop.max_syn_cont()
            bin_size_syn, bin_size_syn_b = max_syn - min_syn, max_syn_b - min_syn_b
            # if len(max_syn_contr) == 2:
            #     bin_size_syn = max_syn_contr[0][0]
            #     bin_size_syn_b = max_syn_contr[1][0]
            # else:
            #     bin_size_syn = max_syn_contr[0][0]

            for i in range(num_freq_exp):
                # ******************************************************************************************************
                # ENTROPY COMPUTATION USING FIXED NUMBER OF BINS
                # Entropy calculations for input
                H_ISI_iw_tr_100bins.append(binned_entropy(ISI_per_freq_iw_tr[i], n_bins=100))
                H_ISI_iw_st_100bins.append(binned_entropy(ISI_per_freq_iw_st[i], n_bins=100))
                H_ISI_mw_tr_100bins.append(binned_entropy(ISI_per_freq_mw_tr[i], n_bins=100))
                H_ISI_mw_st_100bins.append(binned_entropy(ISI_per_freq_mw_st[i], n_bins=100))
                H_ISI_ew_tr_100bins.append(binned_entropy(ISI_per_freq_ew_tr[i], n_bins=100))
                H_ISI_ew_st_100bins.append(binned_entropy(ISI_per_freq_ew_st[i], n_bins=100))
                # Entropy calculations for neuronal responses
                H_PSR_iw_tr_100bins.append(binned_entropy(PSR_per_freq_iw_tr[i], n_bins=100))
                H_PSR_iw_st_100bins.append(binned_entropy(PSR_per_freq_iw_st[i], n_bins=100))
                H_PSR_mw_tr_100bins.append(binned_entropy(PSR_per_freq_mw_tr[i], n_bins=100))
                H_PSR_mw_st_100bins.append(binned_entropy(PSR_per_freq_mw_st[i], n_bins=100))
                H_PSR_ew_tr_100bins.append(binned_entropy(PSR_per_freq_ew_tr[i], n_bins=100))
                H_PSR_ew_st_100bins.append(binned_entropy(PSR_per_freq_ew_st[i], n_bins=100))
                # Entropy calculations for synaptic response
                H_PSR_syn_iw_tr_100bins.append(binned_entropy(PSR_syn_per_freq_iw_tr[i], n_bins=100))
                H_PSR_syn_iw_st_100bins.append(binned_entropy(PSR_syn_per_freq_iw_st[i], n_bins=100))
                H_PSR_syn_mw_tr_100bins.append(binned_entropy(PSR_syn_per_freq_mw_tr[i], n_bins=100))
                H_PSR_syn_mw_st_100bins.append(binned_entropy(PSR_syn_per_freq_mw_st[i], n_bins=100))
                H_PSR_syn_ew_tr_100bins.append(binned_entropy(PSR_syn_per_freq_ew_tr[i], n_bins=100))
                H_PSR_syn_ew_st_100bins.append(binned_entropy(PSR_syn_per_freq_ew_st[i], n_bins=100))
                if PSR_syn_iw_tr.ndim == 2:
                    # Entropy calculations for second synaptic response
                    H_PSR_syn_b_iw_tr_100bins.append(binned_entropy(PSR_syn_b_per_freq_iw_tr[i], n_bins=100))
                    H_PSR_syn_b_iw_st_100bins.append(binned_entropy(PSR_syn_b_per_freq_iw_st[i], n_bins=100))
                    H_PSR_syn_b_mw_tr_100bins.append(binned_entropy(PSR_syn_b_per_freq_mw_tr[i], n_bins=100))
                    H_PSR_syn_b_mw_st_100bins.append(binned_entropy(PSR_syn_b_per_freq_mw_st[i], n_bins=100))
                    H_PSR_syn_b_ew_tr_100bins.append(binned_entropy(PSR_syn_b_per_freq_ew_tr[i], n_bins=100))
                    H_PSR_syn_b_ew_st_100bins.append(binned_entropy(PSR_syn_b_per_freq_ew_st[i], n_bins=100))
                # ******************************************************************************************************
                # ENTROPY COMPUTATION USING FIXED BIN SIZE
                # Entropy calculation for input
                bin_size = 0.1 / f_vector[i]  # 10% of T
                H_ISI_iw_tr.append(H_entropy_dyn_bins(ISI_per_freq_iw_tr[i], bin_size=bin_size))
                H_ISI_iw_st.append(H_entropy_dyn_bins(ISI_per_freq_iw_st[i], bin_size=bin_size))
                H_ISI_ew_tr.append(H_entropy_dyn_bins(ISI_per_freq_ew_tr[i], bin_size=bin_size))
                H_ISI_ew_st.append(H_entropy_dyn_bins(ISI_per_freq_ew_st[i], bin_size=bin_size))
                bin_size = 0.1 / proportional_changes[i]  # 10% of T
                H_ISI_mw_tr.append(H_entropy_dyn_bins(ISI_per_freq_mw_tr[i], bin_size=bin_size))
                H_ISI_mw_st.append(H_entropy_dyn_bins(ISI_per_freq_mw_st[i], bin_size=bin_size))
                # Entropy calculations for neuronal responses
                bin_size = 0.1e-3  # 0.1mV
                H_PSR_iw_tr.append(H_entropy_dyn_bins(PSR_per_freq_iw_tr[i], bin_size=bin_size))
                H_PSR_iw_st.append(H_entropy_dyn_bins(PSR_per_freq_iw_st[i], bin_size=bin_size))
                H_PSR_mw_tr.append(H_entropy_dyn_bins(PSR_per_freq_mw_tr[i], bin_size=bin_size))
                H_PSR_mw_st.append(H_entropy_dyn_bins(PSR_per_freq_mw_st[i], bin_size=bin_size))
                H_PSR_ew_tr.append(H_entropy_dyn_bins(PSR_per_freq_ew_tr[i], bin_size=bin_size))
                H_PSR_ew_st.append(H_entropy_dyn_bins(PSR_per_freq_ew_st[i], bin_size=bin_size))
                # Entropy calculations for synaptic response
                bin_size = 0.01 * bin_size_syn  # 1% of max. synaptic contribution
                H_PSR_syn_iw_tr.append(H_entropy_dyn_bins(PSR_syn_per_freq_iw_tr[i], bin_size=bin_size))
                H_PSR_syn_iw_st.append(H_entropy_dyn_bins(PSR_syn_per_freq_iw_st[i], bin_size=bin_size))
                H_PSR_syn_mw_tr.append(H_entropy_dyn_bins(PSR_syn_per_freq_mw_tr[i], bin_size=bin_size))
                H_PSR_syn_mw_st.append(H_entropy_dyn_bins(PSR_syn_per_freq_mw_st[i], bin_size=bin_size))
                H_PSR_syn_ew_tr.append(H_entropy_dyn_bins(PSR_syn_per_freq_ew_tr[i], bin_size=bin_size))
                H_PSR_syn_ew_st.append(H_entropy_dyn_bins(PSR_syn_per_freq_ew_st[i], bin_size=bin_size))
                if PSR_syn_iw_tr.ndim == 2:
                    # Entropy calculations for second synaptic response
                    bin_size = 0.01 * bin_size_syn_b  # 1% of max. synaptic contribution
                    H_PSR_syn_b_iw_tr.append(H_entropy_dyn_bins(PSR_syn_b_per_freq_iw_tr[i], bin_size=bin_size))
                    H_PSR_syn_b_iw_st.append(H_entropy_dyn_bins(PSR_syn_b_per_freq_iw_st[i], bin_size=bin_size))
                    H_PSR_syn_b_mw_tr.append(H_entropy_dyn_bins(PSR_syn_b_per_freq_mw_tr[i], bin_size=bin_size))
                    H_PSR_syn_b_mw_st.append(H_entropy_dyn_bins(PSR_syn_b_per_freq_mw_st[i], bin_size=bin_size))
                    H_PSR_syn_b_ew_tr.append(H_entropy_dyn_bins(PSR_syn_b_per_freq_ew_tr[i], bin_size=bin_size))
                    H_PSR_syn_b_ew_st.append(H_entropy_dyn_bins(PSR_syn_b_per_freq_ew_st[i], bin_size=bin_size))

            # ******************************************************************************************************
            # ENTROPY COMPUTATION USING FIXED NUMBER OF BINS
            dr['H_ISI_tr_100'] = np.array([H_ISI_iw_tr_100bins, H_ISI_mw_tr_100bins, H_ISI_ew_tr_100bins])
            dr['H_ISI_st_100'] = np.array([H_ISI_iw_st_100bins, H_ISI_mw_st_100bins, H_ISI_ew_st_100bins])
            dr['H_PSR_tr_100'] = np.array([H_PSR_iw_tr_100bins, H_PSR_mw_tr_100bins, H_PSR_ew_tr_100bins])
            dr['H_PSR_st_100'] = np.array([H_PSR_iw_st_100bins, H_PSR_mw_st_100bins, H_PSR_ew_st_100bins])
            dr['H_PSR_syn_tr_100'] = np.array([H_PSR_syn_iw_tr_100bins, H_PSR_syn_mw_tr_100bins,
                                               H_PSR_syn_ew_tr_100bins])
            dr['H_PSR_syn_st_100'] = np.array([H_PSR_syn_iw_st_100bins, H_PSR_syn_mw_st_100bins,
                                               H_PSR_syn_ew_st_100bins])
            if PSR_syn_iw_tr.ndim == 2:
                dr['H_PSR_syn_b_tr_100'] = np.array([H_PSR_syn_b_iw_tr_100bins, H_PSR_syn_b_mw_tr_100bins,
                                                     H_PSR_syn_b_ew_tr_100bins])
                dr['H_PSR_syn_b_st_100'] = np.array([H_PSR_syn_b_iw_st_100bins, H_PSR_syn_b_mw_st_100bins,
                                                     H_PSR_syn_b_ew_st_100bins])
            # ******************************************************************************************************
            # ENTROPY COMPUTATION USING FIXED BIN SIZE
            dr['H_ISI_tr'] = np.array([H_ISI_iw_tr, H_ISI_mw_tr, H_ISI_ew_tr])
            dr['H_ISI_st'] = np.array([H_ISI_iw_st, H_ISI_mw_st, H_ISI_ew_st])
            dr['H_PSR_tr'] = np.array([H_PSR_iw_tr, H_PSR_mw_tr, H_PSR_ew_tr])
            dr['H_PSR_st'] = np.array([H_PSR_iw_st, H_PSR_mw_st, H_PSR_ew_st])
            dr['H_PSR_syn_tr'] = np.array([H_PSR_syn_iw_tr, H_PSR_syn_mw_tr, H_PSR_syn_ew_tr])
            dr['H_PSR_syn_st'] = np.array([H_PSR_syn_iw_st, H_PSR_syn_mw_st, H_PSR_syn_ew_st])
            if PSR_syn_iw_tr.ndim == 2:
                dr['H_PSR_syn_b_tr'] = np.array([H_PSR_syn_b_iw_tr, H_PSR_syn_b_mw_tr, H_PSR_syn_b_ew_tr])
                dr['H_PSR_syn_b_st'] = np.array([H_PSR_syn_b_iw_st, H_PSR_syn_b_mw_st, H_PSR_syn_b_ew_st])

            # Saving bin_size computation min-max limits
            dr['H_PSR_syn_max_contr'] = [[min_syn, max_syn]]
            if PSR_syn_iw_tr.ndim == 2: dr['H_PSR_syn_max_contr'].append([min_syn_b, max_syn_b])

            # """
            # ##########################################################################################################

            # Saving varible to know if
            dr['neuron_noise'] = n_noise

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
                dr['syn_' + stat_list[nam]] = res_real_syn[nam, :]
                if self.stp_prop.get_output().ndim == 3: dr['syn_b_' + stat_list[nam]] = res_real_syn_b[nam, :]

            if self.save_vars:
                saveObject(dr, file_name, folder_vars)

        # if experiment is forced, then update final dictionary to be returned
        if soft_stop_cond:
            # Updating final dictionary
            for nam in range(res_real.shape[0]):
                dr[stat_list[nam]] = res_real[nam, :]

        print_time(m_time() - ini_loop_time, "Total big loop")

        return dr
