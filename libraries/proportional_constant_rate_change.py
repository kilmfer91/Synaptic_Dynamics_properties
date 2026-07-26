from gain_control.utils_gc import *
import cProfile, pstats, tracemalloc, os, psutil


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
        self.imputations = True  # dict_params['imputations']
        self.stoch_input = dict_params['stoch_input']
        self.lif_output = True  # dict_params['lif_output']
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
        self.num_instance_model = int(self.num_realizations * self.num_syn)
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
        if not self.stoch_input:
            self.file_name = model + '_det' + aux_name
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
        max_gain = max(gain_vector)
        aux_max_freq = int(((sfreq / 4) - 10) / (1 + max_gain))  # int(sfreq / 6) - 10
        if max_freq is None:
            max_freq = aux_max_freq
        else:
            if max_freq > aux_max_freq:
                max_freq = aux_max_freq
        # so max. ini freq sfreq/12 | 16kHz:2501, 5kHz:801, 6KHz: 951
        range_f, range_f2, range_f3, range_f4 = [], [], [], []
        if 100 < max_freq:
            range_f = [i for i in range(10, 100, 5)]
            if 500 < max_freq:
                range_f2 = [i for i in range(100, 500, 10)]
                if 1000 < max_freq:
                    range_f3 = [i for i in range(500, 1000, 50)]
                    range_f4 = [i for i in range(1000, max_freq, 100)]
                else:
                    range_f3 = [i for i in range(500, max_freq, 50)]
            else:
                range_f2 = [i for i in range(100, max_freq, 10)]
        else:
            range_f = [i for i in range(10, max_freq, 5)]
        # range_f = [10, 20, 500]
        # range_f2, range_f3, range_f4 = [], [] , []
        self.f_vector = np.array(range_f + range_f2 + range_f3 + range_f4) if f_vec is None else f_vec

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
            aux = [[[] for _ in range(dr['num_freq_exp'])] for _ in range(len(self.neuron_prop.get_state_variables()))]
            if 'time_transition' not in dr: dr['time_transition'] = aux
            aux = [[[] for _ in range(dr['num_freq_exp'])] for _ in range(len(self.stp_prop.get_state_variables()))]
            if 'time_transition_syn' not in dr: dr['time_transition_syn'] = aux
            if 'time_transition_syn_b' not in dr: dr['time_transition_syn_b'] = [[] for _ in range(dr['num_freq_exp'])]
            # if 'stat_tSeries_transition' not in dr: dr['stat_tSeries_transition'] = [[np.zeros((1, 1))]]
            # if 'stat_time_transition' not in dr: dr['stat_time_transition'] = [np.zeros(1)]
            if 'neuron_noise' not in dr: dr['neuron_noise'] = False
            if 'PSR_events' not in dr: dr['PSR_events'] = [[] for _ in range(dr['num_freq_exp'])]
            if 'spike_events' not in dr: dr['spike_events'] = [[] for _ in range(dr['num_freq_exp'])]
            if 'PSR_events_wind' not in dr: dr['PSR_events_wind'] = [[], [], []]
            if 'spike_events_wind' not in dr: dr['spike_events'] = [[], [], []]
            if 'name_neuron_state_variables' not in dr: dr['name_neuron_state_variables'] = None  # Later assigned
            if 'name_syn_state_variables' not in dr: dr['name_syn_state_variables'] = None  # Assigned in (model_creati)

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
            aux = [[[] for _ in range(dr['num_freq_exp'])] for _ in range(len(self.neuron_prop.get_state_variables()))]
            dr['time_transition'] = aux
            aux = [[[] for _ in range(dr['num_freq_exp'])] for _ in range(len(self.stp_prop.get_state_variables()))]
            dr['time_transition_syn'] = aux
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

            # Setting names of state variables
            dr['name_neuron_state_variables'] = list(self.neuron_prop.get_state_variables().keys())
            dr['name_syn_state_variables'] = list(self.stp_prop.get_state_variables().keys())

        self.dict_results = dr
        return self.file_loaded, self.dict_results

    @staticmethod
    def compute_spike_outputs_vectorized(output, time_spike_events, operators_sv, arg_operators_sv, dt):
        """
        Compute per-spike output summaries for all synapses.

        Parameters
        ----------
        output : np.ndarray
            Shape (n_state_vars, n_syn, L)
        time_spike_events : list[np.ndarray]
            List of length n_syn; each is a 1D array of spike times.
        operators_sv : list[callable]
            One operator per state variable, e.g. [np.min, np.max, ...]
        arg_operators_sv : list[callable]
            One operator per state variable returning an index, e.g. [np.argmin, np.argmax, ...]
        dt: float
            Time step in seconds.
        Returns
        -------
        output_spike_events : list[list]
            output_spike_events[s][i] = list of per-state-variable summaries for spike i of synapse s.
        output_spike_events_tonic : list[list]
            Tonic component per spike.
        ind_spike_events : list[list]
        ind_spike_events_tonic : list[list]
        """
        n_state_vars, n_syn, L = output.shape
        assert len(time_spike_events) == n_syn
        assert len(operators_sv) == n_state_vars
        assert len(arg_operators_sv) == n_state_vars

        output_spike_events = [[] for _ in range(n_syn)]
        output_spike_events_tonic = [[] for _ in range(n_syn)]
        ind_spike_events = [[] for _ in range(n_syn)]
        ind_spike_events_tonic = [[] for _ in range(n_syn)]
        time_spike_events_t = [[] for _ in range(n_syn)]

        # Tonic component at t=0 (same for all spikes, but you can repeat per spike if needed)
        tonic_all = output[:, :, 0]  # shape (n_state_vars, n_syn)

        for s in range(n_syn):
            spikes = time_spike_events[s]
            if len(spikes) == 0:
                continue

            # Tonic: for each spike, store the same tonic vector (matching your original logic)
            tonic_s = tonic_all[:, s]  # (n_state_vars,)
            for _ in spikes:
                output_spike_events_tonic[s].append(list(tonic_s))

            # Phasic / interval-based
            for i, t_curr in enumerate(spikes):
                if i == 0:
                    # No previous spike: treat as single-time phasic
                    per_state_variable = list(output[:, s, t_curr])
                    output_spike_events[s].append(per_state_variable)
                    ind_spike_events_tonic[s].append(t_curr - 1)
                    ind_spike_events[s].append(t_curr)
                    continue

                t_prev = spikes[i - 1]
                if t_curr == t_prev:
                    # Same time step (edge case)
                    per_state_variable = list(output[:, s, t_curr])
                    output_spike_events[s].append(per_state_variable)
                    ind_spike_events_tonic[s].append(t_curr - 1)
                    ind_spike_events[s].append(t_curr)
                else:
                    # Interval [t_prev, t_curr)
                    segment = output[:, s, t_prev:t_curr]  # (n_state_vars, interval_len)

                    # Apply operators per state variable
                    per_state_variable = [op(segment[sv, :]) for sv, op in enumerate(operators_sv)]
                    output_spike_events[s].append(per_state_variable)

                    # Arg operators
                    a = np.array([
                        op(segment[sv, :]) for sv, op in enumerate(arg_operators_sv)
                    ])

                    ind_spike_events_tonic[s].append(t_prev - 1)
                    ind_spike_events[s].append(a + t_prev)

            # time spike events
            time_spike_events_t[s] = time_spike_events[s] * dt
        a = (output_spike_events, output_spike_events_tonic, ind_spike_events, ind_spike_events_tonic,
             time_spike_events_t)
        return a

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

        # Setting names of state variables in final dictionary
        # self.dict_results['name_neuron_state_variables'] = list(self.neuron_prop.get_state_variables().keys())
        # self.dict_results['name_syn_state_variables'] = list(self.stp_prop.get_state_variables().keys())

    def validate_dict_params(self, dr):
        pass

    def run(self, gain, fixed_rate_change=None, dr=None, soft_stop_cond=True, plot_ind_figs=False, th_percentage=1e-2,
            y_lims_ind_plot=None, st_prior=None, filtering=False, cutoff=5, profiling=False):
        if dr is None: dr = self.dict_results
        self.validate_dict_params(dr)

        [total_realizations, stoch_input, seeds, num_realizations, num_freq_exp, sfreq, f_vector, num_changes_rate,
         aux_num_r, dyn_synapse, sim_params, t_tra, t_tra_syn, t_tra_syn_b, lif_output, file_name, gain_v,
         fix_rate_change_a, folder_vars, stp_params, n_noise] = [dr['t_realizations'], self.stoch_input, dr['seeds'],
                                                                 dr['realizations'], dr['num_freq_exp'], dr['sfreq'],
                                                                 dr['initial_frequencies'], dr['num_changes_rate'],
                                                                 dr['num_instance_model'], dr['dyn_synapse'],
                                                                 dr['sim_params'], dr['time_transition'],
                                                                 dr['time_transition_syn'], dr['time_transition_syn_b'],
                                                                 self.lif_output, self.file_name, self.gain_vector,
                                                                 dr['fix_rate_change_a'], self.folder_vars,
                                                                 self.stp_params, self.neuron_noise]

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
        shape_stat = 63
        # length state variables
        l_sv = len(self.neuron_prop.get_state_variables())
        l_svs = len(self.stp_prop.get_state_variables())
        res_per_reali = [np.zeros((shape_stat, num_freq_exp, num_realizations)) for _ in range(l_sv)]
        res_per_reali_syn = [np.zeros((shape_stat, num_freq_exp, num_realizations)) for _ in range(l_svs)]
        res_per_reali_syn_b = np.zeros((shape_stat, num_freq_exp, num_realizations))
        res_real = [np.zeros((shape_stat, total_realizations, num_freq_exp)) for _ in range(l_sv)]
        res_real_syn = [np.zeros((shape_stat, total_realizations, num_freq_exp)) for _ in range(l_svs)]
        res_real_syn_b = np.zeros((shape_stat, total_realizations, num_freq_exp))

        # Suprathreshold spikes for neuron model
        ISI_tr_per_freq_i = [[[] for _ in range(num_freq_exp)] for _ in range(l_sv)]
        ISI_st_per_freq_i = [[[] for _ in range(num_freq_exp)] for _ in range(l_sv)]
        ISI_tr_per_freq_m = [[[] for _ in range(num_freq_exp)] for _ in range(l_sv)]
        ISI_st_per_freq_m = [[[] for _ in range(num_freq_exp)] for _ in range(l_sv)]
        ISI_tr_per_freq_e = [[[] for _ in range(num_freq_exp)] for _ in range(l_sv)]
        ISI_st_per_freq_e = [[[] for _ in range(num_freq_exp)] for _ in range(l_sv)]
        num_spike_tr_per_freq_i = [[0 for _ in range(num_freq_exp)] for _ in range(l_sv)]
        num_spike_st_per_freq_i = [[0 for _ in range(num_freq_exp)] for _ in range(l_sv)]
        num_spike_tr_per_freq_m = [[0 for _ in range(num_freq_exp)] for _ in range(l_sv)]
        num_spike_st_per_freq_m = [[0 for _ in range(num_freq_exp)] for _ in range(l_sv)]
        num_spike_tr_per_freq_e = [[0 for _ in range(num_freq_exp)] for _ in range(l_sv)]
        num_spike_st_per_freq_e = [[0 for _ in range(num_freq_exp)] for _ in range(l_sv)]

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

        # **************************************************************************************************************
        # PROFILING
        if profiling:
            profiler = cProfile.Profile()
            profiler.enable()
        # **************************************************************************************************************

        ini_loop_time = m_time()
        print("Ini big loop")
        realization = 0
        while realization < num_loop_realizations and soft_stop_cond:
            loop_time = m_time()
            t_tra_mid_win, t_tra_mid_win_syn = None, None

            # Building reference signal for constant and fixed rate changes
            i = num_freq_exp - 1
            # ****************************************************************************************
            # Figure for PhD dissertation: methodology - temporal filtering - stochastic input
            # fig_syn_filt = plt.figure(figsize=(8, 8))
            # fig_syn_filt.suptitle("Temporal responses of Short-term facilitation", c="black", alpha=0.7, fontsize=20)
            # c_a = 3
            # ax = [fig_syn_filt.add_subplot(4, 1, i) for i in [1, 2, 3, 4]]
            # ****************************************************************************************

            while i >= 0:  # while i < num_freq_exp:
                loop_experiments = m_time()

                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # Input creation
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
                                                 num_realizations=aux_num_r, poisson=self.stoch_input, seeds=seeds1,
                                                 avoid_last_fast_spike_det=not self.stoch_input)
                # ISIs, histograms = inter_spike_intervals(ref_signals, 1 / sfreq, 1e-3)
                # histograms[0][1] *= 1000
                # plot_isi_histogram(histograms, 0)
                cons_aux = simple_spike_train(sfreq, proportional_changes[i], int(L / num_changes_rate),
                                              num_realizations=aux_num_r, poisson=stoch_input, seeds=seeds2,
                                              avoid_last_fast_spike_det=not self.stoch_input)
                fix_aux = simple_spike_train(sfreq, constant_changes[i], int(L / num_changes_rate),
                                             num_realizations=aux_num_r, poisson=stoch_input, seeds=seeds3,
                                             avoid_last_fast_spike_det=not self.stoch_input)

                cons_input = np.concatenate((ref_signals, cons_aux, ref_signals), axis=1)
                fix_input = np.concatenate((ref_signals, fix_aux, ref_signals), axis=1)

                # ******************************************************************************************************
                """
                # Plotting example of input patter
                rate_schema = np.concatenate((np.ones(int(L / 3)) * f_vector[i], np.ones(int(L / 3)) *
                                              proportional_changes[i], np.ones(int(L / 3)) * f_vector[i]))
                fig_gc_input = plot_gc_prop_input_example(time_vector, 1 / self.sfreq, 0, rate_schema, cons_input[0, :])
                path_save = self.folder_plots + file_name + '_input_sample_prop.png'
                fig_gc_input.savefig(path_save, format='png')
                # """
                # ******************************************************************************************************

                # Avoiding spikes in t==0
                cons_input[:, 0], fix_input[:, 0] = 0, 0
                if not stoch_input:
                    cons_input[:, 1], fix_input[:, 1] = 1, 1
                else:
                    for neuron in range(cons_input.shape[0]):
                        cons_input[neuron, neuron + 1] = 1
                        fix_input[neuron, neuron + 1] = 1
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # Running STP model
                if dyn_synapse:
                    # Reseting initial conditions
                    self.stp_prop.set_initial_conditions()
                    self.neuron_prop.set_simulation_params(sim_params, seeds1[0])
                    self.stp_fix.set_initial_conditions()
                    self.neuron_fix.set_simulation_params(sim_params, seeds1[0])
                    # Running the models
                    st_prior_ = None

                    # Getting masks of interspike intervals (to compute spike responses for each state variable)
                    spike_mask = detect_spikes(cons_input)
                    edges_syn = spike_edges_from_mask(spike_mask)
                    syn_spike_masks = [build_interval_masks_from_edges(edges_syn[s], L)
                                       for s in range(self.stp_prop.n_syn)]

                    # Running synapse-neuron model
                    model_stp_parallel(self.stp_prop, self.neuron_prop, stp_params, cons_input, n_seeds, n_noise,
                                       rate_input=f_vector[i], st_prior=st_prior_)

                    # Computing output spike events
                    # self.stp_prop.compute_output_spike_events(syn_spike_masks, edges_syn)
                    # self.neuron_prop.compute_output_spike_events(syn_spike_masks, edges_syn)

                    # Computing inter-spike responses for state variables of neurons and synapses
                    state_variables = self.stp_prop.get_output_state_variables()
                    a = self.compute_spike_outputs_vectorized(state_variables, edges_syn, self.stp_prop.operators_sv,
                                                              self.stp_prop.arg_operators_sv, 1 / self.sfreq)
                    (self.stp_prop.output_spike_events, self.stp_prop.output_spike_events_tonic,
                     self.stp_prop.ind_spike_events, self.stp_prop.ind_spike_events_tonic,
                     self.stp_prop.time_spike_events) = a

                    state_variables = self.neuron_prop.get_output_state_variables()
                    a = self.compute_spike_outputs_vectorized(state_variables, edges_syn, self.neuron_prop.operators_sv,
                                                              self.neuron_prop.arg_operators_sv, 1 / self.sfreq)
                    (self.neuron_prop.output_spike_events, self.neuron_prop.output_spike_events_tonic,
                     self.neuron_prop.ind_spike_events, self.neuron_prop.ind_spike_events_tonic,
                     self.neuron_prop.time_spike_events) = a

                    """
                    for s in range(self.stp_prop.n_syn):
                        s_mask = syn_spike_masks[s]  # [num_masks, L]
                        s_var_syn = state_variables[:, s, :]  # [num state variables, L]
                        # Expand s_mask to (num_state_vars, n_intervals, L)
                        n_intervals = s_mask.shape[0]
                        mask_3d = np.broadcast_to(s_mask[np.newaxis, :, :], (num_state_vars, n_intervals, L))

                        # Expand s_var_syn to (num_state_vars, n_intervals, L)
                        var_3d = np.broadcast_to(s_var_syn[:, np.newaxis, :], (num_state_vars, n_intervals, L))

                        # Element-wise multiplication
                        masked_var_3d = var_3d * mask_3d  # still (num_state_vars, n_intervals, L)

                        # Applying operators
                        per_state_variable = np.array([op(xi) for xi, op in zip(masked_var_3d, operators_sv)])
                    # """
                    # model_stp_parallel(self.stp_fix, self.lif_fix, stp_params, fix_input)
                else:
                    # Reseting initial conditions
                    self.neuron_prop.set_simulation_params(sim_params)
                    # lif_fix.set_simulation_params(sim_params)
                    # Running the models
                    static_synapse(self.neuron_prop, cons_input, 9e0)  # , 0.0125e-6)
                    # static_synapse(lif_fix, fix_input, 9e0)  # , 0.0125e-6)
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # STATE VARIABLES OF THE NEURON
                # For suprathreshold calculations
                t_spikes_gen = [np.array(self.neuron_prop.time_spikes_generated[k]) / self.sfreq
                                for k in range(self.neuron_prop.n_neurons)]
                ind_spikes_gen = [np.array(self.neuron_prop.time_spikes_generated[k])
                                  for k in range(self.neuron_prop.n_neurons)]

                sv = 0
                for k, v in self.neuron_prop.get_state_variables().items():
                    # getting transition time for rate of proportional change if possible
                    aux_cond = np.where(proportional_changes[i] <= f_vector)
                    if len(aux_cond[0]) > 0:
                        aux_i = aux_cond[0][0]
                        t_tra_mid_win = np.max(t_tra[sv][aux_i])

                    # Auxiliar title
                    title_graph_ = title_graph + ", freq. %dHz" % f_vector[i] + ", " + k
                    # Compute statistical descriptors
                    # print(k)
                    signal_prop, signal_fix = v, self.neuron_fix.get_state_variables()[k]
                    signal_prop_spikes = np.copy(signal_prop)

                    # settting variable to check number of spikes generated by neuron
                    ind_neuron_spikes = None

                    # Clipping signals to avoid suprathreshold maxima
                    if k == 'v':  # if not plot_ind_figs and k == 'v':
                        signal_prop = np.clip(signal_prop, None, self.neuron_prop.params['V_threshold'][0])  # 0.0)
                        signal_fix = np.clip(signal_fix, None, self.neuron_prop.params['V_threshold'][0])  # 0.0)
                        ind_neuron_spikes = ind_spikes_gen

                    # Computing statistics of each window, for the whole win. and for the transition- and steady-states
                    a = aux_statistics_prop_cons(signal_prop, signal_fix, Le_time_win, None,
                                                 sim_params, [None, t_tra_mid_win, None], 1 / sfreq,
                                                 th_percentage=th_percentage, filtering=filtering, cutoff=cutoff,
                                                 title=title_graph_, det_in=True, det_r=f_vector[i],
                                                 ind_spikes_gen=ind_neuron_spikes)  # det_in=not self.stoch_input
                    if a[0].shape[1] != res_per_reali[sv].shape[2]:
                        assert a[0].shape[1] == res_per_reali[sv].shape[2], "not same shape"
                    res_per_reali[sv][:, i, :], t_tr_, tr_time_series_i, piw, pmw, pew, t_tr_filt, ISI, num_spikes = a
                    t_tra[sv][i].append(t_tr_)
                    # ISI = [st_pi, tr_pi, st_pm, tr_pm, st_pe, tr_pe]
                    # num_spikes = [st_pi, tr_pi, st_pm, tr_pm, st_pe, tr_pe]

                    # **************************************************************************************************
                    # For Suprathreshold regime
                    ISI_st_per_freq_i[sv][i] = ISI_st_per_freq_i[sv][i] + ISI[0]
                    ISI_tr_per_freq_i[sv][i] = ISI_tr_per_freq_i[sv][i] + ISI[1]
                    ISI_st_per_freq_m[sv][i] = ISI_st_per_freq_m[sv][i] + ISI[2]
                    ISI_tr_per_freq_m[sv][i] = ISI_tr_per_freq_m[sv][i] + ISI[3]
                    ISI_st_per_freq_e[sv][i] = ISI_st_per_freq_e[sv][i] + ISI[4]
                    ISI_tr_per_freq_e[sv][i] = ISI_tr_per_freq_e[sv][i] + ISI[5]
                    # For number of spikes
                    num_spike_st_per_freq_i[sv][i] += num_spikes[0]
                    num_spike_tr_per_freq_i[sv][i] += num_spikes[1]
                    num_spike_st_per_freq_m[sv][i] += num_spikes[2]
                    num_spike_tr_per_freq_m[sv][i] += num_spikes[3]
                    num_spike_st_per_freq_e[sv][i] += num_spikes[4]
                    num_spike_tr_per_freq_e[sv][i] += num_spikes[5]

                    # **************************************************************************************************

                    # Plotting individual figures if indicated
                    if plot_ind_figs:
                        t_tr = t_tr_[0]
                        path_save = self.folder_plots + file_name + '_' + str(f_vector[i]) + '_stat.png'
                        plot_gc_mem_potential_prop_fix(time_vector, i, signal_prop, signal_fix, t_tr, res_per_reali[sv],
                                                       title_graph_, max_t, path_save=path_save,
                                                       save_figs=self.save_figs,
                                                       y_lims_ind_plot=y_lims_ind_plot, plot_stats=True, plt_grid=False)
                    sv += 1
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # STATE VARIABLES OF SYNAPSES
                sv = 0
                for k, v in self.stp_prop.get_state_variables().items():
                    # print(k)
                    # getting transition time for rate of proportional change if possible
                    aux_cond = np.where(proportional_changes[i] <= f_vector)
                    if len(aux_cond[0]) > 0:
                        aux_i = aux_cond[0][0]
                        t_tra_mid_win_syn = np.max(t_tra_syn[sv][aux_i])

                    # Auxiliar title
                    title_graph_ = title_graph + ", freq. %dHz" % f_vector[i] + ", " + k
                    # Compute statistical descriptors
                    signal_prop, signal_fix = v, self.stp_prop.get_state_variables()[k]
                    # Computing statistics of each window for transition- and steady-states
                    a = aux_statistics_prop_cons(signal_prop, signal_fix, Le_time_win, None,
                                                 sim_params, [None, t_tra_mid_win_syn, None], 1 / sfreq,
                                                 th_percentage=th_percentage, filtering=filtering, cutoff=cutoff,
                                                 title=title_graph_, det_in=True, det_r=f_vector[i])
                    # det_in=not self.stoch_input

                    if a[0].shape[1] != res_per_reali_syn[sv].shape[2]:
                        assert a[0].shape[1] == res_per_reali_syn[sv].shape[2], "not same shape"
                    res_per_reali_syn[sv][:, i, :], t_tr_syn, tr_time_series_i, piw, pmw, pew, t_tr_filt, _, _ = a
                    t_tra_syn[sv][i].append(t_tr_syn)

                    # Plotting individual figures if indicated
                    if plot_ind_figs:
                        t_tr = t_tr_syn[0]
                        path_save = self.folder_plots + file_name + '_' + str(f_vector[i]) + '_stat.png'
                        plot_gc_mem_potential_prop_fix(time_vector, i, signal_prop, signal_fix, t_tr,
                                                       res_per_reali_syn[sv],
                                                       title_graph_, max_t, path_save=path_save,
                                                       save_figs=self.save_figs,
                                                       y_lims_ind_plot=y_lims_ind_plot, plot_stats=True, plt_grid=False)

                        # ****************************************************************************************
                        # Figure for PhD dissertation: methodology - temporal filtering - stochastic input
                        # plot_gc_stoch_input(time_vector, i, signal_prop, signal_fix, t_tr, res_per_reali_syn[sv],
                        #                     title_graph_, max_t, path_save=path_save, save_figs=self.save_figs,
                        #                     y_lims_ind_plot=y_lims_ind_plot, ref_rate=f_vector[i], dt=1 / self.sfreq,
                        #                     th_percentage=th_percentage, ax=ax[c_a])
                        # if c_a == 0: ax[c_a].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                        # if c_a == 3: ax[c_a].set_xlabel("Time (s)", color="gray", fontsize=14)
                        # c_a -= 1
                        # ****************************************************************************************
                    sv += 1
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # Computing Postsynaptic response (PSR) to compute H(PSR) -unconditional entropy-
                # Neuron contribution
                PSR_per_freq[i].append(self.neuron_prop.output_spike_events)
                spike_event_per_freq[i].append(self.neuron_prop.time_spike_events)
                # Synaptic contribution
                PSR_per_freq_syn[i].append(self.stp_prop.output_spike_events)
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

                # Final print of the loop
                print_time(m_time() - loop_experiments, file_name + ", Realisation " + str(realization) +
                           ", frequency " + str(f_vector[i]))
                i -= 1

            # ****************************************************************************************
            # Figure for PhD dissertation: methodology - temporal filtering - stochastic input
            # fig_syn_filt.tight_layout()  # pad=0.5, w_pad=1.0, h_pad=1.0)
            # path_save = self.folder_plots + file_name + '_temporal_response_stoc_input.png'
            # if self.save_figs: fig_syn_filt.savefig(path_save, format='png')
            # ****************************************************************************************

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Organising transitory and stationary statistical descriptors of 3 windows in final arrays
            for res_i in range(res_real[0].shape[0]):
                r = realization
                # Iterating through the state variables of the neuron
                for sv in range(l_sv):
                    res_real[sv][res_i, r * num_realizations:(r + 1) * num_realizations] = res_per_reali[sv][res_i, :].T
                # Iterating through the state variables of the synapse
                for sv in range(l_svs):
                    res_real_syn[sv][res_i, r * num_realizations:(r + 1) * num_realizations] = res_per_reali_syn[sv][
                        res_i, :].T
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            print_time(m_time() - loop_time, file_name + ", Realisation " + str(realization))

            realization += 1

        # **************************************************************************************************************
        # PROFILING
        if profiling:
            profiler.disable()
            pstats.Stats(profiler).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(50)
        # **************************************************************************************************************

        #
        if soft_stop_cond:
            # Obtaining time series of statistical descriptors for transition component of signals (ini, mid, end win)
            # self.tr_time_series = tr_time_series
            # st_tr_a, res = get_time_series_statistics_of_transitions(tr_time_series, f_vector, proportional_changes,
            #                                                          1/sfreq, th_percentage)
            # Saving extra variables
            # dr['stat_tSeries_transition'] = res
            # dr['stat_time_transition'] = st_tr_a

            # transition-state for neuron variables
            name_vars = dr['name_neuron_state_variables']
            for sv in range(l_sv):
                for i in range(num_freq_exp):
                    t_tra[sv][i] = np.ravel(t_tra[sv][i])
                # Organising data for final dictionary input
                aux_n = "_" + name_vars[sv]
                if name_vars[sv] == 'v': aux_n = ''
                dr['time_transition' + aux_n] = np.array(t_tra[sv]).T

            # transition-state for synaptic variables
            name_vars = dr['name_syn_state_variables']
            for sv in range(l_svs):
                for i in range(num_freq_exp):
                    t_tra_syn[sv][i] = np.ravel(t_tra_syn[sv][i])
                # Organising data for final dictionary input
                aux_n = "_" + name_vars[sv]
                dr['time_transition_syn' + aux_n] = np.array(t_tra_syn[sv]).T

                # For ampa if synapse is Doorn
                # if self.stp_prop.get_output().ndim == 3: t_tra_syn_b[i] = np.ravel(t_tra_syn_b[i])
            # t_tra = np.array(t_tra).T
            # t_tra_syn = np.array(t_tra_syn).T
            # if self.stp_prop.get_output().ndim == 3: t_tra_syn_b = np.array(t_tra_syn_b).T

            # Saving transition times of each window
            # dr['time_transition'] = t_tra
            # dr['time_transition_syn'] = t_tra_syn
            # if self.stp_prop.get_output().ndim == 3: dr['time_transition_syn_b'] = t_tra_syn_b

            # ##########################################################################################################
            # """
            # Getting information theory analysis

            # Names of state variables
            names_syn_sv = list(self.stp_prop.get_state_variables().keys())
            names_neu_sv = list(self.neuron_prop.get_state_variables().keys())

            # getting times of windows
            l_w = sim_params['max_t']
            i_w, m_w, e_w = int(l_w / 3), int(2 * l_w / 3), sim_params['max_t']

            # Auxiliar variables
            a = [[] for _ in range(num_freq_exp)]
            ISI_per_freq_iw_tr, ISI_per_freq_mw_tr, ISI_per_freq_ew_tr = a.copy(), a.copy(), a.copy()
            ISI_per_freq_iw_st, ISI_per_freq_mw_st, ISI_per_freq_ew_st = a.copy(), a.copy(), a.copy()
            PSR_per_freq_iw_tr, PSR_per_freq_mw_tr, PSR_per_freq_ew_tr = a.copy(), a.copy(), a.copy()
            PSR_per_freq_iw_st, PSR_per_freq_mw_st, PSR_per_freq_ew_st = a.copy(), a.copy(), a.copy()
            SV_neu_per_freq_iw_tr = [[[] for _ in range(num_freq_exp)] for _ in range(l_sv)]
            SV_neu_per_freq_mw_tr = [[[] for _ in range(num_freq_exp)] for _ in range(l_sv)]
            SV_neu_per_freq_ew_tr = [[[] for _ in range(num_freq_exp)] for _ in range(l_sv)]
            SV_neu_per_freq_iw_st = [[[] for _ in range(num_freq_exp)] for _ in range(l_sv)]
            SV_neu_per_freq_mw_st = [[[] for _ in range(num_freq_exp)] for _ in range(l_sv)]
            SV_neu_per_freq_ew_st = [[[] for _ in range(num_freq_exp)] for _ in range(l_sv)]

            # Per neuron state variable containers: one list per variable, each containing [iw, mw, ew]
            SV_syn_per_freq_iw_tr = [[[] for _ in range(num_freq_exp)] for _ in range(l_svs)]
            SV_syn_per_freq_mw_tr = [[[] for _ in range(num_freq_exp)] for _ in range(l_svs)]
            SV_syn_per_freq_ew_tr = [[[] for _ in range(num_freq_exp)] for _ in range(l_svs)]
            SV_syn_per_freq_iw_st = [[[] for _ in range(num_freq_exp)] for _ in range(l_svs)]
            SV_syn_per_freq_mw_st = [[[] for _ in range(num_freq_exp)] for _ in range(l_svs)]
            SV_syn_per_freq_ew_st = [[[] for _ in range(num_freq_exp)] for _ in range(l_svs)]

            # min-max of synaptic contributions
            min_neu, max_neu = [np.inf for _ in range(l_sv)], [-np.inf for _ in range(l_sv)]
            min_syn, max_syn = [np.inf for _ in range(l_svs)], [-np.inf for _ in range(l_svs)]
            # membrane potential threshold
            thr = self.neuron_prop.params['V_threshold'][0]

            # Iterating through general realizations
            realization = 0
            for realization in range(num_loop_realizations):
                # Iterating through frequencies
                i = num_freq_exp - 1
                for i in range(num_freq_exp):
                    f_ = f_vector[i]
                    # Iterating through specific realizations
                    for neuron_realization in range(num_realizations):
                        # Input spike events
                        ta = np.array(spike_event_per_freq[i][realization][neuron_realization])  # / sfreq
                        # Postsynaptic contribution to the neuron
                        PSR_aux = np.array(PSR_per_freq[i][realization][neuron_realization])  # (n_sv, spike_events)
                        # Postsynaptic contribution to the receptor
                        PSR_aux_syn = np.array(PSR_per_freq_syn[i][realization][neuron_realization])  # (n_syn,spi_ev)
                        # Auxiliar index to get correct rate
                        aux_ind = num_realizations * realization + neuron_realization

                        # Updating min-max of neuron contributions
                        max_ = np.max(PSR_aux, axis=0)
                        min_ = np.min(PSR_aux, axis=0)

                        # Updating min-max of synaptic contributions
                        max_s = np.max(PSR_aux_syn, axis=0)
                        min_s = np.min(PSR_aux_syn, axis=0)

                        # Iterating through neuron state variables
                        for sv in range(l_sv):
                            aux_n = "_" + names_neu_sv[sv]
                            if names_neu_sv[sv] == 'v': aux_n = ''
                            # Getting time of reaching steady-state for ini and end windows
                            tr_st_time = dr['time_transition' + aux_n][aux_ind, i]
                            # Getting time of reaching steady-state for mid window
                            tr_st_time_mw = tr_st_time  # In case there is no tr_st_time for rate of mid window
                            aux_cond = np.where(proportional_changes[i] <= f_vector)
                            if len(aux_cond[0]) > 0:
                                aux_i = aux_cond[0][0]
                                tr_st_time_mw = dr['time_transition' + aux_n][aux_ind, aux_i]
                            # Getting masks to separate transitory and stationary states for ini, mid and end windows
                            mask_iw_tr = (ta <= tr_st_time)
                            mask_iw_st = (ta >= tr_st_time) & (ta <= i_w)
                            mask_mw_tr = (ta >= i_w) & (ta <= i_w + tr_st_time_mw)
                            mask_mw_st = (ta >= i_w + tr_st_time_mw) & (ta <= m_w)
                            mask_ew_tr = (ta >= m_w) & (ta <= m_w + tr_st_time)
                            mask_ew_st = (ta >= m_w + tr_st_time)

                            # Computing input
                            if names_neu_sv[sv] == 'v':
                                # Separating ini, mid and end windows of spike events
                                spike_ev_iw_tr = np.append(ta[mask_iw_tr], tr_st_time)
                                spike_ev_iw_st = np.append(ta[mask_iw_st], i_w)
                                spike_ev_mw_tr = np.append(ta[mask_mw_tr], i_w + tr_st_time_mw)
                                spike_ev_mw_st = np.append(ta[mask_mw_st], m_w)
                                spike_ev_ew_tr = np.append(ta[mask_ew_tr], m_w + tr_st_time)
                                spike_ev_ew_st = np.append(ta[mask_ew_st], e_w)

                                # PRESYNAPTIC INPUT SPIKES
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

                            # STATE VARIABLE CONTRIBUTIONS
                            # Separating ini, mid and end windows of neuron PSR
                            PSR_iw_tr = PSR_aux[mask_iw_tr].T
                            PSR_iw_st = PSR_aux[mask_iw_st].T
                            PSR_mw_tr = PSR_aux[mask_mw_tr].T
                            PSR_mw_st = PSR_aux[mask_mw_st].T
                            PSR_ew_tr = PSR_aux[mask_ew_tr].T
                            PSR_ew_st = PSR_aux[mask_ew_st].T

                            # Auxiliar variables for PSRs
                            a, b, c, d, e, f = PSR_iw_tr, PSR_iw_st, PSR_mw_tr, PSR_mw_st, PSR_ew_tr, PSR_ew_st

                            # Updating state varibles of neurons
                            if len(a) == l_sv:
                                aux_l = list(a[sv])
                                if names_neu_sv[sv] == 'v': aux_l = list(a[sv][a[sv] < thr])  # subthreshold for v
                                if len(aux_l) > 0: SV_neu_per_freq_iw_tr[sv][i] = SV_neu_per_freq_iw_tr[sv][i] + aux_l
                            if len(b) == l_sv:
                                aux_l = list(b[sv])
                                if names_neu_sv[sv] == 'v': aux_l = list(b[sv][b[sv] < thr])  # subthreshold for v
                                if len(aux_l) > 0: SV_neu_per_freq_iw_st[sv][i] = SV_neu_per_freq_iw_st[sv][i] + aux_l
                            if len(c) == l_sv:
                                aux_l = list(c[sv])
                                if names_neu_sv[sv] == 'v': aux_l = list(c[sv][c[sv] < thr])  # subthreshold for v
                                if len(aux_l) > 0: SV_neu_per_freq_mw_tr[sv][i] = SV_neu_per_freq_mw_tr[sv][i] + aux_l
                            if len(d) == l_sv:
                                aux_l = list(d[sv])
                                if names_neu_sv[sv] == 'v': aux_l = list(d[sv][d[sv] < thr])  # subthreshold for v
                                if len(aux_l) > 0: SV_neu_per_freq_mw_st[sv][i] = SV_neu_per_freq_mw_st[sv][i] + aux_l
                            if len(e) == l_sv:
                                aux_l = list(e[sv])
                                if names_neu_sv[sv] == 'v': aux_l = list(e[sv][e[sv] < thr])  # subthreshold for v
                                if len(aux_l) > 0: SV_neu_per_freq_ew_tr[sv][i] = SV_neu_per_freq_ew_tr[sv][i] + aux_l
                            if len(f) == l_sv:
                                aux_l = list(f[sv])
                                if names_neu_sv[sv] == 'v': aux_l = list(f[sv][f[sv] < thr])  # subthreshold for v
                                if len(aux_l) > 0: SV_neu_per_freq_ew_st[sv][i] = SV_neu_per_freq_ew_st[sv][i] + aux_l
                            # Updating min-max of synaptic contributions
                            if max_[sv] > max_neu[sv]: max_neu[sv] = max_[sv]
                            if min_[sv] < min_neu[sv]: min_neu[sv] = min_[sv]

                        # Iterating through neuron state variables
                        for sv in range(l_svs):
                            aux_n = "_" + names_syn_sv[sv]
                            # Getting time of reaching steady-state for ini and end windows
                            tr_st_time = dr['time_transition_syn' + aux_n][aux_ind, i]
                            # Getting time of reaching steady-state for mid window
                            tr_st_time_mw = tr_st_time  # In case there is no tr_st_time for rate of mid window
                            aux_cond = np.where(proportional_changes[i] <= f_vector)
                            if len(aux_cond[0]) > 0:
                                aux_i = aux_cond[0][0]
                                tr_st_time_mw = dr['time_transition_syn' + aux_n][aux_ind, aux_i]

                            # Getting masks to separate transitory and stationary states for ini, mid and end windows
                            mask_iw_tr = (ta <= tr_st_time)
                            mask_iw_st = (ta >= tr_st_time) & (ta <= i_w)
                            mask_mw_tr = (ta >= i_w) & (ta <= i_w + tr_st_time_mw)
                            mask_mw_st = (ta >= i_w + tr_st_time_mw) & (ta <= m_w)
                            mask_ew_tr = (ta >= m_w) & (ta <= m_w + tr_st_time)
                            mask_ew_st = (ta >= m_w + tr_st_time)

                            # Separating ini, mid and end windows of synapse PSR
                            PSR_syn_iw_tr = PSR_aux_syn[mask_iw_tr].T
                            PSR_syn_iw_st = PSR_aux_syn[mask_iw_st].T
                            PSR_syn_mw_tr = PSR_aux_syn[mask_mw_tr].T
                            PSR_syn_mw_st = PSR_aux_syn[mask_mw_st].T
                            PSR_syn_ew_tr = PSR_aux_syn[mask_ew_tr].T
                            PSR_syn_ew_st = PSR_aux_syn[mask_ew_st].T

                            # Updating state varibles of synapses
                            if len(PSR_syn_iw_tr) == l_svs:
                                aux_l = list(PSR_syn_iw_tr[sv])
                                if len(aux_l) > 0: SV_syn_per_freq_iw_tr[sv][i] = SV_syn_per_freq_iw_tr[sv][i] + aux_l
                            if len(PSR_syn_iw_st) == l_svs:
                                aux_l = list(PSR_syn_iw_st[sv])
                                if len(aux_l) > 0: SV_syn_per_freq_iw_st[sv][i] = SV_syn_per_freq_iw_st[sv][i] + aux_l
                            if len(PSR_syn_mw_tr) == l_svs:
                                aux_l = list(PSR_syn_mw_tr[sv])
                                if len(aux_l) > 0: SV_syn_per_freq_mw_tr[sv][i] = SV_syn_per_freq_mw_tr[sv][i] + aux_l
                            if len(PSR_syn_mw_st) == l_svs:
                                aux_l = list(PSR_syn_mw_st[sv])
                                if len(aux_l) > 0: SV_syn_per_freq_mw_st[sv][i] = SV_syn_per_freq_mw_st[sv][i] + aux_l
                            if len(PSR_syn_ew_tr) == l_svs:
                                aux_l = list(PSR_syn_ew_tr[sv])
                                if len(aux_l) > 0: SV_syn_per_freq_ew_tr[sv][i] = SV_syn_per_freq_ew_tr[sv][i] + aux_l
                            if len(PSR_syn_ew_st) == l_svs:
                                aux_l = list(PSR_syn_ew_st[sv])
                                if len(aux_l) > 0: SV_syn_per_freq_ew_st[sv][i] = SV_syn_per_freq_ew_st[sv][i] + aux_l
                            # Updating min-max of synaptic contributions
                            if max_s[sv] > max_syn[sv]: max_syn[sv] = max_s[sv]
                            if min_s[sv] < min_syn[sv]: min_syn[sv] = min_s[sv]

                        """      
                        # Getting time of reaching steady-state for ini and end windows
                        tr_st_time = dr['time_transition'][aux_ind, i]
                        # Getting time of reaching steady-state for mid window
                        tr_st_time_mw = tr_st_time  # In case there is no tr_st_time for rate of mid window
                        aux_cond = np.where(proportional_changes[i] <= f_vector)
                        if len(aux_cond[0]) > 0:
                            aux_i = aux_cond[0][0]
                            tr_st_time_mw = dr['time_transition'][aux_ind, aux_i]
                        # Getting masks to separate transitory and stationary states for ini, mid and end windows
                        mask_iw_tr = (ta <= tr_st_time)
                        mask_iw_st = (ta >= tr_st_time) & (ta <= i_w)
                        mask_mw_tr = (ta >= i_w) & (ta <= i_w + tr_st_time_mw)
                        mask_mw_st = (ta >= i_w + tr_st_time_mw) & (ta <= m_w)
                        mask_ew_tr = (ta >= m_w) & (ta <= m_w + tr_st_time)
                        mask_ew_st = (ta >= m_w + tr_st_time)

                        # Getting masks to separate ini, mid and end windows
                        # mask_iniw = ta < i_w
                        # mask_endw = ta >= m_w
                        # mask_midw = np.logical_not(np.logical_xor(mask_iniw, mask_endw))

                        # Separating ini, mid and end windows of spike events
                        spike_ev_iw_tr = np.append(ta[mask_iw_tr], tr_st_time)
                        spike_ev_iw_st = np.append(ta[mask_iw_st], i_w)
                        spike_ev_mw_tr = np.append(ta[mask_mw_tr], i_w + tr_st_time_mw)
                        spike_ev_mw_st = np.append(ta[mask_mw_st], m_w)
                        spike_ev_ew_tr = np.append(ta[mask_ew_tr], m_w + tr_st_time)
                        spike_ev_ew_st = np.append(ta[mask_ew_st], e_w)

                        # PRESYNAPTIC INPUT SPIKES
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

                        # STATE VARIABLE CONTRIBUTIONS
                        # Separating ini, mid and end windows of neuron PSR
                        PSR_iw_tr = PSR_aux[mask_iw_tr].T
                        PSR_iw_st = PSR_aux[mask_iw_st].T
                        PSR_mw_tr = PSR_aux[mask_mw_tr].T
                        PSR_mw_st = PSR_aux[mask_mw_st].T
                        PSR_ew_tr = PSR_aux[mask_ew_tr].T
                        PSR_ew_st = PSR_aux[mask_ew_st].T

                        # Updating min-max of neuron contributions
                        max_ = np.max(PSR_aux, axis=0)
                        min_ = np.min(PSR_aux, axis=0)

                        # Updating state varibles of neurons
                        thr = self.neuron_prop.params['V_threshold'][0]
                        a, b, c, d, e, f = PSR_iw_tr, PSR_iw_st, PSR_mw_tr, PSR_mw_st, PSR_ew_tr, PSR_ew_st
                        for sv in range(l_sv):
                            # Not considering suptrathreshold components of neuronal PSRs of membrane potential
                            # if names_neu_sv[sv] == 'v':
                            #     a = PSR_iw_tr[np.where(PSR_iw_tr < thr)[0]]
                            #     b = PSR_iw_st[np.where(PSR_iw_st < thr)[0]]
                            #     c = PSR_mw_tr[np.where(PSR_mw_tr < thr)[0]]
                            #     d = PSR_mw_st[np.where(PSR_mw_st < thr)[0]]
                            #     e = PSR_ew_tr[np.where(PSR_ew_tr < thr)[0]]
                            #     f = PSR_ew_st[np.where(PSR_ew_st < thr)[0]]
                            # else:
                            # print(f'realization {realization}, rate {f_}, neu realization {neuron_realization},
                            # state variable {sv}')
                            if len(a) == l_sv:
                                aux_l = list(a[sv])
                                if names_neu_sv[sv] == 'v': aux_l = list(a[sv][a[sv] < thr])  # subthreshold for v
                                if len(aux_l) > 0: SV_neu_per_freq_iw_tr[sv][i] = SV_neu_per_freq_iw_tr[sv][i] + aux_l
                            if len(b) == l_sv:
                                aux_l = list(b[sv])
                                if names_neu_sv[sv] == 'v': aux_l = list(b[sv][b[sv] < thr])  # subthreshold for v
                                if len(aux_l) > 0: SV_neu_per_freq_iw_st[sv][i] = SV_neu_per_freq_iw_st[sv][i] + aux_l
                            if len(c) == l_sv:
                                aux_l = list(c[sv])
                                if names_neu_sv[sv] == 'v': aux_l = list(c[sv][c[sv] < thr])  # subthreshold for v
                                if len(aux_l) > 0: SV_neu_per_freq_mw_tr[sv][i] = SV_neu_per_freq_mw_tr[sv][i] + aux_l
                            if len(d) == l_sv:
                                aux_l = list(d[sv])
                                if names_neu_sv[sv] == 'v': aux_l = list(d[sv][d[sv] < thr])  # subthreshold for v
                                if len(aux_l) > 0: SV_neu_per_freq_mw_st[sv][i] = SV_neu_per_freq_mw_st[sv][i] + aux_l
                            if len(e) == l_sv:
                                aux_l = list(e[sv])
                                if names_neu_sv[sv] == 'v': aux_l = list(e[sv][e[sv] < thr])  # subthreshold for v
                                if len(aux_l) > 0: SV_neu_per_freq_ew_tr[sv][i] = SV_neu_per_freq_ew_tr[sv][i] + aux_l
                            if len(f) == l_sv:
                                aux_l = list(f[sv])
                                if names_neu_sv[sv] == 'v': aux_l = list(f[sv][f[sv] < thr])  # subthreshold for v
                                if len(aux_l) > 0: SV_neu_per_freq_ew_st[sv][i] = SV_neu_per_freq_ew_st[sv][i] + aux_l
                            # Updating min-max of synaptic contributions
                            if max_[sv] > max_neu[sv]: max_neu[sv] = max_[sv]
                            if min_[sv] < min_neu[sv]: min_neu[sv] = min_[sv]

                        # Separating ini, mid and end windows of synapse PSR
                        PSR_syn_iw_tr = PSR_aux_syn[mask_iw_tr].T
                        PSR_syn_iw_st = PSR_aux_syn[mask_iw_st].T
                        PSR_syn_mw_tr = PSR_aux_syn[mask_mw_tr].T
                        PSR_syn_mw_st = PSR_aux_syn[mask_mw_st].T
                        PSR_syn_ew_tr = PSR_aux_syn[mask_ew_tr].T
                        PSR_syn_ew_st = PSR_aux_syn[mask_ew_st].T
                        
                        # Updating min-max of synaptic contributions
                        max_ = np.max(PSR_aux_syn, axis=0)
                        min_ = np.min(PSR_aux_syn, axis=0)
                        # Updating state varibles of synapses
                        for sv in range(l_svs):
                            if len(PSR_syn_iw_tr) == l_svs:
                                aux_l = list(PSR_syn_iw_tr[sv])
                                if len(aux_l) > 0: SV_syn_per_freq_iw_tr[sv][i] = SV_syn_per_freq_iw_tr[sv][i] + aux_l
                            if len(PSR_syn_iw_st) == l_svs:
                                aux_l = list(PSR_syn_iw_st[sv])
                                if len(aux_l) > 0: SV_syn_per_freq_iw_st[sv][i] = SV_syn_per_freq_iw_st[sv][i] + aux_l
                            if len(PSR_syn_mw_tr) == l_svs:
                                aux_l = list(PSR_syn_mw_tr[sv])
                                if len(aux_l) > 0: SV_syn_per_freq_mw_tr[sv][i] = SV_syn_per_freq_mw_tr[sv][i] + aux_l
                            if len(PSR_syn_mw_st) == l_svs:
                                aux_l = list(PSR_syn_mw_st[sv])
                                if len(aux_l) > 0: SV_syn_per_freq_mw_st[sv][i] = SV_syn_per_freq_mw_st[sv][i] + aux_l
                            if len(PSR_syn_ew_tr) == l_svs:
                                aux_l = list(PSR_syn_ew_tr[sv])
                                if len(aux_l) > 0: SV_syn_per_freq_ew_tr[sv][i] = SV_syn_per_freq_ew_tr[sv][i] + aux_l
                            if len(PSR_syn_ew_st) == l_svs:
                                aux_l = list(PSR_syn_ew_st[sv])
                                if len(aux_l) > 0: SV_syn_per_freq_ew_st[sv][i] = SV_syn_per_freq_ew_st[sv][i] + aux_l
                            # Updating min-max of synaptic contributions
                            if max_[sv] > max_syn[sv]: max_syn[sv] = max_[sv]
                            if min_[sv] < min_syn[sv]: min_syn[sv] = min_[sv]
                        # """
            # Entropy variables
            # Getting information theory analysis
            # Entropy for input spikes (ts-tr based on neuron output)
            H_ISI_iw_tr, H_ISI_mw_tr, H_ISI_ew_tr = [], [], []
            H_ISI_iw_st, H_ISI_mw_st, H_ISI_ew_st = [], [], []
            # Entropy for neuronal output
            H_SV_neu_iw_tr, H_SV_neu_iw_st = [[] for _ in range(l_sv)], [[] for _ in range(l_sv)]
            H_SV_neu_mw_tr, H_SV_neu_mw_st = [[] for _ in range(l_sv)], [[] for _ in range(l_sv)]
            H_SV_neu_ew_tr, H_SV_neu_ew_st = [[] for _ in range(l_sv)], [[] for _ in range(l_sv)]
            # Entropy for synaptic output(s)
            H_SV_syn_iw_tr, H_SV_syn_iw_st = [[] for _ in range(l_svs)], [[] for _ in range(l_svs)]
            H_SV_syn_mw_tr, H_SV_syn_mw_st = [[] for _ in range(l_svs)], [[] for _ in range(l_svs)]
            H_SV_syn_ew_tr, H_SV_syn_ew_st = [[] for _ in range(l_svs)], [[] for _ in range(l_svs)]

            """
            # ******************************************************************************************************
            # ENTROPY COMPUTATION USING FREEDMAN DIACONIS RULE
            # Entropy for input spikes (ts-tr based on neuron output)
            Hfd_ISI_iw_tr, Hfd_ISI_mw_tr, Hfd_ISI_ew_tr = [], [], []
            Hfd_ISI_iw_st, Hfd_ISI_mw_st, Hfd_ISI_ew_st = [], [], []
            # Entropy for neuronal output
            Hfd_SV_neu_iw_tr, Hfd_SV_neu_iw_st = [[] for _ in range(l_sv)], [[] for _ in range(l_sv)]
            Hfd_SV_neu_mw_tr, Hfd_SV_neu_mw_st = [[] for _ in range(l_sv)], [[] for _ in range(l_sv)]
            Hfd_SV_neu_ew_tr, Hfd_SV_neu_ew_st = [[] for _ in range(l_sv)], [[] for _ in range(l_sv)]
            # Entropy for synaptic output(s)
            Hfd_SV_syn_iw_tr, Hfd_SV_syn_iw_st = [[] for _ in range(l_svs)], [[] for _ in range(l_svs)]
            Hfd_SV_syn_mw_tr, Hfd_SV_syn_mw_st = [[] for _ in range(l_svs)], [[] for _ in range(l_svs)]
            Hfd_SV_syn_ew_tr, Hfd_SV_syn_ew_st = [[] for _ in range(l_svs)], [[] for _ in range(l_svs)]
            # ******************************************************************************************************
            # ENTROPY COMPUTATION USING SCOTT RULE
            # Entropy for input spikes (ts-tr based on neuron output)
            Hs_ISI_iw_tr, Hs_ISI_mw_tr, Hs_ISI_ew_tr = [], [], []
            Hs_ISI_iw_st, Hs_ISI_mw_st, Hs_ISI_ew_st = [], [], []
            # Entropy for neuronal output
            Hs_SV_neu_iw_tr, Hs_SV_neu_iw_st = [[] for _ in range(l_sv)], [[] for _ in range(l_sv)]
            Hs_SV_neu_mw_tr, Hs_SV_neu_mw_st = [[] for _ in range(l_sv)], [[] for _ in range(l_sv)]
            Hs_SV_neu_ew_tr, Hs_SV_neu_ew_st = [[] for _ in range(l_sv)], [[] for _ in range(l_sv)]
            # Entropy for synaptic output(s)
            Hs_SV_syn_iw_tr, Hs_SV_syn_iw_st = [[] for _ in range(l_svs)], [[] for _ in range(l_svs)]
            Hs_SV_syn_mw_tr, Hs_SV_syn_mw_st = [[] for _ in range(l_svs)], [[] for _ in range(l_svs)]
            Hs_SV_syn_ew_tr, Hs_SV_syn_ew_st = [[] for _ in range(l_svs)], [[] for _ in range(l_svs)]
            # ******************************************************************************************************
            # ENTROPY COMPUTATION USING KNUTH RULE
            # Entropy for input spikes (ts-tr based on neuron output)
            Hk_ISI_iw_tr, Hk_ISI_mw_tr, Hk_ISI_ew_tr = [], [], []
            Hk_ISI_iw_st, Hk_ISI_mw_st, Hk_ISI_ew_st = [], [], []
            # Entropy for neuronal output
            Hk_SV_neu_iw_tr, Hk_SV_neu_iw_st = [[] for _ in range(l_sv)], [[] for _ in range(l_sv)]
            Hk_SV_neu_mw_tr, Hk_SV_neu_mw_st = [[] for _ in range(l_sv)], [[] for _ in range(l_sv)]
            Hk_SV_neu_ew_tr, Hk_SV_neu_ew_st = [[] for _ in range(l_sv)], [[] for _ in range(l_sv)]
            # Entropy for synaptic output(s)
            Hk_SV_syn_iw_tr, Hk_SV_syn_iw_st = [[] for _ in range(l_svs)], [[] for _ in range(l_svs)]
            Hk_SV_syn_mw_tr, Hk_SV_syn_mw_st = [[] for _ in range(l_svs)], [[] for _ in range(l_svs)]
            Hk_SV_syn_ew_tr, Hk_SV_syn_ew_st = [[] for _ in range(l_svs)], [[] for _ in range(l_svs)]
            # """

            # Histograms variables
            # Histograms and edges for input spikes (ts-tr based on neuron output)
            b_ISI_iw_tr, b_ISI_mw_tr, b_ISI_ew_tr = [], [], []
            b_ISI_iw_st, b_ISI_mw_st, b_ISI_ew_st = [], [], []
            e_ISI_iw_tr, e_ISI_mw_tr, e_ISI_ew_tr = [], [], []
            e_ISI_iw_st, e_ISI_mw_st, e_ISI_ew_st = [], [], []
            # Histograms for neuronal output
            b_SV_neu_iw_tr, b_SV_neu_iw_st = [[] for _ in range(l_sv)], [[] for _ in range(l_sv)]
            b_SV_neu_mw_tr, b_SV_neu_mw_st = [[] for _ in range(l_sv)], [[] for _ in range(l_sv)]
            b_SV_neu_ew_tr, b_SV_neu_ew_st = [[] for _ in range(l_sv)], [[] for _ in range(l_sv)]
            # Edges for neuronal output
            e_SV_neu_iw_tr, e_SV_neu_iw_st = [[] for _ in range(l_sv)], [[] for _ in range(l_sv)]
            e_SV_neu_mw_tr, e_SV_neu_mw_st = [[] for _ in range(l_sv)], [[] for _ in range(l_sv)]
            e_SV_neu_ew_tr, e_SV_neu_ew_st = [[] for _ in range(l_sv)], [[] for _ in range(l_sv)]
            # Histograms for synaptic output(s)
            b_SV_syn_iw_tr, b_SV_syn_iw_st = [[] for _ in range(l_svs)], [[] for _ in range(l_svs)]
            b_SV_syn_mw_tr, b_SV_syn_mw_st = [[] for _ in range(l_svs)], [[] for _ in range(l_svs)]
            b_SV_syn_ew_tr, b_SV_syn_ew_st = [[] for _ in range(l_svs)], [[] for _ in range(l_svs)]
            # Edges for synaptic output(s)
            e_SV_syn_iw_tr, e_SV_syn_iw_st = [[] for _ in range(l_svs)], [[] for _ in range(l_svs)]
            e_SV_syn_mw_tr, e_SV_syn_mw_st = [[] for _ in range(l_svs)], [[] for _ in range(l_svs)]
            e_SV_syn_ew_tr, e_SV_syn_ew_st = [[] for _ in range(l_svs)], [[] for _ in range(l_svs)]

            # conductance of synaptic contributions to neuron model
            max_syn_contr = self.neuron_prop.max_syn_cont()
            bin_size = [(min_neu[sv], max_neu[sv]) for sv in range(l_sv)]
            bin_size_syn = [(min_syn[sv], max_syn[sv]) for sv in range(l_svs)]
            
            # Plotting histograms 
            plt_f = False
            fig_n = [None for _ in range(l_sv)]
            fig_s = [None for _ in range(l_svs)]
            fig_I = None
            ax_I = [None for i in range(1, 7)]
            ax_n = [[None for i in range(1, 7)] for j in range(l_sv)]
            ax_s = [[None for i in range(1, 7)] for j in range(l_svs)]
            
            # Iterating through all rates
            for i in range(num_freq_exp):
                # Plotting histograms for each rate, state variable and window
                if plt_f:
                    if i == 0 or i == num_freq_exp - 1:
                        fig_I = plt.figure(figsize=(15, 3.6))  # 15, 9
                        ax_I = [fig_I.add_subplot(1, 6, i) for i in range(1, 7)]  # 25
                        fig_I.suptitle('Histograms and entropy for different binning rules for frequency %d, Input ' %
                                       (f_vector[i]), fontsize=20)
                        fig_n = [plt.figure(figsize=(15, 3.6)) for _ in range(l_sv)]  # 15, 9
                        ax_n = [[fig_n[j].add_subplot(1, 6, i) for i in range(1, 7)] for j in range(l_sv)]  # 25
                        aux_title = 'Histograms and entropy for different binning rules for frequency %d, var %s'
                        for n in range(l_sv): fig_n[n].suptitle(aux_title % (f_vector[i], names_neu_sv[n]), fontsize=20)
                        fig_s = [plt.figure(figsize=(15, 3.6)) for _ in range(l_svs)]  # 15, 9
                        aux_title = 'Histograms and entropy for different binning rules for frequency %d, var %s'
                        ax_s = [[fig_s[j].add_subplot(1, 6, i) for i in range(1, 7)] for j in range(l_svs)]  # 25
                        for n in range(l_svs): fig_s[n].suptitle(aux_title % (f_vector[i], names_syn_sv[n]),
                                                                 fontsize=20)
                # ******************************************************************************************************
                # ENTROPY COMPUTATION USING FIXED BIN SIZE
                # ******************************************************************************************************
                # Entropy calculation for input
                b_ = 0.1 / f_vector[i]  # 10% of T
                append_entropy_ad_hoc(H_ISI_iw_tr, b_ISI_iw_tr, e_ISI_iw_tr, ISI_per_freq_iw_tr[i],
                                      b_, plot=plt_f, ax=ax_I[0])
                append_entropy_ad_hoc(H_ISI_iw_st, b_ISI_iw_st, e_ISI_iw_st, ISI_per_freq_iw_st[i],
                                      b_, plot=plt_f, ax=ax_I[1])
                append_entropy_ad_hoc(H_ISI_ew_tr, b_ISI_ew_tr, e_ISI_ew_tr, ISI_per_freq_ew_tr[i],
                                      b_, plot=plt_f, ax=ax_I[4])
                append_entropy_ad_hoc(H_ISI_ew_st, b_ISI_ew_st, e_ISI_ew_st, ISI_per_freq_ew_st[i],
                                      b_, plot=plt_f, ax=ax_I[5])
                b_factor = 0.1 / proportional_changes[i]  # 10% of T
                append_entropy_ad_hoc(H_ISI_mw_tr, b_ISI_mw_tr, e_ISI_mw_tr, ISI_per_freq_mw_tr[i],
                                      b_, plot=plt_f, ax=ax_I[2])
                append_entropy_ad_hoc(H_ISI_mw_st, b_ISI_mw_tr, e_ISI_mw_tr, ISI_per_freq_mw_st[i],
                                      b_,  plot=plt_f, ax=ax_I[3])

                """
                # Entropy calculations for state variables of neuronal responses
                H_a = [H_SV_neu_iw_tr, H_SV_neu_iw_st, H_SV_neu_mw_tr, H_SV_neu_mw_st, H_SV_neu_ew_tr, H_SV_neu_ew_st]
                b_a = [b_SV_neu_iw_tr, b_SV_neu_iw_st, b_SV_neu_mw_tr, b_SV_neu_mw_st, b_SV_neu_ew_tr, b_SV_neu_ew_st]
                e_a = [e_SV_neu_iw_tr, e_SV_neu_iw_st, e_SV_neu_mw_tr, e_SV_neu_mw_st, e_SV_neu_ew_tr, e_SV_neu_ew_st]
                dat_a = [SV_neu_per_freq_iw_tr, SV_neu_per_freq_iw_st, SV_neu_per_freq_mw_tr, SV_neu_per_freq_mw_st, 
                SV_neu_per_freq_ew_tr, SV_neu_per_freq_ew_st]
                entropy_sv(H_a, b_a, e_a, l_sv, bin_size, names_neu_sv, dat_a, i, plt_f, ax_n)

                H_a = [H_SV_syn_iw_tr, H_SV_syn_iw_st, H_SV_syn_mw_tr, H_SV_syn_mw_st, H_SV_syn_ew_tr, H_SV_syn_ew_st]
                b_a = [b_SV_syn_iw_tr, b_SV_syn_iw_st, b_SV_syn_mw_tr, b_SV_syn_mw_st, b_SV_syn_ew_tr, b_SV_syn_ew_st]
                e_a = [e_SV_syn_iw_tr, e_SV_syn_iw_st, e_SV_syn_mw_tr, e_SV_syn_mw_st, e_SV_syn_ew_tr, e_SV_syn_ew_st]
                dat_a = [SV_syn_per_freq_iw_tr, SV_syn_per_freq_iw_st, SV_syn_per_freq_mw_tr, SV_syn_per_freq_mw_st, 
                SV_syn_per_freq_ew_tr, SV_syn_per_freq_ew_st]
                entropy_sv(H_a, b_a, e_a, l_svs, bin_size, names_syn_sv, dat_a, i, plt_f, ax_n)
                # """

                # """
                # Entropy calculations for state variables of neuronal responses
                for sv in range(l_sv):
                    r_d = bin_size[sv]
                    b_ = 0.01 * (bin_size[sv][1] - bin_size[sv][0])  # max - min
                    if names_neu_sv[sv] == 'v': b_ = 0.1e-3
                    append_entropy_ad_hoc(H_SV_neu_iw_tr[sv], b_SV_neu_iw_tr[sv], e_SV_neu_iw_tr[sv],
                                          SV_neu_per_freq_iw_tr[sv][i], b_, range_d=r_d, plot=plt_f, ax=ax_n[sv][0])
                    append_entropy_ad_hoc(H_SV_neu_iw_st[sv], b_SV_neu_iw_st[sv], e_SV_neu_iw_st[sv],
                                          SV_neu_per_freq_iw_st[sv][i], b_, range_d=r_d, plot=plt_f, ax=ax_n[sv][1])
                    append_entropy_ad_hoc(H_SV_neu_mw_tr[sv], b_SV_neu_mw_tr[sv], e_SV_neu_mw_tr[sv],
                                          SV_neu_per_freq_mw_tr[sv][i], b_, range_d=r_d, plot=plt_f, ax=ax_n[sv][2])
                    append_entropy_ad_hoc(H_SV_neu_mw_st[sv], b_SV_neu_mw_st[sv], e_SV_neu_mw_st[sv],
                                          SV_neu_per_freq_mw_st[sv][i], b_, range_d=r_d, plot=plt_f, ax=ax_n[sv][3])
                    append_entropy_ad_hoc(H_SV_neu_ew_tr[sv], b_SV_neu_ew_tr[sv], e_SV_neu_ew_tr[sv],
                                          SV_neu_per_freq_ew_tr[sv][i], b_, range_d=r_d, plot=plt_f, ax=ax_n[sv][4])
                    append_entropy_ad_hoc(H_SV_neu_ew_st[sv], b_SV_neu_ew_st[sv], e_SV_neu_ew_st[sv],
                                          SV_neu_per_freq_ew_st[sv][i], b_, range_d=r_d, plot=plt_f, ax=ax_n[sv][5])
                
                # Entropy calculations for state variables of synaptic responses
                for sv in range(l_svs):
                    b_ = 0.01 * (bin_size_syn[sv][1] - bin_size_syn[sv][0])  # max - min
                    r_d = bin_size_syn[sv]
                    append_entropy_ad_hoc(H_SV_syn_iw_tr[sv], b_SV_syn_iw_tr[sv], e_SV_syn_iw_tr[sv],
                                          SV_syn_per_freq_iw_tr[sv][i], b_, range_d=r_d, plot=plt_f, ax=ax_s[sv][0])
                    append_entropy_ad_hoc(H_SV_syn_iw_st[sv], b_SV_syn_iw_st[sv], e_SV_syn_iw_st[sv],
                                          SV_syn_per_freq_iw_st[sv][i], b_, range_d=r_d, plot=plt_f, ax=ax_s[sv][1])
                    append_entropy_ad_hoc(H_SV_syn_mw_tr[sv], b_SV_syn_mw_tr[sv], e_SV_syn_mw_tr[sv],
                                          SV_syn_per_freq_mw_tr[sv][i], b_, range_d=r_d, plot=plt_f, ax=ax_s[sv][2])
                    append_entropy_ad_hoc(H_SV_syn_mw_st[sv], b_SV_syn_mw_st[sv], e_SV_syn_mw_st[sv],
                                          SV_syn_per_freq_mw_st[sv][i], b_, range_d=r_d, plot=plt_f, ax=ax_s[sv][3])
                    append_entropy_ad_hoc(H_SV_syn_ew_tr[sv], b_SV_syn_ew_tr[sv], e_SV_syn_ew_tr[sv],
                                          SV_syn_per_freq_ew_tr[sv][i], b_, range_d=r_d, plot=plt_f, ax=ax_s[sv][4])
                    append_entropy_ad_hoc(H_SV_syn_ew_st[sv], b_SV_syn_ew_st[sv], e_SV_syn_ew_st[sv],
                                          SV_syn_per_freq_ew_st[sv][i], b_, range_d=r_d, plot=plt_f, ax=ax_s[sv][5])
                # """
                """
                # ******************************************************************************************************
                # ENTROPY COMPUTATION USING FREEDMAN DIACONIS RULE
                # ******************************************************************************************************
                # Entropy calculation for input
                
                append_entropy(Hfd_ISI_iw_tr, ISI_per_freq_iw_tr[i], method='fd', plot=plt_f, ax=ax_I[6])
                append_entropy(Hfd_ISI_iw_st, ISI_per_freq_iw_st[i], method='fd', plot=plt_f, ax=ax_I[7])
                append_entropy(Hfd_ISI_ew_tr, ISI_per_freq_ew_tr[i], method='fd', plot=plt_f, ax=ax_I[10])
                append_entropy(Hfd_ISI_ew_st, ISI_per_freq_ew_st[i], method='fd', plot=plt_f, ax=ax_I[11])
                append_entropy(Hfd_ISI_mw_tr, ISI_per_freq_mw_tr[i], method='fd', plot=plt_f, ax=ax_I[8])
                append_entropy(Hfd_ISI_mw_st, ISI_per_freq_mw_st[i], method='fd', plot=plt_f, ax=ax_I[9])

                # Entropy calculations for state variables of neuronal responses
                for sv in range(l_sv):
                    min_max = bin_size[sv]
                    append_entropy(Hfd_SV_neu_iw_tr[sv], SV_neu_per_freq_iw_tr[sv][i], range_d=min_max, method='fd', 
                    plot=plt_f, ax=ax_n[sv][6])
                    append_entropy(Hfd_SV_neu_iw_st[sv], SV_neu_per_freq_iw_st[sv][i], range_d=min_max, method='fd', 
                    plot=plt_f, ax=ax_n[sv][7])
                    append_entropy(Hfd_SV_neu_mw_tr[sv], SV_neu_per_freq_mw_tr[sv][i], range_d=min_max, method='fd', 
                    plot=plt_f, ax=ax_n[sv][8])
                    append_entropy(Hfd_SV_neu_mw_st[sv], SV_neu_per_freq_mw_st[sv][i], range_d=min_max, method='fd', 
                    plot=plt_f, ax=ax_n[sv][9])
                    append_entropy(Hfd_SV_neu_ew_tr[sv], SV_neu_per_freq_ew_tr[sv][i], range_d=min_max, method='fd', 
                    plot=plt_f, ax=ax_n[sv][10])
                    append_entropy(Hfd_SV_neu_ew_st[sv], SV_neu_per_freq_ew_st[sv][i], range_d=min_max, method='fd', 
                    plot=plt_f, ax=ax_n[sv][11])

                # Entropy calculations for state variables of synaptic responses
                for sv in range(l_svs):
                    if names_syn_sv[sv] != 'x_nmda':
                        min_max = bin_size_syn[sv]
                        append_entropy(Hfd_SV_syn_iw_tr[sv], SV_syn_per_freq_iw_tr[sv][i], range_d=min_max, 
                        method='fd', plot=plt_f, ax=ax_s[sv][6])
                        append_entropy(Hfd_SV_syn_iw_st[sv], SV_syn_per_freq_iw_st[sv][i], range_d=min_max, 
                        method='fd', plot=plt_f, ax=ax_s[sv][7])
                        append_entropy(Hfd_SV_syn_mw_tr[sv], SV_syn_per_freq_mw_tr[sv][i], range_d=min_max, 
                        method='fd', plot=plt_f, ax=ax_s[sv][8])
                        append_entropy(Hfd_SV_syn_mw_st[sv], SV_syn_per_freq_mw_st[sv][i], range_d=min_max, 
                        method='fd', plot=plt_f, ax=ax_s[sv][9])
                        append_entropy(Hfd_SV_syn_ew_tr[sv], SV_syn_per_freq_ew_tr[sv][i], range_d=min_max, 
                        method='fd', plot=plt_f, ax=ax_s[sv][10])
                        append_entropy(Hfd_SV_syn_ew_st[sv], SV_syn_per_freq_ew_st[sv][i], range_d=min_max, 
                        method='fd', plot=plt_f, ax=ax_s[sv][11])
                
                # ******************************************************************************************************
                # ENTROPY COMPUTATION USING SCOTT'S RULE
                # ******************************************************************************************************
                # Entropy calculation for input
                append_entropy(Hs_ISI_iw_tr, ISI_per_freq_iw_tr[i], method='scott', plot=plt_f, ax=ax_I[12])
                append_entropy(Hs_ISI_iw_st, ISI_per_freq_iw_st[i], method='scott', plot=plt_f, ax=ax_I[13])
                append_entropy(Hs_ISI_ew_tr, ISI_per_freq_ew_tr[i], method='scott', plot=plt_f, ax=ax_I[16])
                append_entropy(Hs_ISI_ew_st, ISI_per_freq_ew_st[i], method='scott', plot=plt_f, ax=ax_I[17])
                append_entropy(Hs_ISI_mw_tr, ISI_per_freq_mw_tr[i], method='scott', plot=plt_f, ax=ax_I[14])
                append_entropy(Hs_ISI_mw_st, ISI_per_freq_mw_st[i], method='scott', plot=plt_f, ax=ax_I[15])

                # Entropy calculations for state variables of neuronal responses
                for sv in range(l_sv):
                    min_max = bin_size[sv]
                    append_entropy(Hs_SV_neu_iw_tr[sv], SV_neu_per_freq_iw_tr[sv][i], range_d=min_max, 
                    method='scott', plot=plt_f, ax=ax_n[sv][12])
                    append_entropy(Hs_SV_neu_iw_st[sv], SV_neu_per_freq_iw_st[sv][i], range_d=min_max, 
                    method='scott', plot=plt_f, ax=ax_n[sv][13])
                    append_entropy(Hs_SV_neu_mw_tr[sv], SV_neu_per_freq_mw_tr[sv][i], range_d=min_max, 
                    method='scott', plot=plt_f, ax=ax_n[sv][14])
                    append_entropy(Hs_SV_neu_mw_st[sv], SV_neu_per_freq_mw_st[sv][i], range_d=min_max, 
                    method='scott', plot=plt_f, ax=ax_n[sv][15])
                    append_entropy(Hs_SV_neu_ew_tr[sv], SV_neu_per_freq_ew_tr[sv][i], range_d=min_max, 
                    method='scott', plot=plt_f, ax=ax_n[sv][16])
                    append_entropy(Hs_SV_neu_ew_st[sv], SV_neu_per_freq_ew_st[sv][i], range_d=min_max, 
                    method='scott', plot=plt_f, ax=ax_n[sv][17])

                # Entropy calculations for state variables of synaptic responses
                for sv in range(l_svs):
                    min_max = bin_size_syn[sv]
                    append_entropy(Hs_SV_syn_iw_tr[sv], SV_syn_per_freq_iw_tr[sv][i], range_d=min_max, 
                    method='scott', plot=plt_f, ax=ax_s[sv][12])
                    append_entropy(Hs_SV_syn_iw_st[sv], SV_syn_per_freq_iw_st[sv][i], range_d=min_max, 
                    method='scott', plot=plt_f, ax=ax_s[sv][13])
                    append_entropy(Hs_SV_syn_mw_tr[sv], SV_syn_per_freq_mw_tr[sv][i], range_d=min_max, 
                    method='scott', plot=plt_f, ax=ax_s[sv][14])
                    append_entropy(Hs_SV_syn_mw_st[sv], SV_syn_per_freq_mw_st[sv][i], range_d=min_max, 
                    method='scott', plot=plt_f, ax=ax_s[sv][15])
                    append_entropy(Hs_SV_syn_ew_tr[sv], SV_syn_per_freq_ew_tr[sv][i], range_d=min_max, 
                    method='scott', plot=plt_f, ax=ax_s[sv][16])
                    append_entropy(Hs_SV_syn_ew_st[sv], SV_syn_per_freq_ew_st[sv][i], range_d=min_max, 
                    method='scott', plot=plt_f, ax=ax_s[sv][17])
                # ******************************************************************************************************
                # ENTROPY COMPUTATION USING KNUTH'S RULE
                # ******************************************************************************************************
                # Entropy calculation for input
                append_entropy(Hk_ISI_iw_tr, ISI_per_freq_iw_tr[i], method='knuth', plot=plt_f, ax=ax_I[18])
                append_entropy(Hk_ISI_iw_st, ISI_per_freq_iw_st[i], method='knuth', plot=plt_f, ax=ax_I[19])
                append_entropy(Hk_ISI_ew_tr, ISI_per_freq_ew_tr[i], method='knuth', plot=plt_f, ax=ax_I[22])
                append_entropy(Hk_ISI_ew_st, ISI_per_freq_ew_st[i], method='knuth', plot=plt_f, ax=ax_I[23])
                append_entropy(Hk_ISI_mw_tr, ISI_per_freq_mw_tr[i], method='knuth', plot=plt_f, ax=ax_I[20])
                append_entropy(Hk_ISI_mw_st, ISI_per_freq_mw_st[i], method='knuth', plot=plt_f, ax=ax_I[21])

                # Entropy calculations for state variables of neuronal responses
                for sv in range(l_sv):
                    min_max = bin_size[sv]
                    append_entropy(Hk_SV_neu_iw_tr[sv], SV_neu_per_freq_iw_tr[sv][i], range_d=min_max, 
                    method='knuth', plot=plt_f, ax=ax_n[sv][18])
                    append_entropy(Hk_SV_neu_iw_st[sv], SV_neu_per_freq_iw_st[sv][i], range_d=min_max, 
                    method='knuth', plot=plt_f, ax=ax_n[sv][19])
                    append_entropy(Hk_SV_neu_mw_tr[sv], SV_neu_per_freq_mw_tr[sv][i], range_d=min_max, 
                    method='knuth', plot=plt_f, ax=ax_n[sv][20])
                    append_entropy(Hk_SV_neu_mw_st[sv], SV_neu_per_freq_mw_st[sv][i], range_d=min_max, 
                    method='knuth', plot=plt_f, ax=ax_n[sv][21])
                    append_entropy(Hk_SV_neu_ew_tr[sv], SV_neu_per_freq_ew_tr[sv][i], range_d=min_max, 
                    method='knuth', plot=plt_f, ax=ax_n[sv][22])
                    append_entropy(Hk_SV_neu_ew_st[sv], SV_neu_per_freq_ew_st[sv][i], range_d=min_max, 
                    method='knuth', plot=plt_f, ax=ax_n[sv][23])

                # Entropy calculations for state variables of synaptic responses
                for sv in range(l_svs):
                    min_max = bin_size_syn[sv]
                    append_entropy(Hk_SV_syn_iw_tr[sv], SV_syn_per_freq_iw_tr[sv][i], range_d=min_max, 
                    method='knuth', plot=plt_f, ax=ax_s[sv][18])
                    append_entropy(Hk_SV_syn_iw_st[sv], SV_syn_per_freq_iw_st[sv][i], range_d=min_max, 
                    method='knuth', plot=plt_f, ax=ax_s[sv][19])
                    append_entropy(Hk_SV_syn_mw_tr[sv], SV_syn_per_freq_mw_tr[sv][i], range_d=min_max, 
                    method='knuth', plot=plt_f, ax=ax_s[sv][20])
                    append_entropy(Hk_SV_syn_mw_st[sv], SV_syn_per_freq_mw_st[sv][i], range_d=min_max, 
                    method='knuth', plot=plt_f, ax=ax_s[sv][21])
                    append_entropy(Hk_SV_syn_ew_tr[sv], SV_syn_per_freq_ew_tr[sv][i], range_d=min_max, 
                    method='knuth', plot=plt_f, ax=ax_s[sv][22])
                    append_entropy(Hk_SV_syn_ew_st[sv], SV_syn_per_freq_ew_st[sv][i], range_d=min_max, 
                    method='knuth', plot=plt_f, ax=ax_s[sv][23])
                # """
                if plt_f:
                    for n in range(l_sv): fig_n[n].tight_layout()
                    for n in range(l_svs): fig_s[n].tight_layout()
                    fig_I.tight_layout()
                    if plt_f: plt_f = False
            # ******************************************************************************************************
            # ENTROPY COMPUTATION USING FIXED BIN SIZE
            # Entropies, histograms and edges - fixed bin size
            dr['H_ISI_tr'] = np.array([H_ISI_iw_tr, H_ISI_mw_tr, H_ISI_ew_tr])
            dr['H_ISI_st'] = np.array([H_ISI_iw_st, H_ISI_mw_st, H_ISI_ew_st])
            dr['hist_ISI_tr'] = [b_ISI_iw_tr, b_ISI_mw_tr, b_ISI_ew_tr]
            dr['hist_ISI_st'] = [b_ISI_iw_st, b_ISI_mw_st, b_ISI_ew_st]
            dr['edge_ISI_tr'] = [e_ISI_iw_tr, e_ISI_mw_tr, e_ISI_ew_tr]
            dr['edge_ISI_st'] = [e_ISI_iw_st, e_ISI_mw_st, e_ISI_ew_st]
            """
            # Entropies - binning freedman-diaconis rule
            dr['H_fd_ISI_tr'] = np.array([Hfd_ISI_iw_tr, Hfd_ISI_mw_tr, Hfd_ISI_ew_tr])
            dr['H_fd_ISI_st'] = np.array([Hfd_ISI_iw_st, Hfd_ISI_mw_st, Hfd_ISI_ew_st])
            # Entropies - binning scott's rule
            dr['H_s_ISI_tr'] = np.array([Hs_ISI_iw_tr, Hs_ISI_mw_tr, Hs_ISI_ew_tr])
            dr['H_s_ISI_st'] = np.array([Hs_ISI_iw_st, Hs_ISI_mw_st, Hs_ISI_ew_st])
            # Entropies - binning knuth's rule
            dr['H_k_ISI_tr'] = np.array([Hk_ISI_iw_tr, Hk_ISI_mw_tr, Hk_ISI_ew_tr])
            dr['H_k_ISI_st'] = np.array([Hk_ISI_iw_st, Hk_ISI_mw_st, Hk_ISI_ew_st])
            # """

            # State variables of synaptic responses
            for sv in range(l_sv):
                name_sv = names_neu_sv[sv]
                # Entropies - fixed bin size
                dr[f'H_{name_sv}_neu_tr'] = np.array([H_SV_neu_iw_tr[sv], H_SV_neu_mw_tr[sv], H_SV_neu_ew_tr[sv]])
                dr[f'H_{name_sv}_neu_st'] = np.array([H_SV_neu_iw_st[sv], H_SV_neu_mw_st[sv], H_SV_neu_ew_st[sv]])
                # Histograms - fixed bin size
                dr[f'hist_{name_sv}_neu_tr'] = [b_SV_neu_iw_tr[sv], b_SV_neu_mw_tr[sv], b_SV_neu_ew_tr[sv]]
                dr[f'hist_{name_sv}_neu_st'] = [b_SV_neu_iw_st[sv], b_SV_neu_mw_st[sv], b_SV_neu_ew_st[sv]]
                # Edges - fixed bin size
                dr[f'edge_{name_sv}_neu_tr'] = [e_SV_neu_iw_tr[sv], e_SV_neu_mw_tr[sv], e_SV_neu_ew_tr[sv]]
                dr[f'edge_{name_sv}_neu_st'] = [e_SV_neu_iw_st[sv], e_SV_neu_mw_st[sv], e_SV_neu_ew_st[sv]]
                # min-max limits
                dr[f'range_{name_sv}_neu'] = bin_size[sv]
                """
                # Entropies - binning freedman-diaconis rule
                dr[f'H_fd_{name_sv}_neu_tr'] = np.array(
                    [Hfd_SV_neu_iw_tr[sv], Hfd_SV_neu_mw_tr[sv], Hfd_SV_neu_ew_tr[sv]])
                dr[f'H_fd_{name_sv}_neu_st'] = np.array(
                    [Hfd_SV_neu_iw_st[sv], Hfd_SV_neu_mw_st[sv], Hfd_SV_neu_ew_st[sv]])
                # Entropies - binning scott's rule
                dr[f'H_k_{name_sv}_neu_tr'] = np.array([Hk_SV_neu_iw_tr[sv], Hk_SV_neu_mw_tr[sv], Hk_SV_neu_ew_tr[sv]])
                dr[f'H_k_{name_sv}_neu_st'] = np.array([Hk_SV_neu_iw_st[sv], Hk_SV_neu_mw_st[sv], Hk_SV_neu_ew_st[sv]])
                # Entropies - binning knuth's rule
                dr[f'H_s_{name_sv}_neu_tr'] = np.array([Hs_SV_neu_iw_tr[sv], Hs_SV_neu_mw_tr[sv], Hs_SV_neu_ew_tr[sv]])
                dr[f'H_s_{name_sv}_neu_st'] = np.array([Hs_SV_neu_iw_st[sv], Hs_SV_neu_mw_st[sv], Hs_SV_neu_ew_st[sv]])
                # """

            # State variables of synaptic responses
            for sv in range(l_svs):
                name_sv = names_syn_sv[sv]
                # Entropies - fixed bin size
                dr[f'H_{name_sv}_syn_tr'] = np.array([H_SV_syn_iw_tr[sv], H_SV_syn_mw_tr[sv], H_SV_syn_ew_tr[sv]])
                dr[f'H_{name_sv}_syn_st'] = np.array([H_SV_syn_iw_st[sv], H_SV_syn_mw_st[sv], H_SV_syn_ew_st[sv]])
                # Histograms - fixed bin size
                dr[f'hist_{name_sv}_syn_tr'] = [b_SV_syn_iw_tr[sv], b_SV_syn_mw_tr[sv], b_SV_syn_ew_tr[sv]]
                dr[f'hist_{name_sv}_syn_st'] = [b_SV_syn_iw_st[sv], b_SV_syn_mw_st[sv], b_SV_syn_ew_st[sv]]
                # Edges - fixed bin size
                dr[f'edge_{name_sv}_syn_tr'] = [e_SV_syn_iw_tr[sv], e_SV_syn_mw_tr[sv], e_SV_syn_ew_tr[sv]]
                dr[f'edge_{name_sv}_syn_st'] = [e_SV_syn_iw_st[sv], e_SV_syn_mw_st[sv], e_SV_syn_ew_st[sv]]
                # min-max limits
                dr[f'range_{name_sv}_syn'] = bin_size_syn[sv]
                """
                # Entropies - binning freedman-diaconis rule
                dr[f'H_fd_{name_sv}_syn_tr'] = np.array(
                    [Hfd_SV_syn_iw_tr[sv], Hfd_SV_syn_mw_tr[sv], Hfd_SV_syn_ew_tr[sv]])
                dr[f'H_fd_{name_sv}_syn_st'] = np.array(
                    [Hfd_SV_syn_iw_st[sv], Hfd_SV_syn_mw_st[sv], Hfd_SV_syn_ew_st[sv]])
                # Entropies - binning scott's rule
                dr[f'H_k_{name_sv}_syn_tr'] = np.array([Hk_SV_syn_iw_tr[sv], Hk_SV_syn_mw_tr[sv], Hk_SV_syn_ew_tr[sv]])
                dr[f'H_k_{name_sv}_syn_st'] = np.array([Hk_SV_syn_iw_st[sv], Hk_SV_syn_mw_st[sv], Hk_SV_syn_ew_st[sv]])
                # Entropies - binning knuth's rule
                dr[f'H_s_{name_sv}_syn_tr'] = np.array([Hs_SV_syn_iw_tr[sv], Hs_SV_syn_mw_tr[sv], Hs_SV_syn_ew_tr[sv]])
                dr[f'H_s_{name_sv}_syn_st'] = np.array([Hs_SV_syn_iw_st[sv], Hs_SV_syn_mw_st[sv], Hs_SV_syn_ew_st[sv]])
                # """
            # """
            # ##########################################################################################################

            # Saving varible to know if
            dr['neuron_noise'] = n_noise

        # Saving final dictionary if file does not exist
        if not os.path.isfile(folder_vars + file_name):

            # Updating final dictionary
            for nam in range(res_real[0].shape[0]):
                # Looping through state variables of neuron
                names_sv = list(self.neuron_prop.get_state_variables().keys())
                for sv in range(l_sv):
                    name_sv = names_sv[sv]
                    if name_sv == 'v':
                        dr[stat_list[nam]] = res_real[sv][nam, :]
                    else:
                        dr[name_sv + '_' + stat_list[nam]] = res_real[sv][nam, :]

                # For synapses
                names_sv = list(self.stp_prop.get_state_variables().keys())
                for sv in range(l_svs):
                    name_sv = names_sv[sv]
                    dr[name_sv + '_' + stat_list[nam]] = res_real_syn[sv][nam, :]
                # dr['syn_' + stat_list[nam]] = res_real_syn[nam, :]
                # if self.stp_prop.get_output().ndim == 3: dr['syn_b_' + stat_list[nam]] = res_real_syn_b[nam, :]

            # **********************************************************************************************************
            # For ISI of postsynaptic neuron in suprathreshold
            names_sv = list(self.neuron_prop.get_state_variables().keys())
            nam = res_real[0].shape[0]
            for sv in range(l_sv):
                name_sv = names_sv[sv] + '_'
                if name_sv == 'v_': name_sv = ''
                # For number of spikes
                dr[name_sv + stat_list[nam + 0]] = num_spike_st_per_freq_i[sv]
                dr[name_sv + stat_list[nam + 1]] = num_spike_st_per_freq_m[sv]
                dr[name_sv + stat_list[nam + 2]] = num_spike_st_per_freq_e[sv]
                dr[name_sv + stat_list[nam + 3]] = num_spike_tr_per_freq_i[sv]
                dr[name_sv + stat_list[nam + 4]] = num_spike_tr_per_freq_m[sv]
                dr[name_sv + stat_list[nam + 5]] = num_spike_tr_per_freq_e[sv]
                # For ISI
                dr[name_sv + stat_list[nam + 6]] = ISI_st_per_freq_i[sv]
                dr[name_sv + stat_list[nam + 7]] = ISI_st_per_freq_m[sv]
                dr[name_sv + stat_list[nam + 8]] = ISI_st_per_freq_e[sv]
                dr[name_sv + stat_list[nam + 9]] = ISI_tr_per_freq_i[sv]
                dr[name_sv + stat_list[nam + 10]] = ISI_tr_per_freq_m[sv]
                dr[name_sv + stat_list[nam + 11]] = ISI_tr_per_freq_e[sv]
            # **********************************************************************************************************

            if self.save_vars:
                saveObject(dr, file_name, folder_vars)

        # if experiment is forced, then update final dictionary to be returned
        if soft_stop_cond:
            # Updating final dictionary
            for nam in range(res_real[0].shape[0]):
                # Looping through state variables of neuron
                names_sv = list(self.neuron_prop.get_state_variables().keys())
                for sv in range(l_sv):
                    name_sv = names_sv[sv]
                    if name_sv == 'v':
                        dr[stat_list[nam]] = res_real[sv][nam, :]
                    else:
                        dr[name_sv + '_' + stat_list[nam]] = res_real[sv][nam, :]

                # For synapses
                names_sv = list(self.stp_prop.get_state_variables().keys())
                for sv in range(l_svs):
                    name_sv = names_sv[sv]
                    dr[name_sv + '_' + stat_list[nam]] = res_real_syn[sv][nam, :]

        print_time(m_time() - ini_loop_time, "Total big loop")

        return dr
