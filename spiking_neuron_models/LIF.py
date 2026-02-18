import numpy as np


class LIF_model:
    def __init__(self, n_neu=1):

        # Model params
        self.g_L = None
        self.V_equilibrium = None
        self.tau_m = None
        self.V_init = None
        self.V_reset = None
        self.V_threshold = None
        self.t_refractory = None

        # Auxiliar vars
        self.t_r_counter = None
        self.dt = None
        self.time_vector = None
        self.L = None
        self.n_neurons = n_neu
        self.seed = None

        # Spike event tracking
        self.edge_detection = None
        self.membrane_potential_events = None
        self.time_spike_events = None
        self.output_spike_events = None
        self.output_spike_events_tonic = None
        self.ind_spike_events = None
        self.ind_spike_events_tonic = None

        # Output
        self.membrane_potential = None
        self.rec_spikes = [[] for _ in range(n_neu)]
        self.rec_spikes2 = np.expand_dims(np.zeros(n_neu), axis=1)  # [[] for _ in range(n_neu)]

        self.params = {'V_threshold': -55.0, 'V_reset': -75.0, 'tau_m': 10.0, 'g_L': 10.0, 'V_init': -75.0,
                       'V_equilibrium': -75.0, 't_refractory': 2.0, 'T': 400.0, 'dt': 0.1}

        self.sim_params = {'sfreq': 1000, 'max_t': 0.8}
        self.set_simulation_params()

    def set_model_params(self, model_params):
        assert isinstance(model_params, dict), 'params should be a dict'
        for key, value in model_params.items():
            if key in self.params.keys():
                self.params[key] = value

        self.g_L = self.params['g_L']
        self.V_equilibrium = self.params['V_equilibrium']
        self.tau_m = self.params['tau_m']
        self.V_init = self.params['V_init']
        self.V_reset = self.params['V_reset']
        self.V_threshold = self.params['V_threshold']
        self.t_refractory = self.params['t_refractory']
        self.t_r_counter = np.zeros(self.n_neurons)

    def set_seed(self, seed):
        # Assign seed
        if seed is not None: self.seed = seed
        np.random.seed(seed)

    def set_simulation_params(self, sim_params=None, seed=None):
        if sim_params is not None:
            assert isinstance(sim_params, dict), 'params should be a dict'
            for key, value in sim_params.items():
                if key in self.sim_params.keys():
                    self.sim_params[key] = value

        # Set seed
        # self.set_seed(seed)

        # time simulation variables
        self.dt = 1 / self.sim_params['sfreq']
        self.time_vector = np.arange(0, self.sim_params['max_t'], self.dt)
        self.L = int(self.sim_params['sfreq'] * self.sim_params['max_t'])

        # Output variables
        self.membrane_potential = np.zeros((self.n_neurons, self.L))
        self.membrane_potential[:, 0] = self.V_init

        # Spike event tracking
        self.membrane_potential_events = [[] for _ in range(self.n_neurons)]
        self.output_spike_events = [[] for _ in range(self.n_neurons)]
        self.output_spike_events_tonic = [[] for _ in range(self.n_neurons)]
        self.ind_spike_events = [[] for _ in range(self.n_neurons)]
        self.ind_spike_events_tonic = [[] for _ in range(self.n_neurons)]
        self.time_spike_events = [[] for _ in range(self.n_neurons)]

    def update_state(self, it, seed=None, use_noise=False, *args):
        """
        Update neuron state using explicit Euler.
        it: current time step
        seed: random seed
        use_noise: whether to use noise or not.
        args:
            I_input: Input current [pA]
        """
        # Seed
        if seed is not None: np.random.seed(seed)

        I_input = args[0][0]
        g_L = self.g_L
        E_L = self.V_equilibrium
        tau_m = self.tau_m
        dt = self.dt
        v = E_L
        cond_threshold = []
        if it > 0:
            v = self.membrane_potential[:, it - 1]

        # """
        # If the state of LIF is in refractory period, decrease counter and set membrane potential to reset
        cond_counter = np.where(self.t_r_counter > 0)[0]  # selecting neurons in the refractory period
        if len(cond_counter) > 0:
            self.membrane_potential[cond_counter, it] = self.V_reset[cond_counter]  #
            v[cond_counter] = self.V_reset[cond_counter]
            self.t_r_counter[cond_counter] -= 1

        # If the membrane potential reaches the threshold, then fire a spike, set membrane potential to reset and set
        # refractory time
        cond_n = np.where(self.t_r_counter == 0)[0]  # selecting neurons that are active
        if it > 0:
            # selecting neurons that pass the firing threshold
            # cond_threshold = np.where(self.membrane_potential[cond_n, it - 1] >= self.V_threshold[cond_n])[0]
            cond_threshold = np.where(self.membrane_potential[cond_n, it - 1] >= self.V_threshold)[0]

        if len(cond_threshold) > 0:
            # Computing neurons that fire a spike
            aux_spikes = np.zeros(self.n_neurons)
            aux_spikes[cond_threshold] = it
            """
            # Store the times (it) where spikes are fired for each neuron
            self.rec_spikes2 = np.concatenate((self.rec_spikes2, np.expand_dims(aux_spikes, axis=1)), axis=1)
            """
            v_thres = self.V_threshold[cond_n]
            self.membrane_potential[cond_threshold, it] = v_thres + np.abs(0.2 * v_thres)
            v[cond_threshold] = v_thres + np.abs(0.2 * v_thres)  # self.V_reset[cond_threshold]
            self.t_r_counter[cond_threshold] = self.t_refractory[cond_threshold] / self.dt

        # dv = (-(v - E_L) * dt + np.sum(I_input, axis=0) / g_L) / tau_m
        dv = (-(v - E_L) + np.sum(I_input, axis=0) / g_L) * (dt / tau_m)
        # if np.sum(I_input) == 0:
        #     print(it)
        # dv = (-(v - E_L) * dt + g_L / np.sum(I_input)) / tau_m
        self.membrane_potential[:, it] = dv + v
        """
        # Looping all neurons
        for neuron in range(self.n_neurons):
            # If the state of LIF is in refractory period, decrease counter and set membrane potential to reset
            if self.t_r_counter[neuron] > 0:
                self.membrane_potential[neuron, it] = self.V_reset[neuron]
                v = self.V_reset[neuron]
                self.t_r_counter[neuron] -= 1

            # If the membrane potential reaches the threshold, then fire a spike, set membrane potential to reset and 
            # set refractory time
            elif self.membrane_potential[neuron, it] >= self.V_threshold[neuron]:
                self.rec_spikes[neuron].append(it)
                self.membrane_potential[neuron, it] = self.V_reset[neuron]
                v = self.V_reset[neuron]
                self.t_r_counter[neuron] = self.t_refractory[neuron] / self.dt

            # Calculate the increment of the membrane potential
            # dv = (-(v - E_L) + (I_input / g_L)) * (dt / tau_m)
            # Calculate the increment of the membrane potential
            dv = (-g_L * (v - E_L) * dt + I_input) / (tau_m * g_L)
            self.membrane_potential[neuron, it] = dv[neuron] + v[neuron]

            # Update the membrane potential
            # v[it + 1] = dv + v[it]
        # """

    def detect_spike_event(self, t, Input, output):
        """
        Parameters
        ----------
        t
        Input
        output
        """
        # Detecting raising edges
        # When t is 0
        if t == 0:
            # Detecting raising edges
            self.edge_detection = np.where(Input[:, t] > 0.0)[0]
        else:
            # Edge detector
            self.edge_detection = Input[:, t] > Input[:, t - 1]

        if np.sum(self.edge_detection) > 0:
            self.append_spike_event(t, self.edge_detection, output)

    def append_spike_event(self, t, output):
        """Store spike events for analysis (override parent)."""
        self.membrane_potential_events.append(self.membrane_potential[:, t])
        self.time_spike_events.append(t)

        if len(self.time_spike_events) > 1:
            spike_range = (self.time_spike_events[-2], self.time_spike_events[-1])
            self.compute_output_spike_event(spike_range, output)

    def compute_output_spike_event(self, spike_range, output):
        # print("TM, append_spike_event(), spike range ", spike_range, " in time ", spike_range[0])
        assert isinstance(spike_range, tuple), "Param 'spike_range' must be a tuple"
        assert len(spike_range) == 2, "Param 'spike_range' must be a tuple of 2 values"
        assert isinstance(spike_range[0], int), "first element of param 'spike_range' must be integer"
        assert isinstance(spike_range[1], int), "second element of param 'spike_range' must be integer"
        assert spike_range[1] >= spike_range[0], "Param 'spike_range' must contain order elements"
        assert isinstance(output, np.ndarray), "Param 'output' must be a numpy array"
        assert len(output.shape) == 2, "Param 'output' must be a 2D-array"
        assert output.shape[1] >= spike_range[1], ("second element of param 'spike_range' must be less or equal than "
                                                   "the length of param 'output'")
        """
        if spike_range[1] == spike_range[0]:
            # phasic component of spiking responses
            self.output_spike_events.append(output[:, spike_range[0]])
            # tonic component of spiking responses
            self.output_spike_events_tonic.appen(output[:, 0])

            # Updating index of phasic and tonic spike event occurences
            self.ind_spike_events_tonic.append(spike_range[0] - 1)
            self.ind_spike_events.append(a + spike_range[0])

        else: # """
        # Tonic component of the spiking response
        self.output_spike_events_tonic.append(output[:, spike_range[0] - 1])

        # Excitatory response
        if np.sum(output) > 0:
            self.output_spike_events.append(np.max(output[:, spike_range[0]: spike_range[1]], axis=1))
            a = np.argmax(output[:, spike_range[0]: spike_range[1]], axis=1)
        # Inhibitory response
        else:
            self.output_spike_events.append(np.min(output[:, spike_range[0]: spike_range[1]], axis=1))
            a = np.argmin(output[:, spike_range[0]: spike_range[1]], axis=1)

        # Updating index of phasic and tonic spike event occurences
        self.ind_spike_events_tonic.append(spike_range[0] - 1)
        self.ind_spike_events.append(a + spike_range[0])

    def append_spike_event(self, t, activated_neurons, output, append_time=True):
        """Store spike events for analysis (override parent)."""
        neurons_with_input_event = np.array(range(self.n_neurons))[activated_neurons]
        for n in neurons_with_input_event:
            if append_time: self.membrane_potential_events[n].append(self.membrane_potential[n, t])
            if append_time: self.time_spike_events[n].append(t)

            if len(self.time_spike_events[n]) > 1:
                spike_range = (self.time_spike_events[n][-2], self.time_spike_events[n][-1])
                self.compute_output_spike_event(spike_range, n, output)

    def compute_output_spike_event(self, spike_range, n, output):
        """
        assert isinstance(spike_range, tuple), "Param 'spike_range' must be a tuple"
        assert len(spike_range) == 2, "Param 'spike_range' must be a tuple of 2 values"
        assert isinstance(spike_range[0], int), "first element of param 'spike_range' must be integer"
        assert isinstance(spike_range[1], int), "second element of param 'spike_range' must be integer"
        assert spike_range[1] >= spike_range[0], "Param 'spike_range' must contain order elements"
        assert isinstance(output, np.ndarray), "Param 'output' must be a numpy array"
        assert len(output.shape) == 2, "Param 'output' must be a 2D-array"
        assert output.shape[1] >= spike_range[1], ("second element of param 'spike_range' must be less or equal than "
                                                   "the length of param 'output'")

        if spike_range[1] == spike_range[0]:
            # phasic component of spiking responses
            self.output_spike_events.append(output[:, spike_range[0]])
            # tonic component of spiking responses
            self.output_spike_events_tonic.appen(output[:, 0])

            # Updating index of phasic and tonic spike event occurences
            self.ind_spike_events_tonic.append(spike_range[0] - 1)
            self.ind_spike_events.append(a + spike_range[0])

        else: # """
        # Tonic component of the spiking response
        self.output_spike_events_tonic[n].append(output[n, spike_range[0] - 1])

        # Excitatory response
        # if np.sum(output) > 0:
        self.output_spike_events[n].append(np.max(output[n, spike_range[0]: spike_range[1]]))
        a = np.argmax(output[n, spike_range[0]: spike_range[1]])
        # Inhibitory response
        # else:
        #     self.output_spike_events.append(np.min(output[:, spike_range[0]: spike_range[1]], axis=1))
        #     a = np.argmin(output[:, spike_range[0]: spike_range[1]], axis=1)

        # Updating index of phasic and tonic spike event occurences
        self.ind_spike_events_tonic[n].append(spike_range[0] - 1)
        self.ind_spike_events[n].append(a + spike_range[0])
