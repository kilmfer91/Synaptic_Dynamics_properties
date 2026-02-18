import numpy as np


class HH_AHP_model:
    """
    Hodgkin-Huxley type neuron with slow after-hyperpolarization (sAHP) current.
    Units: V [mV], t [ms], I [pA], g [nS], Cm [pF]
    """
    def __init__(self, n_neu=1):
        # Model parameters (all in physical units, scalars or arrays of length n_neu)
        self.Cm = None  # membrane capacitance [pF]
        self.g_na = None  # max Na conductance [nS]
        self.g_kd = None  # max K delayed rectifier conductance [nS]
        self.g_l = None  # leak conductance [nS]
        self.El = None  # leak reversal [mV]
        self.EK = None  # K reversal [mV]
        self.ENa = None  # Na reversal [mV]
        self.VT = None  # voltage shift for gating [mV]
        self.g_AHP = None  # max sAHP conductance [nS]
        self.E_AHP = None  # sAHP reversal [mV]
        self.tau_Ca = None  # Ca recovery time constant [ms]
        self.alpha_Ca = None  # Ca increment per spike [unitless]
        self.sigma = None  # noise std dev [mV]
        self.g_ampa = None  # maximal conductance of AMPA channels
        self.g_nmda = None  # maximal conductance of NMDA channels
        self.E_ampa = None  # Nernst potentials of AMPA synaptic channels
        self.E_nmda = None  # Nernst potentials of NMDA synaptic channels
        self.V_init = None
        self.V_reset = None
        self.V_threshold = None
        self.t_refractory = None

        # Auxiliary variables
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

        # State variables (all shape: [n_neurons, L])
        self.membrane_potential = None  # V [mV]
        self.m_gate = None  # Na activation [0-1]
        self.h_gate = None  # Na inactivation [0-1]
        self.n_gate = None  # K activation [0-1]
        # self.hp_gate = None  # slow inactivation [0-1]  # I doubt this is used!
        self.Ca = None  # sAHP calcium [unitless] # This Ca is gAHP in the paper "Breaking the burst"

        # Auxiliar variables
        self.alp_m = None
        self.bet_m = None
        self.alp_h = None
        self.bet_h = None
        self.alp_n = None
        self.bet_n = None
        self.I_Na = None
        self.I_K = None
        self.I_L = None
        self.I_AHP = None
        self.I_ampa = None
        self.I_nmda = None

        # Default parameters (converted to consistent units: mV, ms, pA, nS, pF)
        self.params = {
            'Cm': 6e-12,  # pF (from area=300 um², Cm=2 uF/cm²)
            'g_na': 240e-9,  # 80.0e-9,  # nS (1.6*50 mS/cm² * 300 um²)
            'g_kd': 19.5e-9,  # nS (1.3*5 mS/cm² * 300 um²)
            'g_l': 0.9e-9,  # nS (0.3 mS/cm² * 300 um²)
            'El': -39.2e-3,  # mV
            'EK': -80.0e-3,  # mV
            'ENa': 70.0e-3,  # mV
            'VT': -30.4e-3,  # mV
            'sigma': 6.0e-3,  # mV (noise std dev)
            'g_AHP': 5.0e-9,  # nS
            'E_AHP': -80.0e-3,  # mV (= EK)
            'g_ampa': 1.6e-9,  # nS
            'g_nmda': 0.4e-9,  # nS
            'E_ampa': 0e-3,  # mV
            'E_nmda': 0e-3,  # mV
            'tau_Ca': 8000.0e-3,  # ms
            'alpha_Ca': 0.00035,  # per spike
            'V_init': -39.0e-3,  # mV
            'V_reset': -39.0e-3,  # mV (no voltage reset, just refractory)
            'V_threshold': 0.0e-3,  # mV (from Brian2: V>0*mV)
            't_refractory': 2.0e-3,  # ms
        }
        self.sim_params = {'sfreq': 1000, 'max_t': 0.8}
        self.set_simulation_params()

    def set_model_params(self, model_params):
        """Set model parameters from dictionary (only known keys)."""
        assert isinstance(model_params, dict), 'params should be a dict'
        for key, value in model_params.items():
            if key in self.params.keys():
                self.params[key] = value

        # Assign to instance variables (support per-neuron heterogeneity)
        self.Cm = np.full(self.n_neurons, self.params['Cm'])
        self.g_na = np.full(self.n_neurons, self.params['g_na'])
        self.g_kd = np.full(self.n_neurons, self.params['g_kd'])
        self.g_l = np.full(self.n_neurons, self.params['g_l'])
        self.El = np.full(self.n_neurons, self.params['El'])
        self.EK = np.full(self.n_neurons, self.params['EK'])
        self.ENa = np.full(self.n_neurons, self.params['ENa'])
        self.VT = np.full(self.n_neurons, self.params['VT'])
        self.g_AHP = np.full(self.n_neurons, self.params['g_AHP'])
        self.E_AHP = np.full(self.n_neurons, self.params['E_AHP'])
        self.tau_Ca = np.full(self.n_neurons, self.params['tau_Ca'])
        self.alpha_Ca = np.full(self.n_neurons, self.params['alpha_Ca'])
        self.sigma = np.full(self.n_neurons, self.params['sigma'])
        self.g_ampa = np.full(self.n_neurons, self.params['g_ampa'])
        self.g_nmda = np.full(self.n_neurons, self.params['g_nmda'])
        self.E_ampa = np.full(self.n_neurons, self.params['E_ampa'])
        self.E_nmda = np.full(self.n_neurons, self.params['E_nmda'])
        self.V_init = np.full(self.n_neurons, self.params['V_init'])
        self.V_reset = np.full(self.n_neurons, self.params['V_reset'])
        self.V_threshold = np.full(self.n_neurons, self.params['V_threshold'])
        self.t_refractory = np.full(self.n_neurons, self.params['t_refractory'])
        self.t_r_counter = np.zeros(self.n_neurons)
        self.initialize_state_variables()

    def set_seed(self, seed):
        # Assign seed
        if seed is not None: self.seed = seed

    def set_simulation_params(self, sim_params=None, seed=None):
        """Set simulation parameters and allocate state arrays."""
        if sim_params is not None:
            assert isinstance(sim_params, dict), 'params should be a dict'
            for key, value in sim_params.items():
                if key in self.sim_params.keys():
                    self.sim_params[key] = value

        # Set seed
        self.set_seed(seed)

        # Time variables (ms)
        self.dt = 1.0 / self.sim_params['sfreq']  # ms
        self.time_vector = np.arange(0, self.sim_params['max_t'], self.dt)
        self.L = int(self.sim_params['sfreq'] * self.sim_params['max_t'])

        # Allocate state arrays [n_neurons, L]
        self.membrane_potential = np.zeros((self.n_neurons, self.L))
        self.m_gate = np.zeros((self.n_neurons, self.L))
        self.h_gate = np.zeros((self.n_neurons, self.L))
        self.n_gate = np.zeros((self.n_neurons, self.L))
        self.alp_m = np.zeros((self.n_neurons, self.L))
        self.bet_m = np.zeros((self.n_neurons, self.L))
        self.alp_h = np.zeros((self.n_neurons, self.L))
        self.bet_h = np.zeros((self.n_neurons, self.L))
        self.alp_n = np.zeros((self.n_neurons, self.L))
        self.bet_n = np.zeros((self.n_neurons, self.L))
        self.I_Na = np.zeros((self.n_neurons, self.L))
        self.I_K = np.zeros((self.n_neurons, self.L))
        self.I_L = np.zeros((self.n_neurons, self.L))
        self.I_AHP = np.zeros((self.n_neurons, self.L))
        self.I_ampa = np.zeros((self.n_neurons, self.L))
        self.I_nmda = np.zeros((self.n_neurons, self.L))
        # self.hp_gate = np.zeros((self.n_neurons, self.L))  # I doubt this is used!
        self.Ca = np.zeros((self.n_neurons, self.L))  # This Ca is gAHP in the paper "Breaking the burst"
        self.set_model_params(self.params)

        # Spike event tracking
        self.membrane_potential_events = [[] for _ in range(self.n_neurons)]
        self.output_spike_events = [[] for _ in range(self.n_neurons)]
        self.output_spike_events_tonic = [[] for _ in range(self.n_neurons)]
        self.ind_spike_events = [[] for _ in range(self.n_neurons)]
        self.ind_spike_events_tonic = [[] for _ in range(self.n_neurons)]
        self.time_spike_events = [[] for _ in range(self.n_neurons)]

    def initialize_state_variables(self):
        # Initialize state variables
        V = self.V_init
        self.membrane_potential[:, 0] = V
        alpha_m, beta_m = self.alpha_m(V), self.beta_m(V)
        self.m_gate[:, 0] = alpha_m / (alpha_m + beta_m)  # approximate steady-state values
        alpha_h, beta_h = self.alpha_h(V), self.beta_h(V)
        self.h_gate[:, 0] = alpha_h / (alpha_h + beta_h)
        alpha_n, beta_n = self.alpha_n(V), self.beta_n(V)
        self.n_gate[:, 0] = alpha_n / (alpha_n + beta_n)
        # self.hp_gate[:, 0] = 0.6  # I doubt this is used!
        self.Ca[:, 0] = 0.0  # This Ca is gAHP in the paper "Breaking the burst"

    def compute_steady_state(self, V):
        """Compute exact steady-state gating values at voltage V."""
        alpha_m = self.alpha_m(V)
        beta_m = self.beta_m(V)
        m_inf = alpha_m / (alpha_m + beta_m)

        alpha_h = self.alpha_h(V)
        beta_h = self.beta_h(V)
        h_inf = alpha_h / (alpha_h + beta_h)

        alpha_n = self.alpha_n(V)
        beta_n = self.beta_n(V)
        n_inf = alpha_n / (alpha_n + beta_n)

        return m_inf, h_inf, n_inf

    @staticmethod
    def exprel(x):
        x = np.array(x, dtype=float)
        out = np.empty_like(x)
        small = np.abs(x) < 1e-6  # threshold can be adjusted
        out[small] = 1.0
        out[~small] = (np.exp(x[~small]) - 1.0) / x[~small]
        return out

    def alpha_m(self, V):
        """Na activation rate [1/ms]. V in mV."""
        # alpha_m = 0.32*(mV**-1)*4*mV/exprel((13*mV-V+VT)/(4*mV))/ms : Hz
        return 0.32e3 * 4e-3 / self.exprel((13e-3 - V + self.VT) / 4e-3)  # [1/ms]

    def beta_m(self, V):
        """Na activation rate [1/ms]. V in mV."""
        # 0.28*(mV**-1)*5*mV/exprel((V-VT-40*mV)/(5*mV))/ms : Hz
        return 0.28e3 * 5e-3 / self.exprel((V - self.VT - 40e-3) / 5e-3)  # [1/ms]

    def alpha_h(self, V):
        """Na inactivation rate [1/ms]. V in mV."""
        # 0.128 * exp((17 * mV - V + VT) / (18 * mV)) / ms: Hz
        # return 0.128 * np.exp((17.0 - V + self.VT) / 18.0)  # 1/ms
        return 0.128 * np.exp(-(V - self.VT - 17e-3) / 18e-3)  # 1/ms

    def beta_h(self, V):
        """Na inactivation rate [1/ms]. V in mV."""
        # 4./(1+exp((40*mV-V+VT)/(5*mV)))/ms : Hz
        # return 4.0 / (1.0 + np.exp((40.0 - V + self.VT) / 5.0))  # 1/ms
        return 4 / (1 + np.exp(-(V - self.VT - 40e-3) / 5e-3))  # 1/ms

    def alpha_n(self, V):
        """K activation rate [1/ms]. V in mV."""
        # 0.032*(mV**-1)*5*mV/exprel((15*mV-V+VT)/(5*mV))/ms : Hz
        # return 0.032 * (5.0 / (np.exp((15.0 - V + self.VT) / 5.0) + 1e-6))  # Hz -> 1/ms
        # return (-0.032 * (V - self.VT - 15.))/(np.exp(-(V - self.VT - 15.)/5.) - 1.)  # Hz -> 1/ms
        return 0.032e3 * 5e-3 / self.exprel((15e-3 - V + self.VT) / 5e-3)  # [1/ms]

    def beta_n(self, V):
        """K activation rate [1/ms]. V in mV."""
        # .5*exp((10*mV-V+VT)/(40*mV))/ms : Hz
        # return 0.5 * np.exp((10.0 - V + self.VT) / 40.0)  # 1/ms
        return 0.5 * np.exp(-(V - self.VT - 10e-3) / 40e-3)  # 1/ms

    def update_state(self, it, seed=None, use_noise=False, *args):
        """
        Update neuron state using explicit Euler.
        it: current time step
        seed: random seed
        use_noise: whether to add noise to neuron state
        args:
            I_ext: [n_neurons] external current [pA]
            s_ampa_tot: [n_neurons] total AMPA conductance [unitless, summed across synapses]
            s_nmda_tot: [n_neurons] total NMDA conductance [unitless, summed across synapses]
        """
        # Extracting inputs
        s_ampa_tot = args[0][0]
        s_nmda_tot = args[0][1]
        I_ext = args[0][2]

        # Seed
        if seed is not None: np.random.seed(seed)

        if it == 0:
            V = self.V_init
            m = self.m_gate[:, 0]
            h = self.h_gate[:, 0]
            n = self.n_gate[:, 0]
            # hp = self.hp_gate[:, 0]  # I doubt this is used!
            Ca = self.Ca[:, 0]  # This Ca is gAHP in the paper "Breaking the burst"
        else:
            V = self.membrane_potential[:, it - 1]
            m = self.m_gate[:, it - 1]
            h = self.h_gate[:, it - 1]
            n = self.n_gate[:, it - 1]
            # hp = self.hp_gate[:, it - 1]  # I doubt this is used!
            Ca = self.Ca[:, it - 1]  # This Ca is gAHP in the paper "Breaking the burst"

        # Refractory period: clamp V to reset and decrement counter
        cond_counter = np.where(self.t_r_counter > 0)[0]
        if len(cond_counter) > 0:
            self.membrane_potential[cond_counter, it] = self.V_reset[cond_counter]
            V[cond_counter] = self.V_reset[cond_counter]
            self.t_r_counter[cond_counter] -= 1

        # Threshold crossing (only for non-refractory neurons)
        cond_n = np.where(self.t_r_counter == 0)[0]
        if it > 0 and len(cond_n) > 0:
            cond_threshold = cond_n[np.where(V[cond_n] >= self.V_threshold[cond_n])[0]]
        else:
            cond_threshold = np.array([])

        if len(cond_threshold) > 0:
            # Spike occurred: increment Ca, reset refractory counter
            Ca[cond_threshold] += self.alpha_Ca[cond_threshold]  # This Ca is gAHP in the paper "Breaking the burst"
            self.t_r_counter[cond_threshold] = self.t_refractory[cond_threshold] / self.dt

        # Gating variable rates
        const = 1e3
        alpha_m = self.alpha_m(V) * const  # For the units 1/ms
        beta_m = self.beta_m(V) * const    # For the units 1/ms
        alpha_h = self.alpha_h(V) * const  # For the units 1/ms
        beta_h = self.beta_h(V) * const    # For the units 1/ms
        alpha_n = self.alpha_n(V) * const  # For the units 1/ms
        beta_n = self.beta_n(V) * const    # For the units 1/ms
        self.alp_m[:, it] = alpha_m
        self.bet_m[:, it] = beta_m
        self.alp_h[:, it] = alpha_h
        self.bet_h[:, it] = beta_h
        self.alp_n[:, it] = alpha_n
        self.bet_n[:, it] = beta_n
        # alpha_hp = 0.128 * np.exp((17.0 - V + (self.VT * 1e3)) / 18.0)  # I doubt this is used!
        # beta_hp = 4.0 / (1.0 + np.exp((30.0 - V + (self.VT * 1e3)) / 5.0))  # I doubt this is used!

        # Euler integration (all in consistent units)
        dm = (alpha_m * (1.0 - m) - beta_m * m) * self.dt
        dh = (alpha_h * (1.0 - h) - beta_h * h) * self.dt
        dn = (alpha_n * (1.0 - n) - beta_n * n) * self.dt
        # dhp = (alpha_hp * (1.0 - hp) - beta_hp * hp) * self.dt  # I doubt this is used!
        dCa = -Ca / self.tau_Ca * self.dt  # This Ca is gAHP in the paper "Breaking the burst"

        # Membrane equation: C_m dV/dt = I_ext - I_ampa - I_nmda - I_Na - I_K - I_L + I_AHP + noise
        # Convert conductances nS -> S, currents pA -> A, Cm pF -> F for consistency
        I_Na = self.g_na * (m ** 3) * h * (V - self.ENa)  # * 1e-3  # fA (e-15) # * 1e12  # pA
        I_K = self.g_kd * (n ** 4) * (V - self.EK)  # * 1e-3  # aA (e-18) # 1e12
        I_L = self.g_l * (V - self.El)  # * 1e-3  # fA (e-15) # 1e12 # pA
        I_AHP = -self.g_AHP * Ca * (V - self.E_AHP)  # pA  # *********** sign of g_AHP

        # Synaptic currents
        I_ampa = self.g_ampa * s_ampa_tot * (V - self.E_ampa)  # pA
        mg_block = 1.0 / (1.0 + np.exp(-0.062 * V * 1e3) / 3.57)  # V in mV
        I_nmda = self.g_nmda * s_nmda_tot * (V - self.E_nmda) * mg_block  # pA

        # Monitoring the currents
        self.I_Na[:, it] = I_Na
        self.I_K[:, it] = I_K
        self.I_L[:, it] = I_L
        self.I_AHP[:, it] = I_AHP
        self.I_ampa[:, it] = I_ampa
        self.I_nmda[:, it] = I_nmda

        # Noise term (discretized): sigma * sqrt(2*g_L/C_m) * randn() / sqrt(dt)
        noise = 0
        if use_noise:
            noise_scale = self.sigma * np.sqrt(2.0 * self.g_l / self.Cm)
            noise = noise_scale * np.random.randn(self.n_neurons) / np.sqrt(self.dt)

        # dV = (I_ext + I_ampa + I_nmda - I_Na - I_K - I_L + I_AHP + noise) / self.Cm * self.dt  # *********
        dV = (noise + (I_ext - I_ampa - I_nmda - I_Na - I_K - I_L + I_AHP) / self.Cm) * self.dt

        # Store updated states
        self.membrane_potential[:, it] = V + dV
        self.m_gate[:, it] = m + dm
        self.h_gate[:, it] = h + dh
        self.n_gate[:, it] = n + dn
        # self.hp_gate[:, it] = hp + dhp  # I doubt this is used!
        self.Ca[:, it] = Ca + dCa

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
