import numpy as np


class HH_Simple_model:
    """
    Adaptive Exponential Integrate-and-Fire (AdEx) neuron with sAHP from Doorn et al. 2024.
    Units: V [mV], t [ms], I [pA], g [nS], C [pF]
    """

    def __init__(self, n_neu=1):
        # Model parameters (all in physical units, scalars or arrays of length n_neu)
        self.C = None  # membrane capacitance [pF]
        self.gL = None  # leak conductance [nS]
        self.taum = None  # membrane time constant [ms]
        self.EL = None  # leak reversal [mV]
        self.VTL = None  # EIF rheobase threshold [mV]
        self.DeltaT = None  # EIF sharpness [mV]
        self.Vcut = None  # spike cutoff [mV]
        self.sigma = None  # noise std dev [mV]
        self.Vr = None  # reset potential [mV]
        self.g_AHP = None  # max sAHP conductance [nS]
        self.E_AHP = None  # sAHP reversal [mV]
        self.tau_Ca = None  # Ca recovery time constant [ms]
        self.alpha_Ca = None  # Ca increment per spike [unitless]
        self.g_syn = None  # synaptic conductance [nS]
        self.E_syn = None  # synaptic reversal [mV]
        self.tau_syn = None  # synaptic time constant [ms]
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

        # State variables (all shape: [n_neurons, L])
        self.membrane_potential = None  # V [mV]
        self.Ca = None  # sAHP calcium [unitless]
        self.s_syn = None  # synaptic conductance [unitless]
        self.x_d = None  # synaptic depression [unitless]

        # Auxiliar variables
        self.I_AHP = None
        self.I_syn = None

        # Default parameters (converted to consistent units: mV, ms, pA, nS, pF)
        self.params = {
            'C': 100e-12,  # pF
            'gL': 1.8e-9,  # nS
            'EL': -40.0e-3,  # mV
            'VTL': -20.0e-3,  # mV
            'DeltaT': 2.0e-3,  # mV
            'Vcut': 20.0e-3,  # mV
            'sigma': 8.0e-3,  # mV
            'Vr': -40.0e-3,  # mV
            'g_AHP': 100e-9,  # nS
            'E_AHP': -80.0e-3,  # mV
            'tau_Ca': 6000.0e-3,  # ms
            'alpha_Ca': 0.00035,  # per spike
            'g_syn': 6.0e-9,  # nS
            'E_syn': 0.0e-3,  # mV
            'tau_syn': 20.0e-3,  # ms
            'V_init': -40.0e-3,  # mV
            'V_reset': -40.0e-3,  # mV
            'V_threshold': 20.0e-3,  # mV
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
        self.C = np.full(self.n_neurons, self.params['C'])
        self.gL = np.full(self.n_neurons, self.params['gL'])
        self.EL = np.full(self.n_neurons, self.params['EL'])
        self.VTL = np.full(self.n_neurons, self.params['VTL'])
        self.DeltaT = np.full(self.n_neurons, self.params['DeltaT'])
        self.Vcut = np.full(self.n_neurons, self.params['Vcut'])
        self.sigma = np.full(self.n_neurons, self.params['sigma'])
        self.Vr = np.full(self.n_neurons, self.params['Vr'])
        self.g_AHP = np.full(self.n_neurons, self.params['g_AHP'])
        self.E_AHP = np.full(self.n_neurons, self.params['E_AHP'])
        self.tau_Ca = np.full(self.n_neurons, self.params['tau_Ca'])
        self.alpha_Ca = np.full(self.n_neurons, self.params['alpha_Ca'])
        self.g_syn = np.full(self.n_neurons, self.params['g_syn'])
        self.E_syn = np.full(self.n_neurons, self.params['E_syn'])
        self.tau_syn = np.full(self.n_neurons, self.params['tau_syn'])
        self.V_init = np.full(self.n_neurons, self.params['V_init'])
        self.V_reset = np.full(self.n_neurons, self.params['V_reset'])
        self.V_threshold = np.full(self.n_neurons, self.params['V_threshold'])
        self.t_refractory = np.full(self.n_neurons, self.params['t_refractory'])
        self.t_r_counter = np.zeros(self.n_neurons)
        self.taum = self.C * 1e-12 / (self.gL * 1e-9) * 1e3  # ms
        self.initialize_state_variables()

    def set_seed(self, seed):
        if seed is not None:
            self.seed = seed

    def set_simulation_params(self, sim_params=None, seed=None):
        """Set simulation parameters and allocate state arrays."""
        if sim_params is not None:
            assert isinstance(sim_params, dict), 'params should be a dict'
            for key, value in sim_params.items():
                if key in self.sim_params.keys():
                    self.sim_params[key] = value

        self.set_seed(seed)

        # Time variables (ms)
        self.dt = 1.0 / self.sim_params['sfreq']  # ms
        self.time_vector = np.arange(0, self.sim_params['max_t'], self.dt)
        self.L = int(self.sim_params['sfreq'] * self.sim_params['max_t'])

        # Allocate state arrays [n_neurons, L]
        self.membrane_potential = np.zeros((self.n_neurons, self.L))
        self.Ca = np.zeros((self.n_neurons, self.L))
        self.s_syn = np.zeros((self.n_neurons, self.L))
        self.x_d = np.zeros((self.n_neurons, self.L))
        self.I_AHP = np.zeros((self.n_neurons, self.L))
        self.I_syn = np.zeros((self.n_neurons, self.L))
        self.set_model_params(self.params)

    def initialize_state_variables(self):
        """Initialize state variables to steady-state."""
        self.membrane_potential[:, 0] = self.V_init
        self.Ca[:, 0] = 0.0
        self.s_syn[:, 0] = 0.0
        self.x_d[:, 0] = 1.0  # fully recovered

    def update_state(self, it, seed=None, *args):
        """
        Update neuron state using explicit Euler.
        it: current time step
        args:
            I_ext: [n_neurons] external current [pA]
            s_syn_tot: [n_neurons] total synaptic conductance [unitless, summed across synapses]
            x_d_tot: [n_neurons] total depression factor [unitless, averaged or summed]
        """
        # Extracting inputs
        s_syn_tot = args[0][0]  # total synaptic conductance from all synapses
        x_d_tot = args[0][1]  # total depression factor from all synapses
        I_ext = args[0][2]  # external current [pA]

        # Seed
        if seed is not None:
            np.random.seed(seed)

        if it == 0:
            V = self.V_init
            Ca = self.Ca[:, 0]
            s_syn = self.s_syn[:, 0]
            x_d = self.x_d[:, 0]
        else:
            V = self.membrane_potential[:, it - 1]
            Ca = self.Ca[:, it - 1]
            s_syn = self.s_syn[:, it - 1]
            x_d = self.x_d[:, it - 1]

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
            # Spike occurred: increment Ca, apply depression, reset refractory
            Ca[cond_threshold] += self.alpha_Ca[cond_threshold]
            x_d[cond_threshold] *= 0.8  # (1-U) with U=0.2
            self.t_r_counter[cond_threshold] = self.t_refractory[cond_threshold] / self.dt

        # Euler integration for state variables
        dCa = -Ca / self.tau_Ca * self.dt
        ds_syn = -s_syn / self.tau_syn * self.dt
        dx_d = (1.0 - x_d) / (self.gL * 1e-9 / self.C * 1e-12 * 1e3) * self.dt  # tau_d = 813 ms

        # AdEx currents (all in consistent units: pA, nS, mV, pF, ms)
        I_leak = self.gL * (self.EL - V) * 1e3  # pA
        I_AHP = -self.g_AHP * Ca * (V - self.E_AHP) * 1e3  # pA
        I_syn = self.g_syn * s_syn_tot * (V - self.E_syn) * 1e3  # pA

        # Monitor currents
        self.I_AHP[:, it] = I_AHP
        self.I_syn[:, it] = I_syn

        # Noise term: sigma * sqrt(2/tau_m) * randn() * sqrt(dt) [mV]
        noise_scale = self.sigma * np.sqrt(2.0 / self.taum) * np.sqrt(self.dt)
        noise = noise_scale * np.random.randn(self.n_neurons)

        # AdEx membrane equation: C dV/dt = I_leak + gL*DeltaT*exp((V-VT)/DeltaT) + I_ext + I_syn + I_AHP + noise
        exp_term = self.gL * self.DeltaT * np.exp((V - self.VTL) / self.DeltaT) * 1e3  # pA
        dV = (I_leak + exp_term + I_ext + I_syn + I_AHP) / (self.C * 1e-12) * 1e12 * self.dt + noise  # mV

        # Store updated states
        self.membrane_potential[:, it] = V + dV
        self.Ca[:, it] = Ca + dCa
        self.s_syn[:, it] = s_syn + ds_syn
        self.x_d[:, it] = x_d + dx_d
