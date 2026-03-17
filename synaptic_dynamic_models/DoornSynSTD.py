from synaptic_dynamic_models.SynDynModel import SynDynModel
import numpy as np


class DoornSTD_model(SynDynModel):
    """
    # BE AWARE THAT I AM MULTIPLYING alpha_nmda BY 1e-1 TO MATCH BRIAN SIMULATIONS
    AMPA/NMDA synaptic dynamics with short-term depression (STD) from Doorn et al. 2024.
    Units: t [s], g [nS], I [pA]
    Implements Brian2 synapse model without STF/Asynchronous release (base case).
    """

    def __init__(self, n_syn=1):
        super().__init__(n_syn)

        # Model variables [n_syn, L]
        self.s_ampa = None  # AMPA conductance [unitless 0-1]
        self.s_nmda = None  # NMDA total conductance [unitless 0-1]
        self.x_nmda = None  # NMDA availability [unitless 0-1]
        self.x_d = None  # Depression variable [unitless 0-1]

        # Spike events
        self.s_ampa_spike_events = None
        self.s_nmda_spike_events = None
        self.x_nmda_spike_events = None
        self.x_d_spike_events = None
        self.s_ampa_steady_state = None
        self.s_nmda_steady_state = None
        self.x_nmda_steady_state = None
        self.x_d_steady_state = None

        # Derivative variables [n_syn, L]
        self.d_s_ampa = None
        self.d_s_nmda = None
        self.d_x_nmda = None
        self.d_x_d = None

        # Auxiliar variables
        self.E_ampa, self.tau_ampa = None, None
        self.E_nmda, self.tau_nmda_rise = None, None
        self.tau_nmda_decay, self.alpha_nmda = None, None
        self.tau_d, self.U, self.S = None, None, None

        # Parameters (converted to consistent units)
        self.params = {
            'tau_ampa': 0.00205042,  # 0.002,  # s (2 ms)
            'tau_nmda_rise': 0.002,  # s (2 ms)
            'tau_nmda_decay': 0.100,  # s (100 ms)
            'alpha_nmda': 0.5,  # Hz (0.5 kHz) # BE AWARE THAT I AM MULTIPLYING THIS TERM BY 1e-1 TO MATCH BRIAN SIM.
            'tau_d': 0.200,  # s (200 ms) - STD recovery
            'U': 0.2,  # unitless - STD release probability
            'S': 0.4  # unitless - overall strength
        }

        # Set model and simulation parameters
        self.set_model_params(self.params)
        self.set_simulation_params()

    def set_model_params(self, model_params):
        """Set synaptic parameters from dictionary.
        # BE AWARE THAT I AM MULTIPLYING alpha_nmda BY 1e-1 TO MATCH BRIAN SIMULATIONS"""
        assert isinstance(model_params, dict), 'params should be a dict'
        for key, value in model_params.items():
            if key in self.params.keys():
                self.params[key] = value

        self.tau_ampa, self.S = self.params['tau_ampa'], self.params['S']
        self.tau_nmda_decay, self.alpha_nmda = self.params['tau_nmda_decay'], self.params['alpha_nmda'] * 1e-1
        self.tau_d, self.U, self.tau_nmda_rise = self.params['tau_d'], self.params['U'], self.params['tau_nmda_rise']

    def set_initial_conditions(self, Input=None):
        """Initialize all state variables to steady-state."""
        # Model variables
        self.s_ampa = np.zeros((self.n_syn, self.L))
        self.s_nmda = np.zeros((self.n_syn, self.L))
        self.x_nmda = np.zeros((self.n_syn, self.L))
        self.x_d = np.zeros((self.n_syn, self.L))

        # Derivatives
        self.d_s_ampa = np.zeros((self.n_syn, self.L))
        self.d_s_nmda = np.zeros((self.n_syn, self.L))
        self.d_x_nmda = np.zeros((self.n_syn, self.L))
        self.d_x_d = np.zeros((self.n_syn, self.L))

        # Spike event tracking (inherited + model-specific)
        self.s_ampa_spike_events = []
        self.s_nmda_spike_events = []
        self.x_nmda_spike_events = []
        self.x_d_spike_events = []
        self.output_spike_events = []
        self.output_spike_events_tonic = []
        self.ind_spike_events = []
        self.ind_spike_events_tonic = []
        self.time_spike_events = []

        # Steady-state tracking
        self.s_ampa_steady_state = None
        self.s_nmda_steady_state = None
        self.x_nmda_steady_state = None
        self.x_d_steady_state = None
        self.output_steady_state = None
        self.efficacy = [0.0 for _ in range(self.n_syn)]
        self.efficacy_2 = [0.0 for _ in range(self.n_syn)]
        self.efficacy_3 = [0.0 for _ in range(self.n_syn)]
        self.t_steady_state = [0.0 for _ in range(self.n_syn)]
        self.t_max = [0.0 for _ in range(self.n_syn)]

        # Initial conditions (steady-state at rest)
        self.s_ampa[:, 0] = 0.0
        self.s_nmda[:, 0] = 0.0
        self.x_nmda[:, 0] = 0.0
        self.x_d[:, 0] = 1.0  # fully recovered

        if Input is None:
            self.Input = np.zeros((self.n_syn, self.L))
        else:
            self.Input = Input
        self.edge_detection = False

    def evaluate_model_euler(self, I_it, it):
        """Update synaptic state using explicit Euler. I_it: spike events [0/1]."""
        self.Input[:, it] = I_it

        # Update dynamics (order matters per Brian2)
        self.update_x_nmda(it)  # NMDA availability
        self.update_x_d(it)     # Depression variable
        self.update_s_nmda(it)  # NMDA conductance
        self.update_s_ampa(it)  # AMPA conductance

    def get_output(self):
        """Return synaptic conductances for neuron: [s_ampa, s_nmda_tot] shape (2, n_syn, L)."""
        # s_nmda_tot = w * S * x_d * s_nmda (per synapse, to be summed by user code)
        s_nmda_tot = self.S * self.x_d * self.s_nmda  # w=1 for now
        return np.stack([self.s_ampa, s_nmda_tot])   # [2, n_syn, L] unitless [0-1]

    def update_x_nmda(self, it):
        """NMDA availability: dx_nmda/dt = -x_nmda/tau_nmda_rise [clock-driven]."""
        dt = self.dt
        I_it = self.Input[:, it]

        if it == 0:
            dx = -self.x_nmda[:, 0] / self.tau_nmda_rise * dt + I_it
            self.d_x_nmda[:, 0] = dx
            self.x_nmda[:, 0] = self.x_nmda[:, 0] + dx
        else:
            dx = -self.x_nmda[:, it - 1] / self.tau_nmda_rise * dt + I_it
            self.d_x_nmda[:, it] = dx
            self.x_nmda[:, it] = self.x_nmda[:, it - 1] + dx

    def update_x_d(self, it):
        """Depression recovery: dx_d/dt = (1-x_d)/tau_d - U*x_d*spike [clock-driven]."""
        dt = self.dt
        I_it = self.Input[:, it]

        if it == 0:
            self.d_x_d[:, 0] = (dt / self.tau_d) * (1.0 - self.x_d[:, 0]) - self.U * self.x_d[:, 0] * I_it
            self.x_d[:, 0] = self.x_d[:, 0] + self.d_x_d[:, 0]
        else:
            dx = (dt / self.tau_d) * (1.0 - self.x_d[:, it - 1]) - self.U * self.x_d[:, it - 1] * I_it
            self.d_x_d[:, it] = dx
            self.x_d[:, it] = self.x_d[:, it - 1] + dx

    def update_s_ampa(self, it):
        """AMPA: ds_ampa/dt = -s_ampa/tau_ampa + w*S*x_d*spike [clock-driven]."""
        dt = self.dt
        w_S_xd = self.S * self.x_d[:, it]  # w=1 for now (distributed weights external)
        I_it = self.Input[:, it]

        if it == 0:
            ds = -self.s_ampa[:, 0] / self.tau_ampa * dt + w_S_xd * I_it
            self.d_s_ampa[:, 0] = ds
            self.s_ampa[:, 0] = self.s_ampa[:, 0] + ds
        else:
            ds = -self.s_ampa[:, it - 1] / self.tau_ampa * dt + w_S_xd * I_it
            self.d_s_ampa[:, it] = ds
            self.s_ampa[:, it] = self.s_ampa[:, it - 1] + ds

    def update_s_nmda(self, it):
        """NMDA: ds_nmda/dt = -s_nmda/tau_decay + alpha_nmda*x_nmda*(1-s_nmda) [clock-driven]."""
        dt = self.dt
        if it == 0:
            ds = -self.s_nmda[:, 0] / self.tau_nmda_decay * dt + \
                 self.alpha_nmda * self.x_nmda[:, 0] * (1.0 - self.s_nmda[:, 0])
            self.d_s_nmda[:, 0] = ds
            self.s_nmda[:, 0] = self.s_nmda[:, 0] + ds
        else:
            ds = -self.s_nmda[:, it - 1] / self.tau_nmda_decay * dt + \
                 self.alpha_nmda * self.x_nmda[:, it] * (1.0 - self.s_nmda[:, it - 1])
            self.d_s_nmda[:, it] = ds
            self.s_nmda[:, it] = self.s_nmda[:, it - 1] + ds

    def append_spike_event(self, t, output):
        """Store spike events for analysis (override parent)."""
        self.s_ampa_spike_events.append(self.s_ampa[:, t])
        self.s_nmda_spike_events.append(self.s_nmda[:, t])
        self.x_nmda_spike_events.append(self.x_nmda[:, t])
        self.x_d_spike_events.append(self.x_d[:, t])
        self.time_spike_events.append(t)

        if len(self.time_spike_events) > 1:
            spike_range = (self.time_spike_events[-2], self.time_spike_events[-1])
            self.compute_output_spike_event(spike_range, output)
