from synaptic_dynamic_models.SynDynModel import SynDynModel
import numpy as np


class DoornAsyn_model(SynDynModel):
    """
    # BE AWARE THAT I AM MULTIPLYING alpha_nmda BY 1e-1 TO MATCH BRIAN SIMULATIONS
    AMPA/NMDA synaptic dynamics with short-term depression (STD) + asynchronous release
    from Doorn et al. 2024 (AsynchronousRelease=True case).
    Units: t [s], conductances unitless (0–1) mapped later to currents via neuron.
    Mirrors Brian2 qar stochastic update exactly.
    """

    def __init__(self, n_syn=1):
        super().__init__(n_syn)

        # Model variables [n_syn, L]
        self.s_ampa = None  # AMPA conductance [unitless 0-1]
        self.s_nmda = None  # NMDA total conductance [unitless 0-1]
        self.x_nmda = None  # NMDA availability [unitless 0-1]
        self.x_d = None  # Depression variable [unitless 0-1]
        self.qar = None  # Asynchronous release rate [Hz]
        self.uar = None  # Asynchronous facilitation [unitless]

        # Spike events
        self.s_ampa_spike_events = None
        self.s_nmda_spike_events = None
        self.x_nmda_spike_events = None
        self.x_d_spike_events = None
        self.qar_spike_events = None
        self.uar_spike_events = None
        self.s_ampa_steady_state = None
        self.s_nmda_steady_state = None
        self.x_nmda_steady_state = None
        self.x_d_steady_state = None
        self.qar_steady_state = None
        self.uar_steady_state = None

        # Derivative variables [n_syn, L]
        self.d_s_ampa = None
        self.d_s_nmda = None
        self.d_x_nmda = None
        self.d_x_d = None
        self.d_qar = None
        self.d_uar = None

        # Parameters (converted to consistent units)
        self.E_ampa, self.tau_ampa = None, None
        self.E_nmda, self.tau_nmda_rise = None, None
        self.tau_nmda_decay, self.alpha_nmda = None, None
        self.tau_d, self.U, self.S = None, None, None
        self.x0, self.tau_ar, self.Uar, self.Umax = None, None, None, None
        self.params = {
            'E_ampa': 0.0,  # mV
            'E_nmda': 0.0,  # mV
            'tau_ampa': 0.00205042,  # s (2 ms)
            'tau_nmda_rise': 0.002,  # s (2 ms) = taux_nmda
            'tau_nmda_decay': 0.100,  # s (100 ms) = taus_nmda
            'alpha_nmda': 0.05,  # Hz (0.5 kHz)
            'tau_d': 0.200,  # s (200 ms) - STD recovery
            'U': 0.2,  # unitless - STD release probability
            'S': 0.4,  # unitless - overall strength
            # Asynchronous parameters
            'x0': 5.0,  # quantum size
            'tau_ar': 700e-3,  # s (700 ms) - async recovery
            'Uar': 0.003,  # async facilitation increment per spike
            'Umax': 0.5e3  # Hz - async saturation level
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

        self.E_ampa = self.params['E_ampa']
        self.tau_ampa = self.params['tau_ampa']
        self.E_nmda = self.params['E_nmda']
        self.tau_nmda_rise = self.params['tau_nmda_rise']
        self.tau_nmda_decay = self.params['tau_nmda_decay']
        self.alpha_nmda = self.params['alpha_nmda'] * 1e-1
        self.tau_d = self.params['tau_d']
        self.U = self.params['U']
        self.S = self.params['S']
        self.x0 = self.params['x0']
        self.tau_ar = self.params['tau_ar']
        self.Uar = self.params['Uar']
        self.Umax = self.params['Umax']

    def set_initial_conditions(self, Input=None):
        """Initialize all state variables to steady-state."""
        # Model variables
        self.s_ampa = np.zeros((self.n_syn, self.L))
        self.s_nmda = np.zeros((self.n_syn, self.L))
        self.x_nmda = np.zeros((self.n_syn, self.L))
        self.x_d = np.zeros((self.n_syn, self.L))
        self.qar = np.zeros((self.n_syn, self.L))
        self.uar = np.zeros((self.n_syn, self.L))

        # Derivatives
        self.d_s_ampa = np.zeros((self.n_syn, self.L))
        self.d_s_nmda = np.zeros((self.n_syn, self.L))
        self.d_x_nmda = np.zeros((self.n_syn, self.L))
        self.d_x_d = np.zeros((self.n_syn, self.L))
        self.d_qar = np.zeros((self.n_syn, self.L))
        self.d_uar = np.zeros((self.n_syn, self.L))

        # Spike event tracking
        self.s_ampa_spike_events = [[] for _ in range(self.n_syn)]
        self.s_nmda_spike_events = [[] for _ in range(self.n_syn)]
        self.x_nmda_spike_events = [[] for _ in range(self.n_syn)]
        self.x_d_spike_events = [[] for _ in range(self.n_syn)]
        self.qar_spike_events = [[] for _ in range(self.n_syn)]
        self.uar_spike_events = [[] for _ in range(self.n_syn)]
        self.output_spike_events = [[] for _ in range(self.n_syn)]
        self.output_spike_events_tonic = [[] for _ in range(self.n_syn)]
        self.ind_spike_events = [[] for _ in range(self.n_syn)]
        self.ind_spike_events_tonic = [[] for _ in range(self.n_syn)]
        self.time_spike_events = [[] for _ in range(self.n_syn)]

        # Steady-state tracking
        self.s_ampa_steady_state = None
        self.s_nmda_steady_state = None
        self.x_nmda_steady_state = None
        self.x_d_steady_state = None
        self.qar_steady_state = None
        self.uar_steady_state = None
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
        self.qar[:, 0] = 0.0
        self.uar[:, 0] = 0.0

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
        self.update_uar(it)  # Async facilitation
        self.update_x_d(it)  # Depression (now subtracts qar)
        self.update_qar(it)  # Asynchronous rate (stochastic, uses current x_d)
        self.update_s_nmda(it)  # NMDA conductance (+ x0*qar term)
        self.update_s_ampa(it)  # AMPA conductance

    def update_x_nmda(self, it):
        """NMDA availability: dx_nmda/dt = -x_nmda/tau_nmda_rise + spike [clock-driven]."""
        dt = self.dt
        I_it = self.Input[:, it]
        if it == 0:
            dx = -self.x_nmda[:, 0] / self.tau_nmda_rise * dt + I_it
            self.d_x_nmda[:, 0] = dx
            self.x_nmda[:, 0] += dx
        else:
            dx = -self.x_nmda[:, it - 1] / self.tau_nmda_rise * dt + I_it
            self.d_x_nmda[:, it] = dx
            self.x_nmda[:, it] = self.x_nmda[:, it - 1] + dx

    def update_uar(self, it):
        """Async facilitation: duar/dt = -uar/tau_ar [clock-driven]."""
        dt = self.dt
        I_it = self.Input[:, it]

        if it == 0:
            du = -(dt / self.tau_ar) * self.uar[:, 0] + self.Uar * (self.Umax - self.uar[:, 0]) * I_it
            self.d_uar[:, 0] = du
            self.uar[:, 0] = self.uar[:, 0] + du
        else:
            du = -(dt / self.tau_ar) * self.uar[:, it - 1] + self.Uar * (self.Umax - self.uar[:, it - 1]) * I_it
            self.d_uar[:, it] = du
            self.uar[:, it] = self.uar[:, it - 1] + du

    def update_x_d(self, it):
        """Depression: dx_d/dt = (1-x_d)/tau_d - qar - U*x_d*spike [clock-driven]."""
        dt = self.dt
        I_it = self.Input[:, it]

        if it == 0:
            dqar = self.qar[:, 0]
            dx = (dt / self.tau_d) * (1.0 - self.x_d[:, 0]) - dqar * dt - self.U * self.x_d[:, 0] * I_it
            self.d_x_d[:, 0] = dx
            self.x_d[:, 0] += dx
        else:
            dqar = self.qar[:, it - 1]
            dx = (dt / self.tau_d) * (1.0 - self.x_d[:, it - 1]) - dqar * dt - self.U * self.x_d[:, it - 1] * I_it
            self.d_x_d[:, it] = dx
            self.x_d[:, it] = self.x_d[:, it - 1] + dx

    def update_qar(self, it):
        """Asynchronous release: qar = clip(randn()*sqrt(...)+..., 0, ...) / dt [Hz]."""
        dt = self.dt
        x_d = self.x_d[:, it] if it > 0 else self.x_d[:, 0]

        # Brian2: qar = clip(randn()*sqrt(x_d/x0*uar*dt*(1-uar*dt))+uar*dt*x_d/x0, 0, 2*x_d/x0*uar*dt)/dt
        uar = self.uar[:, it] if it > 0 else self.uar[:, 0]
        noise_term = np.random.randn(self.n_syn) * np.sqrt(x_d / self.x0 * uar * dt * (1 - uar * dt))
        drift_term = uar * dt * x_d / self.x0
        qar_raw = np.clip(noise_term + drift_term, 0, 2 * x_d / self.x0 * uar * dt)
        qar_hz = qar_raw / dt  # Convert to Hz

        self.qar[:, it] = qar_hz  # * 1e3
        self.d_qar[:, it] = qar_hz  # Note: qar has no ODE, it's discrete

    def update_s_nmda(self, it):
        """NMDA: ds_nmda/dt = -s_nmda/tau_decay + alpha_nmda*x_nmda*(1-s_nmda) + x0*qar."""
        dt = self.dt

        if it == 0:
            ds = -self.s_nmda[:, 0] / self.tau_nmda_decay * dt + \
                 self.alpha_nmda * self.x_nmda[:, 0] * (1.0 - self.s_nmda[:, 0]) + self.x0 * self.qar[:, 0] * dt
            self.d_s_nmda[:, 0] = ds
            self.s_nmda[:, 0] = self.s_nmda[:, 0] + ds
        else:
            ds = -self.s_nmda[:, it - 1] / self.tau_nmda_decay * dt + \
                 self.alpha_nmda * self.x_nmda[:, it] * (1.0 - self.s_nmda[:, it - 1]) + self.x0 * self.qar[:, it] * dt
            self.d_s_nmda[:, it] = ds
            self.s_nmda[:, it] = self.s_nmda[:, it - 1] + ds

    def update_s_ampa(self, it):
        """AMPA: ds_ampa/dt = -s_ampa/tau_ampa + w*S*x_d*spike [clock-driven] + qar_tot_post [summed]
           qar_tot_post = w * S * x0 * qar
           ds_ampa/dt = -s_ampa/tau_ampa + w * S * (x_d * spike + x0 * qar)
        """
        dt = self.dt
        I_it = self.Input[:, it]
        w_S_xd_qar = self.S * (self.x_d[:, it] * I_it + self.x0 * self.qar[:, it] * dt)  # w=1 for now

        if it == 0:
            ds = -self.s_ampa[:, 0] / self.tau_ampa * dt + w_S_xd_qar  # * I_it
            self.d_s_ampa[:, 0] = ds
            self.s_ampa[:, 0] = self.s_ampa[:, 0] + ds
        else:
            ds = -self.s_ampa[:, it - 1] / self.tau_ampa * dt + w_S_xd_qar  # * I_it
            self.d_s_ampa[:, it] = ds
            self.s_ampa[:, it] = self.s_ampa[:, it - 1] + ds

    def get_output(self):
        """Return synaptic conductances: [s_ampa, s_nmda_tot] shape (2, n_syn, L)."""
        # s_nmda_tot = w * S * x_d * s_nmda (per synapse, to be summed by user code)
        s_nmda_tot = self.S * self.x_d * self.s_nmda  # w=1 for now
        return np.stack([self.s_ampa, s_nmda_tot])  # [2, n_syn, L] unitless [0-1]

    def append_spike_event(self, t, active_synapses, output, append_time=True):
        """Store spike events for analysis (override parent)."""
        synapses_with_input_event = np.array(range(self.n_syn))[active_synapses]

        for s in synapses_with_input_event:
            if append_time:
                self.s_ampa_spike_events[s].append(self.s_ampa[s, t])
                self.s_nmda_spike_events[s].append(self.s_nmda[s, t])
                self.x_nmda_spike_events[s].append(self.x_nmda[s, t])
                self.x_d_spike_events[s].append(self.x_d[s, t])
                self.qar_spike_events[s].append(self.qar[s, t])
                self.uar_spike_events[s].append(self.uar[s, t])

                # In case of appending a spike event externally
                if t not in self.time_spike_events[s]:
                    self.time_spike_events[s].append(t)
                else:
                    break

            if len(self.time_spike_events[s]) > 1:
                spike_range = (self.time_spike_events[s][-2], self.time_spike_events[s][-1])
                self.compute_output_spike_event(spike_range, s, output)

    def compute_output_spike_event(self, spike_range, s, output):
        # print("TM, append_spike_event(), spike range ", spike_range, " in time ", spike_range[0])
        assert isinstance(spike_range, tuple), "Param 'spike_range' must be a tuple"
        assert len(spike_range) == 2, "Param 'spike_range' must be a tuple of 2 values"
        assert isinstance(spike_range[0], int), "first element of param 'spike_range' must be integer"
        assert isinstance(spike_range[1], int), "second element of param 'spike_range' must be integer"
        assert spike_range[1] >= spike_range[0], "Param 'spike_range' must contain order elements"
        assert isinstance(output, np.ndarray), "Param 'output' must be a numpy array"
        assert len(output.shape) == 3, "Param 'output' must be a 2D-array, current size is " + str(output.shape)
        assert output.shape[2] >= spike_range[1], ("second element of param 'spike_range' must be less or equal than "
                                                   "the length of param 'output'")
        if spike_range[1] == spike_range[0]:
            # phasic component of spiking responses
            self.output_spike_events[s].append(output[:, s, spike_range[0]])
            # tonic component of spiking responses
            self.output_spike_events_tonic[s].appen(output[:, s, 0])

            # Updating index of phasic and tonic spike event occurences
            self.ind_spike_events_tonic[s].append(spike_range[0] - 1)
            self.ind_spike_events[s].append(spike_range[0])

        else:
            # Tonic component of the spiking response
            self.output_spike_events_tonic[s].append(output[:, s,  spike_range[0] - 1])

            # EPSP
            if np.sum(output) > 0:
                self.output_spike_events[s].append(np.max(output[:, s, spike_range[0]: spike_range[1]], axis=1))
                a = np.argmax(output[:, s, spike_range[0]: spike_range[1]], axis=1)
            # EPSC
            else:
                self.output_spike_events[s].append(np.min(output[:, s, spike_range[0]: spike_range[1]], axis=1))
                a = np.argmin(output[:, s, spike_range[0]: spike_range[1]], axis=1)

            # Updating index of phasic and tonic spike event occurences
            self.ind_spike_events_tonic[s].append(spike_range[0] - 1)
            self.ind_spike_events[s].append(a + spike_range[0])