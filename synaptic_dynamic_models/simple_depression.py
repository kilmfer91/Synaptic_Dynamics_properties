from synaptic_dynamic_models.SynDynModel import SynDynModel
import numpy as np
from scipy.integrate import odeint


class Simple_Depression(SynDynModel):

    def __init__(self, n_syn=1):
        super().__init__(n_syn)

        # model variables
        self.alpha = None
        self.g = None

        # Spike events
        self.alpha_spike_events = None
        self.alpha_steady_state = None

        # derivative variables
        self.d_alpha = None

        # Parameters
        self.tau_g = None
        self.tau_alpha = None
        self.g0 = None
        self.f = None
        self.params = {'tau_g': 100e-3, 'tau_alpha': 2e-3, 'g0': 0.075, 'f': 0.75}

        # Calling the main methods for setting the model
        self.set_model_params(self.params)
        self.set_simulation_params()
        self.set_initial_conditions()

    def set_model_params(self, model_params):
        """
        Set the paramateres for simple depression
        :param model_params: (dict) parameters of s_dep. A dictionary with at least one of the following keys: ['tau']
        """
        assert isinstance(model_params, dict), 'params should be a dict'
        for key, value in model_params.items():
            if key in self.params.keys():
                self.params[key] = value

        self.tau_g = self.params['tau_g']
        self.tau_alpha = self.params['tau_alpha']
        self.g0 = self.params['g0']
        self.f = self.params['f']

    def set_initial_conditions(self, Input=None):
        # model variables
        self.alpha = np.zeros((self.n_syn, self.L))
        self.g = np.zeros((self.n_syn, self.L))

        # derivative variables
        self.d_alpha = np.zeros((self.n_syn, self.L))

        # Variables for spike events and steady-state calculations
        self.alpha_spike_events = []
        self.output_spike_events = []
        self.time_spike_events = []
        self.alpha_steady_state = None
        self.output_steady_state = None
        self.efficacy = [0.0 for _ in range(self.n_syn)]
        self.efficacy_2 = [0.0 for _ in range(self.n_syn)]
        self.efficacy_3 = [0.0 for _ in range(self.n_syn)]
        self.t_steady_state = [0.0 for _ in range(self.n_syn)]
        self.t_max = [0.0 for _ in range(self.n_syn)]

        # Initial conditions
        self.alpha[:, 0] = 1.0
        self.g[:, 0] = self.g0
        self.edge_detection = False

        if Input is None:
            self.Input = np.zeros((self.n_syn, self.L))
        else:
            assert isinstance(Input, np.ndarray), "'Input' must be a numpy array"
            assert len(spike_range) == 2, "'Input' must have 2-dimensions"
            self.Input = Input

    def evaluate_model_euler(self, I_t, t):
        """
        Compute the time functions of the simple depression model by solving the ODE using the euler method
        :param I_t: (numpy array (n, t)) value of n-inputs at time t
        :param t: (int) time value
        """
        # Input
        self.Input[:, t] = I_t

        # model evaluation
        if t == 0:
            if I_t.any() > 0:
                self.alpha[:, 0] *= self.f * I_t
                ind_spike = list(np.where(I_t > 0)[0])
                self.g[ind_spike, 0] = self.g0 * self.alpha[ind_spike, 0]

        else:
            """
            d_alpha = (((1 - self.alpha[:, t - 1]) / self.tau_alpha) * self.dt)  # + 0.75 * I_t
            self.alpha[:, t] = d_alpha + self.alpha[:, t - 1]
            if I_t.any() > 0:
                self.alpha[:, t] *= 0.75 * I_t
            d_g = ((-self.g[:, t - 1]) / self.tau_g * self.dt) + self.g0 * self.alpha[:, t] * I_t
            self.g[:, t] = d_g + self.g[:, t - 1]
            # """
            d_alpha = (((1 - self.alpha[:, t - 1]) / self.tau_alpha) * self.dt)  # + 0.75 * I_t
            self.alpha[:, t] = d_alpha + self.alpha[:, t - 1]
            # d_g = ((-self.g[:, t - 1]) / self.tau_g * self.dt) + self.g0 * self.alpha[:, t] * I_t
            d_g = (-self.g[:, t - 1]) / self.tau_g * self.dt
            self.g[:, t] = d_g + self.g[:, t - 1]

            if I_t.any() > 0:
                ind_spike = list(np.where(I_t > 0)[0])
                # self.alpha[:, t] *= self.f * I_t
                self.alpha[ind_spike, t] = self.alpha[ind_spike, t] * self.f
                self.g[ind_spike, t] = self.g0 * self.alpha[ind_spike, t]

    def get_output(self):
        return self.g

    def append_spike_event(self, t, output):
        """
        Storing spike events for each state variable given a t-time
        Parameters
        ----------
        t
        output
        """
        self.alpha_spike_events.append(self.alpha[:, t])
        self.time_spike_events.append(t)

        # Computing maximum alpha response between the last and the current spike
        # If this is the first spike event
        if len(self.time_spike_events) > 1:
            spike_range = (self.time_spike_events[-2], self.time_spike_events[-1])
            self.compute_output_spike_event(spike_range, output)
