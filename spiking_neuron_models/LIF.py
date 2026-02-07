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

        # Output
        self.membrane_potential = None
        self.rec_spikes = [[] for _ in range(n_neu)]
        self.rec_spikes2 = np.expand_dims(np.zeros(n_neu), axis=1)  # [[] for _ in range(n_neu)]

        self.params = {'V_threshold': -55.0, 'V_reset': -75.0, 'tau_m': 10.0, 'g_L': 10.0, 'V_init': -75.0,
                       'V_equilibrium': -75.0, 't_refractory': 2.0, 'T': 400.0, 'dt': 0.1}

        self.sim_params = {'sfreq': 1000, 'max_t': 0.8}
        self.set_simulation_params()

    def set_model_params(self, model_params):
        assert isinstance(sim_params, dict), 'params should be a dict'
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

    def set_simulation_params(self, sim_params=None):
        if sim_params is not None:
            assert isinstance(sim_params, dict), 'params should be a dict'
            for key, value in sim_params.items():
                if key in self.sim_params.keys():
                    self.sim_params[key] = value

        # time simulation variables
        self.dt = 1 / self.sim_params['sfreq']
        self.time_vector = np.arange(0, self.sim_params['max_t'], self.dt)
        self.L = int(self.sim_params['sfreq'] * self.sim_params['max_t'])

        # Output variables
        self.membrane_potential = np.zeros((self.n_neurons, self.L))
        self.membrane_potential[:, 0] = self.V_init

    def update_state(self, I_input, it):
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
