import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, sosfiltfilt


class SlidingWindowTransitoryAnalyser:
    """
    Sliding-window + steady-state analysis for a single HH neuron driven by STP synapse(s).

    Usage pattern (inside main loop):

        analyzer = SlidingWindowTransitionAnalyzer(
            n_model=n_model,
            Input=Input,
            Input_copy=Input_copy,
            seeds=seeds,
            L=L,
            window_length_ms=200,
            sliding_step_ms=20,
            num_tail_wins=20,
            num_slid_wins=5,
            epsilon=None,
            epsilon_min_max=None,
            plot=False,
            verbose=True,
        )

        it = 0
        while it < L:
            # ... your synapse + neuron update here, including flex_t and n_model.update_state() ...

            it, finish = analyzer.update(flex_t=it)
            if finish:
                break

        if analyzer.finish_sliding_window:
            results = analyzer.analyse()
            (stat_descriptors_tr_st,
             time_series_tr,
             tr_st_time,
             th_tr_an,
             th_tr_ab) = results

    """

    def __init__(self, n_model, Input, seeds, L, window_length_ms=200.0, sliding_step_ms=20.0,
                 num_tail_wins=20, num_slid_wins=5, epsilon=None, epsilon_min_max=None, c_tol_dev=None,
                 plot=False, verbose=True):
        self.n_model = n_model
        self.Input = Input
        self.Input_copy = np.copy(Input)
        self.seeds = seeds
        self.L = L

        self.window_length_ms = window_length_ms
        self.sliding_step_ms = sliding_step_ms
        self.num_tail_wins = num_tail_wins
        self.num_slid_wins = num_slid_wins

        self.plot = plot
        self.verbose = verbose

        # Basic info
        self.num_neu = n_model.n_neurons if hasattr(n_model, "n_neurons") else n_model.membrane_potential.shape[0]
        self.dt = n_model.dt  # in seconds
        self.t_vec_ms = n_model.time_vector * 1e3  # ms

        # Build sliding windows index list (time -> indices)
        self.windows = self._sliding_window_indices(
            self.t_vec_ms, self.window_length_ms, self.sliding_step_ms
        )
        self.num_windows = len(self.windows)

        # Statistics arrays
        self.supra_rate = np.zeros((self.num_neu, self.num_windows))
        self.supra_mean_ISI = np.zeros((self.num_neu, self.num_windows))
        self.supra_std_ISI = np.zeros((self.num_neu, self.num_windows))

        self.sub_mean_v = np.zeros((self.num_neu, self.num_windows))
        self.sub_median_v = np.zeros((self.num_neu, self.num_windows))
        self.sub_q10_v = np.zeros((self.num_neu, self.num_windows))
        self.sub_q90_v = np.zeros((self.num_neu, self.num_windows))
        self.sub_q5_v = np.zeros((self.num_neu, self.num_windows))
        self.sub_q95_v = np.zeros((self.num_neu, self.num_windows))
        self.sub_min_v = np.zeros((self.num_neu, self.num_windows))
        self.sub_max_v = np.zeros((self.num_neu, self.num_windows))

        # Labels for printing/plotting
        self.labels = ['rate', 'μISI', 'sISI', 'V_mu', 'V_me', 'V_10', 'V_90', 'Vmin', 'Vmax']
        self.num_stat_des_supra = 3
        self.num_stat_des_sub = 6  # mean, median, q10, q90, min, max

        # Tolerances (if None, set your current defaults)
        if epsilon is None:
            # per-descriptor epsilon for relative change (only used for completeness)
            self.epsilon = np.array([1e-1, 1e-2, 1e-2, 2e-3, 2e-3, 3e-3, 4e-3, 3e-3, 4e-3])
        else:
            self.epsilon = np.array(epsilon)

        if epsilon_min_max is None:
            self.epsilon_min_max = np.array([1e1, 1e0, 1e0, 2e-1, 2e-1, 2e-1, 3e-1, 2e-1, 4e-1])
        else:
            self.epsilon_min_max = np.array(epsilon_min_max)

        if c_tol_dev is None:
            self.c_tol_dev = np.array([1, 1, 1, 1.5, 1.5, 1, 1, 1, 1, 1, 1])
        else:
            self.c_tol_dev = np.array(c_tol_dev)

        # Bias & colors for plotting
        self.bias_plot = np.array([5, 0.04, 0.012, 0.004, 0.004, 0.004, 0.01, 0.004, 0.04])
        self.colors = ['black', 'yellow', 'green', 'tab:orange', 'tab:blue',
                       'tab:green', 'tab:green', 'tab:red', 'tab:red']

        # Max/min trackers for sliding windows
        self.maxi_v_supra = np.full((self.num_stat_des_supra, self.num_neu), -np.inf)
        self.mini_v_supra = np.full((self.num_stat_des_supra, self.num_neu), np.inf)
        self.maxi_v_sub = np.full((self.num_stat_des_sub, self.num_neu), -np.inf)
        self.mini_v_sub = np.full((self.num_stat_des_sub, self.num_neu), np.inf)

        # Stimulus windows: 3 segments per neuron; fields: [start_step, end_step, start_win_idx, end_win_idx]
        # Initialize start_step for [0:L/3), [L/3:2L/3), [2L/3:L)
        L = int(L)
        seg_bounds = np.array([0, L // 3, 2 * L // 3, L], dtype=int)
        # len_stimuli_windows[neuron, seg_index, field(0..3)]
        self.len_stimuli_windows = np.zeros((self.num_neu, 3, 4), dtype=int)
        for n in range(self.num_neu):
            for s in range(3):
                self.len_stimuli_windows[n, s, 0] = seg_bounds[s]     # start_step
                self.len_stimuli_windows[n, s, 1] = 0                 # end_step (to fill)
                self.len_stimuli_windows[n, s, 2] = self._step_to_win_idx(seg_bounds[s])
                self.len_stimuli_windows[n, s, 3] = 0                 # end_win_idx (to fill)

        # Trackers for mini/maxi, rel_cha, tol_dev across 3 segments
        # shape: [segment_index (0..2), method(=0), neuron]
        self.supra_mini_maxi = np.zeros((3, 3, self.num_neu))
        self.sub_mini_maxi = np.zeros((3, 3, self.num_neu))
        self.supra_rel_cha = np.zeros((3, 3, self.num_neu))  # kept but not in final condition
        self.sub_rel_cha = np.zeros((3, 3, self.num_neu))
        self.supra_tol_dev = np.zeros((3, 3, self.num_neu))
        self.sub_tol_dev = np.zeros((3, 3, self.num_neu))

        # Tracks first time each neuron reaches transition-state for methods (min-max, tol_dev)
        self.tr_st_3_cond = np.zeros((2, self.num_neu))  # row 0: min-max, row 1: tol_dev

        # Sliding-window bookkeeping
        self.counter_slid_win = 0
        self.slid_win_start = 0
        self.counter_stim_win = 0
        self.finish_sliding_window = False

        # Plot figure if requested
        if self.plot:
            self.fig_stat = plt.figure(figsize=[12, 4])
            self.ax = []
            num_stat_des = self.num_stat_des_supra + self.num_stat_des_sub
            for i in range(num_stat_des):
                ax_i = plt.subplot(int(np.ceil(num_stat_des / 5)), 5, i + 1)
                ax_i.grid()
                ax_i.set_xlim(0, 100)
                self.ax.append(ax_i)
            self.fig_stat.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
        else:
            self.ax = None

    # ----------------- helper: sliding windows -----------------

    @staticmethod
    def _sliding_window_indices(t_ms, win_len_ms, step_ms):
        windows = []
        t0 = t_ms[0]
        t_end = t_ms[-1]
        start = t0
        while start + win_len_ms <= t_end:
            i0 = np.searchsorted(t_ms, start)
            i1 = np.searchsorted(t_ms, start + win_len_ms)
            if i1 > i0:
                windows.append((i0, i1))
            start += step_ms
        return windows

    def _step_to_win_idx(self, step_index):
        """
        Approximate: find window index whose end time is >= time_vector[step_index].
        """
        if len(self.windows) == 0:
            return 0
        t_step = self.n_model.time_vector[step_index] * 1e3
        for i, (i0, i1) in enumerate(self.windows):
            if self.t_vec_ms[i1 - 1] >= t_step:
                return i
        return len(self.windows) - 1

    # ----------------- update: called each time step -----------------

    def update(self, flex_t, it):
        """
        Called inside main loop at each time step (with current index flex_t).
        Computes sliding-window stats when flex_t passes next window end,
        updates steady-state conditions, and may set finish_sliding_window=True
        and override 'it' (returned as first output).

        Returns:
            new_it (int), finish_sliding_window (bool)
        """
        if self.finish_sliding_window:
            # Already finished; nothing more to do
            return flex_t, True

        # If we ran out of windows, nothing to update
        if self.counter_slid_win >= self.num_windows:
            return flex_t, False

        # Tracking it
        override_it = it

        window = self.windows[self.counter_slid_win]
        t0, t1 = window
        win_duration_s = (t1 - t0) * self.dt

        # Only compute stats if we've passed the end of this window
        if flex_t >= t1:
            # ---- suprathreshold & subthreshold stats in this window ----
            spike_in_window = [[] for _ in range(self.num_neu)]
            rate_in_window = np.zeros(self.num_neu)
            mean_ISI_in_window = np.zeros(self.num_neu)
            std_ISI_in_window = np.zeros(self.num_neu)

            mean_v_in_window = np.zeros(self.num_neu)
            median_v_in_window = np.zeros(self.num_neu)
            q5_v_in_window = np.zeros(self.num_neu)
            q95_v_in_window = np.zeros(self.num_neu)
            q10_v_in_window = np.zeros(self.num_neu)
            q90_v_in_window = np.zeros(self.num_neu)
            min_v_in_window = np.zeros(self.num_neu)
            max_v_in_window = np.zeros(self.num_neu)

            for n in range(self.num_neu):
                # spikes
                spk_mask = ((self.n_model.time_spikes_generated[n] >= t0) &
                            (self.n_model.time_spikes_generated[n] < t1))
                spike_times = np.array(self.n_model.time_spikes_generated[n])[spk_mask] * self.dt
                spike_in_window[n] = spike_times

                if len(spike_times) == 0:
                    if np.sum(self.supra_rate[n, :self.counter_slid_win]) > 0:
                        rate_in_window[n] = 0.0
                    else:
                        rate_in_window[n] = 0.0
                else:
                    rate_in_window[n] = len(spike_times) / win_duration_s

                if len(spike_times) >= 2:
                    isis = np.diff(spike_times)
                    mean_ISI_in_window[n] = isis.mean()
                    std_ISI_in_window[n] = isis.std()

                # subthreshold V
                mask_v_aux = self.n_model.membrane_potential[n, t0:t1] < self.n_model.V_threshold[n]
                v_aux = self.n_model.membrane_potential[n, t0:t1][mask_v_aux]
                if len(v_aux) == 0:
                    # avoid NaNs; keep zeros or previous?
                    v_aux = self.n_model.membrane_potential[n, t0:t1]

                mean_v_in_window[n] = np.mean(v_aux)
                median_v_in_window[n] = np.median(v_aux)
                q10_v_in_window[n] = np.quantile(v_aux, 0.1)
                q90_v_in_window[n] = np.quantile(v_aux, 0.9)
                q5_v_in_window[n] = np.quantile(v_aux, 0.05)
                q95_v_in_window[n] = np.quantile(v_aux, 0.95)
                min_v_in_window[n] = np.min(v_aux)
                max_v_in_window[n] = np.max(v_aux)

            # store stats
            k = self.counter_slid_win
            self.supra_rate[:, k] = rate_in_window
            self.supra_mean_ISI[:, k] = mean_ISI_in_window
            self.supra_std_ISI[:, k] = std_ISI_in_window

            self.sub_mean_v[:, k] = mean_v_in_window
            self.sub_median_v[:, k] = median_v_in_window
            # self.sub_q5_v[:, k] = q5_v_in_window
            # self.sub_q95_v[:, k] = q95_v_in_window
            self.sub_q10_v[:, k] = q10_v_in_window
            self.sub_q90_v[:, k] = q90_v_in_window
            self.sub_min_v[:, k] = min_v_in_window
            self.sub_max_v[:, k] = max_v_in_window

            # ---- steady-state checks if enough windows ----
            if k >= self.slid_win_start + self.num_tail_wins + self.num_slid_wins:
                override_it = self._check_steady_state(flex_t, k, it)

            self.counter_slid_win += 1
            if self.verbose:
                print(f"Computing window {self.counter_slid_win}/{self.num_windows} at t={flex_t * self.dt:.3f}")

        return override_it, self.finish_sliding_window

    # ----------------- steady-state check (min-max + tol_dev) -----------------

    def _check_steady_state(self, flex_t, counter_slid_win, it):
        # Build descriptor lists
        stat_des_supra = [self.supra_rate, self.supra_mean_ISI, self.supra_std_ISI]
        stat_des_sub = [self.sub_mean_v, self.sub_median_v, self.sub_q10_v,
                        self.sub_q90_v, self.sub_min_v, self.sub_max_v]

        # Condition arrays
        conds_supra_max_min = np.zeros((self.num_stat_des_supra, self.num_neu), dtype=bool)
        conds_supra_tol_dev = np.zeros((self.num_stat_des_supra, self.num_neu), dtype=bool)
        conds_sub_max_min = np.zeros((self.num_stat_des_sub, self.num_neu), dtype=bool)
        conds_sub_tol_dev = np.zeros((self.num_stat_des_sub, self.num_neu), dtype=bool)

        # Tail indices for stationary mean/std
        min_tail = counter_slid_win - self.num_tail_wins - self.num_slid_wins
        if self.counter_stim_win > 0:
            # Start tail after previous segment windows
            prev_end_win = np.max(self.len_stimuli_windows[:, self.counter_stim_win - 1, 3])
            min_tail = max(min_tail, prev_end_win)
        max_tail = counter_slid_win - self.num_slid_wins

        c_sd = 0

        # ---- suprathreshold descriptors ----
        for i in range(self.num_stat_des_supra):
            tail_stat_des = stat_des_supra[i][:, min_tail:max_tail]
            aux_stat_des = stat_des_supra[i][:, counter_slid_win - self.num_slid_wins:counter_slid_win]

            aux_mean = np.repeat(np.mean(tail_stat_des, axis=1, keepdims=True), self.num_slid_wins, axis=1)
            aux_std = np.repeat(np.std(tail_stat_des, axis=1, ddof=1, keepdims=True),
                                self.num_slid_wins, axis=1)

            # max-min condition
            local_max = np.max(aux_stat_des, axis=1)
            local_min = np.min(aux_stat_des, axis=1)
            diff_max_min_local = local_max - local_min

            # update global max/min trackers
            self.maxi_v_supra[i, :] = np.maximum(self.maxi_v_supra[i, :], local_max)
            self.mini_v_supra[i, :] = np.minimum(self.mini_v_supra[i, :], local_min)

            ep = (self.maxi_v_supra[i, :] - self.mini_v_supra[i, :]) * self.epsilon_min_max[c_sd]
            conds_supra_max_min[i, :] = diff_max_min_local <= ep

            # tol_dev condition: |stat - mean| / std <= 1
            delta_k2 = np.abs(aux_stat_des - aux_mean) / (aux_std + 1e-9)
            aux2_per_win = delta_k2 <= self.c_tol_dev[c_sd]  # 1.0
            conds_supra_tol_dev[i, :] = np.sum(aux2_per_win, axis=1) == self.num_slid_wins

            if self.verbose:
                print(
                    f"{self.labels[c_sd]}, maxi-mini: {conds_supra_max_min[i, 0]}, "
                    f"max-min {diff_max_min_local[0]:.2E}, ep {ep[0]:.2E}. "
                    f"Cond tol_dev: {conds_supra_tol_dev[i, 0]}, "
                    f"d_k2 {np.max(delta_k2, axis=1)[0]:.2E}"
                )

            # plotting
            if self.plot:
                for n in range(self.num_neu):
                    self.ax[c_sd].plot(
                        np.arange(counter_slid_win),
                        stat_des_supra[i][n, :counter_slid_win] + n * self.bias_plot[c_sd],
                        color=self.colors[c_sd]
                    )

            c_sd += 1

        # ---- subthreshold descriptors ----
        for i in range(self.num_stat_des_sub):
            # only neurons that have been in subthreshold (no spikes) in [min_tail, counter_slid_win)
            neurons_not_spiking = self.supra_rate[:, min_tail:counter_slid_win] == 0
            mask_in_sub = np.sum(neurons_not_spiking, axis=1) == (counter_slid_win - min_tail)

            if not np.any(mask_in_sub):
                if self.verbose:
                    print(f"{self.labels[c_sd]}, all neurons suprathreshold in this interval")
                c_sd += 1
                continue

            tail_stat_des = stat_des_sub[i][mask_in_sub, min_tail:max_tail]
            aux_stat_des = stat_des_sub[i][mask_in_sub, counter_slid_win - self.num_slid_wins:counter_slid_win]

            if tail_stat_des.shape[0] > 0:
                aux_mean = np.repeat(np.mean(tail_stat_des, axis=1, keepdims=True), self.num_slid_wins, axis=1)
                aux_std = np.repeat(np.std(tail_stat_des, axis=1, ddof=1, keepdims=True),
                                    self.num_slid_wins, axis=1)
            else:
                aux_mean = np.zeros((0, self.num_slid_wins))
                aux_std = np.zeros((0, self.num_slid_wins))

            """
            # Updating maxi and mini masks for a specific statistica descriptor
            # aux_mask_ma = np.max(aux_stat_des, axis=1) > maxi_v_sub[i, :]
            aux_mask_ma = np.max(aux_stat_des, axis=1) > maxi_v_sub[i, mask_in_sub]
            if np.any(aux_mask_ma):
                aux = maxi_v_sub[i, mask_in_sub]
                aux[aux_mask_ma] = np.max(aux_stat_des, axis=1)[aux_mask_ma]
                maxi_v_sub[i, mask_in_sub] = aux
            # aux_mask_mi = np.min(aux_stat_des, axis=1) < mini_v_sub[i, :]
            aux_mask_mi = np.min(aux_stat_des, axis=1) < mini_v_sub[i, mask_in_sub]
            if np.any(aux_mask_mi):
                aux = mini_v_sub[i, mask_in_sub]
                aux[aux_mask_mi] = np.min(aux_stat_des, axis=1)[aux_mask_mi]
                mini_v_sub[i, mask_in_sub] = aux
            # """
            local_max = np.max(aux_stat_des, axis=1)
            local_min = np.min(aux_stat_des, axis=1)
            diff_max_min_local = local_max - local_min

            # update global max/min for sub
            self.maxi_v_sub[i, mask_in_sub] = np.maximum(self.maxi_v_sub[i, mask_in_sub], local_max)
            self.mini_v_sub[i, mask_in_sub] = np.minimum(self.mini_v_sub[i, mask_in_sub], local_min)

            ep = (self.maxi_v_sub[i, mask_in_sub] - self.mini_v_sub[i, mask_in_sub]) * self.epsilon_min_max[c_sd]
            cond_local_mm = diff_max_min_local <= ep
            conds_sub_max_min[i, mask_in_sub] = cond_local_mm

            # tol_dev with threshold 1.5
            delta_k2 = np.abs(aux_stat_des - aux_mean) / (aux_std + 1e-9)
            aux2_per_win = delta_k2 <= self.c_tol_dev[c_sd]  # 1.5
            cond_local_td = np.sum(aux2_per_win, axis=1) == self.num_slid_wins
            conds_sub_tol_dev[i, mask_in_sub] = cond_local_td

            if tail_stat_des.shape[0] > 0 and self.verbose:
                print(
                    f"{self.labels[c_sd]}, maxi-mini: {cond_local_mm[0]}, "
                    f"max-min {diff_max_min_local[0]:.2E}, ep {ep[0]:.2E}. "
                    f"Cond tol_dev: {cond_local_td[0]}, "
                    f"d_k2 {np.max(delta_k2, axis=1)[0]:.2E}, c {self.c_tol_dev[0]:.2E}. "
                )

                if self.plot:
                    for n in range(self.num_neu):
                        self.ax[c_sd].plot(np.arange(counter_slid_win), stat_des_sub[i][n, :counter_slid_win] +
                                           n * self.bias_plot[c_sd], color=self.colors[c_sd])

            c_sd += 1

        # ---- update trackers via update_tr_st_trackers-like logic ----
        a = self.counter_stim_win
        self.supra_mini_maxi[a, :] = self.update_tr_st_trackers(conds_supra_max_min, self.supra_mini_maxi[a, :], flex_t)
        self.sub_mini_maxi[a, :] = self.update_tr_st_trackers(conds_sub_max_min, self.sub_mini_maxi[a, :], flex_t)
        self.supra_tol_dev[a, :] = self.update_tr_st_trackers(conds_supra_tol_dev, self.supra_tol_dev[a, :], flex_t)
        self.sub_tol_dev[a, :] = self.update_tr_st_trackers(conds_sub_tol_dev, self.sub_tol_dev[a, :], flex_t)

        if self.verbose:
            print(
                f"Stim win {a}, supra(min-max): {self.supra_mini_maxi[a, 0, :]}, "
                f"supra(tol_dev): {self.supra_tol_dev[a, 0, :]}"
            )
            print(
                f"Stim win {a}, sub(min-max): {self.sub_mini_maxi[a, 0, :]}, "
                f"sub(tol_dev): {self.sub_tol_dev[a, 0, :]}"
            )

        # masks of neurons that reached via each metric
        mask_supr_mm = np.where(self.supra_mini_maxi[a, 0, :] > 0)[0]
        mask_sub_mm = np.where(self.sub_mini_maxi[a, 0, :] > 0)[0]
        mask_supr_td = np.where(self.supra_tol_dev[a, 0, :] > 0)[0]
        mask_sub_td = np.where(self.sub_tol_dev[a, 0, :] > 0)[0]

        # Updating time of reaching st based on each approach (mini-maxi, tolarance deviation)
        mask_to_update = (self.tr_st_3_cond[0, mask_sub_mm] == 0)
        if np.any(mask_to_update):
            self.tr_st_3_cond[0, mask_sub_mm[mask_to_update]] = flex_t
        mask_to_update = (self.tr_st_3_cond[1, mask_sub_td] == 0)
        if np.any(mask_to_update):
            self.tr_st_3_cond[1, mask_sub_td[mask_to_update]] = flex_t

        # which neurons have at least one method satisfied?
        mask_sub = np.sum(self.tr_st_3_cond > 0, axis=0) == 2  # both min-max and tol_dev > 0

        # update len_stimuli_windows for these neurons (first time)
        if np.any(mask_sub):
            ind_to_update = (self.len_stimuli_windows[:, self.counter_stim_win, 1] == 0)
            if np.any(ind_to_update):
                mask_update = np.logical_and(mask_sub, ind_to_update)
                self.len_stimuli_windows[mask_update, self.counter_stim_win, 1] = flex_t
                self.len_stimuli_windows[mask_update, self.counter_stim_win, 3] = self.counter_slid_win

        cond_sub_mm_count = mask_sub_mm.shape[0]
        cond_sub_td_count = mask_sub_td.shape[0]

        # if all neurons reached via both min-max and tol_dev in subthreshold descriptors
        if cond_sub_mm_count == self.num_neu and cond_sub_td_count == self.num_neu:
            if self.verbose:
                print("All neurons reach steady-state in segment", self.counter_stim_win)

            if self.plot:
                fig_mem = plt.figure()
                axfm = fig_mem.add_subplot(111)
                # axfm.plot(n_model.time_vector[:flex_t], n_model.membrane_potential[0, :flex_t], c="gray")
                mem_pot = self.n_model.membrane_potential[:, :flex_t]
                min_m = np.min(mem_pot)
                max_m = np.max(mem_pot)
                for n in range(self.num_neu):
                    # Plotting membrane potential
                    axfm.plot(self.n_model.time_vector[:flex_t], mem_pot[n, :] + n * .01, c="gray")
                    for b in range(a + 1):
                        # Plotting tr_st_time for each method
                        lim_a, lim_b = min_m + 0.01 * n, max_m + 0.01 * n
                        axfm.plot([self.sub_mini_maxi[b, 0, n], self.sub_mini_maxi[b, 0, n]], [lim_a, lim_b], c="tab:red")
                        axfm.plot([self.sub_rel_cha[b, 0, n], self.sub_rel_cha[b, 0, n]], [lim_a, lim_b], c="tab:green")
                        axfm.plot([self.sub_tol_dev[b, 0, n], self.sub_tol_dev[b, 0, n]], [lim_a, lim_b], c="tab:blue")
                        aux_p = self.len_stimuli_windows[n, b, 0] * self.dt
                        axfm.plot([aux_p, aux_p], [min_m, max_m + 0.01 * n], c="black")
                axfm.grid()

            # finalize segment boundaries
            seg_idx = self.counter_stim_win
            max_tr_st = np.max(self.len_stimuli_windows[:, seg_idx, 1])
            self.len_stimuli_windows[:, seg_idx, 1] = max_tr_st
            if seg_idx < 2:
                # next segment start at max_tr_st
                self.len_stimuli_windows[:, seg_idx + 1, 0] = max_tr_st
                self.len_stimuli_windows[:, seg_idx + 1, 2] = (self.len_stimuli_windows[:, seg_idx, 3] +
                                                               int(self.window_length_ms / self.sliding_step_ms))

            # update iteration index for main loop
            if seg_idx == 0: new_it = self.L // 3 - 1
            elif seg_idx == 1: new_it = 2 * self.L // 3 - 1
            else:
                new_it = self.L - 1
                self.finish_sliding_window = True

            # reset trackers for next segment
            self.counter_stim_win += 1
            self.counter_slid_win += int(self.window_length_ms / self.sliding_step_ms) - 1
            self.slid_win_start = self.counter_slid_win + 1

            # re-splice Input according to flex_t / new_it
            self.Input = np.concatenate((self.Input[:, :flex_t], self.Input_copy[:, new_it + 1:]), axis=1)
            # reset min/max trackers and tr_st_3_cond
            self.maxi_v_supra[:] = -np.inf
            self.mini_v_supra[:] = np.inf
            self.maxi_v_sub[:] = -np.inf
            self.mini_v_sub[:] = np.inf
            self.tr_st_3_cond[:, :] = 0

            # override flex_t in caller via return value
            # we emulate by setting attribute; caller must use returned value from update()
            return new_it
        else:
            return it

    @staticmethod
    def _update_tr_st_trackers(conds, tracker, flex_t):
        """
        Equivalent of your update_tr_st_trackers:
        If conds[i, n] is True for any descriptor i for neuron n, and tracker was 0, set to flex_t.
        """
        new_tracker = tracker.copy()
        # conds: [num_descriptors, num_neurons]
        any_true = np.any(conds, axis=0)
        for n in range(conds.shape[1]):
            if any_true[n] and new_tracker[0, n] == 0:
                new_tracker[0, n] = flex_t
        return new_tracker

    def update_tr_st_trackers(self, conds, tr_st_array, t):
        # Number of neurons
        num_neu = self.num_neu
        # copy of tr_st_array
        aux_array = np.copy(tr_st_array)
        # If the statistical descriptors of some neurons reach steady-state, update steady-state trackers
        mask_neurons = np.all(conds, axis=0)
        if np.sum(mask_neurons) > 0:
            # Computing tr_st_time
            tr_st_time = self.compute_time_tr_st(t)

            # Storing tr_st_time for all neurons that reach steady-state for the first time
            if np.any(aux_array[0, mask_neurons]) == 0:
                aux_array[0, mask_neurons] = \
                    np.array([[tr_st_time] for _ in range(num_neu)])[mask_neurons, 0]

            # updating last time that tr_st_time is computed
            aux_array[1, mask_neurons] = tr_st_time
            # Counting times that neurons reach steady-state
            aux_array[2, mask_neurons] += 1

        return aux_array

    def compute_time_tr_st(self, t):
        # Computing time of end of transition
        is_tr_st_time = (self.window_length_ms + (self.num_slid_wins * self.sliding_step_ms)) * 1e-3  # in s
        ind_tr_st = t - int(is_tr_st_time / self.dt)
        tr_st_time = ind_tr_st * self.dt
        return tr_st_time

    # ----------------- analyse: called once finish_sliding_window is True -----------------

    def analyse(self):
        """
        Run after finish_sliding_window is True.
        Computes:
            stat_descriptors_tr_st,
            time_series_tr,
            tr_st_time,
            th_tr_an,
            th_tr_ab
        """
        windows_a = np.array(self.windows)
        window_length = self.window_length_ms
        sliding_step = self.sliding_step_ms
        st_win_l = self.num_tail_wins + self.num_slid_wins
        num_neu = self.num_neu
        mem_pot = self.n_model.membrane_potential

        # indices of windows for initial, mid, end segments (per neuron, but we take neuron 0 as reference)
        # len_stimuli_windows[n, seg, 0:2] in steps
        i_iw = (self.len_stimuli_windows[0, 0, :2] * self.dt * 1e3) / sliding_step
        i_mw = (self.len_stimuli_windows[0, 1, :2] * self.dt * 1e3) / sliding_step
        i_ew = (self.len_stimuli_windows[0, 2, :2] * self.dt * 1e3) / sliding_step
        l_win = window_length / sliding_step
        i_iw[1] -= l_win
        i_mw[1] -= l_win
        i_ew[1] -= l_win
        i_iw = i_iw.astype(int)
        i_mw = i_mw.astype(int)
        i_ew = i_ew.astype(int)

        # steady-state membrane potential segments
        mem_pot_iw_st = mem_pot[:, self.windows[i_iw[1] - st_win_l][0]: self.windows[i_iw[1]][1]]
        mem_pot_mw_st = mem_pot[:, self.windows[i_mw[1] - st_win_l][0]: self.windows[i_mw[1]][1]]
        mem_pot_ew_st = mem_pot[:, self.windows[i_ew[1] - st_win_l][0]: self.windows[i_ew[1]][1]]

        iw_st = self._statistics_signal(mem_pot_iw_st, axis=1)
        mw_st = self._statistics_signal(mem_pot_mw_st, axis=1)
        ew_st = self._statistics_signal(mem_pot_ew_st, axis=1)

        tr_signals = [self.sub_mean_v, self.sub_median_v, self.sub_q5_v, self.sub_q10_v, self.sub_q90_v, self.sub_q95_v,
                      self.sub_min_v, self.sub_max_v]
        # we only used mean, median, q10, q90, min, max => indices [0,1,3,4,6,7]

        iw_tr = self.stat_tr_slid(tr_signals, i_iw, st_win_l)  # self._stat_tr_slid
        mw_tr = self.stat_tr_slid(tr_signals, i_mw, st_win_l)  # self._stat_tr_slid
        ew_tr = self.stat_tr_slid(tr_signals, i_ew, st_win_l)  # self._stat_tr_slid

        # time to reach steady-state for each segment (your updated version)
        diff_idx = np.diff(self.len_stimuli_windows[:, :, 2:4], axis=2)[:, :, 0]  # shape [neuron, seg]
        tr_st_time_idx = np.array([windows_a[diff_idx[:, i], 0] for i in range(3)])
        tr_st_time = tr_st_time_idx * self.dt

        # steady-state descriptors (8 per segment: μ, med, q5, q10, q90, q95, min, max; you have 8 in iw_st)
        stat_descriptors_tr_st = np.array([
            iw_st[0], iw_st[1], iw_st[2], iw_st[3], iw_st[4], iw_st[5], iw_st[6], iw_st[7],
            mw_st[0], mw_st[1], mw_st[2], mw_st[3], mw_st[4], mw_st[5], mw_st[6], mw_st[7],
            ew_st[0], ew_st[1], ew_st[2], ew_st[3], ew_st[4], ew_st[5], ew_st[6], ew_st[7],
        ])

        time_series_tr = [
            iw_tr[0], iw_tr[1], iw_tr[2], iw_tr[3], iw_tr[4], iw_tr[5], iw_tr[6], iw_tr[7],
            mw_tr[0], mw_tr[1], mw_tr[2], mw_tr[3], mw_tr[4], mw_tr[5], mw_tr[6], mw_tr[7],
            ew_tr[0], ew_tr[1], ew_tr[2], ew_tr[3], ew_tr[4], ew_tr[5], ew_tr[6], ew_tr[7],
        ]

        # "old method" transition threshold between initial and end signals
        piw = self.n_model.membrane_potential[:, :self.len_stimuli_windows[0, 0, 1]]
        lim_a, lim_b = self.len_stimuli_windows[0, 2, :2]
        pew = self.n_model.membrane_potential[:, lim_a:lim_b]
        a_len = min(piw.shape[1], pew.shape[1])

        if self.plot:
            # auxiliar plots
            plt.figure()
            for n in range(num_neu):
                aux = piw[n, :] + n * 0.005
                plt.plot(self.n_model.time_vector[:aux.shape[0]], aux)
                aux = pew[n, :] + n * 0.005
                plt.plot(self.n_model.time_vector[:aux.shape[0]], aux)
            plt.grid()
            plt.figure()
            for n in range(num_neu):
                plt.plot(self.sub_mean_v[n, i_iw[0]:i_iw[1]] + n * 0.005)
                plt.plot(self.sub_mean_v[n, i_ew[0]:i_ew[1]] + n * 0.005)
            plt.grid()

        th_tr_an = self.get_transition_time_from_2_signals(piw[:, :a_len], pew[:, :a_len], th_percentage=7e-2) * self.dt
        th_tr_ab = self.get_transition_time_from_2_signals(piw[:, :a_len], pew[:, :a_len], th_percentage=1e-1,
                                                           filtering=True, sfreq=1 / self.dt) * self.dt

        return stat_descriptors_tr_st, time_series_tr, tr_st_time, th_tr_an, th_tr_ab

    # ----------------- helper functions you said you'd provide -----------------

    @staticmethod
    def _statistics_signal(signal, axis=1):
        """
        Compute mean, median, q5, q10, q90, q95, min, max along given axis.
        Returns list of 8 arrays.
        """
        mean_ = np.mean(signal, axis=axis)
        med_ = np.median(signal, axis=axis)
        q5_ = np.quantile(signal, 0.05, axis=axis)
        q10_ = np.quantile(signal, 0.10, axis=axis)
        q90_ = np.quantile(signal, 0.90, axis=axis)
        q95_ = np.quantile(signal, 0.95, axis=axis)
        min_ = np.min(signal, axis=axis)
        max_ = np.max(signal, axis=axis)
        return [mean_, med_, q5_, q10_, q90_, q95_, min_, max_]

    @staticmethod
    def _stat_tr_slid(tr_signals, i_seg, st_win_l):
        """
        Analog of stat_tr_slid(tr_signals, i_seg, st_win_l).
        tr_signals: list of arrays per descriptor [num_neu, num_windows]
        i_seg: [start_win, end_win]
        """
        start_idx, end_idx = i_seg.astype(int)
        seg_idx_range = np.arange(end_idx - st_win_l, end_idx)
        stats = []
        for s in tr_signals:
            if s is None:
                stats.append(None)
            else:
                seg_vals = s[:, seg_idx_range]
                stats.append(np.mean(seg_vals, axis=1))
        return stats

    @staticmethod
    def stat_tr_slid(signals, win_r, win_l):
        res = []
        for signal in signals:
            res.append(signal[:, win_r[0]:win_r[1] - win_l])
        return res

    @staticmethod
    def _get_transition_time_from_2_signals(sig_a, sig_b, th_percentage=7e-2,
                                            filtering=False, sfreq=None):
        """
        Your get_transition_time_from_2_signals:
        Given two membrane potential signals sig_a, sig_b [num_neu, T], find the earliest
        time index where |a(t) - b(t)| <= th_percentage * (max-min) or similar.
        Filtering flag: apply lowpass filtering before difference if True.
        """
        diff = sig_a - sig_b
        if filtering and sfreq is not None:
            # simple lowpass butterworth as example
            b, a = butter(N=2, Wn=5.0 / (sfreq / 2), btype='low')
            diff = filtfilt(b, a, diff, axis=1)

        # e.g., threshold on normalized difference
        max_diff = np.max(np.abs(diff), axis=1, keepdims=True)
        norm_diff = np.abs(diff) / (max_diff + 1e-12)
        # first time all neurons below threshold
        cond = np.all(norm_diff <= th_percentage, axis=0)
        idx = np.argmax(cond) if np.any(cond) else 0
        return idx

    def get_transition_time_from_2_signals(self, signal1, signal2, th_percentage=1e-5, filtering=False, cutoff=5,
                                           sfreq=None):
        # Substract ini and end window to define the transition period (exclude lasst 10 samples to avoid errors)
        ini_minus_end_windows = np.abs(signal1[:, :-10] - signal2[:, :-10])
        # Find the 0.1% of the maximum for each realization (the threshold to define that the difference between ini and
        # end windows is sufficiently low to be considered zero)
        thresholds = np.max(ini_minus_end_windows, axis=1) * th_percentage
        shapes_diff = ini_minus_end_windows.shape
        # Create the mask to compare each realization with the 0.1% of their maximums
        mask_thr = np.repeat(np.reshape(thresholds, (shapes_diff[0], 1)), shapes_diff[1], axis=1)
        # If filtering is activated, apply the thresholds on the filtered signals
        if filtering:
            assert sfreq is not None, "sfreq must be given"
            ini_minus_end_windows = self.lowpass(ini_minus_end_windows, cutoff, sfreq)
        # find indices where the difference is bigger than 0.1% of maximum
        ind_tr = np.where(ini_minus_end_windows > mask_thr)
        # Getting indices of unique values (i.e. realizations)
        val_unique, ind_unique = np.unique(ind_tr[0], return_index=True)
        # getting last index (indicating that after that, diff. is lower than 1e-6)
        first_indtr = np.roll(ind_tr[1][list(np.array(ind_unique) - 1)], -1)

        return first_indtr

    @staticmethod
    def lowpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
        sos = butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
        filtered_data = sosfiltfilt(sos, data)
        return filtered_data
