import sys
import os
import time
import re
from datetime import timedelta
import pickle
from utils_plot import *

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# **********************************************************************************************************************
# PATHS TO STORE/LOAD DATA
path_signals_ext = "../../reference data/"
path_outputs_ext = "../../outputs/fitting_test/"


# **********************************************************************************************************************
# EXAMPLE OF PARAMETERS FOR MSSM AND TM MODELS
"""MSSM"""
# External parameters for MSSM with 1 synapse. Facilitation
params_name_mssm = ['tau_c', 'alpha', 'V0', 'tau_v', 'P0', 'k_NtV', 'k_Nt', 'tau_Nt', 'k_EPSP', 'tau_EPSP']
ext_par_mssm = [3e-3, 905e-4, 3.45, 8.4e-3, 0.002, 1, 1, 13.5e-3, 10, 9e-3, 0.0, 0.0]  # From Karim C0 0.05
selected_params_mssm = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

"""TM Model"""
# From the paper "Differential signaling via the same axon of neocortical pyramidal neurons" Markram, Wang, Tsodyks
params_name_tm = ['U0', 'tau_f', 'tau_d', 'Ase', 'tau_syn']
ext_par_tm = [0.03, 530e-3, 130e-3, 1540, 2.5e-3]
selected_params_tm = [0, 1, 2, 3, 4]

"""LAP Model"""
# From the paper "Differential signaling via the same axon of neocortical pyramidal neurons" Markram, Wang, Tsodyks
params_name_lap = ['KCa', 'tau_Cai', 'Krel_half', 'Krecov_0', 'Krecov_max', 'Prel0', 'Prel_max', 'Krecov_half',
                   'tau_EPSC', 'KGlu', 'n', 'Ntotal']
ext_par_lap = [120, 100e-3, 9, 2.2e-2, 2.2e-2, 0.06, 0.9, 0, 5e-3, 1e-10/5.5, 1, 9.2e5]  # facilitation
ext_par_lap = [9e2*4/1.5, 90e-3, 0.015e2*4/1.5, 1e-4, 6.6e-3, 0.46, 0.57, 0.43e2*4/1.5, 10e-3, 1.9*3.5e-11, 1, 30e5]  # depression
ext_par_lap = [515, 450e-3, 20, 7.5e-3, 7.5e-3, 0.02, 1, 0, 15e-4, 3.7e-4, 1, 9.5e6]  # facilitation-depression
selected_params_lap = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# labels for uploading examples of facilitation and depression
prefix_v = ['depression', 'facilitation']

# ******************************************************************************************************************
# TIME CONDITIONS
r_ext = [10, 30]  # Frequency of the spike train (Hz)
sfreq_ext = [2e3, 2e3]  # [2e3, 5e3]  # Sampling frequency (Hz)
ini_t_ext = [.1, .04]  # Initial time of simulation (s)
end_t_ext = [3.1, .55]  # Maximum time of spike train (s) (MUST BE LESS THAN max_t_v)
max_t_ext = [2.0, 1.2]  # [3.5, .8] # Maximum time of simulation (s)
# Input-Output factors
input_factor_ext = [1.0, 1.0]
output_factor_ext = [-1.0, -1.0]


# **********************************************************************************************************************
# SPIKING NEURONS FUNCTIONS
def input_spike_train(sfreq, freq_signal, max_time, min_time=0.0):
    """
    spike train at a given frequency
    :param sfreq: sample frequency
    :param freq_signal: frequency of the output spike train
    :param max_time: Duration (in seconds) of the spike train
    :param min_time: time to start a spike train (in seconds)
    :return spike_train: Spike train
    """
    dt = 1 / sfreq  # size of steps from the sample frequency
    T = 1 / freq_signal  # Signal period
    step = int(T / dt)  # Where to generate an impulse
    L = int(max_time / dt)  # Number of samples in the desire time (max_time)
    spike_train = signal.unit_impulse(L, [i * step for i in range(int(np.ceil(L / step)))])
    if min_time > 0.0:
        spike_train[:int(min_time / dt) - 1] = 0.0
    return spike_train


# **********************************************************************************************************************
# ADDITIONAL FUNCTIONS
def m_time():
    return time.time()


def check_create_folder(aux_fold):
    if not os.path.isdir(aux_fold):
        os.mkdir(aux_fold)


def print_time(ts, msg):
    ms_res = ts * 1000
    min_res = ts / 60
    # print('Execution time:', ts, 'milliseconds')
    print(str(msg) + '. Execution time:', str(timedelta(seconds=ts)))


def loadObject(name, path='./'):
    """
    This function loads and returns an object, which should be located in path
    :param name: (String) name of the object-file
    :param path: (String)
    :return:
    res_object: (object)
    """
    pickleFile = open(path + name, 'rb')
    res_object = pickle.load(pickleFile)
    pickleFile.close()
    return res_object


def saveObject(obj, name, path='./'):
    """
    This function saves an object in path
    :param obj: object to be saved
    :param name: (String) name of the object-file
    :param path: (String)
    """
    if not os.path.isfile(path + name):
        pickleFile = open(path + name, 'wb')
        pickle.dump(obj, pickleFile)
        pickleFile.close()


def check_file(file):
    return os.path.isfile(file)


def rmse(x, y):
    if x.ndim != y.ndim:
        while x.ndim < y.ndim:
            x = np.expand_dims(x, axis=x.ndim)
        while y.ndim < x.ndim:
            y = np.expand_dims(y, axis=x.ndim)

    # aligning time dimension in last dimension
    if x.shape[-1] != y.shape[-1]:
        x = x.T

    # Computing RMSE in dim 1 (last dimension) for preserving the num of synapses
    return np.square(np.subtract(x, y)).mean(axis=1)


def su(params):
    return np.random.uniform(params[0], params[1])


def sn(params):
    return np.random.normal(params[0], params[1])


def plot_temp_mssm(model_stp, fig, titleSize=12, labelSize=9, ind_interest=0):
    time_vector = model_stp.time_vector
    output = model_stp.get_output()

    ax = fig.add_subplot(231)
    if ind_interest is None:
        ax.fill_between(time_vector, np.min(model_stp.C, axis=0), np.max(model_stp.C, axis=0), color="darkgrey",
                        alpha=0.3)
        ax.plot(time_vector, np.mean(model_stp.C, axis=0), c='black')
    else:
        ax.plot(time_vector, model_stp.C[ind_interest, :])
    ax.set_title(r'C(t) [$\mu$M]', fontsize=titleSize)
    ax.set_xlabel('time (s)', c='gray', fontsize=labelSize)
    # ax.set_ylabel(r'', c='gray', fontsize=labelSize)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(8)
    ax.grid()

    ax = fig.add_subplot(232)
    if ind_interest is None:
        ax.fill_between(time_vector, np.min(model_stp.V, axis=0), np.max(model_stp.V, axis=0), color="darkgrey",
                        alpha=0.3)
        ax.plot(time_vector, np.mean(model_stp.V, axis=0), c='black')
    else:
        ax.plot(time_vector, model_stp.C[ind_interest, :])
    ax.set_title(r'V(t) [$\mu$M]', fontsize=titleSize)
    ax.set_xlabel('time (s)', c='gray', fontsize=labelSize)
    # ax.set_ylabel(r'', fontsize=labelSize)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(8)
    ax.grid()

    ax = fig.add_subplot(233)
    if ind_interest is None:
        ax.fill_between(time_vector, np.min(model_stp.P, axis=0), np.max(model_stp.P, axis=0), color="darkgrey",
                        alpha=0.3)
        ax.plot(time_vector, np.mean(model_stp.P, axis=0), c='black')
    else:
        ax.plot(time_vector, model_stp.P[ind_interest, :])
    ax.set_title('$P(t)$', fontsize=titleSize)
    ax.set_xlabel('time (s)', c='gray', fontsize=labelSize)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(8)
    ax.grid()

    ax = fig.add_subplot(234)
    if ind_interest is None:
        ax.fill_between(time_vector, np.min(model_stp.N, axis=0), np.max(model_stp.N, axis=0), color="darkgrey",
                        alpha=0.3)
        ax.plot(time_vector, np.mean(model_stp.N, axis=0), c='black')
    else:
        ax.plot(time_vector, model_stp.N[ind_interest, :])
    ax.set_title(r'$N(t)$ [$\mu$M]', fontsize=titleSize)
    ax.set_xlabel('time (s)', c='gray', fontsize=labelSize)
    # ax.set_ylabel('$\mu$M')
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(8)
    ax.grid()

    ax = fig.add_subplot(235)
    if ind_interest is None:
        ax.plot(time_vector, np.mean(output, axis=0), c='black')
        ax.fill_between(time_vector, np.min(output, axis=0), np.max(output, axis=0), color="darkgrey",
                        alpha=0.3)
    else:
        ax.plot(time_vector, output[ind_interest, :])
    ax.set_title('$EPSP(t)$ [mV]', fontsize=titleSize)
    ax.set_xlabel('time (s)', c='gray', fontsize=labelSize)
    # ax.set_ylabel('$\mu$M')
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(8)
    ax.grid()

    fig.subplots_adjust(top=0.92)
    # Final adjustments
    fig.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)


def plot_temp_lap(model_stp, fig, titleSize=12, labelSize=9, ind_interest=0):
    time_vector = model_stp.time_vector
    output = model_stp.get_output()

    ax = fig.add_subplot(231)
    if ind_interest is None:
        ax.fill_between(time_vector, np.min(model_stp.Cai, axis=0), np.max(model_stp.Cai, axis=0), color="darkgrey",
                        alpha=0.3)
        ax.plot(time_vector, np.mean(model_stp.Cai, axis=0), c='black')
    else:
        ax.plot(time_vector, model_stp.Cai[ind_interest, :])
    ax.set_title(r'C_{ai}(t) [$\mu$M]', fontsize=titleSize)
    ax.set_xlabel('time (s)', c='gray', fontsize=labelSize)
    # ax.set_ylabel(r'', c='gray', fontsize=labelSize)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(8)
    ax.grid()

    ax = fig.add_subplot(232)
    if ind_interest is None:
        ax.fill_between(time_vector, np.min(model_stp.Rrel, axis=0), np.max(model_stp.Rrel, axis=0), color="darkgrey",
                        alpha=0.3)
        ax.plot(time_vector, np.mean(model_stp.Rrel, axis=0), c='black')
    else:
        ax.plot(time_vector, model_stp.Rrel[ind_interest, :])
    ax.set_title(r'R_{rel}(t)', fontsize=titleSize)
    ax.set_xlabel('time (s)', c='gray', fontsize=labelSize)
    # ax.set_ylabel(r'', fontsize=labelSize)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(8)
    ax.grid()

    ax = fig.add_subplot(233)
    if ind_interest is None:
        ax.fill_between(time_vector, np.min(model_stp.Krecov, axis=0), np.max(model_stp.Krecov, axis=0), color="darkgrey",
                        alpha=0.3)
        ax.plot(time_vector, np.mean(model_stp.Krecov, axis=0), c='black')
    else:
        ax.plot(time_vector, model_stp.Krecov[ind_interest, :])
    ax.set_title(r'K_{recov}(t) [$ms^{-1}$]', fontsize=titleSize)
    ax.set_xlabel('time (s)', c='gray', fontsize=labelSize)
    # ax.set_ylabel(r'', fontsize=labelSize)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(8)
    ax.grid()

    ax = fig.add_subplot(234)
    if ind_interest is None:
        ax.fill_between(time_vector, np.min(model_stp.Prel, axis=0), np.max(model_stp.Prel, axis=0), color="darkgrey",
                        alpha=0.3)
        ax.plot(time_vector, np.mean(model_stp.Prel, axis=0), c='black')
    else:
        ax.plot(time_vector, model_stp.Prel[ind_interest, :])
    ax.set_title('$P_{rel}(t)$', fontsize=titleSize)
    ax.set_xlabel('time (s)', c='gray', fontsize=labelSize)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(8)
    ax.grid()

    ax = fig.add_subplot(235)
    if ind_interest is None:
        ax.fill_between(time_vector, np.min(model_stp.FluxGlu, axis=0), np.max(model_stp.FluxGlu, axis=0), color="darkgrey",
                        alpha=0.3)
        ax.plot(time_vector, np.mean(model_stp.FluxGlu, axis=0), c='black')
    else:
        ax.plot(time_vector, model_stp.FluxGlu[ind_interest, :])
    ax.set_title(r'$Flux_{Glu}(t)$ [#$ms^{-1}$]', fontsize=titleSize)
    ax.set_xlabel('time (s)', c='gray', fontsize=labelSize)
    # ax.set_ylabel('$\mu$M')
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(8)
    ax.grid()

    ax = fig.add_subplot(236)
    if ind_interest is None:
        ax.plot(time_vector, np.mean(output, axis=0), c='black')
        ax.fill_between(time_vector, np.min(output, axis=0), np.max(output, axis=0), color="darkgrey",
                        alpha=0.3)
    else:
        ax.plot(time_vector, output[ind_interest, :])
    ax.set_title('$E_{psc}(t)$ [pA]', fontsize=titleSize)
    ax.set_xlabel('time (s)', c='gray', fontsize=labelSize)
    # ax.set_ylabel('$\mu$M')
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(8)
    ax.grid()

    fig.subplots_adjust(top=0.92)
    # Final adjustments
    fig.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)


def plot_temp_tm(model_stp, fig, titleSize=12, labelSize=9, ind_interest=0):
    time_vector = model_stp.time_vector
    output = model_stp.get_output()

    ax = fig.add_subplot(131)
    if ind_interest is None:
        ax.fill_between(time_vector, np.min(model_stp.U, axis=0), np.max(model_stp.U, axis=0), color="darkgrey",
                        alpha=0.3)
        ax.plot(time_vector, np.mean(model_stp.U, axis=0), c='black')
    else:
        ax.plot(time_vector, model_stp.U[ind_interest, :])
    ax.set_title(r'U(t) - Utilisation', fontsize=titleSize)
    ax.set_xlabel('time (s)', c='gray', fontsize=labelSize)
    # ax.set_ylabel(r'', c='gray', fontsize=labelSize)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(8)
    ax.grid()

    ax = fig.add_subplot(132)
    if ind_interest is None:
        ax.fill_between(time_vector, np.min(model_stp.R, axis=0), np.max(model_stp.R, axis=0), color="darkgrey",
                        alpha=0.3)
        ax.plot(time_vector, np.mean(model_stp.R, axis=0), c='black')
    else:
        ax.plot(time_vector, model_stp.R[ind_interest, :])
    ax.set_title(r'R(t) - Resources', fontsize=titleSize)
    ax.set_xlabel('time (s)', c='gray', fontsize=labelSize)
    # ax.set_ylabel(r'', fontsize=labelSize)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(8)
    ax.grid()

    ax = fig.add_subplot(133)
    if ind_interest is None:
        ax.fill_between(time_vector, np.min(model_stp.I_out, axis=0), np.max(model_stp.I_out, axis=0), color="darkgrey",
                        alpha=0.3)
        ax.plot(time_vector, np.mean(model_stp.I_out, axis=0), c='black')
    else:
        ax.plot(time_vector, model_stp.P[ind_interest, :])
    ax.set_title('$output(t)$', fontsize=titleSize)
    ax.set_xlabel('time (s)', c='gray', fontsize=labelSize)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(8)
    ax.grid()

    fig.subplots_adjust(top=0.92)
    # Final adjustments
    fig.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)


def plot_freq_efficacies(pf):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(pf.fa.loop_frequencies, pf.reference_signal, label='reference')
    ax.plot(pf.fa.loop_frequencies, pf.fa.efficacy[0, :], label='example output')
    ax.plot(pf.fa.loop_frequencies, np.mean(pf.fa.efficacy, axis=0), c='black', label='output mean')
    ax.fill_between(pf.fa.loop_frequencies, np.min(pf.fa.efficacy, axis=0), np.max(pf.fa.efficacy, axis=0),
                     color="darkgrey", alpha=0.3)
    ax.set_title("Frequency responses of reference and output")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Efficacy")
    ax.legend()
    ax.grid()
    fig.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)

    fig1 = plt.figure()
    fig1.suptitle("Frequency responses of efficacies")
    ax = fig1.add_subplot(211)
    ax.plot(pf.fa.loop_frequencies, pf.fa.efficacy[0, :], label="phasic eff", color='tab:red', alpha=1)
    ax.plot(pf.fa.loop_frequencies, pf.fa.efficacy_2[0, :], label="phasic eff2", color='tab:blue', alpha=1)
    ax.plot(pf.fa.loop_frequencies, pf.fa.efficacy_3[0, :], label="phasic eff3", color='tab:green', alpha=1)
    ax.plot(pf.fa.loop_frequencies, pf.fa.efficacy_tonic[0, :], label="tonic eff", color='tab:red', alpha=0.5)
    ax.plot(pf.fa.loop_frequencies, pf.fa.efficacy_2_tonic[0, :], label="tonic eff2", color='tab:blue', alpha=0.5)
    ax.plot(pf.fa.loop_frequencies, pf.fa.efficacy_3_tonic[0, :], label="tonic eff3", color='tab:green', alpha=0.5)
    ax.set_title("Phasic and tonic components", color='gray')
    ax.set_xlabel("Efficacies")
    ax.set_xlabel("Frequency (Hz)")
    ax.legend()
    ax.grid()
    ax3 = fig1.add_subplot(212)
    ax3.plot(pf.fa.loop_frequencies, pf.fa.eff_st[0, :], label="phasic eff")
    ax3.plot(pf.fa.loop_frequencies, pf.fa.eff_st_tonic[0, :], label="tonic eff")
    ax3.set_title("Non-normalize efficacy", color='gray')
    ax3.set_xlabel("Efficacies")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.legend()
    ax3.grid()
    fig1.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)


def plot_freq_analysis(fa, title_aux, plot=True, ind=0):
    plot_title = "Frequency analysis of Model" + title_aux
    subtitle_size = 12
    subtitle_color = 'gray'
    org_plots = 230
    ind_plots = [org_plots + 1, org_plots + 2, org_plots + 3, org_plots + 4, org_plots + 5]
    time_vectors = [fa.loop_frequencies for _ in range(len(ind_plots))]
    plots = [fa.efficacy[ind, :], fa.efficacy_2[ind, :], fa.efficacy_3[ind, :], fa.time_ss[ind, :], fa.time_max[ind, :]]
    subplot_title = ["Efficacy st/0", "Efficacy st/max", "Efficacy max/0", "time to steady state", "time to max"]
    ylabels = ['EPSPst/EPSP0', 'EPSPst/EPSPmax', 'EPSPmax/EPSP0', 'time(s)', 'time(s)']
    colorplot = ['black', 'black', 'black', 'black', 'black']
    xlabels = ['freq (Hz)', 'freq (Hz)', 'freq (Hz)', 'freq (Hz)', 'freq (Hz)']
    plot_syn_dyn(time_vectors, plot_title, ind_plots, plots, subplot_title=subplot_title, xlabels=xlabels,
                 ylabels=ylabels, plot=plot, color_plot=colorplot)


def poisson_generator2(dt, Lt, rate, n, myseed=None):
    """
    Generates poisson trains

    Args:
    dt         : dt
    Lt         : Lt
    rate       : noise amplitute [Hz]
    n          : number of Poisson trains
    myseed     : random seed. int or boolean

    Returns:
    pre_spike_train : spike train matrix, ith row represents whether
                      there is a spike in ith spike train over time
                      (1 if there is a spike, 0 otherwise)
    """

    # set random seed
    if myseed is not None:
        np.random.seed(seed=myseed)
    else:
        np.random.seed(1807)

    # generate uniformly distributed random variables
    u_rand = np.random.rand(n, Lt)

    # generate Poisson train
    poisson_train = 1. * (u_rand < (rate * dt))

    return poisson_train


def lowpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = signal.sosfiltfilt(sos, data)
    return filtered_data


# **********************************************************************************************************************
# LOADING EXAMPLES OF FITTING
class Example_fitting:
    def __init__(self, model_str, path_signals=None, path_outputs=None):

        # Model
        self.model_str = model_str

        # Definition of time parameters
        self.sfreq = None
        self.dt = None
        self.max_t = None
        self.input_factor = None
        self.output_factor = None
        self.time_vector = None
        self.L = None
        self.r = None

        # params dictionaries
        self.dict_params = None
        self.sim_params = None
        self.DE_params = []

        # parameters of paths
        self.path_signals = path_signals if path_signals is not None else path_signals_ext
        self.path_outputs = path_outputs if path_outputs is not None else path_outputs_ext

    def initial_params(self, ind, *args_):
        """
        args =
        :param ind:
        :param args_: tuple() [sfreq_v, max_t_v, input_factor_v, output_factor_v, description, r_v]
        :return:
        """
        args = args_[0]

        if args[0] is None:
            sfreq_v = sfreq_ext
        if args[1] is None:
            max_t_v = max_t_ext
        if args[2] is None:
            input_factor_v = input_factor_ext
        if args[3] is None:
            output_factor_v = output_factor_ext
        if args[5] is None:
            r_v = r_ext

        self.sfreq = args[0]
        self.dt = 1 / self.sfreq
        self.max_t = args[1]
        self.input_factor = args[2]
        self.output_factor = args[3]
        self.time_vector = np.arange(0, self.max_t, self.dt)
        self.L = self.time_vector.shape[0]
        self.r = args[5]

        # assigning dictionaries of parameters
        self.params_sim()
        self.params_dict(description=args[4])

    def params_DE(self, strategy='best1bin', generations=1000, popsize=15, tol=0.01, mutation=(0.5, 1),
                  recombination=0.7, seed=None, callback=None, disp=True, polish=True, init='latinhypercube', atol=0,
                  updating='immediate', workers=1, constraints=(), x0=None, integrality=None, vectorized=False):
        self.DE_params.append(strategy)
        self.DE_params.append(generations)
        self.DE_params.append(popsize)
        self.DE_params.append(tol)
        self.DE_params.append(mutation)
        self.DE_params.append(recombination)
        self.DE_params.append(seed)
        self.DE_params.append(callback)
        self.DE_params.append(disp)
        self.DE_params.append(polish)
        self.DE_params.append(init)
        self.DE_params.append(atol)
        self.DE_params.append(updating)
        self.DE_params.append(workers)
        self.DE_params.append(constraints)
        self.DE_params.append(x0)
        self.DE_params.append(integrality)
        self.DE_params.append(vectorized)

    def params_dict(self, ext_par=None, description=''):
        dt = self.dt
        if self.model_str == 'MSSM':
            if ext_par is None:
                ext_pa = ext_par_mssm
            else:
                ext_pa = ext_par
            min_n = 2 * dt
            self.dict_params = {'model_str': self.model_str,
                                'params_name': params_name_mssm,
                                'bo': ((2 * dt, 0.0,  0.02, 2 * dt, 0.0,  1.0, 1e-1, 2 * dt,  1e-3, 2 * dt),
                                       (1e-1,   1e-1,  1e1,  1.0,   2e-2, 1e2,  1.0,  10 * dt, 1e0,  10 * dt)),
                                'ODE_mode': 'ODE',
                                'ind_experiment': 0,
                                'only_spikes': False,
                                'path': self.path_outputs,
                                'description_file': description,
                                'output_factor': self.output_factor,
                                'frequency_reference': False,
                                'ref_freq_vector': None,
                                }

        if self.model_str == 'TM':
            if ext_par is None:
                ext_pa = ext_par_tm
            else:
                ext_pa = ext_par
            self.dict_params = {'model_str': self.model_str,
                                'params_name': params_name_tm,
                                'bo': ((0.0, 2 * dt, 2 * dt, 0.0, 2 * dt),
                                       (1.0, 1.0,    1.0,    1e4, 10.0)),
                                'ODE_mode': 'ODE',
                                'ind_experiment': 0,
                                'only_spikes': False,
                                'path': self.path_outputs,
                                'description_file': description,
                                'output_factor': self.output_factor,
                                'frequency_reference': False,
                                'ref_freq_vector': None,
                                }

        if self.model_str == "LAP":
            if ext_par is None:
                ext_pa = ext_par_tm
            else:
                ext_pa = ext_par
            dt = self.dt
            max_k = 1 / (2 * dt * 1e3)
            self.dict_params = {'model_str': self.model_str,
                                'params_name': params_name_lap,
                                'bo': ((1e2, 2 * dt, 0.0, 0.0,   0.0,   0.0,  0.98, 0.0,   2 * dt,  0.0, 0.0, 0.0),
                                       (1e3, 1.0,    1e2, max_k, max_k, 0.98, 1.0,  500.0, 1e-2,    1.0, 1e4, 1e4)),
                                'ODE_mode': 'ODE',
                                'ind_experiment': 0,
                                'only_spikes': False,
                                'path': self.path_outputs,
                                'description_file': description,
                                'output_factor': self.output_factor,
                                'frequency_reference': False,
                                'ref_freq_vector': None,
                                }

    def params_sim(self):
        self.sim_params = {'sfreq': self.sfreq, 'max_t': self.max_t, 'L': self.L, 'time_vector': self.time_vector}
