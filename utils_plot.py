import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.signal as signal

colors = ["#ED0000", "#FFF200", "#0000ED", "#FF7E27", "#00ED00", "#7F3F3F", "#B97A57", "#FFDBB7", "#ED00ED", "#00EDED",
          "#FF7F7F", "#FFC90D", "#6C6CFF", "#FFB364", "#22B14C", "#D1A5A5", "#008040", "#AEFFAE", "#FF84FF", "#95FFFF",
          "#FFB7B7", "#808000", "#0080FF", "#C46200", "#008080", "#800000", "#B0B0FF", "#FFFFB3", "#FF3E9E", "#BBBB00",
          "#00ED00", "#00FFF2", "#ED0000", "#27FF7E", "#0000ED", "#3F7F3F", "#57B97A", "#B7FFDB", "#EDED00", "#ED00ED"]


def check_timeVector_series(time_vector, time_s, index):
    cnt = 0
    time_v = time_vector
    # while len(time_vector) != len(time_s) and cnt < 3:
    while len(time_v) != len(time_s) and cnt < 3:
        time_v = time_vector[index]
        cnt += 1
    assert cnt < 3, "Mismatch of shapes between time_vector and signal"
    return time_v, time_s


def lowpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = signal.sosfiltfilt(sos, data)
    return filtered_data


def norm_array(np_array, compute_norm=True, min_n=None, max_n=None):
    """
    Min max normalization of np_array if compute_norm is True. If min_n and max_n are specified, then use
    them as the source range to normalize to [0, 1]
    :param np_array:
    :param compute_norm:
    :param min_n:
    :param max_n:
    :return:
    either normalized array or np_array in case compute_norm is False
    """
    if np_array is not None:
        min_n = np.min(np_array) if min_n is None else min_n
        max_n = np.max(np_array) if max_n is None else max_n
        if compute_norm:
            return np.copy((np_array - min_n) / (max_n - min_n))
        else:
            return np_array
    else:
        return None


def plot_res_mssm(time_vector, input_signal, reference_signal, label):
    fig = plt.figure(figsize=(12, 2.8))
    plt.suptitle(label)
    ax = fig.add_subplot(121)
    ax.plot(time_vector, input_signal, c='gray')
    ax.set_xlabel("time(s)")
    ax.set_ylabel("Spikes")
    ax.set_title("Input Spike train")
    ax.grid()
    ax = fig.add_subplot(122)
    for i in range(reference_signal.shape[0]): ax.plot(time_vector, reference_signal[i, :])
    ax.set_xlabel("time(s)")
    ax.set_ylabel("A")
    ax.set_title("Postsynaptic response")
    ax.grid()
    fig.tight_layout(pad=0.5, w_pad=0.1, h_pad=1.0)


def plot_syn_dyn(time_vector, plot_title, ind_plots, plots, plot_leg=None, color_plot=None, alphas=None,
                 subplot_title=None, xlabels=None, ylabels=None, ylim_ranges=None, subtitle_size=12,
                 subtitle_color='gray', fig_size=(17, 8.5), fig_pad=.8, save=False, plot=True, path_to_save="",
                 yerr=None, xerr=None, uplims=None, lolims=None, vlines=None, xscale='linear', std_plt1=None,
                 std_plt2=None, x_axis_log=False, y_axis_log=False):
    # Default graphical parameters
    alph = None
    labl = None
    col = None
    yer = None
    xer = None
    uplim = False
    lolim = False
    # Creating plot
    fig = plt.figure(figsize=fig_size)
    fig.suptitle(plot_title)

    # Looping through the plots
    for index in range(len(ind_plots)):

        # Adding a new axis for each subplot
        ax = fig.add_subplot(ind_plots[index])

        # if more than 1 graph should be plotted in ax
        if isinstance(plots[index], list) or (isinstance(plots[index], np.ndarray) and plots[index].ndim > 1):

            # Looping through the graphs
            for pl in range(len(plots[index])):

                # Specifying graphical parameters
                if alphas is not None:
                    alph = alphas[index][pl]
                if plot_leg is not None:
                    labl = plot_leg[index][pl]
                if color_plot is not None:
                    col = color_plot[index][pl]
                if yerr is not None:
                    yer = yerr[index][pl]
                if xerr is not None:
                    xer = xerr[index][pl]
                if uplims is not None:
                    uplim = uplims[index][pl]
                if lolims is not None:
                    lolim = lolims[index][pl]
                if std_plt1 is not None:
                    std_pl1 = std_plt1[index][pl]
                if std_plt2 is not None:
                    std_pl2 = std_plt2[index][pl]
                # Plotting
                time_v, time_s = check_timeVector_series(time_vector, plots[index][pl])
                ax.errorbar(time_v, time_s, label=labl, c=col, alpha=alph, yerr=yer, xerr=xer,
                            uplims=uplim, lolims=lolim, capsize=0.1)
                if std_plt1 is not None or std_plt2 is not None:
                    ax.fill_between(time_vector[index], plots[index][pl] - std_pl1, plots[index][pl] + std_pl2,
                                    color="grey", alpha=0.5)
            # Adding legends
            ax.legend(framealpha=0.3)

        # Otherwise plot only once
        else:

            # Specifying graphical parameters
            if alphas is not None:
                alph = alphas[index]
            if color_plot is not None:
                col = color_plot[index]
            if yerr is not None:
                yer = yerr[index]
            if xerr is not None:
                xer = xerr[index]
            if uplims is not None:
                uplim = uplims[index]
            if lolims is not None:
                lolim = lolims[index]
            if std_plt1 is not None:
                std_pl1 = std_plt1[index]
            if std_plt2 is not None:
                std_pl2 = std_plt2[index]
            # Plotting
            time_v, time_s = check_timeVector_series(time_vector, np.squeeze(plots[index]), index)
            ax.errorbar(time_v, time_s, alpha=alph, c=col, yerr=yer, xerr=xer,
                        uplims=uplim, lolims=lolim, capsize=0.1)
            if std_plt1 is not None or std_plt2 is not None:
                ax.fill_between(time_vector[index], plots[index] - std_pl1, plots[index] + std_pl2,
                                color="grey", alpha=0.5)
        ax.set_xscale(xscale)
        ax.grid()

        # Plotting vertical line (if necessary)
        if vlines is not None:
            if vlines[index]:
                ax.axvline(vlines[index], c='red')

        # Specifying graphical parameters
        if subplot_title is not None:
            ax.set_title(subplot_title[index], size=subtitle_size, c=subtitle_color)
        if ylabels is not None:
            ax.set_ylabel(ylabels[index])
        if xlabels is not None:
            ax.set_xlabel(xlabels[index])
        if ylim_ranges is not None:
            ax.set_ylim(ylim_ranges[index])
        if x_axis_log:
            ax.set_xscale('log')
        if y_axis_log:
            ax.set_yscale('log')

    # Adjust graph
    fig.tight_layout(pad=fig_pad)

    # If save is True, then save figure in the path_to_save
    if save:
        path = path_to_save
        path += ".png"
        plt.savefig(path)

    # Close figure if condition is True
    if not plot or save:
        plt.close(fig)

    # If True, show the figure
    if plot:
        plt.show()


def plot_hist_pdf(data, labels, title, medians=False, colors=None, ylimMax=None, ylimMin=None, rotationLabels=0,
                  sizeLabels=8, sizeTitle=12, figSize=None, plotFig=False, returnAx=False, fig=None, pos=None,
                  posInset=None, xAxisSci=False, yAxisFontSize=8, hatchs=None, binSize=30):
    # Create a figure instance
    if (plotFig and fig is None) or fig is None:
        if figSize is None:
            fig = plt.figure()
        else:
            fig = plt.figure(figsize=figSize)

    if returnAx:
        # Create an axes instance
        ax = fig.add_subplot(pos[0], pos[1], pos[2])
    else:
        # Create an axes instance
        ax = fig.add_subplot(111)

    # Getting rid of the nan values
    if isinstance(data, list):
        data = np.array(data).T
    # data_list = [[] for _ in range(data.shape[1])]
    # for cols in range(data.shape[1]):
    #     d_aux = data[:, cols]
    #     data_list[cols] = d_aux[~np.isnan(d_aux)]
    data_list = list(data)

    # Create the boxplot
    # colors_hist = [colors[i + 4] for i in range(data.shape[1])]  # [colors[i + 32] for i in range(data.shape[1])]
    colors_hist = colors[4]  # [colors[i + 32] for i in range(data.shape[1])]
    bp = ax.hist(data_list, bins=binSize, density=False, orientation='horizontal', cumulative=False,
                 histtype='stepfilled', label=labels, color=colors_hist, alpha=0.5)
    bp = ax.hist(data_list, bins=binSize, density=False, orientation='horizontal', cumulative=False, histtype='step',
                 color='#808080')  # colors_hist)
    #              color = ['#808080' for i in range(data.shape[1])])  # colors_hist)
    for bar, hatch in zip(ax.patches, hatchs):  # loop over bars and hatches to set hatches in correct order
        bar.set_hatch(hatch)
        # bar.set_edgecolor('#000000')
    # Add a horizontal grid to the plot, but make it very light in color
    ax.yaxis.grid(True, linestyle=':', which='major', color='lightgrey', alpha=0.5, label=labels)
    # Hide these grid behind plot objects
    ax.set_axisbelow(True)

    # Custom x-axis labels
    # ax.set_xticklabels(labels, rotation=rotationLabels, fontsize=sizeLabels)
    # ax.set_xticks([i for i in range(1, len(labels) + 1)], labels, rotation=rotationLabels, fontsize=sizeLabels)
    ax.tick_params(axis='x', labelsize=sizeLabels, colors='gray')
    ax.tick_params(axis='y', labelsize=sizeLabels, colors='gray')
    ax.set_title(title, size=sizeTitle)

    # Format of y-axis
    if xAxisSci:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.offsetText.set_color('black')  # set_position((1.3, 0))
        ax.yaxis.offsetText.set_fontsize(yAxisFontSize)

    # Setting limits to y-axis
    if ylimMax is not None:
        plt.ylim(top=ylimMax)
    if ylimMin is not None:
        plt.ylim(bottom=ylimMin)


def plot_net_depolarization(fa, loop_frequencies):
    netmem = fa.efficacy[0, :] * loop_frequencies
    inv_freq = 1 / np.array(loop_frequencies)
    plt.figure()
    plt.plot(loop_frequencies, netmem)  # * 60e-3)
    plt.grid()
    plt.xlabel("f (Hz)")
    plt.ylabel("Net depolarization (mV)")
    plt.title("EPSP_st * 1/f", c='gray')

    plt.figure()
    fac_mul = 8.7e-2
    plt.plot(loop_frequencies, fa.efficacy_2[0, :] * fac_mul, label='EPSP_st')
    plt.plot(loop_frequencies, inv_freq, "--", label="1/f", c='orange')
    plt.grid()
    plt.xlabel("f (Hz)")
    plt.ylabel("EPSPst * " + str(fac_mul))
    plt.ylim((0, 40e-3))
    plt.title("EPSP_st", c='gray')
    plt.legend()


def plot_isi_histogram(histograms, ind):
    """
    Plot histograms of ISI from function inter_spike_intervals()
    Parameters
    ----------
    histograms
    ind

    Returns
    -------

    """
    counts, bin_edges = histograms[ind]
    if len(counts) == 0:
        print(f"No data for histogram index {ind}")
        return
    width = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + width / 2
    plt.figure()
    plt.bar(bin_centers, counts, width=width, edgecolor='black')
    plt.xlabel('Inter-spike interval (seconds)')
    plt.ylabel('Count')
    plt.title(f'ISI Histogram {ind}')
    plt.grid()
    plt.show()


def plot_gc_sin_three_scenarios(fig, i, time_vector, mean_rate, max_oscil, lif, coff, sfreq, modulation_signal1,
                                modulation_signal2, modulated_signal1, modulated_signal2):
    # Organising signals for plotting
    plot_mod1 = modulation_signal1
    plot_mod2 = modulation_signal2
    plot_s1 = modulated_signal1[0, :]
    plot_s2 = modulated_signal2[0, :]

    if i == 1:
        plot_mod1 = modulation_signal2
        plot_mod2 = modulation_signal1
        plot_s1 = modulated_signal2[0, :]
        plot_s2 = modulated_signal1[0, :]

    col = "#71BE56"
    if i == 1: col = "#0192C8"
    ax7 = fig.add_subplot(3, 2, (i * 2) + 1)
    ax7.plot(time_vector, plot_mod1, c="#71BE56")
    ax7.set_ylabel("Rate (Hz)")
    if i == 0 or i == 2: ax7.grid()
    ax7.set_ylim(10, 850)
    ax7.set_title(r'%dsin(0.2$\pi$t) + %d' % (mean_rate[i], max_oscil[i]), c=col)
    if i == 2: ax7.set_ylim(mean_rate[0] - 20, mean_rate[0] + 20)
    ax7.tick_params(axis='y', labelcolor="#71BE56")
    ax7b = ax7.twinx()
    ax7b.plot(time_vector, plot_mod2, c="#0192C8")
    if i == 1: ax7b.grid()
    ax7b.tick_params(axis='y', labelcolor="#0192C8")
    ax7b.set_ylim(0, 50)

    ax8 = fig.add_subplot(3, 2, (i * 2) + 2)
    ax8.plot(time_vector, lif.membrane_potential[0, :], c='black', alpha=0.8)
    ax8.plot(time_vector, lowpass(lif.membrane_potential[0, :], coff, sfreq), c='tab:red', alpha=0.8)
    ax8.grid()
    ax8.set_ylim(-65.5, -27)
    ax8.set_ylabel("mV")
    # ax8.set_title(f"LIF, diff median {diff_median_mempot:.3f}mV", c='gray')

    if i == 2:
        ax7.set_xlabel("time (s)")  # , fontsize=18)
        ax8.set_xlabel("time (s)")


def plot_gc_prop_input_example(time_vector, dt, ind_exp, sin_high_rate, high_rate_spikes):
    fonts = 8  # 12  12
    c1 = "tab:orange"  # "#AFAFAF"  # "#0192C8"
    c2 = "tab:blue"  # "#000000"  # "#71BE56"
    fig_esann3 = plt.figure(figsize=(8, 3))
    ax1s3 = fig_esann3.add_subplot(211)
    ax1s3.plot(time_vector, sin_high_rate, c=c2)
    if ind_exp == 0: ax1s3.set_ylabel("Rate (Hz)", fontsize=fonts)
    ax1s3.set_ylim(80, 170)
    ax1s3.set_title(r'Proportional change schema ($\delta = 50\%$)', fontsize=fonts + 6, c="gray")
    # ax1s3.set_title('Sinusoidal pattern for proportional change of firing rate at baseline rate 100Hz',
    #                 fontsize=fonts + 6, c="gray")
    # ax1s3.grid()
    ax1s3.set_ylabel("Rate (Hz)", fontsize=fonts - 1)
    ax1s3.yaxis.set_tick_params(labelsize=fonts)
    ax1s3.set_xlabel("Time  (s)", fontsize=fonts)
    ax1s3.xaxis.set_tick_params(labelsize=fonts)
    ax1s3.legend(framealpha=0.3)

    t0, t1, t2, t3 = 0, 2, 4, 6

    # Separating bars
    for x in [t1, t2]:
        ax1s3.axvline(x, color="gray", linestyle="--", linewidth=1)
    # Labels centered in each window
    ymin, ymax = ax1s3.get_ylim()
    ypos = ymax - 0.05 * (ymax - ymin)
    ax1s3.text(1, ypos, "ini-window", ha="center", va="top", color='gray')
    ax1s3.text(3, ypos, "mid-window", ha="center", va="top", color='gray')
    ax1s3.text(5, ypos, "end-window", ha="center", va="top", color='gray')

    ax2s3 = fig_esann3.add_subplot(234)
    ax2s3.plot(time_vector[int(1 / dt):int(1.5 / dt)], high_rate_spikes[int(1.0 / dt):int(1.5 / dt)], c=c2)
    ax2s3.set_xlabel("Time (s)", fontsize=fonts)
    ax2s3.xaxis.set_tick_params(labelsize=fonts)
    # ax2s3.grid()
    ax2s3.get_yaxis().set_visible(False)
    ax2s3.set_title("ini-window (100Hz)", c="gray", fontsize=fonts + 4)
    ax3s3 = fig_esann3.add_subplot(235)
    ax3s3.plot(time_vector[int(3 / dt):int(3.5 / dt)], high_rate_spikes[int(3 / dt):int(3.5 / dt)], c=c2)
    ax3s3.set_xlabel("Time (s)", fontsize=fonts)
    ax3s3.xaxis.set_tick_params(labelsize=fonts)
    # ax3s3.grid()
    ax3s3.get_yaxis().set_visible(False)
    ax3s3.set_title("mid-window (150Hz)", c="gray", fontsize=fonts + 4)
    ax4s3 = fig_esann3.add_subplot(236)
    ax4s3.plot(time_vector[int(5 / dt):int(5.5 / dt)], high_rate_spikes[int(5 / dt):int(5.5 / dt)], c=c2)
    ax4s3.set_xlabel("Time (s)", fontsize=fonts)
    ax4s3.xaxis.set_tick_params(labelsize=fonts)
    # ax4s3.grid()
    ax4s3.get_yaxis().set_visible(False)
    ax4s3.set_title("end-window (100Hz)", c="gray", fontsize=fonts + 4)
    fig_esann3.tight_layout(pad=0.5, w_pad=1.0, h_pad=0.1)
    return fig_esann3


def plot_gc_sin_input_example(time_vector, dt, ind_exp, sin_high_rate, sin_low_rate, high_rate_spikes, low_rate_spikes):
    fonts = 8  # 12  12
    c1 = "tab:blue"  # "#AFAFAF"  # "#0192C8"
    c2 = "tab:orange"  # "#000000"  # "#71BE56"
    fig_esann3 = plt.figure(figsize=(8, 3))
    ax1s3 = fig_esann3.add_subplot(211)
    ax1s3.plot(time_vector, sin_high_rate, label="high-firing rates", c=c2)
    if ind_exp == 0: ax1s3.set_ylabel("Rate (Hz)", fontsize=fonts)
    ax1s3.set_ylim(0, 160)
    ax1s3.set_title('Sinusoidal pattern for high firing rates',
                    fontsize=fonts + 6, c="gray")
    # ax1s3.set_title('Sinusoidal pattern for proportional change of firing rate at baseline rate 100Hz',
    #                 fontsize=fonts + 6, c="gray")
    ax1s3.plot(time_vector, sin_low_rate, label="low-firing rates", c=c1)
    # ax1s3.grid()
    ax1s3.set_ylabel("Rate (Hz)", fontsize=fonts - 1)
    ax1s3.yaxis.set_tick_params(labelsize=fonts)
    ax1s3.set_xlabel("Time  (s)", fontsize=fonts)
    ax1s3.xaxis.set_tick_params(labelsize=fonts)
    ax1s3.legend(framealpha=0.3)
    ax2s3 = fig_esann3.add_subplot(234)
    ax2s3.plot(time_vector[:int(0.5 / dt)], high_rate_spikes[:int(0.5 / dt)] + 1.1, c=c2)
    ax2s3.plot(time_vector[:int(0.5 / dt)], low_rate_spikes[:int(0.5 / dt)], c=c1)
    ax2s3.set_xlabel("Time (s)", fontsize=fonts)
    ax2s3.xaxis.set_tick_params(labelsize=fonts)
    # ax2s3.grid()
    ax2s3.get_yaxis().set_visible(False)
    ax2s3.set_title("Firing rates (100Hz)", c="gray", fontsize=fonts + 4)
    ax3s3 = fig_esann3.add_subplot(235)
    ax3s3.plot(time_vector[int(7 / dt):int(7.5 / dt)], high_rate_spikes[int(7 / dt):int(7.5 / dt)] + 1.1, c=c2)
    ax3s3.plot(time_vector[int(7 / dt):int(7.5 / dt)], low_rate_spikes[int(7 / dt):int(7.5 / dt)], c=c1)
    ax3s3.set_xlabel("Time (s)", fontsize=fonts)
    ax3s3.xaxis.set_tick_params(labelsize=fonts)
    # ax3s3.grid()
    ax3s3.get_yaxis().set_visible(False)
    ax3s3.set_title("Firing rates (50Hz)", c="gray", fontsize=fonts + 4)
    ax4s3 = fig_esann3.add_subplot(236)
    ax4s3.plot(time_vector[int(12 / dt):int(12.5 / dt)], high_rate_spikes[int(12 / dt):int(12.5 / dt)] + 1.1, c=c2)
    ax4s3.plot(time_vector[int(12 / dt):int(12.5 / dt)], low_rate_spikes[int(12 / dt):int(12.5 / dt)], c=c1)
    ax4s3.set_xlabel("Time (s)", fontsize=fonts)
    ax4s3.xaxis.set_tick_params(labelsize=fonts)
    # ax4s3.grid()
    ax4s3.get_yaxis().set_visible(False)
    ax4s3.set_title("Firing rates (150Hz)", c="gray", fontsize=fonts + 4)
    fig_esann3.tight_layout(pad=0.5, w_pad=1.0, h_pad=0.1)
    return fig_esann3


def plot_gc_sin_mp_high_rates_esann(fig, ind, ind_exp, time_vector, mean_rate, output_mp_esann, out_ylim_min,
                                    out_ylim_max, output_mp_low_filt_esann):
    # Figure for ESANN paper, 2026
    fonts = 12
    ax1_s = fig.add_subplot(1, 4, ind_exp + 1)
    ax1_s.plot(time_vector, output_mp_esann, c="#AFAFAF")
    if ind_exp == 0: ax1_s.set_ylabel("mem. pot. (mv)")
    ax1_s.yaxis.set_tick_params(labelsize=fonts - 2)
    ax1_s.set_ylim(out_ylim_min, out_ylim_max)
    # ax1_s.set_title(r'%dsin(0.2$\pi$t) + %d' % (mean_rate[0], max_oscil[0]), c='gray')
    if ind == 4: ax1_s.set_title('For baseline rate %dHz' % mean_rate[0], c='gray')
    if ind == 5: ax1_s.set_xlabel("time (s)")
    # ax1_s.tick_params(axis='y', labelcolor="#71BE56")
    ax1_s.plot(time_vector, output_mp_low_filt_esann, c="black")
    ax1_s.xaxis.set_tick_params(labelsize=fonts - 2)
    # ax1_s.grid()

    fig.tight_layout(pad=0.1, w_pad=0.5, h_pad=0.3)


def plot_gc_sin_mp_high_rates(fig, ind, ind_exp, time_vector, mean_rate, output_mp_esann, out_ylim_min, out_ylim_max,
                              output_mp_low_filt_esann, num_graphs=4, pathsave=None, savefig=False):
    # Figure for ESANN paper, 2026
    rows_plot = int(np.ceil(num_graphs / 6))
    fonts = 12
    ax1_s = fig.add_subplot(rows_plot, 6, ind_exp + 1)
    ax1_s.plot(time_vector, output_mp_esann, c="#AFAFAF")
    if ind_exp == 0: ax1_s.set_ylabel("mem. pot. (mv)")
    ax1_s.yaxis.set_tick_params(labelsize=fonts - 2)
    ax1_s.set_ylim(out_ylim_min, out_ylim_max)
    # ax1_s.set_title(r'%dsin(0.2$\pi$t) + %d' % (mean_rate[0], max_oscil[0]), c='gray')
    ax1_s.set_title('B. rate %dHz' % mean_rate[0], c='gray')
    ax1_s.set_xlabel("time (s)")
    # ax1_s.tick_params(axis='y', labelcolor="#71BE56")
    ax1_s.plot(time_vector, output_mp_low_filt_esann, c="black")
    ax1_s.xaxis.set_tick_params(labelsize=fonts - 2)
    # ax1_s.grid()

    fig.tight_layout(pad=0.1, w_pad=0.5, h_pad=0.3)
    if savefig and pathsave is not None: fig.savefig(pathsave, format='png')


def plot_gc_sin_freq_response_efficacy(loop_frequencies, fa, title):
    fig_esann2 = plt.figure(figsize=(5, 2))
    fonts = 12
    ax1s2 = fig_esann2.add_subplot(1, 1, 1)
    ax1s2.plot(loop_frequencies, fa.efficacy[0, :], color="black")
    # ax1s2.set_xscale('log')
    # ax1s2.grid()
    ax1s2.set_xlabel("Frequency (Hz)", fontsize=fonts)
    ax1s2.xaxis.set_tick_params(labelsize=fonts)
    ax1s2.set_ylabel("Current (pA)", fontsize=fonts)
    ax1s2.yaxis.set_tick_params(labelsize=fonts)
    ax1s2.set_title(title, fontsize=fonts + 4)
    range_eff = np.max(fa.efficacy[0]) - np.min(fa.efficacy[0])
    ind_eff = np.where(fa.efficacy[0] < (0.01 * range_eff) + np.min(fa.efficacy[0]))
    freq_st = loop_frequencies[ind_eff[0][0]]
    ax1s2.plot([freq_st, freq_st], [np.min(fa.efficacy[0]), np.max(fa.efficacy[0])], color="#AFAFAF")
    empty_patch = mpatches.Patch(color='none', label=r'$freq_{st}=$%dHz' % freq_st)
    ax1s2.legend(handles=[empty_patch], loc='upper right', fontsize=fonts)
    fig_esann2.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)


def plot_gc_sin_statistics(res_per_reali, mean_rates):
    prop_high_rate_max_pos = res_per_reali[0, 0, :]
    prop_high_rate_min_neg = res_per_reali[1, 0, :]
    prop_high_rate_q1_pos = res_per_reali[2, 0, :]
    prop_high_rate_q90_pos = res_per_reali[3, 0, :]
    prop_high_rate_q1_neg = res_per_reali[4, 0, :]
    prop_high_rate_q90_neg = res_per_reali[5, 0, :]
    prop_high_freq_vector = np.array(mean_rates)[:len(mean_rates), 0]
    prop_high_rate_amplitude = prop_high_rate_max_pos - prop_high_rate_min_neg
    prop_high_rate_var_pos = prop_high_rate_q90_pos - prop_high_rate_q1_pos
    prop_high_rate_var_neg = prop_high_rate_q90_neg - prop_high_rate_q1_neg
    prop_high_rate_max = res_per_reali[6, 0, :]
    prop_high_rate_min = res_per_reali[7, 0, :]
    prop_high_rate_q1 = res_per_reali[8, 0, :]
    prop_high_rate_q90 = res_per_reali[9, 0, :]
    prop_high_rate_amplitude2 = prop_high_rate_max - prop_high_rate_min
    prop_high_rate_var = prop_high_rate_q90 - prop_high_rate_q1

    # Statistics for proportional changes of low-firing rates
    prop_low_rate_max_pos = res_per_reali[0, 1, :]
    prop_low_rate_min_neg = res_per_reali[1, 1, :]
    prop_low_rate_q1_pos = res_per_reali[2, 1, :]
    prop_low_rate_q90_pos = res_per_reali[3, 1, :]
    prop_low_rate_q1_neg = res_per_reali[4, 1, :]
    prop_low_rate_q90_neg = res_per_reali[5, 1, :]
    prop_low_freq_vector = np.array(mean_rates)[:len(mean_rates), 1]
    prop_low_rate_amplitude = prop_low_rate_max_pos - prop_low_rate_min_neg
    prop_low_rate_var_pos = prop_low_rate_q90_pos - prop_low_rate_q1_pos
    prop_low_rate_var_neg = prop_low_rate_q90_neg - prop_low_rate_q1_neg
    prop_low_rate_max = res_per_reali[6, 1, :]
    prop_low_rate_min = res_per_reali[7, 1, :]
    prop_low_rate_q1 = res_per_reali[8, 1, :]
    prop_low_rate_q90 = res_per_reali[9, 1, :]
    prop_low_rate_amplitude2 = prop_low_rate_max - prop_low_rate_min
    prop_low_rate_var = prop_low_rate_q90 - prop_low_rate_q1

    # Statistics for small changes of high-firing rates
    small_high_rate_max_pos = res_per_reali[0, 2, :]
    small_high_rate_min_neg = res_per_reali[1, 2, :]
    small_high_rate_q1_pos = res_per_reali[2, 2, :]
    small_high_rate_q90_pos = res_per_reali[3, 2, :]
    small_high_rate_q1_neg = res_per_reali[4, 2, :]
    small_high_rate_q90_neg = res_per_reali[5, 2, :]
    small_high_freq_vector = np.array(mean_rates)[:len(mean_rates), 2]
    small_high_rate_amplitude = small_high_rate_max_pos - small_high_rate_min_neg
    small_high_rate_var_pos = small_high_rate_q90_pos - small_high_rate_q1_pos
    small_high_rate_var_neg = small_high_rate_q90_neg - small_high_rate_q1_neg
    small_high_rate_max = res_per_reali[6, 2, :]
    small_high_rate_min = res_per_reali[7, 2, :]
    small_high_rate_q1 = res_per_reali[8, 2, :]
    small_high_rate_q90 = res_per_reali[9, 2, :]
    small_high_rate_amplitude2 = small_high_rate_max - small_high_rate_min
    small_high_rate_var = small_high_rate_q90 - small_high_rate_q1

    fig = plt.figure(figsize=(10, 3))
    fig.suptitle(description)
    ax11 = fig.add_subplot(131)
    ax12 = fig.add_subplot(132)
    ax13 = fig.add_subplot(133)
    ax11.plot(prop_high_freq_vector, prop_high_rate_amplitude, c="black")
    ax11.fill_between(prop_high_freq_vector, 0, prop_high_rate_var_pos, color="tab:red", alpha=0.3)
    ax11.fill_between(prop_high_freq_vector, 0, prop_high_rate_var_neg, color="tab:blue", alpha=0.3)
    ax11.grid()
    ax11.set_title("High-rate Proportional", c="gray")
    ax11.set_ylabel("mV")
    ax11.set_xlabel("Frequency (Hz)")
    # ax11.set_ylim(-66.7, -59.5)

    ax12.plot(prop_low_freq_vector, prop_low_rate_amplitude, c="black")
    ax12.fill_between(prop_low_freq_vector, 0, prop_low_rate_var_pos, color="tab:red", alpha=0.3)
    ax12.fill_between(prop_low_freq_vector, 0, prop_low_rate_var_neg, color="tab:blue", alpha=0.3)
    ax12.grid()
    ax12.set_ylabel("mV")
    ax12.set_xlabel("Frequency (Hz)")
    ax12.set_title("Low-rate Proportional", c="gray")
    # ax12.set_ylim(-66.7, -59.5)

    ax13.plot(small_high_freq_vector, small_high_rate_amplitude, c="black")
    ax13.fill_between(small_high_freq_vector, 0, small_high_rate_var_pos, color="tab:red", alpha=0.3)
    ax13.fill_between(small_high_freq_vector, 0, small_high_rate_var_neg, color="tab:blue", alpha=0.3)
    ax13.grid()
    ax13.set_ylabel("mV")
    ax13.set_xlabel("Frequency (Hz)")
    ax13.set_title("High-rate small variation", c="gray")
    # ax13.set_ylim(-66.7, -59.5)

    fig.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)


def plot_gc_mem_potential_prop_fix(time_vector, i, s1, s2, t_tr, statis, title, max_t, path_save="", save_figs=False,
                                   y_lims_ind_plot=None, plot_stats=False, plt_grid=False, th_percentage=1e-3):
    # t_tr = t_tr_[0]
    a, b, c = int(max_t / 3), int(2 * max_t / 3), max_t

    # ******************************************************************************************************************
    # Figure for PhD thesis: methodology-Experimental setup-Stimuli schema-proportional change of rate
    """
    fig_tr_st = plt.figure(figsize=(8, 5))
    plt.suptitle("Example of system response to three window schema", color="black", alpha=0.7)
    axa = fig_tr_st.add_subplot(2, 1, 1)
    axa.set_ylabel('Mem. pot. (mV)', color="gray")
    axa.plot(time_vector, s1[0, :], c="black", alpha=0.5)
    # axa.grid()
    t0, t1, t2, t3 = 0, 2, 4, 6
    # Window shading
    axa.axvspan(t0, t1, color="tab:blue", alpha=0.12)
    axa.axvspan(t1, t2, color="tab:orange", alpha=0.12)
    axa.axvspan(t2, t3, color="tab:green", alpha=0.12)
    # Separating bars
    for x in [t1, t2]:
        axa.axvline(x, color="k", linestyle="--", linewidth=1)
    # Labels centered in each window
    ymin, ymax = axa.get_ylim()
    ypos = ymax - 0.08 * (ymax - ymin)
    axa.text(1, ypos, "ini-window", ha="center", va="top")
    axa.text(3, ypos, "mid-window", ha="center", va="top")
    axa.text(5, ypos, "end-window", ha="center", va="top")

    axb = fig_tr_st.add_subplot(2, 1, 2)
    axb.set_xlabel("Time (s)", color="gray")
    axb.set_ylabel('Mem. pot. (mV)', color="gray")
    L = s1.shape[1]
    ini_minus_end_windows = np.abs(s1[0, int(2 * L / 3):L - 10] - s1[0, :int(L / 3) - 10])
    # Find the 0.1% of the maximum
    thresholds = np.max(ini_minus_end_windows) * th_percentage
    # find indices where the difference is bigger than 0.1% of maximum
    # ind_tr = np.where(ini_minus_end_windows < thresholds)[0][0]
    ind_tr = np.where(ini_minus_end_windows > thresholds)
    val_unique, ind_unique = np.unique(ind_tr[0], return_index=True)
    ind_tr = ind_tr[0][list(np.array(ind_unique) - 1)][0]

    # Plotting
    axb.plot(time_vector[:int(2 * ind_tr)], s1[0, :int(2 * ind_tr)], c="tab:blue", label=r'ini-window')
    axb.plot(time_vector[:int(2 * ind_tr)], s1[0, int(2 * L / 3):int(2 * L / 3) + 2 * ind_tr], c="tab:green", label=r'end-window')
    axb.axvline(time_vector[ind_tr], color="k", linestyle="--", linewidth=1)
    # Labels centered in each window
    ymin, ymax = axb.get_ylim()
    ypos = ymax - 0.1 * (ymax - ymin)
    axb.text(time_vector[int(ind_tr * 1.01)], ypos, r'$t_{tr/st} = %.1f$ms' % (time_vector[ind_tr] * 1e3), ha="left", va="top")
    # axb.grid()
    axb.set_title("Time to reach steady-state from difference between ini- and end windows", color="black", alpha=0.7)
    axb.legend(loc="best")
    plt.tight_layout()
    if save_figs: fig_tr_st.savefig(path_save, format='png')
    # """
    # ******************************************************************************************************************

    figc = plt.figure(figsize=(8, 3))  # (10, 3))
    plt.suptitle(title)
    # tm=30/syn=100 [-65.7,-52.5], tm=1/syn=100[-70,-35], tm=1/syn=1 [-70.05,-67.4]
    ylims = y_lims_ind_plot if y_lims_ind_plot is not None else [-71, -43]
    ax1 = figc.add_subplot(1, 1, 1)  # (1, 2, 1)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Mem. potential (V)")
    ax1.plot(time_vector, s1[0, :], c="black", alpha=0.4)
    if plot_stats:
        # np.array(mean_st_pi), np.array(median_st_pi), np.array(q5_st_pi), np.array(q10_st_pi),
        # np.array(q90_st_pi), np.array(q95_st_pi), np.array(min_st_pi), np.array(max_st_pi),  # 7

        # np.array(mean_st_pm), np.array(median_st_pm), np.array(q5_st_pm), np.array(q10_st_pm),
        # np.array(q90_st_pm), np.array(q95_st_pm), np.array(min_st_pm), np.array(max_st_pm),  # 15

        #  np.array(mean_st_pe), np.array(median_st_pe), np.array(q5_st_pe), np.array(q10_st_pe),
        # np.array(q90_st_pe), np.array(q95_st_pe), np.array(min_st_pe), np.array(max_st_pe),  # 23

        # np.array(mean_st_ci), np.array(median_st_ci), np.array(q5_st_ci), np.array(q10_st_ci),
        # np.array(q90_st_ci), np.array(q95_st_ci), np.array(min_st_ci), np.array(max_st_ci),  # 31
        # np.array(mean_st_cm), np.array(median_st_cm), np.array(q5_st_cm), np.array(q10_st_cm),
        # np.array(q90_st_cm), np.array(q95_st_cm), np.array(min_st_cm), np.array(max_st_cm),  # 39
        # np.array(mean_st_ce), np.array(median_st_ce), np.array(q5_st_ce), np.array(q10_st_ce),
        # np.array(q90_st_ce), np.array(q95_st_ce), np.array(min_st_ce), np.array(max_st_ce),  # 47

        # np.array(max_tr_pi), np.array(max_tr_pm), np.array(max_tr_pe)]  # 50

        ax1.plot([0 + t_tr, a], [statis[0, i, 0], statis[0, i, 0]], c="tab:orange", label=r'$\mu$')  # mean ini window
        ax1.plot([a + t_tr, b], [statis[8, i, 0], statis[8, i, 0]], c="tab:orange")  # mean mid window
        ax1.plot([b + t_tr, c], [statis[16, i, 0], statis[16, i, 0]], c="tab:orange")  # mean end window

        ax1.plot([0 + t_tr, a], [statis[6, i, 0], statis[6, i, 0]], c="tab:red", alpha=0.8, label='min')  # min ini win
        ax1.plot([a + t_tr, b], [statis[14, i, 0], statis[14, i, 0]], c="tab:red", alpha=0.8)  # min mid window
        ax1.plot([b + t_tr, c], [statis[22, i, 0], statis[22, i, 0]], c="tab:red", alpha=0.8)  # min end window
        ax1.plot([0, 0 + t_tr], [statis[52, i, 0], statis[52, i, 0]], c="tab:red", alpha=0.8)  # tr min ini window
        ax1.plot([a, a + t_tr], [statis[57, i, 0], statis[57, i, 0]], c="tab:red", alpha=0.8)  # tr min mid window
        ax1.plot([b, b + t_tr], [statis[62, i, 0], statis[62, i, 0]], c="tab:red", alpha=0.8)  # tr min end window

        ax1.plot([0 + t_tr, a], [statis[7, i, 0], statis[7, i, 0]], c="tab:red", alpha=0.8, label='max')  # max ini win
        ax1.plot([a + t_tr, b], [statis[15, i, 0], statis[15, i, 0]], c="tab:red", alpha=0.8)  # max mid window
        ax1.plot([b + t_tr, c], [statis[23, i, 0], statis[23, i, 0]], c="tab:red", alpha=0.8)  # max end window
        ax1.plot([0, 0 + t_tr], [statis[51, i, 0], statis[51, i, 0]], c="tab:red", alpha=0.8)  # tr max ini window
        ax1.plot([a, a + t_tr], [statis[56, i, 0], statis[56, i, 0]], c="tab:red", alpha=0.8)  # tr max mid window
        ax1.plot([b, b + t_tr], [statis[61, i, 0], statis[61, i, 0]], c="tab:red", alpha=0.8)  # tr max end window

        ax1.plot([0 + t_tr, a], [statis[3, i, 0], statis[3, i, 0]], c="tab:green", label='q10%')  # q10 ini window
        ax1.plot([a + t_tr, b], [statis[11, i, 0], statis[11, i, 0]], c="tab:green")  # q10 mid window
        ax1.plot([b + t_tr, c], [statis[19, i, 0], statis[19, i, 0]], c="tab:green")  # q10 end window
        ax1.plot([0, 0 + t_tr], [statis[50, i, 0], statis[50, i, 0]], c="tab:green", alpha=0.8)  # tr q10 ini window
        ax1.plot([a, a + t_tr], [statis[55, i, 0], statis[55, i, 0]], c="tab:green", alpha=0.8)  # tr q10 mid window
        ax1.plot([b, b + t_tr], [statis[60, i, 0], statis[60, i, 0]], c="tab:green", alpha=0.8)  # tr q10 end window

        ax1.plot([0 + t_tr, a], [statis[4, i, 0], statis[4, i, 0]], c="tab:green", label='q90%')  # q90 ini window
        ax1.plot([a + t_tr, b], [statis[12, i, 0], statis[12, i, 0]], c="tab:green")  # q90 mid window
        ax1.plot([b + t_tr, c], [statis[20, i, 0], statis[20, i, 0]], c="tab:green")  # q90 end window
        ax1.plot([0, 0 + t_tr], [statis[49, i, 0], statis[49, i, 0]], c="tab:green", alpha=0.8)  # tr q90 ini window
        ax1.plot([a, a + t_tr], [statis[54, i, 0], statis[54, i, 0]], c="tab:green", alpha=0.8)  # tr q90 mid window
        ax1.plot([b, b + t_tr], [statis[59, i, 0], statis[59, i, 0]], c="tab:green", alpha=0.8)  # tr q90 end window

        ax1.plot([0 + t_tr, a], [statis[1, i, 0], statis[1, i, 0]], c="tab:blue", label='median')  # median ini window
        ax1.plot([a + t_tr, b], [statis[9, i, 0], statis[9, i, 0]], c="tab:blue")  # median mid window
        ax1.plot([b + t_tr, c], [statis[17, i, 0], statis[17, i, 0]], c="tab:blue")  # median end window
        ax1.plot([0, 0 + t_tr], [statis[48, i, 0], statis[48, i, 0]], c="tab:blue", alpha=0.8)  # tr median ini window
        ax1.plot([a, a + t_tr], [statis[53, i, 0], statis[53, i, 0]], c="tab:blue", alpha=0.8)  # tr median mid window
        ax1.plot([b, b + t_tr], [statis[58, i, 0], statis[58, i, 0]], c="tab:blue", alpha=0.8)  # tr median end window
        # ax1.grid()
        # ax1.legend(loc="upper right")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax1.set_title("Proportional changes", color="gray")
    ax1.set_ylim(ylims)
    if plt_grid: ax1.grid()
    figc.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
    if save_figs: figc.savefig(path_save, format='png')


def aux_plot_features_win_prop_fix(dr, lbl, st_lbl, normalise=False, min_n=None, max_n=None):
    plot_sign = False
    sign = None
    if lbl + st_lbl in dr:
        sign = dr[lbl + st_lbl]
        plot_sign = True
    else:
        if st_lbl == '_q1':
            if lbl + '_q10' in dr:
                sign = dr[lbl + '_q10']
                plot_sign = True
        if st_lbl == '_q10':
            if lbl + '_q1' in dr:
                sign = dr[lbl + '_q1']
                plot_sign = True
    if normalise:
        return plot_sign, norm_array(sign, compute_norm=True, min_n=min_n, max_n=max_n)
    else:
        return plot_sign, sign


def plot_features_windows_prop(f_vector, dr, lbl, st_lbl, cols, suptitle_="", path_save="", save_figs=False,
                                   y_lims_ind_plot=None, titles=None, normalise=False, min_n=None, max_n=None):
    fig_st = plt.figure(figsize=(10, 6))
    plt.suptitle(suptitle_)
    ylims = y_lims_ind_plot if y_lims_ind_plot is not None else None  # [-70.15, -67.3]  # [-70.05, -52]
    for i in range(len(lbl)):
        ax_st = fig_st.add_subplot(int(len(lbl) / 3), 3, i + 1)
        for j in range(len(st_lbl)):
            plot_sign, sign = aux_plot_features_win_prop_fix(dr, lbl[i], st_lbl[j],
                                                             normalise=normalise, min_n=min_n, max_n=max_n)
            if plot_sign:
                ax_st.plot(f_vector, np.median(sign, axis=0), c=cols[j], label=st_lbl[j][1:])
                ax_st.fill_between(f_vector, np.quantile(sign, 0.1, axis=0), np.quantile(sign, 0.9, axis=0),
                                   color=cols[j], alpha=0.3)
        ax_st.set_xlabel("Rate (Hz)")
        ax_st.set_ylabel("mem. pot. (mV)")
        aux_title = titles[i] if titles is not None else lbl[i].split("_")[1] + " win. (" + lbl[i].split("_")[2] + ")"
        ax_st.set_title(aux_title, color='gray')
        # ax_st.set_title("Frequency response", color='gray')
        ax_st.grid()
        ax_st.set_xscale('log')
        if (i + 1) % 3 == 0: ax_st.legend(loc='upper right')
        # ax_st.set_ylim(ylims)
    fig_st.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    if save_figs: fig_st.savefig(path_save, format='png')


def plot_features_windows_prop_fix(f_vector, dr, lbl, st_lbl, cols, suptitle_="", path_save="", save_figs=False,
                                   y_lims_ind_plot=None, titles=None, normalise=False, min_n=None, max_n=None):
    fig_st = plt.figure(figsize=(10, 6))
    plt.suptitle(suptitle_)
    ylims = y_lims_ind_plot if y_lims_ind_plot is not None else None  # [-70.15, -67.3]  # [-70.05, -52]
    for i in range(len(lbl)):
        ax_st = fig_st.add_subplot(2, 3, i + 1)
        for j in range(len(st_lbl)):
            plot_sign, sign = aux_plot_features_win_prop_fix(dr, lbl[i], st_lbl[j],
                                                             normalise=normalise, min_n=min_n, max_n=max_n)
            if plot_sign:
                ax_st.plot(f_vector, np.median(sign, axis=0), c=cols[j], label=st_lbl[j][1:])
                ax_st.fill_between(f_vector, np.quantile(sign, 0.1, axis=0), np.quantile(sign, 0.9, axis=0),
                                   color=cols[j], alpha=0.3)
        ax_st.set_xlabel("Rate (Hz)")
        ax_st.set_ylabel("mem. pot. (mV)")
        aux_title = titles[i] if titles is not None else lbl[i].split("_")[1] + " win. (" + lbl[i].split("_")[2] + ")"
        ax_st.set_title(aux_title, color='gray')
        ax_st.grid()
        ax_st.set_xscale('log')
        if (i + 1) % 3 == 0: ax_st.legend(loc='upper right')
        # ax_st.set_ylim(ylims)
    fig_st.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    if save_figs: fig_st.savefig(path_save, format='png')


def plot_features_tr_st_3windows(f_vector, dr, lbl, lbl2, st_lbl, cols, t_, title_graph, path_save, save_figs,
                                    y_lims_ind_plot=None, ls=None, normalise=False, min_n=None, max_n=None, y_lbl=None):
    ls = ['-' for _ in range(len(st_lbl))] if ls is None else ls
    fig_st2 = plt.figure(figsize=(10, 3.2))
    plt.suptitle(title_graph)
    ylims = y_lims_ind_plot if y_lims_ind_plot is not None else None  # [-70.15, -67.3]  # [-70.05, -52]
    y_label = y_lbl if y_lbl is not None else None
    ax_st2 = None
    for i in range(len(lbl)):
        ax_st2 = fig_st2.add_subplot(int(len(lbl) / 3), 3, i + 1)
        for j in range(len(st_lbl)):
            plot_sign, sign = aux_plot_features_win_prop_fix(dr, lbl[i], st_lbl[j],
                                                             normalise=normalise, min_n=min_n, max_n=max_n)
            if plot_sign:
                if i == 2:
                    ax_st2.plot(f_vector, np.median(sign, axis=0), c='gray', label=lbl[i][:3] + st_lbl[j],
                                linestyle=ls[j])
                else:
                    ax_st2.plot(f_vector, np.median(sign, axis=0), c='gray', linestyle=ls[j])
                ax_st2.fill_between(f_vector, np.quantile(sign, 0.1, axis=0), np.quantile(sign, 0.9, axis=0),
                                    color='gray', alpha=0.3)
        for j in range(len(st_lbl)):
            plot_sign, sign = aux_plot_features_win_prop_fix(dr, lbl2[i], st_lbl[j],
                                                             normalise=normalise, min_n=min_n, max_n=max_n)
            if plot_sign:
                if i == 2:
                    ax_st2.plot(f_vector, np.median(sign, axis=0), c=cols[j], label=lbl2[i][:3] + st_lbl[j],
                                linestyle=ls[j])
                else:
                    ax_st2.plot(f_vector, np.median(sign, axis=0), c=cols[j], linestyle=ls[j])
                ax_st2.fill_between(f_vector, np.quantile(sign, 0.1, axis=0), np.quantile(sign, 0.9, axis=0),
                                    color=cols[j], alpha=0.3)
        ax_st2.set_title(t_[i], color='gray')
        ax_st2.set_xlabel("Rate (Hz)")
        ax_st2.set_ylabel(y_label)
        ax_st2.grid()
        ax_st2.set_xscale('log')
        if y_lims_ind_plot is not None: ax_st2.set_ylim(ylims)

    ax_st2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # fig_st2.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    if save_figs: fig_st2.savefig(path_save, format='png')


def plot_features_2windows_prop_fix(f_vector, dr, lbl, lbl2, st_lbl, cols, t_, title_graph, path_save, save_figs,
                                    y_lims_ind_plot=None, normalise=None, min_n=None, max_n=None):
    num_rows = int(len(lbl) / 2)
    fig_st2 = plt.figure(figsize=(8, 6))
    plt.suptitle(title_graph)
    ylims = y_lims_ind_plot if y_lims_ind_plot is not None else None  # [-70.15, -67.3]  # [-70.05, -52]
    ax_st2 = None
    for i in range(len(lbl)):
        ax_st2 = fig_st2.add_subplot(int(len(lbl) / 2), 2, i + 1)
        for j in range(len(st_lbl)):
            plot_sign, sign = aux_plot_features_win_prop_fix(dr, lbl[i], st_lbl[j],
                                                             normalise=normalise, min_n=min_n, max_n=max_n)
            if plot_sign:
                ax_st2.plot(f_vector, np.median(sign, axis=0), c='gray')
                ax_st2.fill_between(f_vector, np.quantile(sign, 0.1, axis=0), np.quantile(sign, 0.9, axis=0),
                                    color='gray', alpha=0.3)
        for j in range(len(st_lbl)):
            plot_sign, sign = aux_plot_features_win_prop_fix(dr, lbl2[i], st_lbl[j],
                                                             normalise=normalise, min_n=min_n, max_n=max_n)
            if plot_sign:
                ax_st2.plot(f_vector, np.median(sign, axis=0), c=cols[j])
                ax_st2.fill_between(f_vector, np.quantile(sign, 0.1, axis=0), np.quantile(sign, 0.9, axis=0),
                                    color=cols[j], alpha=0.3)
        ax_st2.set_title(t_[i], color='gray')
        ax_st2.set_xlabel("Rate (Hz)")
        ax_st2.set_ylabel("mem. pot. (mV)")
        ax_st2.grid()
        ax_st2.set_xscale('log')

    ax_st2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    fig_st2.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    if save_figs: fig_st2.savefig(path_save, format='png')


def plot_diff_windows(f_vector, dr, lbl, lbl2, st_lbl, cols_, t_, title_graph="", name_save="", save_figs=False,
                      y_lims_ind_plot=None, ls=None, normalise=False, min_n=None, max_n=None):
    ls = ['-' for _ in range(len(st_lbl))] if ls is None else ls
    fig2 = plt.figure(figsize=(10, 3.2))  # (7, 2.5)
    plt.suptitle(title_graph)
    alpha = 0.1
    ylims = y_lims_ind_plot if y_lims_ind_plot is not None else None  # [-70.15, -67.3]  # [-70.05, -52]
    for i in range(len(lbl)):
        ax_3 = fig2.add_subplot(1, len(lbl), i + 1)
        for j in range(len(st_lbl)):
            plot_sign1, sign1 = aux_plot_features_win_prop_fix(dr, lbl[i], st_lbl[j],
                                                               normalise=normalise, min_n=min_n, max_n=max_n)
            plot_sign2, sign2 = aux_plot_features_win_prop_fix(dr, lbl2[i], st_lbl[j],
                                                               normalise=normalise, min_n=min_n, max_n=max_n)
            if j > 3:
                alpha = 0.3
            else:
                alpha = 0.1
            if plot_sign1 and plot_sign2:
                aux = sign1 - sign2
                ax_3.plot(f_vector, np.median(aux, axis=0), c=cols_[j], label=st_lbl[j][1:], linestyle=ls[j])
                ax_3.fill_between(f_vector, np.quantile(aux, 0.1, axis=0), np.quantile(aux, 0.9, axis=0),
                                  color=cols_[j], alpha=alpha)
        ax_3.set_xlabel("Rate (Hz)")
        ax_3.set_ylabel("mem. pot. (mV)")
        ax_3.set_title(t_[i], c="gray")
        ax_3.grid()
        ax_3.set_xscale('log')
        if (i + 1) % 3 == 0: ax_3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax_3.set_ylim(ylims)
    fig2.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    if save_figs: fig2.savefig(name_save, format='png')


def plot_diff_windows_tr_st(f_vector, dr, mid_st_lbl, mid_tr_lbl, ini_st_lbl, st_lbl, cols_, t_, title_graph="",
                            name_save="", save_figs=False, ax=None, y_lims_ind_plot=None, ls=None, lbls=None,
                            fillBetween=True, normalise=None, min_n=None, max_n=None, y_lbl=None):
    ls = ['-' for _ in range(len(st_lbl))] if ls is None else ls
    ext_ax = False
    if ax is not None: ext_ax = True
    fig2 = None
    if not ext_ax: fig2 = plt.figure(figsize=(6, 3.2))  # (7, 2.5)
    plt.suptitle(title_graph)
    alpha = 0.1
    ylims = y_lims_ind_plot if y_lims_ind_plot is not None else None  # [-70.15, -67.3]  # [-70.05, -52]
    y_label = y_lbl if y_lbl is not None else None
    for i in range(len(mid_st_lbl)):
        if not ext_ax: ax = fig2.add_subplot(1, len(mid_st_lbl), i + 1)
        i_leg = 0
        for j in range(len(st_lbl)):
            plot_sign1, sign_m_st = aux_plot_features_win_prop_fix(dr, mid_st_lbl[i], st_lbl[j],
                                                                   normalise=normalise, min_n=min_n, max_n=max_n)
            plot_sign2, sign_m_tr = aux_plot_features_win_prop_fix(dr, mid_tr_lbl[i], st_lbl[j],
                                                                   normalise=normalise, min_n=min_n, max_n=max_n)
            plot_sign3, sign_i_st = aux_plot_features_win_prop_fix(dr, ini_st_lbl[i], st_lbl[j],
                                                                   normalise=normalise, min_n=min_n, max_n=max_n)
            if j > 3:
                alpha = 0.3
            else:
                alpha = 0.1
            if plot_sign1 and plot_sign3:
                aux = sign_m_st - sign_i_st
                aux_lbl = st_lbl[j][1:] if lbls is None else lbls[i_leg] + st_lbl[j][1:] + ")"
                ax.plot(f_vector, np.median(aux, axis=0), c=cols_[j], label=aux_lbl, linestyle=ls[j])
                if fillBetween:
                    ax.fill_between(f_vector, np.quantile(aux, 0.1, axis=0), np.quantile(aux, 0.9, axis=0),
                                    color=cols_[j], alpha=alpha)
            i_leg += 1
            if plot_sign2:
                aux = sign_m_tr - sign_i_st
                aux_lbl = st_lbl[j][1:] if lbls is None else lbls[i_leg] + st_lbl[j][1:] + ")"
                ax.plot(f_vector, np.median(aux, axis=0), c='black', label=aux_lbl, linestyle=ls[j])
                if fillBetween:
                    ax.fill_between(f_vector, np.quantile(aux, 0.1, axis=0), np.quantile(aux, 0.9, axis=0),
                                    color='black', alpha=alpha)
            i_leg += 1
        ax.set_xlabel("Rate (Hz)")
        ax.set_ylabel(y_label)
        ax.set_title(t_[i], c="gray")
        ax.grid()
        ax.set_xscale('log')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax.set_ylim(ylims)
    if not ext_ax:
        # fig2.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
        if save_figs: fig2.savefig(name_save, format='png')
        return None
    else:
        return ax


def plot_gain_filtering(dr_gain, dr_filt, lbl, lbl2, st_lbl, cols_, title, path_save, save_figs, fig2=None):
    if fig2 is None:
        fig2 = plt.figure(figsize=(9, 5))  # 6.5, 5
        plt.suptitle(title)
    alpha = 0.3
    ax_ = []
    markers = ['+', '*']
    alphas = [1.0, 0.5]

    for i in [0]:  # range(len(lbl)):
        # i = 0
        for j in range(len(st_lbl)):
            if i == 0: ax_.append(fig2.add_subplot(2, 4, j + 1))
            # j = 5
            aux = dr_gain[lbl[i] + st_lbl[j]] - dr_gain[lbl2[i] + st_lbl[j]]
            ax_[j].scatter(dr_filt[lbl2[i] + st_lbl[j]][0, :], np.median(aux, axis=0), marker=markers[i], c=cols_[j],
                           alpha=alphas[i])
            ax_[j].plot(dr_filt[lbl2[i] + st_lbl[j]][0, :], np.median(aux, axis=0), c=cols_[j], alpha=alphas[i])
            if i == 0: ax_[j].fill_between(dr_filt[lbl2[i] + st_lbl[j]][0, :], np.quantile(aux, 0.1, axis=0),
                                           np.quantile(aux, 0.9, axis=0), color=cols_[j], alpha=0.3)
            ax_[j].scatter(dr_filt[lbl2[i] + st_lbl[j]][0, 0], np.median(aux, axis=0)[0], c='black')

            ax_[j].set_xlabel("Synaptic filtering (mV)", color='gray')
            ax_[j].set_ylabel("Gain-Control (mV)", color='gray')
            # ax_[j].set_ylim(ylims)
            ax_[j].set_title(st_lbl[j][1:], c="gray")
            if i == 0: ax_[j].grid()
            # ax_3.set_xscale('log')
            if (i + 1) % 3 == 0: ax_[j].legend(loc='upper right')
            # ax_st.set_ylim(ylims)
    fig2.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    if save_figs: fig2.savefig(path_save, format='png')


def avg_f(vec):
    return np.median(vec, axis=0)


def plot_properties_in_freq(dr_, var_, f_vec, H_list, aux_l, axb_, tr_time, c_g, norm_neuron=True, min_n=None,
                            max_n=None, plot_filt=False):
    min_n = None if min_n is None else min_n
    max_n = None if max_n is None else max_n
    alphas = [1.0, 0.5]
    gain = aux_l
    n_sto_m_st_max = norm_array(dr_[var_[0]], compute_norm=norm_neuron, min_n=min_n, max_n=max_n)
    n_sto_m_tr_max = norm_array(dr_[var_[1]], compute_norm=norm_neuron, min_n=min_n, max_n=max_n)
    n_sto_i_st_max = norm_array(dr_[var_[2]], compute_norm=norm_neuron, min_n=min_n, max_n=max_n)
    n_sto_i_tr_max = norm_array(dr_[var_[3]], compute_norm=norm_neuron, min_n=min_n, max_n=max_n)
    n_sto_m_st_var = norm_array(dr_[var_[4]] - dr_[var_[5]], compute_norm=norm_neuron, min_n=min_n, max_n=max_n)
    n_sto_i_st_var = norm_array(dr_[var_[6]] - dr_[var_[7]], compute_norm=norm_neuron, min_n=min_n, max_n=max_n)
    n_sto_m_st_min = norm_array(dr_[var_[8]], compute_norm=norm_neuron, min_n=min_n, max_n=max_n)
    n_sto_m_tr_min = norm_array(dr_[var_[9]], compute_norm=norm_neuron, min_n=min_n, max_n=max_n)
    n_sto_i_st_min = norm_array(dr_[var_[10]], compute_norm=norm_neuron, min_n=min_n, max_n=max_n)
    n_sto_i_tr_min = norm_array(dr_[var_[11]], compute_norm=norm_neuron, min_n=min_n, max_n=max_n)
    n_sto_m_st_med = norm_array(dr_[var_[12]], compute_norm=norm_neuron, min_n=min_n, max_n=max_n)
    n_sto_i_st_med = norm_array(dr_[var_[13]], compute_norm=norm_neuron, min_n=min_n, max_n=max_n)
    n_sto_tr_time = tr_time

    aux_gain = [np.copy(n_sto_m_st_max - n_sto_i_st_max), np.copy(n_sto_m_st_min - n_sto_i_st_min),
                np.copy(n_sto_m_st_var - n_sto_i_st_var), np.copy(n_sto_m_st_med - n_sto_i_st_med)]
    aux_gain_tr = [np.copy(n_sto_m_tr_max - n_sto_i_st_max), np.copy(n_sto_m_tr_min - n_sto_i_st_min),
                   np.zeros(n_sto_m_tr_max.shape), np.zeros(n_sto_m_tr_max.shape)]

    aux_filt = [np.copy(n_sto_i_st_max), np.copy(n_sto_i_st_min), np.copy(n_sto_i_st_med)]
    aux_filt_tr = [np.copy(n_sto_i_tr_max), np.copy(n_sto_i_tr_min)]
    # For Information theory analysis
    aux_H_st = [H_list[3], H_list[4] - H_list[3]]
    aux_H_tr = [H_list[0], H_list[1] - H_list[3]]
    # Stochastic plots
    # Plotting Entropy
    for j in range(2):
        axb_[j].plot(f_vec, aux_H_st[j], alpha=alphas[1], label="st - " + str(gain), c=c_g)
    for j in range(2):
        axb_[j].plot(f_vec, aux_H_tr[j], alpha=alphas[0], label="tr - " + str(gain), c=c_g)
    # Plotting Gain-Control
    for j in range(4):
        # axb_[j + 2].plot(f_vec, avg_f(aux_filt[j]), alpha=alphas[1], c=c_g)
        # axb_[j].fill_between(f_vec, np.quantile(aux_filt[j], 0.1, axis=0),
        #                     np.quantile(aux_filt[j], 0.9, axis=0), color=cols_[j], alpha=0.1)
        # """
        axb_[j + 4].plot(f_vec, avg_f(aux_gain[j]), alpha=alphas[1], c=c_g, label="st-" + str(gain))
        axb_[j + 4].fill_between(f_vec, np.quantile(aux_gain[j], 0.1, axis=0),
                                 np.quantile(aux_gain[j], 0.9, axis=0), color=c_g, alpha=0.1)
    for j in range(4):
        axb_[j + 4].plot(f_vec, avg_f(aux_gain_tr[j]), alpha=alphas[0], label="tr-" + str(gain), c=c_g)
        axb_[j + 4].fill_between(f_vec, np.quantile(aux_gain_tr[j], 0.01, axis=0),
                                 np.quantile(aux_gain_tr[j], 0.99, axis=0), color=c_g, alpha=0.1)
    if plot_filt:
        a = 2  # 6
        b = 3  # 7
        # Transition time
        axb_[a].plot(f_vec, avg_f(n_sto_tr_time), alpha=alphas[1], c='black')
        axb_[a].fill_between(f_vec, np.quantile(n_sto_tr_time, 0.01, axis=0),
                             np.quantile(n_sto_tr_time, 0.99, axis=0), color='black', alpha=0.1)
        # Plotting filtering property
        c_f = ['tab:red', 'tab:red', 'tab:blue']
        l_f = ['Max', 'Min', 'Med']
        ls = ['-', '--', '-']
        for j in range(3):
            axb_[b].plot(f_vec, avg_f(aux_filt[j]), alpha=alphas[1], c=c_f[j], label="st-" + l_f[j],
                         linestyle=ls[j])
            axb_[b].fill_between(f_vec, np.quantile(aux_filt[j], 0.1, axis=0),
                                 np.quantile(aux_filt[j], 0.9, axis=0), color=c_f[j], alpha=0.1)
        for j in range(2):
            axb_[b].plot(f_vec, avg_f(aux_filt_tr[j]), c=c_f[j], label="tr-" + l_f[j], linestyle=ls[j])
            axb_[b].fill_between(f_vec, np.quantile(aux_filt_tr[j], 0.1, axis=0),
                                 np.quantile(aux_filt_tr[j], 0.9, axis=0), color=c_f[j], alpha=0.1)

    return axb_
