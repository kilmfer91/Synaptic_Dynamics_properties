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
    # ax8.set_ylim(-65.5, -60)
    ax8.set_ylabel("mV")
    # ax8.set_title(f"LIF, diff median {diff_median_mempot:.3f}mV", c='gray')

    if i == 2:
        ax7.set_xlabel("time (s)")  # , fontsize=18)
        ax8.set_xlabel("time (s)")


def plot_gc_sin_input_example(time_vector, dt, ind_exp, sin_high_rate, sin_low_rate, high_rate_spikes, low_rate_spikes):
    fonts = 12
    fig_esann3 = plt.figure(figsize=(11, 3))
    ax1s3 = fig_esann3.add_subplot(211)
    ax1s3.plot(time_vector, sin_high_rate, label="high-firing rates", c="#000000")  # , c="#71BE56"
    if ind_exp == 0: ax1s3.set_ylabel("Rate (Hz)", fontsize=fonts)
    ax1s3.set_ylim(0, 160)
    ax1s3.set_title('Sinusoidal pattern for proportional change of firing rate at baseline rate 100Hz',
                    fontsize=fonts + 6, c="gray")
    ax1s3.plot(time_vector, sin_low_rate, label="low-firing rates", c="#AFAFAF")  # , c="#0192C8"
    # ax1s3.grid()
    ax1s3.set_ylabel("Rate (Hz)", fontsize=fonts - 1)
    ax1s3.yaxis.set_tick_params(labelsize=fonts)
    ax1s3.set_xlabel("Time  (s)", fontsize=fonts)
    ax1s3.xaxis.set_tick_params(labelsize=fonts)
    ax1s3.legend(framealpha=0.3)
    ax2s3 = fig_esann3.add_subplot(234)
    ax2s3.plot(time_vector[:int(0.5 / dt)], high_rate_spikes[:int(0.5 / dt)] + 1.1, c="#000000")  # , c="#71BE56")
    ax2s3.plot(time_vector[:int(0.5 / dt)], low_rate_spikes[:int(0.5 / dt)], c="#AFAFAF")  # , c="#0192C8")
    ax2s3.set_xlabel("Time (s)", fontsize=fonts)
    ax2s3.xaxis.set_tick_params(labelsize=fonts)
    # ax2s3.grid()
    ax2s3.get_yaxis().set_visible(False)
    ax2s3.set_title("Firing rates around 100Hz", c="gray", fontsize=fonts + 4)
    ax3s3 = fig_esann3.add_subplot(235)
    ax3s3.plot(time_vector[int(7 / dt):int(7.5 / dt)], high_rate_spikes[int(7 / dt):int(7.5 / dt)] + 1.1,
               c="#000000")  # , c="#71BE56")
    ax3s3.plot(time_vector[int(7 / dt):int(7.5 / dt)], low_rate_spikes[int(7 / dt):int(7.5 / dt)],
               c="#AFAFAF")  # , c="#0192C8")
    ax3s3.set_xlabel("Time (s)", fontsize=fonts)
    ax3s3.xaxis.set_tick_params(labelsize=fonts)
    # ax3s3.grid()
    ax3s3.get_yaxis().set_visible(False)
    ax3s3.set_title("Firing rates around 50Hz", c="gray", fontsize=fonts + 4)
    ax4s3 = fig_esann3.add_subplot(236)
    ax4s3.plot(time_vector[int(12 / dt):int(12.5 / dt)], high_rate_spikes[int(12 / dt):int(12.5 / dt)] + 1.1,
               c="#000000")  # , c="#71BE56")
    ax4s3.plot(time_vector[int(12 / dt):int(12.5 / dt)], low_rate_spikes[int(12 / dt):int(12.5 / dt)],
               c="#AFAFAF")  # , c="#0192C8")
    ax4s3.set_xlabel("Time (s)", fontsize=fonts)
    ax4s3.xaxis.set_tick_params(labelsize=fonts)
    # ax4s3.grid()
    ax4s3.get_yaxis().set_visible(False)
    ax4s3.set_title("Firing rates around 150Hz", c="gray", fontsize=fonts + 4)
    fig_esann3.tight_layout(pad=0.5, w_pad=1.0, h_pad=0.1)


def plot_gc_sin_mp_high_rates(fig, ind, ind_exp, time_vector, mean_rate, output_mp_esann, out_ylim_min, out_ylim_max,
                              output_mp_low_filt_esann):
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