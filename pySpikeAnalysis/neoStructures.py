# -*- coding: utf-8 -*-
from __future__ import division
import matplotlib
matplotlib.use('TkAgg')
import h5py
import numpy as np
import neo
from quantities import Hz, s, ms
import quantities
import os
from datetime import datetime
import csv
import re
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import *
import _pickle
import mne
import elephant
import seaborn as sns
import matplotlib.pyplot as plt
import spykingcircus_utils
import neo_utils
import eeg_utils
import spikeAnalysis_utils
from scipy import signal
# from numbers import Number
from neoStructures_params import *


class NeoMother:
    """ Superclass of NeoAll and NeoEpoch (both NeoAll and NeoEpoch inherits from NeoMother). Used to define methods
    for both NeoAll and NeoEpochs
    """

    def __init__(self):
        """
        Nothing here
        """

    def channelpos2tetrode(self, channel_pos):
        """  From the channel (micro-wire) position, return the information about the tetrode of this channel

        Parameters
        ----------
        channel_pos : int
            Channel's (micro-wire) position

        Returns
        -------
        tetrode_num : int
            Corresponding tetrode number
        tetrode_channels_pos : array
            Position of the 4 channels of the tetrode
        wire_num_in_tetrode : int
            Position of the input channel in the tetrode (between 0 and 3)
        tetrode_channels_pos_glob : array
            ??? same as tetrode_channels_pos ???

        """
        if not self.channel_names:
            print('Cannot access channel names')
            return [], [], [], []
        else:
            channel_name = self.channel_names[channel_pos]
            wire_num = int(re.search('\d+', channel_name).group())
            tetrode_num = int(np.ceil(wire_num / 4))
            tetrode_channels_pos = np.arange(4 * (tetrode_num - 1), 4 * tetrode_num, 1).astype(int)
            wire_num_in_tetrode = wire_num % 4 - 1 if wire_num % 4 != 0 else 3
            tetrode_channels_pos_glob = np.arange(channel_pos-wire_num_in_tetrode,
                                                  channel_pos-wire_num_in_tetrode+4, 1)
            return tetrode_num, tetrode_channels_pos, wire_num_in_tetrode, tetrode_channels_pos_glob

    def channelpos2elec(self, channel_pos):
        """ From the channel (micro-wire) position, return the corresponding electrode's number and name

        Parameters
        ----------
        channel_pos : int
            Channel's (micro-wire) position

        Returns
        -------
            el_num : int
                Corresponding electrode's number
            el_name : str
                Corresponding electrode's name

        """
        if not self.channel_names:
            print('Cannot access channel names')
        else:
            el_name = self.electrode_names[int(channel_pos)]
            el_num = self.electrode_names_unique.index(el_name)
            return el_num, el_name

    def elecnum2channelpos(self, elec_num):
        """ From the electrode's number return the electrode's channels position and name

        Parameters
        ----------
        elec_num : int
            Electrode position

        Returns
        -------
        channel_pos : array
            Position of the input electrode's channels
        channel_names : array
            Name of the input electrode's channels

        """
        el_name = self.electrode_names_unique[elec_num]
        el_channel = [1 if el_name == el_name_2 else 0 for el_name_2 in self.electrode_names]
        channel_pos, = np.where(el_channel)
        channel_names = [self.channel_names[chan_pos] for chan_pos in channel_pos]
        return channel_pos, channel_names

    def getunitpreferredchannelname(self, unit_name):
        """ Get the preferred channel of a unit from the unit's name

        Parameters
        ----------
        unit_name : str
            Name of the unit

        Returns
        -------
        channel_name : str
            Name of the preferred channel (micro-wire) of the input unit

        """
        if isinstance(unit_name, str):
            if unit_name not in self.unit_names:
                print('No unit names {}'.format(unit_name))
                return
            else:
                unit_pos = self.unit_names.index(unit_name)
        else:
            try:
                unit_pos = int(unit_name)
            except:
                raise ValueError('Wrong argument unit_name : {}'.format(unit_name))
        pref_chan_pos = self.unit_preferred_chan[unit_pos]
        return self.channel_names[pref_chan_pos]

    def subfun_plot_unit_shape(self, unit_pos, plot_mean=0, ax=[], raster_plot_call=0):
        """ Sub-routine for plotting shape of the unit

        Parameters
        ----------
        unit_pos : int
            Unit's position
        plot_mean : bool (default: False)
            If True, plot the mean of the unit's shape over the 4 micro-wires of the tetrode
        ax : axis handle | None
            If provided, plot the figure on this axis
        raster_plot_call : bool (default: False)
            Should be true, when this method is called from the ``plot_rasterplot`` method

        """
        n_t = self.unit_shapes[0].shape[0]
        temp_duration_ms = n_t / self.srate * 1000
        tetrode_num, tetrode_channels_pos, preferred_wire_pos, channel_pos = \
            self.channelpos2tetrode(self.unit_preferred_chan[unit_pos])
        t_vect = np.linspace(-temp_duration_ms / 2, temp_duration_ms / 2, n_t)
        # Plot
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        for i in range(0, 4):
            linewidth = 3 if i == preferred_wire_pos else 2
            plt.plot(t_vect, self.unit_shapes[unit_pos][:, i], lw=linewidth)
        if plot_mean:
            plt.plot(t_vect, self.unit_shapes[unit_pos].mean(1), 'k', alpha=0.7, lw=3)

        ax.autoscale(axis='x', tight=True)
        plt.plot([0, 0], ax.get_ylim(), 'k--', lw=1, alpha=0.6, zorder=1)
        ax.autoscale(axis='y', tight=True)
        plt.plot(ax.get_xlim(), [0, 0], 'k--', lw=1, alpha=0.6, zorder=1)
        if not raster_plot_call:
            ax.set(title='Unit {}'.format(self.unit_names[unit_pos]), xlabel='Time (ms)', ylabel='Amplitude (uV)')
        else:
            ax.yaxis.tick_right()
        # Get tetrode position
        if self.channel_names:
            _, el_name = self.channelpos2elec(self.unit_preferred_chan[unit_pos])
            tetrode_channel_names = [self.channel_names[i] for i in channel_pos]
            ax.legend(tetrode_channel_names, loc='best')
            if not raster_plot_call:
                ax.set(title='Unit {} - {} tetrode {}'.format(self.unit_names[unit_pos], el_name, tetrode_num))
        if self.save_fig and not raster_plot_call:
            fig.savefig(os.path.join(self.fig_dirpath_unitshape, self.unit_names[unit_pos]+'_shape.'+RES_FIG_FORMAT),
                        dpi=RES_FIG_DPI)


class NeoAll(NeoMother):
    """ Contain spike sorting data from Spyking-Circus results file. Create one Neo Segment per unit.


    Attributes :
    -----------
    spikesorting_dirpath : str
        Path of the directory containng the spyking circus result file
    spikesorting_filename : str
        Name of the Spiking Circus result file
    probe_filepath : str
        Path to the Spiking Circus probe file
    signal_dirpath :str | none (default)
        Path of the directory containing the LFP signals files (must be in EDF format)
    event_filepath : str | none (default)
        Path to the event file
    event_csv_sep : str (default ';')
        Separator used in the event file
    fig_dirpath : str | none (default)
        Path to the directory where figures will be saved. If none and save_fig is true, then save the figures in the
        current directory.
    save_fig : bool
        If True, save the figures
    srate : float
        Sampling rate (Hz)
    t_max : float
        Maximum time of spike. If not provided will be computed from the spiketrains.
    n_units : int
        Number of units
    unit_names : array
        Name of each unit
    unit_shapes : array
        Shape/Template of each unit
    unit_preferred_chan : array
        Preferred channel for each unit
    channel_names : array
        Name of the channels of the LFP files
    electrode_names : array
        Name of the electrode for each channel
    electrode_names_unique : array
        Unique electrode names
    fig_dirpath : str
        Path where the figure are saved
    segments : List of Neo Segments
        Neo Segment, 1 per unit
    channel_indexes : Neo ChannelIndex
        Neo ChannelIndex, 1 per tetrode
    signal_provided : bool
        True if LFP signal was provided, False otherwise
    """

    def __init__(self, spikesorting_dirpath, spikesorting_filename, probe_filepath, signal_dirpath=[],
                 event_filepath=[], event_csv_sep=';', fig_dirpath=[], save_fig=0, srate=30000, t_max=-1):
        NeoMother.__init__(self)
        self.spikesorting_dirpath = spikesorting_dirpath
        self.spikesorting_filename = spikesorting_filename
        self.probe_filepath = probe_filepath
        self.signal_dirpath = signal_dirpath
        self.srate = srate
        self.n_units = 0
        self.unit_names = ''
        self.unit_shapes = []
        self.unit_preferred_chan = []
        self.channel_names = []
        self.electrode_names = []
        self.electrode_names_unique = []
        self.save_fig = save_fig
        self.channel_indexes = []
        self.segments = []

        # Create result figures directory
        if save_fig:
            if not fig_dirpath:
                fig_dirpath = '.'
            self.fig_dirpath = spikeAnalysis_utils.createuniquedir(os.path.join(fig_dirpath, spikesorting_filename))
            self.fig_dirpath_srev = spikeAnalysis_utils.createuniquedir(os.path.join(self.fig_dirpath, 'SpikeRateEvolution'))
            self.fig_dirpath_raster = spikeAnalysis_utils.createuniquedir(os.path.join(self.fig_dirpath, 'RasterPlots'))
            self.fig_dirpath_crosscorr = spikeAnalysis_utils.createuniquedir(os.path.join(self.fig_dirpath, 'CrossCorr'))
            self.fig_dirpath_autocorr = spikeAnalysis_utils.createuniquedir(os.path.join(self.fig_dirpath, 'AutoCorr'))
            self.fig_dirpath_isi = spikeAnalysis_utils.createuniquedir(os.path.join(self.fig_dirpath, 'ISI'))
            self.fig_dirpath_unitshape = spikeAnalysis_utils.createuniquedir(os.path.join(self.fig_dirpath, 'UnitShape'))

        # Read spyking-circus result file "mydata.result.hdf5"
        f = h5py.File(os.path.join(spikesorting_dirpath, spikesorting_filename + '.result.hdf5'), 'r')
        spktimes_grp = f['spiketimes']
        self.n_units = len(spktimes_grp)

        # Sort data in increasing number of the unit
        unit_names_temp = list(spktimes_grp.keys())
        spktrains_temp = list(spktimes_grp.values())
        unit_num = np.zeros(self.n_units, dtype=int)
        for idx, name in enumerate(unit_names_temp):
            unit_num[idx] = re.findall('\d+', name)[0]
        sort_vect = unit_num.argsort()

        # Order the spike trains and unit names
        self.unit_names = [unit_names_temp[i] for i in sort_vect]
        spktrainsdataset = [spktrains_temp[i] for i in sort_vect]

        # Get maximum time
        if t_max == -1:
            for i in range(0, self.n_units):
                t_max_i = spktrainsdataset[i].value.max() / float(self.srate)
                if t_max_i > t_max:
                    t_max = t_max_i

        # Create the Neo Segment from the spiketimes and unit number, 1 Neo Segment per unit,
        # Each Neo Segment contain 1 SpikeTrain
        for i in range(0, self.n_units):
            seg = neo.Segment(name=self.unit_names[i], file_origin=spikesorting_filename)
            spktimes_i = (spktrainsdataset[i].value / srate).ravel()
            spktrain = neo.core.SpikeTrain(times=spktimes_i, units='sec', t_stop=t_max, name=self.unit_names[i],
                                           sampling_rate=srate, file_origin=spikesorting_filename)
            seg.spiketrains.append(spktrain)
            self.segments.append(seg)

        # Get unit preferred micro wire
        f_clusters = h5py.File(os.path.join(spikesorting_dirpath, spikesorting_filename + '.clusters.hdf5'), 'r')
        self.unit_preferred_chan = np.array(f_clusters['electrodes']).ravel().astype(int)

        # Get template of each unit : spike shape on each wire of the unit's preferred tetrode
        # First get number of time points in the templates
        for i in range(0, self.n_units):
            self.unit_shapes.append(spykingcircus_utils.get_templates_from_spykingcircus_file(spikesorting_dirpath,
                                                                                              spikesorting_filename, i))

        # Read data signal to get signal channels' name, if LFP signal is provided
        if signal_dirpath:
            filelist = os.listdir(signal_dirpath)
            r = re.compile(".*\.edf")
            edffilelist = list(filter(r.match, filelist))
            if not edffilelist:
                print('Could not find any edf file in {}'.format(signal_dirpath))
            else:
                self.channel_names = eeg_utils.getchannelnamesfromedffile(os.path.join(signal_dirpath, edffilelist[0]))
                self.electrode_names = [re.search('\D+', chan_name).group() for chan_name in self.channel_names]
                for el_name in self.electrode_names:
                    if el_name not in self.electrode_names_unique:
                        self.electrode_names_unique.append(el_name)
            self.signal_provided = True
        # If LFP signal is not provided, then create some
        else:
            self.signal_provided = False
            # Get tretrode groups
            tetrode_groups = spykingcircus_utils.get_channel_groups_from_spykingcircus_probefile(probe_filepath)
            n_tetrodes = len(tetrode_groups)
            channel_names = []
            for i in range(n_tetrodes):
                for j in range(4):
                    channel_names.append('Tetrode {} - channel {}'.format(i+1, j+1))
            self.channel_names = channel_names
            self.electrode_names = ['El unknown']*(4*n_tetrodes)
            for el_name in self.electrode_names:
                if el_name not in self.electrode_names_unique:
                    self.electrode_names_unique.append(el_name)

        # Create one ChannelIndex per tetrode
        tetrode_groups = spykingcircus_utils.get_channel_groups_from_spykingcircus_probefile(probe_filepath)
        for i, grp in enumerate(tetrode_groups):
            grp_channel_names = [self.channel_names[j] for j in grp] if self.channel_names else []
            tetrode_num = int(np.ceil(int(re.search('\d+', self.channel_names[grp[0]]).group()) / 4)) if \
                self.channel_names else -1
            electrode_name = self.electrode_names[grp[0]] if self.electrode_names else ''
            chx = neo.ChannelIndex(index=2, channel_names=grp_channel_names, channel_ids=grp,
                                   name='{} - tetrode {}'.format(electrode_name, tetrode_num))
            # Add units from this tetrode
            tetrode_units_pos, = np.where(np.in1d(self.unit_preferred_chan, grp))
            for upos in tetrode_units_pos:
                unit = neo.Unit(name=self.unit_names[upos])
                unit.spiketrains.append(self.segments[upos].spiketrains[0])
                chx.units.append(unit)
            self.channel_indexes.append(chx)

        # Create one ChannelIndex per electrode
        for i, elec_name in enumerate(self.electrode_names_unique):
            elec_chan_pos, elec_chan_name = self.elecnum2channelpos(i)
            chx = neo.ChannelIndex(index=1, name=elec_name, channel_names=elec_chan_name, channel_ids=elec_chan_pos)
            # Add units from this electrode
            el_units_pos, = np.where(np.in1d(self.unit_preferred_chan, elec_chan_pos))
            for upos in el_units_pos:
                unit = neo.Unit(name=self.unit_names[upos])
                unit.spiketrains.append(self.segments[upos].spiketrains[0])
                chx.units.append(unit)
            self.channel_indexes.append(chx)

        # Create event if event file is provided
        if event_filepath:
            self.read_event_file(event_filepath, event_csv_sep)

        # Set seaborn style for matplotlib
        sns.set()
        sns.set_context('paper')

    def __str__(self):
        desc_str = 'NeoAll Instance with {} units. 1 Neo segment per unit. Each segment contains 1 Neo spiketrain\n'.format(self.n_units)
        desc_str += '{} channel indexes\n'.format(len(self.channel_indexes))
        if self.signal_provided:
            desc_str += '{} channels and {} electrodes'.format(len(self.channel_names), len(self.electrode_names))
        # if len(self.segments) > 0 and self.segments[0].e

        return desc_str

    def read_event_file(self, event_filepath, event_csv_sep=';'):
        """ Read the event file, specified by `event_filepath` and add the event to each segments using the
        :func:`neoStructures.NeoAll.create_event_list` method. This can be done directly when instanciating the
        NeoAll instance by providin the `event_filepath` parameter or by calling this function.

        Parameters
        ----------
        event_filepath : str
            Path to the event file
        event_csv_sep : str
            CSV file separator (default: ';')

        """
        # Read event file (csv file)
        with open(event_filepath, 'r') as f:
            reader = csv.reader(f, delimiter=event_csv_sep)
            next(reader, None)  # skip the headers
            event_data = [row for row in reader]
        ev_time = np.array([float(row[EVENT_TIME_COL]) for row in event_data])
        ev_type = np.array([row[EVENT_TYPE_COL] for row in event_data])
        ev_type_unique = list(set(ev_type))
        for i_ev, ev_type_i in enumerate(ev_type_unique):
            ev_pos_i = np.where(ev_type == ev_type_i)
            self.create_event_list(ev_time[ev_pos_i], ev_type_i)

    def create_event_list(self, event_times, event_names, segment_pos=-1):
        """ Create Neo Event objects for each event specified by event_times and event_names. Add these events to all
        segments specified by segment_pos

        Parameters
        ----------
        event_times : array
            Time of each event (s)
        event_names : array
            Name of each event
        segment_pos : int | array
            Position of the segments in which events must be added

        """
        if np.isscalar(segment_pos) and segment_pos == -1:
            segment_pos = np.arange(0, len(self.segments))
        if len(event_times) != len(event_names) and isinstance(event_names, list):
            print('Arguments event_times and event_names must have the same size')
        n_events = len(event_names) if isinstance(event_names, list) else 1

        for i_ev in range(0, n_events):
            event_times_i = event_times if n_events == 1 else event_times[i_ev]
            event_name_i = event_names if isinstance(event_names, str) else event_names[i_ev]
            event = neo.Event(times=event_times_i*s, name=event_name_i)
            for seg_pos in segment_pos:
                self.segments[seg_pos].events.append(event)

    def reject_spikes_around_events(self, event_name, time_pre, time_post):
        """ Reject spikes around events specified by ``event_name``.
        All spikes occuring in the window [t_event-time_pre < t < t_event+time_post] will be rejected, for each event.
        Modify the spiketrains of the segments in which an event is defined.

        Parameters
        ----------
        event_name : str
            Name of the events to select
        time_pre : float
            Beginning of the window, relative to the event time
        time_post : float
            End of the window, relative to the event time

        """
        if not isinstance(event_name, str):
            print('Argument event_name should be a string')
            return
        for i_seg, seg in enumerate(self.segments):
            if not seg.events:
                continue
            event_names_seg = np.array([event.name for event in seg.events])
            if event_name not in event_names_seg:
                continue
            event_pos, = np.where(event_names_seg == event_name)
            event_times = seg.events[int(event_pos)].times
            new_spiketrain, a = neo_utils.spiketrainrejectaroundevent(self.segments[i_seg].spiketrains[0],
                                                                      event_times, time_pre, time_post)
            self.segments[i_seg].spiketrains[0] = new_spiketrain

    def create_epochs_around_events(self, event_name, time_offset, epoch_duration, epoch_names):
        """  Create Neo Epochs objects around events specified by event name. The epoch times start at the event time
        plus an possible offset time_offset, the duration is specified by epoch_duration.
        Epochs are added to all segments

        Parameters
        ----------
        event_name : str
            Event name from where to create epochs
        time_offset : float
            Time offset from the event time
        epoch_duration : float
            Duration of the epoch
        epoch_names : str
            Name of the epoch

        """
        if not isinstance(event_name, str):
            print('Argument event_name should be a string')
            return
        if not len(time_offset) == len(epoch_duration) == len(epoch_names):
            print('Arguments time_offset, epoch_duration and epoch_names must have the same size')
            return
        n_epochs = len(epoch_names)
        # For all segments
        event_name_found = 0
        for i_seg, seg in enumerate(self.segments):
            if not seg.events:
                continue
            event_names_seg = np.array([event.name for event in seg.events])
            if event_name not in event_names_seg:
                continue
            event_name_found = 1
            event_pos, = np.where(event_names_seg == event_name)
            event_times = seg.events[int(event_pos)].times
            for i_epoch in range(0, n_epochs):
                time_offset_i = time_offset[i_epoch]*s
                time_offset_i = time_offset_i if isinstance(time_offset_i, quantities.quantity.Quantity) else time_offset_i*s
                epoch_duration_i = epoch_duration[i_epoch]
                epoch_duration_i = epoch_duration_i if isinstance(epoch_duration_i, quantities.quantity.Quantity) else epoch_duration_i*s
                epoch_i = neo.Epoch(times=event_times + time_offset_i, durations=epoch_duration_i,
                                    name=epoch_names[i_epoch])
                self.segments[i_seg].epochs.append(epoch_i)
        if not event_name_found:
            print('Could not find any events named {} in any of the segments event list'.format(event_name))

    def plot_isi(self, unit_pos=-1, bin_duration_ms=1, tmax_ms=100, logscale=0):
        """ Plot the Inter-Spike Intervals histogram for unit specified by unit_pos

        Parameters
        ----------
        unit_pos :  int
            Unit position
        bin_duration_ms : float (default: 1)
            Bin duration for the histogram in ms
        tmax_ms : float (default: 100)
            Maximal time interval
        logscale : bool
            If True, the scale of the x-axis is logarithmic

        """
        if np.isscalar(unit_pos) and unit_pos == -1:
            unit_pos = np.arange(0, self.n_units, 1)
        elif np.isscalar(unit_pos):
            unit_pos = np.array([unit_pos])
        for u_pos in unit_pos:
            isi = elephant.statistics.isi(self.segments[u_pos].spiketrains).ravel()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if not logscale:
                isi_hist, bins = np.histogram(1000 * isi, bins=int(tmax_ms/bin_duration_ms), range=(0, tmax_ms))
                ax.bar(bins[:-1], isi_hist, width=1, linewidth=0)
            else:
                bins = np.logspace(np.log10(0.5), np.log10(tmax_ms), 300)
                isi_hist, bins = np.histogram(1000 * isi, bins=bins, range=(0, tmax_ms))
                bin_width = bins[1:] - bins[:-1]
                plt.semilogx(bins[:-1], isi_hist/bin_width)
            plt.xlim(0, tmax_ms)
            ax.set(title='Interspike Interval - unit : {}'.format(self.unit_names[u_pos]),
                   xlabel='ISI (ms)', ylabel='Incidence')
            if self.save_fig:
                fig.savefig(os.path.join(self.fig_dirpath_isi, 'isi_{}.{}'.format(self.unit_names[u_pos],
                                                                                  RES_FIG_FORMAT)), dpi=RES_FIG_DPI)

    def plot_spikerate_evolution_group(self, grp_index, sigma_gauss_kernel=30*s, merge_plots=0,
                                       norm_spikerate=0, mean_plot=0, plot_events=1):
        """ Plot the spike rate evolution across all the length of the segment of all units of the channel index
        specified by grp_index. The spike rate is computed using the Elephant module

        Parameters
        ----------
        grp_index : int | str
            Either the position of the channel index, or its name
        sigma_gauss_kernel : float | duration quantity (default: 100*s)
            Time of the smoothing gaussian kernel
        merge_plots : bool (default: False)
            If True, merge the plots
        norm_spikerate : bool (default: False)
            If True, normalize the spiking rate between 0 and 1
        mean_plot : bool (default: False)
            If True, only the mean spikerate across all units of the group/channel_index is plot
        plot_events : bool (default: True)
            If True, plot a vertical line for each event

        """
        if type(sigma_gauss_kernel) == int or type(sigma_gauss_kernel) == float:
            sigma_gauss_kernel = sigma_gauss_kernel*s
        smoothing_kernel = elephant.kernels.GaussianKernel(sigma=sigma_gauss_kernel)
        channel_index_names = [channel_index.name for channel_index in self.channel_indexes]
        # if np.isscalar(grp_index) and isinstance(grp_index, Number):
        if np.isscalar(grp_index):
            grp_index = np.array([int(grp_index)])
        elif np.isscalar(grp_index) and isinstance(grp_index, str):
            if grp_index not in channel_index_names:
                print('Could not find channel index named {}'.format(grp_index))
                print('Current channel_indexes are : {}'.format(channel_index_names))
            else:
                grp_index = channel_index_names.index(grp_index)
        else:
            if isinstance(grp_index[0], str):
                grp_names = grp_index
                grp_index = list()
                for name in grp_names:
                    if name not in channel_index_names:
                        print('Could not find channel index named {}'.format(name))
                    else:
                        grp_index.append(channel_index_names.index(name))

        spkrate_grp_list = list()
        for channel_index_pos in grp_index:
            if channel_index_pos < 0 or channel_index_pos >= len(self.channel_indexes):
                print('Channel index out of range : {}. {} channel indexes are defined'.format(channel_index_pos, len(self.channel_indexes)))
                continue
            channel_index = self.channel_indexes[channel_index_pos]
            spkrate_list = list()
            for unit in channel_index.units:
                if len(unit.spiketrains) > 1:
                    print('Cannot deal with multiple spiketrains for one unit')
                if unit.spiketrains:
                    spkrate_u = np.array(elephant.statistics.instantaneous_rate(unit.spiketrains[0], 1*s,
                                                                       kernel=smoothing_kernel)).ravel()
                    if norm_spikerate:
                        spkrate_u = (spkrate_u - spkrate_u.min()).astype(float) / (spkrate_u.max() - spkrate_u.min())
                    spkrate_list.append(spkrate_u)
            spkrate_grp_list.append(spkrate_list)
            if not spkrate_list:
                print('No unit for group {}'.format(channel_index.name))

        events = self.segments[0].events
        event_names = [ev.name for ev in events]
        ev_color = sns.husl_palette(len(events))
        # Plot
        plot_unite = 'normalized' if norm_spikerate else 'Hz'
        if not merge_plots:  # 1 figure per group/channel_index if any spiketrains
            for i, channel_index_pos in enumerate(grp_index):
                if channel_index_pos < 0 or channel_index_pos >= len(self.channel_indexes):
                    continue
                channel_index = self.channel_indexes[channel_index_pos]
                spkrate_list = spkrate_grp_list[i]
                if not spkrate_list:
                    continue
                unit_names = [unit.name for unit in channel_index.units]
                legend_labels = ['{} ({})'.format(unit_names[i], self.getunitpreferredchannelname(unit_names[i]))
                                 for i in range(0, len(unit_names))]
                fig = plt.figure()
                ax = fig.add_subplot(111)
                if not mean_plot:
                    for spkrate in spkrate_list:
                        plt.plot(spkrate, zorder=4)
                    plt.autoscale(axis='x', tight=True)
                    ax.set(title='Spike Rate Evolution ({}) - {} - Gaussian smoothing kernel, sigma = {}'.format(
                        plot_unite, channel_index.name, sigma_gauss_kernel), xlabel='time (s)',
                        ylabel='Spike Rate ({})'.format(plot_unite))
                    fig_name = 'spikerate_evolution_{}.{}'.format(channel_index.name, RES_FIG_FORMAT)
                else:
                    plt.plot(np.array(spkrate_list).mean(0), zorder=4)
                    plt.autoscale(axis='x', tight=True)
                    ax.set(
                        title='Mean Spike Rate Evolution ({}) - {} - Gaussian smoothing kernel, sigma = {}'.format(
                            plot_unite, channel_index.name, sigma_gauss_kernel), xlabel='time (s)',
                        ylabel='Spike Rate ({})'.format(plot_unite))
                    legend_labels = ['mean({})'.format(', '.join(legend_labels))]
                    fig_name = 'spikerate_evolution_mean_{}.{}'.format(channel_index.name, RES_FIG_FORMAT)
                if plot_events and events:
                    for i_ev, ev in enumerate(events):
                        ax.plot((np.array(ev.times), np.array(ev.times)), ax.get_ylim(), alpha=0.3, zorder=3,
                                color=ev_color[i_ev])
                    ax.autoscale(axis='both', tight=True)
                    ax.legend(legend_labels+event_names)
                else:
                    ax.legend(legend_labels, loc='best')
                if self.save_fig:
                    fig.savefig(os.path.join(self.fig_dirpath_srev, fig_name), dpi=RES_FIG_DPI)
        else:  # Merge plots
            fig = plt.figure()
            ax = fig.add_subplot(111)
            sns_colors = sns.color_palette(n_colors=len(grp_index))
            legend_labels = list()
            grp_names = list()
            for i, channel_index_pos in enumerate(grp_index):
                if channel_index_pos < 0 or channel_index_pos >= len(self.channel_indexes):
                    continue
                channel_index = self.channel_indexes[channel_index_pos]
                spkrate_list = spkrate_grp_list[i]
                if not spkrate_list:
                    continue
                grp_names.append(channel_index.name)
                unit_names = [unit.name for unit in channel_index.units]
                legend_labels_i = ['{} ({})'.format(unit_names[k], self.getunitpreferredchannelname(unit_names[k]))
                                   for k in range(0, len(unit_names))]
                if not mean_plot:
                    sns_col_palette = sns.dark_palette(sns_colors[i], input="rgb", n_colors=2+len(spkrate_list))
                    for j, spkrate in enumerate(spkrate_list):
                        plt.plot(spkrate, color=sns_col_palette[2+j], zorder=4)
                    legend_labels.append(legend_labels_i)
                else:
                    plt.plot(np.array(spkrate_list).mean(0), color=sns_colors[i])
                    legend_labels.append('{} (n={})'.format(channel_index.name, len(spkrate_list)))
                    plt.autoscale(axis='x', tight=True)
            ax.legend(np.hstack(legend_labels), loc='best')
            ax.set(title='Spike Rate Evolution ({}) - {} - Gaussian smoothing kernel, sigma = {}'.format(
                plot_unite, ' '.join(grp_names), sigma_gauss_kernel), xlabel='time (s)',
                ylabel='Spike Rate ({})'.format(plot_unite))
            if plot_events and events:
                for i_ev, ev in enumerate(events):
                    ax.plot((np.array(ev.times), np.array(ev.times)), ax.get_ylim(), alpha=0.3, zorder=3,
                            color=ev_color[i_ev])
                ax.autoscale(axis='both', tight=True)
                ax.legend(legend_labels + event_names)
            if self.save_fig:
                fig.savefig(os.path.join(self.fig_dirpath_srev, 'spikerate_evolution_{}.{}'
                                         .format('_'.join(grp_names), RES_FIG_FORMAT)), dpi=RES_FIG_DPI)

    def plot_spikerate_evolution(self, unit_pos=-1, sigma_gauss_kernel=30*s, merge_plots=0, norm_spikerate=0,
                                 plot_events=1):
        """ Plot the spikerate evolution across all the segment duration for units specified by unit_pos

        Parameters
        ----------
        unit_pos : int (default: -1)
            Position of the unit. If equal to -1
        sigma_gauss_kernel : float | duration quantity (default: 150s)
            Duration of the smoothing gaussian kernel
        merge_plots : bool (default: False)
            If True, merge the plots
        norm_spikerate : bool (default: False)
            If True, normalize the spiking rate between 0 and 1
        plot_events : bool (default: True)
            If True, plot a vertical line for each event

        """
        if type(sigma_gauss_kernel) == int or type(sigma_gauss_kernel) == float:
            sigma_gauss_kernel = sigma_gauss_kernel*s
        smoothing_kernel = elephant.kernels.GaussianKernel(sigma=sigma_gauss_kernel)
        if np.isscalar(unit_pos) and unit_pos == -1:
            unit_pos = np.arange(0, self.n_units, 1)
        elif np.isscalar(unit_pos):
            unit_pos = np.array([unit_pos])

        spkrate_list = list()
        for u_pos in unit_pos:
            spkrate_i = np.array(elephant.statistics.instantaneous_rate(self.segments[u_pos].spiketrains[0], 1*s,
                                                                        kernel=smoothing_kernel)).ravel()
            if norm_spikerate:
                spkrate_i = (spkrate_i - spkrate_i.min()).astype(float) / (spkrate_i.max() - spkrate_i.min())
            spkrate_list.append(spkrate_i)

        plot_unite = 'normalized' if norm_spikerate else 'Hz'
        if not merge_plots:
            for i, u_pos in enumerate(unit_pos):
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.plot(spkrate_list[i], zorder=4)
                ax.autoscale(axis='both', tight=True)
                if plot_events:
                    events = self.segments[u_pos].events
                    event_names = [ev.name for ev in events]
                    ev_color = sns.husl_palette(len(events))
                    for i_ev, ev in enumerate(events):
                        ax.plot((np.array(ev.times), np.array(ev.times)), ax.get_ylim(), alpha=0.3, zorder=3,
                                color=ev_color[i_ev])
                    if events:
                        ax.autoscale(axis='both', tight=True)
                        ax.legend(['Smoothed Spiking rate']+event_names)

                ax.set(title='Spike Rate Evolution ({}) - unit : {} - Gaussian smoothing kernel, sigma = {}'.format(
                    plot_unite, self.unit_names[u_pos], sigma_gauss_kernel), xlabel='time (s)',
                    ylabel='Spike Rate ({})'.format(plot_unite))
                if self.save_fig:
                    fig.savefig(os.path.join(self.fig_dirpath_srev, 'spikerate_evolution_{}.{}'
                                             .format(self.unit_names[u_pos], RES_FIG_FORMAT)), dpi=RES_FIG_DPI)

        if merge_plots:
            fig_merge = plt.figure()
            ax = fig_merge.add_subplot(111)
            for spkrate in spkrate_list:
                plt.plot(spkrate, zorder=4)
            unit_sel_names = [self.unit_names[i] for i in unit_pos]
            ax.set(title='Spike Rate Evolution ({}) - Gaussian smoothing kernel, sigma = {}'.format(plot_unite,
                   sigma_gauss_kernel), xlabel='time (s)', ylabel='Spike Rate ({})'.format(plot_unite))
            ax.legend(unit_sel_names, loc='best')
            ax.autoscale(axis='x', tight=True)
            if plot_events:
                events = self.segments[unit_pos[0]].events
                event_names = [ev.name for ev in events]
                ev_color = sns.husl_palette(len(events))
                for i_ev, ev in enumerate(events):
                    ax.plot((np.array(ev.times), np.array(ev.times)), ax.get_ylim(), alpha=0.3, zorder=3,
                            color=ev_color[i_ev])
                ax.legend(unit_sel_names + event_names, loc='best')
                ax.autoscale(axis='both', tight=True)
            if self.save_fig:
                fig_merge.savefig(os.path.join(self.fig_dirpath_srev, 'spikerate_evolution_{}.{}'.
                                               format('_'.join(unit_sel_names), RES_FIG_FORMAT)), dpi=RES_FIG_DPI)

    def get_spikerate_change_ratio_on_epochs(self, epoch_name_target, epoch_name_ref):
        """ Compute the mean spikerate on all target and reference epochs, calcul the ratio of these two, for each unit

        Parameters
        ----------
        epoch_name_target : str
            Name of the target epochs
        epoch_name_ref : str
            Name of the reference epochs

        Returns
        -------
        spikerate_ratio : list
        spikerate_target : list
        spikerate_ref : list

        """
        if isinstance(epoch_name_ref, list):
            print('Argument epoch_name_ref should contain only one epoch name')
            return
        n_segments = len(self.segments)
        if isinstance(epoch_name_target, list):
            spikerate_ratio, spikerate_target, spikerate_ref = [], [], []
            for epoch_name_target_i in epoch_name_target:
                spikerate_ratio_i, spikerate_target_i, spikerate_ref_i = \
                    self.get_spikerate_change_ratio_on_epochs(epoch_name_target_i, epoch_name_ref)
                spikerate_ratio.append(spikerate_ratio_i)
                spikerate_target.append(spikerate_target_i)
                spikerate_ref.append(spikerate_ref_i)
        else:
            # For each segments (i.e. for each unit)
            spikerate_target, spikerate_ref = np.zeros(n_segments), np.zeros(n_segments)
            spikerate_target_mat, spikerate_ref_mat = [], []
            for i, seg in enumerate(self.segments):
                if not seg.epochs:
                    continue
                epoch_names = [epoch.name for epoch in seg.epochs]
                if epoch_name_target not in epoch_names:
                    print('No epoch named {} in segment {}'.format(epoch_name_target, seg.name))
                    continue
                if epoch_name_ref not in epoch_names:
                    print('No epoch named {} in segment {}'.format(epoch_name_ref, seg.name))
                    continue
                spkrate_per_epoch_target_i = neo_utils.getmeanspikerateonepoch(seg, epoch_name_target)
                spkrate_per_epoch_ref_i = neo_utils.getmeanspikerateonepoch(seg, epoch_name_ref)
                spikerate_target[i] = spkrate_per_epoch_target_i.mean()
                spikerate_ref[i] = spkrate_per_epoch_ref_i.mean()
                spikerate_target_mat.append(spkrate_per_epoch_target_i)
                spikerate_ref_mat.append(spkrate_per_epoch_ref_i)
            spikerate_ratio = spikerate_target / spikerate_ref if spikerate_ref.sum() != 0 else []
        return spikerate_ratio, spikerate_target, spikerate_ref

    def plot_spikerate_change_ratio_on_epochs(self, epoch_name_target, epoch_name_ref, plot_ratio_only=0,
                                              single_plot=1, merge_plot=0):
        """ Plot the mean spikerate ratio between 2 epochs. Call the method get_spikerate_change_ratio_on_epochs.

        Parameters
        ----------
        epoch_name_target : str
            Name of the target epochs
        epoch_name_ref : str
            Name of the reference epochs
        plot_ratio_only : bool (default: False)
            If True, plot only the ratio and not the target and reference spiking rate
        single_plot : bool (default: True)
            If True, plot individual figures
        merge_plot : bool (default: False)
            If True, merge the figures

        Returns
        -------
        spikerate_ratio : array


        """
        spikerate_ratio, spikerate_target, spikerate_ref = self.get_spikerate_change_ratio_on_epochs(epoch_name_target,
                                                                                                     epoch_name_ref)
        if spikerate_ratio.size == 0:
            print('No spikes')
            return
        n_segments = len(self.segments)
        # Plot
        x_vect = np.arange(0, n_segments, 1)
        if not isinstance(spikerate_ratio, list):
            spikerate_ratio, spikerate_target, spikerate_ref = [spikerate_ratio], [spikerate_target], [spikerate_ref]
            epoch_name_target = [epoch_name_target]
        # Individual plot
        if single_plot:
            for i in range(len(spikerate_ratio)):
                fig = plt.figure()
                if not plot_ratio_only:
                    ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
                    plt.bar(x_vect-0.2, spikerate_target[i], 0.4)
                    plt.bar(x_vect+0.2, spikerate_ref[i], 0.4)
                    ax0.set(title='Mean firing rate per condition : {} / {}'.format(epoch_name_target[i], epoch_name_ref),
                            ylabel='Mean firing rate (Hz)')
                    ax0.legend([epoch_name_target[i], epoch_name_ref], loc='best')
                    ax0.set_xlim(-0.5, x_vect[-1]+0.5)
                    ax0.xaxis.set(ticks=x_vect, ticklabels=[])
                    ax1 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
                else:
                    ax1 = fig.add_subplot(111)
                plt.bar(x_vect, spikerate_ratio[i], 0.5, color=sns.color_palette()[3])
                ax1.set_xlim(-0.5, x_vect[-1]+0.5)
                plt.plot(ax1.get_xlim(), [1, 1], color=sns.color_palette()[5], zorder=0)
                ax1.set(ylabel='{} / {}'.format(epoch_name_target[i], epoch_name_ref))
                ax1.xaxis.set(ticks=x_vect, ticklabels=self.unit_names)
        # Merge plot
        if merge_plot:
            fig_merge = plt.figure()
            n_axes = len(spikerate_ratio)
            x_vect = np.arange(0, n_segments, 1)
            for i in range(0, n_axes):
                ax = plt.subplot2grid((n_axes, 1), (i, 0)) if i == 0 else plt.subplot2grid((n_axes, 1), (i, 0),
                                                                                           sharey=ax, sharex=ax)
                plt.bar(x_vect, spikerate_ratio[i], 0.5, color=sns.color_palette()[3])
                ax.set_xlim(-0.5, x_vect[-1]+0.5)
                plt.plot(ax.get_xlim(), [1, 1], color=sns.color_palette()[5], zorder=0)
                ax.set(ylabel='{} / {}'.format(epoch_name_target[i], epoch_name_ref))
                if i == n_axes-1:
                    ax.xaxis.set(ticks=x_vect, ticklabels=self.unit_names)
                else:
                    ax.xaxis.set(ticklabels=[])
                if i == 0:
                    ax.set(title='Mean firing rate ratio between time periods')

        return spikerate_ratio

    def plot_crosscorrelogram(self, unit_pos_a, unit_pos_b, bin_time=1*ms, max_lag_time=80*ms, smooth_time=10*ms,
                              merge_plots=0, same_yscale=0, fill_under_plot=0, do_stat=0, n_surrogates=100,
                              normal_dist_sd=10*ms, do_plot=1):
        """ Plot the cross-correlogram between spiketrains of units specified by ``unit_pos_a`` and ``unit_pos_b``

        Parameters
        ----------
        unit_pos_a : int
            Position of first unit
        unit_pos_b : int
            Position of second unit
        bin_time : float | duration quantity (default: 1*ms)
            Bin duration (in ms)
        max_lag_time : float | duration quantity (default: 80*ms)
            Maximal lag time (in ms)
        smooth_time : float | duration quantity (default: 10*ms)
            Smoothing time (in ms)
        merge_plots : bool (default: False)
            If True, merge all the cross-correlogram in one figure
        same_yscale : bool (default: False)
            If True, the y-axis limits are the same
        fill_under_plot : bool (default: False)
            If True, fill the area under the plot
        do_stat : bool (default: False)
            If True, do some wrong statistics
        n_surrogates : int (default: 100)
            Number of jittered spike-trains when doing the wrong stats
        normal_dist_sd : float | duration quantity (default: 50*ms)
            Parameter for the stats

        Returns
        -------
        cch : array
            Cross-correlation histogram between the spike-trains of input units
        bins : array
            Contains the IDs of the individual histogram bins
        """
        if type(bin_time) == int or type(bin_time) == float:
            bin_time = bin_time*ms
        if type(max_lag_time) == int or type(max_lag_time) == float:
            max_lag_time = max_lag_time*ms
        if type(smooth_time) == int or type(smooth_time) == float:
            smooth_time = smooth_time*ms

        unit_pos_a = np.atleast_1d(unit_pos_a).astype(int)
        unit_pos_b = np.atleast_1d(unit_pos_b).astype(int)
        if unit_pos_a.size > 1 or unit_pos_b.size > 1:
            cch_list = list()
            maxs_crosscorr, maxs_autocorr = list(), list()
            for upos_a in unit_pos_a:
                for upos_b in unit_pos_b:
                    [cch, bins] = self.plot_crosscorrelogram(int(upos_a), int(upos_b), bin_time=bin_time, max_lag_time=max_lag_time,
                                                             smooth_time=smooth_time, merge_plots=merge_plots, same_yscale=same_yscale,
                                                             fill_under_plot=fill_under_plot, do_stat=do_stat, n_surrogates=n_surrogates,
                                                             normal_dist_sd=normal_dist_sd, do_plot=not merge_plots)
                    cch_list.append(cch)
                    maxs_autocorr.append(cch.max()) if upos_a == upos_b else maxs_crosscorr.append(cch.max())
            max_crosscorr, max_autocorr = np.array(maxs_crosscorr).max(), np.array(maxs_autocorr).max()
            if merge_plots:
                plt.figure()
                for i_row, upos_b in enumerate(unit_pos_b):
                    for i_col, upos_a in enumerate(unit_pos_a):
                        cch_smooth = eeg_utils.smooth1d(cch_list[i_col*len(unit_pos_b)+i_row],
                                                        np.hamming(int(smooth_time / bin_time)), 'same')
                        ax = plt.subplot2grid((len(unit_pos_b), len(unit_pos_a)), (i_row, i_col))
                        if not fill_under_plot:
                            plt.bar(bins, cch_list[i_col*len(unit_pos_b)+i_row], width=1, linewidth=0)
                            # plt.plot(bins, cch_smooth, c='y', alpha=0.7, lw=3)
                            # plt.bar(bins - float(bin_time) / 2, cch_list[i_col*len(unit_pos_b)+i_row], 1)
                            # plt.plot(bins - float(bin_time) / 2, cch_smooth, c='y', alpha=0.7, lw=3)
                        else:
                            plt.fill_between(bins, cch_smooth)
                            # plt.fill_between(bins - float(bin_time)/2, cch_smooth)
                            plt.autoscale(axis='x', tight=True)
                        if same_yscale:
                            ax.set_ylim(0, max_autocorr) if upos_a == upos_b else ax.set_ylim(0, max_crosscorr)
                        if i_row == 0:
                            ax.set(title=self.unit_names[upos_a])
                        if i_col == 0:
                            ax.set(ylabel=self.unit_names[upos_b])
                        ax.autoscale(axis='x', tight=True)
        else:
            unit_pos_a, unit_pos_b = int(unit_pos_a), int(unit_pos_b)
            # Compute binned spike trains
            binned_spktrain_a = elephant.conversion.BinnedSpikeTrain(self.segments[unit_pos_a].spiketrains[0], bin_time)
            if unit_pos_a == unit_pos_b:
                binned_spktrain_b = binned_spktrain_a
            else:
                binned_spktrain_b = elephant.conversion.BinnedSpikeTrain(self.segments[unit_pos_b].spiketrains[0],
                                                                         bin_time)
            # Compute cross correlogram
            cch, bins = elephant.spike_train_correlation.cch(binned_spktrain_a, binned_spktrain_b,
                                                             (-max_lag_time, max_lag_time)*bin_time)
            cch = np.array(cch).flatten()
            # If auto correlogram, remove center peak
            if unit_pos_a == unit_pos_b:
                cch[int(np.floor(len(bins) / 2.0))] = 0
            # cch_smooth = eeg_utils.smooth1d(cch, np.hamming(int(smooth_time/bin_time)), 'same')

            # Statistics
            if do_stat:
                spktrain_b = self.segments[unit_pos_b].spiketrains[0]
                cch_jitter_array = np.zeros((n_surrogates, len(cch)))
                for i in tqdm(range(0, n_surrogates)):
                    jitter_spktrain_b = np.array(spktrain_b)*spktrain_b.units + normal_dist_sd * \
                                                                                np.random.randn(len(spktrain_b))
                    jitter_spktrain_b[jitter_spktrain_b < spktrain_b.t_start] = spktrain_b.t_start
                    jitter_spktrain_b[jitter_spktrain_b > spktrain_b.t_stop] = spktrain_b.t_stop
                    jitter_spktrain_b = neo.SpikeTrain(jitter_spktrain_b, units=spktrain_b.units,
                                                       t_start=spktrain_b.t_start, t_stop=spktrain_b.t_stop)
                    jitter_binned_spktrain_b = elephant.conversion.BinnedSpikeTrain(jitter_spktrain_b, bin_time)
                    cch_jitter, _ = elephant.spike_train_correlation.cch(binned_spktrain_a, jitter_binned_spktrain_b,
                                                                         (-max_lag_time, max_lag_time)*bin_time)
                    cch_jitter_array[i] = np.array(cch_jitter).ravel()
                cch_jitter_mean = cch_jitter_array.mean(0)
                cch_jitter_mean_smooth = eeg_utils.smooth1d(cch_jitter_mean, np.hamming(int(smooth_time / bin_time)), 'same')
                cch_jitter_ci_high = cch_jitter_mean + 2.58 * cch_jitter_array.std(0) / np.sqrt(n_surrogates)
                cch_jitter_ci_high_smooth = eeg_utils.smooth1d(cch_jitter_ci_high, np.hamming(int(smooth_time / bin_time)), 'same')
                cch_jitter_ci_low = cch_jitter_mean - 2.58 * cch_jitter_array.std(0) / np.sqrt(n_surrogates)
                cch_jitter_ci_low_smooth = eeg_utils.smooth1d(cch_jitter_ci_low, np.hamming(int(smooth_time / bin_time)),
                                                              'same')

            if do_plot:
                # Plot
                fig_corr = plt.figure()
                ax = fig_corr.add_subplot(111)
                plt.bar(bins, cch, width=1, linewidth=0)
                # plt.bar(bins-float(bin_time)/2, cch, 1)
                plt.autoscale(tight='True')
                ax.set(title='Cross-correlogram - unit : {} VS {} '.format(self.unit_names[unit_pos_a],
                                                                           self.unit_names[unit_pos_b]),
                       xlabel='Lag (ms)', ylabel='counts')
                if do_stat:
                    # ax.grid(False)
                    plt.plot(bins, cch_jitter_mean_smooth, c='y', alpha=0.7, lw=3)
                    plt.plot(bins, cch_jitter_ci_high_smooth, c='y', alpha=0.7, lw=2, ls=':')
                    plt.plot(bins, cch_jitter_ci_low_smooth, c='y', alpha=0.7, lw=2,  ls=':')

                if self.save_fig:
                    fig_name = 'autocorr_{}'.format(self.unit_names[unit_pos_a]) if unit_pos_a == unit_pos_b else \
                        'crosscorr_{}_{}'.format(self.unit_names[unit_pos_a], self.unit_names[unit_pos_b])
                    fig_corr.savefig(os.path.join(self.fig_dirpath_crosscorr, fig_name+'.'+RES_FIG_FORMAT), dpi=RES_FIG_DPI)

            return cch, bins

    def plot_unit_shape(self, unit_pos=-1, plot_mean=0, ax=[]):
        """ Plot the shape of unit specified by unit_pos

        Parameters
        ----------
        unit_pos : int
            Position of the unit
        plot_mean : bool
            If True, plot the mean of all tetrode templates/shapes
        ax : axis | None
            Axis to plot on. If none create a new figure and an axis

        """
        if np.isscalar(unit_pos) and unit_pos == -1:
            unit_pos = np.arange(0, self.n_units, 1)
        elif np.isscalar(unit_pos):
            unit_pos = np.array([unit_pos])
        if not self.unit_shapes:
            print('No unit templates')
        else:
            for upos in unit_pos:
                self.subfun_plot_unit_shape(upos, plot_mean, ax=ax)

    def plot_raw_unit_shape(self, unit_pos, signal_dirpath, artefact_free_periods_filepath, sep=';', fir_order=400,
                            iir_type='bessel', iir_order=2, fn_hz=[300, 3000], half_duration=0.0018, realign=False,
                            plot_mean_shape=True, plot_density_plot=True):
        """ Get the raw signal of each action potential for the unit specified by ``unit_pos``.
        This allows to investigate the filtering effects on the shape of the unit.

        Parameters
        ----------
        unit_pos : int
            Unit's position
        signal_dirpath : str
            Path to the directory containing the EEG files
        artefact_free_periods_filepath : str
            Path to the file containing the periods free of artefacts. Must be a CSV file with 3 columns in the order
            {'Filename', 't_start', 't_end'}, with a 1 row header.
        sep : str (default: ';')
            CSV separator
        fir_order : int (default : 100)
            FIR filter order
        iir_type : str (default: 'bessel')
            IIR filter type - can be 'bessel', 'butter'
        iir_order : int (default : 2)
            IIR filter order
        fn_hz : float | array
            Cut-off frequency in Hertz. If a scalar, filter will be a high-pass. If contains 2 frequencies, filter
            will be a band-pass filter. Default : [300, 3000]
        half_duration : float (default : 0.0018)
            Half-duration of the observed spikes
        realign : bool (default : False)
            If true, tries to realign the spike traces based on the extremum.
        plot_mean_shape : bool (default : True)
            If true, plot the mean unit shape
        plot_density_plot : bool (default : True)
            If true, plot the density plot

        """
        fn_hz = np.array(fn_hz)
        if fn_hz.size > 2 or fn_hz.size < 1:
            raise ValueError('Argument fn_hz must be either a scalar or an array with 2 elements')
        spike_train = self.segments[unit_pos].spiketrains[0]
        n_spikes = len(spike_train)
        print('{} spikes in the spike train of unit {}'.format(n_spikes, self.unit_names[unit_pos]))
        # Get unit preferred channel
        unit_chan_pos = self.unit_preferred_chan[unit_pos]
        unit_chan_name = self.channel_names[unit_chan_pos]

        # Read csv file
        df = pd.read_csv(artefact_free_periods_filepath, sep=sep, decimal=',')
        filename, keep_start, keep_end = np.array(df['Filename']), np.array(df['t_start']).astype(float), np.array(df['t_end']).astype(float)
        if not keep_start.size == keep_end.size == filename.size:
            raise ValueError('Columns Filename and keep_start and keep_end must have the same size')

        # Pre-load all the EEG files and filter the unit's preferred channel
        unique_filenames, n_files = np.unique(filename), np.unique(filename).size
        data_chan_raw, data_chan_filtered_fir, data_chan_filtered_for, data_chan_filtered_forback = [], [], [], []
        srates = np.zeros(n_files)
        btype = 'highpass' if fn_hz.size == 1 else 'bandpass'
        for i, filename_i in enumerate(unique_filenames):
            raw_i = mne.io.read_raw_edf(os.path.join(signal_dirpath, filename_i))
            data_chan_raw_i = raw_i.get_data(picks=unit_chan_pos)
            srates[i] = raw_i.info['sfreq']
            # Filter design
            b_fir, a_fir = signal.firwin(fir_order, cutoff=(2 / srates[i]) * fn_hz, pass_zero=False), 1
            if iir_type == 'butter' or iir_type == 'butterworth':
                b_iir, a_iir = signal.butter(iir_order, (2 / srates[i]) * fn_hz, btype=btype)
            elif iir_type == 'bessel':
                b_iir, a_iir = signal.bessel(iir_order, (2 / srates[i]) * fn_hz, btype=btype)
            # Filtering
            data_chan_filtered_fir_i = signal.lfilter(b_fir, a_fir, data_chan_raw_i).squeeze()
            data_chan_filtered_fir_i = data_chan_filtered_fir_i[int(fir_order/2):]
            data_chan_filtered_i_for = signal.lfilter(b_iir, a_iir, data_chan_raw_i)
            data_chan_filtered_i_forback = signal.filtfilt(b_iir, a_iir, data_chan_raw_i)
            data_chan_raw.append(data_chan_raw_i.squeeze())
            data_chan_filtered_fir.append(data_chan_filtered_fir_i.squeeze())
            data_chan_filtered_for.append(data_chan_filtered_i_for.squeeze())
            data_chan_filtered_forback.append(data_chan_filtered_i_forback.squeeze())

        if np.unique(srates).size > 1:
            raise ValueError('Multiple sampling rate found. Each file must have the same sampling rate.')
        srate = np.unique(srates)
        n_half_sample = int(half_duration*srate)

        # Interpolation parameters
        resample_factor, interp_method = 4, 'cubic'
        x, x_interp = np.arange(0, 2*n_half_sample), np.linspace(0, 2*n_half_sample-1, resample_factor*2*n_half_sample)
        raw_unit_shape = np.zeros((n_spikes, 2*n_half_sample))
        filt_unit_shape_fir, filt_unit_shape_for, filt_unit_shape_forback = np.zeros((3, n_spikes, 2*n_half_sample))
        filt_unit_shape_fir_r, filt_unit_shape_for_r, filt_unit_shape_forback_r = np.zeros((3, n_spikes, resample_factor*2*n_half_sample))
        raw_unit_shape_r = np.zeros((n_spikes, resample_factor * 2 * n_half_sample))
        spktime_ori = []
        for i, spk_time_i in tqdm(enumerate(spike_train)):
            # spk_time_ori_i = spk_time_i
            spk_time_ori_i, filename_i = spikeAnalysis_utils.find_original_time(float(spk_time_i), filename,
                                                                                keep_start, keep_end)
            if filename_i:
                spk_samp_ori_i = int(np.round(spk_time_ori_i * srate))
                file_pos = int(np.where(unique_filenames == filename_i)[0])
                raw_unit_shape[i, :] = data_chan_raw[file_pos][spk_samp_ori_i-n_half_sample:spk_samp_ori_i+n_half_sample]
                # Remove median
                raw_unit_shape[i, :] = raw_unit_shape[i, :] - np.median(raw_unit_shape[i, :])
                filt_unit_shape_fir[i, :] = data_chan_filtered_fir[file_pos][spk_samp_ori_i-n_half_sample:spk_samp_ori_i+n_half_sample]
                filt_unit_shape_for[i, :] = data_chan_filtered_for[file_pos][spk_samp_ori_i-n_half_sample:spk_samp_ori_i+n_half_sample]
                filt_unit_shape_forback[i, :] = data_chan_filtered_forback[file_pos][spk_samp_ori_i-n_half_sample:spk_samp_ori_i+n_half_sample]
                if realign:
                    # Realigne the extremum of the spike trace in the middle of the window
                    raw_unit_shape_r[i, :] = realign_spike_trace(raw_unit_shape[i, :], filt_unit_shape_fir[i, :], resample_factor, interp_method)
                    filt_unit_shape_fir_r[i, :] = realign_spike_trace(filt_unit_shape_fir[i, :], [], resample_factor, interp_method)
                    filt_unit_shape_for_r[i, :] = realign_spike_trace(filt_unit_shape_for[i, :], [], resample_factor, interp_method)
                    filt_unit_shape_forback_r[i, :] = realign_spike_trace(filt_unit_shape_forback[i, :], [], resample_factor, interp_method)
                spktime_ori.append(spk_time_ori_i)

        n_spikes_to_plot = 40000
        alpha_val = np.max([10 / np.min([n_spikes_to_plot, n_spikes]), 0.0005])

        if not realign:
            t = 1000 * np.linspace(-n_half_sample / 30000, n_half_sample / 30000, 2 * n_half_sample)

            if plot_mean_shape:
                f = plt.figure()
                ax1 = f.add_subplot(211)
                ax1.plot(t, 1E6 * raw_unit_shape.mean(0))
                ax1.set(xlabel='Time (ms)', ylabel='Amplitude (uV)',
                        title='Unit {} - {} - Mean Raw Unit Shape'.format(self.unit_names[unit_pos], unit_chan_name))
                ax2 = f.add_subplot(212, sharex=ax1)
                ax2.plot(t, 1E6 * filt_unit_shape_fir.mean(0))
                ax2.plot(t, 1E6 * filt_unit_shape_for.mean(0))
                ax2.plot(t, 1E6 * filt_unit_shape_forback.mean(0))
                ax2.set(xlabel='Time (ms)', ylabel='Amplitude (uV)',
                        title='Unit : {} - {} - Mean Filtered Unit Shape - {} : {} Hz'.format(self.unit_names[unit_pos], unit_chan_name,
                                                                                              btype, str(fn_hz)))
                ax2.autoscale(axis='x', tight=True)
                plt.legend(['FIR - order {}'.format(fir_order), 'IIR Forward - {} order {}'.format(iir_order, iir_type),
                            'IIR Forward-backward - {} order {}'.format(iir_order, iir_type)])

            # Density plot
            if plot_density_plot:
                f = plt.figure()
                random_pos = np.arange(n_spikes)[np.random.permutation(np.arange(n_spikes))]
                random_pos_sel = random_pos[:np.min([n_spikes_to_plot, n_spikes])]
                ax1 = f.add_subplot(211)
                ax1.plot(t, 1E6 * raw_unit_shape[random_pos_sel, :].T, color='k', alpha=alpha_val)
                ax1.set(ylabel='Amplitude', title='Raw Density plot')
                ax2 = f.add_subplot(212, sharex=ax1)
                ax2.plot(t, 1E6 * filt_unit_shape_fir[random_pos_sel, :].T, color='k', alpha=alpha_val)
                ax2.set(xlabel='time (ms)', ylabel='Amplitude', title='Filtered Density plot')
                ax2.autoscale(axis='x', tight=True)

        else:
            # Re-aligned
            t_interp = 1000 * np.linspace(-n_half_sample / 30000, n_half_sample / 30000,
                                          2 * resample_factor * n_half_sample)
            if plot_mean_shape:
                f = plt.figure()
                ax1 = f.add_subplot(211)
                ax1.plot(t_interp, 1E6 * raw_unit_shape_r.mean(0))
                ax1.set(xlabel='Time (ms)', ylabel='Amplitude (uV)',
                        title='Unit {} - {} - Mean Raw Unit Shape - Realigned'.format(self.unit_names[unit_pos], unit_chan_name))
                ax2 = f.add_subplot(212, sharex=ax1)
                ax2.plot(t_interp, 1E6 * filt_unit_shape_fir_r.mean(0))
                ax2.plot(t_interp, 1E6 * filt_unit_shape_for_r.mean(0))
                ax2.plot(t_interp, 1E6 * filt_unit_shape_forback_r.mean(0))
                ax2.set(xlabel='Time (ms)', ylabel='Amplitude (uV)',
                        title='Unit : {} - {} - Mean Filtered Unit Shape - {} : {} Hz - Realigned'.format(self.unit_names[unit_pos],
                                                                                              unit_chan_name,
                                                                                              btype, str(fn_hz)))
                ax2.autoscale(axis='x', tight=True)
                plt.legend(['FIR - order {}'.format(fir_order), 'IIR Forward - {} order {}'.format(iir_order, iir_type),
                            'IIR Forward-backward - {} order {}'.format(iir_order, iir_type)])

            if plot_density_plot:
                # Density plot - Re-aligned
                f = plt.figure()
                random_pos = np.arange(n_spikes)[np.random.permutation(np.arange(n_spikes))]
                random_pos_sel = random_pos[:np.min([n_spikes_to_plot, n_spikes])]
                ax1 = f.add_subplot(211)
                ax1.plot(t_interp, 1E6 * raw_unit_shape_r[random_pos_sel, :].T, color='k', alpha=alpha_val)
                ax1.set(ylabel='Amplitude', title='Raw Density Plot - Realigned')
                ax2 = f.add_subplot(212, sharex=ax1)
                ax2.plot(t_interp, 1E6 * filt_unit_shape_fir_r[random_pos_sel, :].T, color='k', alpha=alpha_val)
                ax2.set(xlabel='time (ms)', ylabel='Amplitude', title='Filtered Density plot - Realigned')
                ax2.autoscale(axis='x', tight=True)

    def save(self, dirpath, filename=[]):
        """ Save the instance as a pickle (.p) file
        The filename is appended with the currendt date and time

        Parameters
        ----------
        dirpath : str
            Directory path
        filename : str | None
            Output filename. If none is 'neoall_{currentdate}.p'

        """
        if not os.path.isdir(dirpath):
            print('Creating save directory : {}'.format(dirpath))
            os.mkdir(dirpath)
        if not filename:
            filename = 'neoall_{}.p'.format(datetime.strftime(datetime.now(), '%d%m%y_%H%M'))
        with open(os.path.join(dirpath, filename), 'wb') as f:
            _pickle.dump(self, f)

    def export_spike_file_time(self, artefact_free_periods_filepath, output_path, unit_pos=[], sep=';'):
        unit_pos = np.array(unit_pos)
        if unit_pos.size == 0:
            unit_pos = np.arange(self.n_units)

        # Read csv file
        df = pd.read_csv(artefact_free_periods_filepath, sep=sep, decimal=',')
        filename, keep_start, keep_end = np.array(df['Filename']), np.array(df['t_start']).astype(float), np.array(df['t_end']).astype(float)
        if not keep_start.size == keep_end.size == filename.size:
            raise ValueError('Columns Filename and keep_start and keep_end must have the same size')

        # Count the total number of spikes for each unit :
        n_spikes = 0
        for upos in unit_pos:
            n_spikes += len(self.segments[upos].spiketrains[0])

        spk_time_all, spk_time_ori_all, spk_unit_all = np.zeros((3, n_spikes))
        spk_file_all = np.zeros(n_spikes, dtype=object)
        spk_counter = 0
        for upos in tqdm(unit_pos):
            spike_train_i = self.segments[upos].spiketrains[0]
            spk_time_i = [spikeAnalysis_utils.find_original_time(float(spk_time_i), filename, keep_start, keep_end)[0]
                          for spk_time_i in spike_train_i]
            spk_file_i = [spikeAnalysis_utils.find_original_time(float(spk_time_i), filename, keep_start, keep_end)[1]
                          for spk_time_i in spike_train_i]
            try:
                spk_time_offset_i = [int(re.search('_\d+_', re.search('p\d+_\d+_', spk_file_i_j)[0])[0][1:-1]) for
                                     spk_file_i_j in spk_file_i]
            except:
                print('Could not determine offset')
                spk_time_offset_i = []
            spk_time_all[spk_counter:spk_counter+len(spk_time_i)] = spk_time_i
            if spk_time_offset_i:
                spk_time_ori_all[spk_counter:spk_counter+len(spk_time_i)] = np.array(spk_time_i) + np.array(spk_time_offset_i)
            spk_file_all[spk_counter:spk_counter+len(spk_file_i)] = spk_file_i
            spk_unit_all[spk_counter:spk_counter+len(spk_time_i)] = upos
            spk_counter += len(spk_time_i)

        if spk_time_offset_i:
            df = pd.DataFrame({'filename':spk_file_all, 'unit':spk_unit_all, 'time (file)': spk_time_all, 'time (whole file)': spk_time_ori_all})
        else:
            df = pd.DataFrame({'filename':spk_file_all, 'unit':spk_unit_all, 'time (file)': spk_time_all})
        excel_writer = pd.ExcelWriter(output_path)
        df.to_excel(excel_writer, 'Sheet', index_label=None)
        excel_writer.save()




def realign_spike_trace(spike_trace, spike_trace_filtered=[], resample_factor=5, interp_method='cubic',
                        half_extremum_win=15):
    """ Re-align the spike trace so that the extremum in on the center of the window. The extremum position is computed
    on the filtered spike trace. Before taking the extremum, the filtered spike trace is upsampled and interpolated.
    The returned spike trace is the upsampled and interpolated trace, aligned on the center of the window.

    Parameters
    ----------
    spike_trace : array
        Spike trace
    spike_trace_filtered : array | none (default)
        Filtered spike trace. If empty is equal to ``spike_trace``
    resample_factor : int (default : 5)
        Upsampling factor
    interp_method : str (default : 'cubic')
        Interpolation method used
    half_extremum_win : int
        The extremum will be search only around the center of the window.
        In a sub-window around the center : [center-half_extremum_win, center+half_extremum_win]
        Spikes should already be more or less centered.

    Returns
    -------
    spike_trace_realigned : array
        Spike trace aligned and upsampled
    """
    spike_trace_filtered = np.array(spike_trace_filtered)
    same_spike_trace = 0
    if spike_trace_filtered.size == 0:
        spike_trace_filtered = spike_trace
        same_spike_trace = 1
    n_pnts = spike_trace.size
    x_ori, x_interp = np.arange(n_pnts), np.linspace(0, n_pnts-1, n_pnts*resample_factor)
    f_interp = interp1d(x_ori, spike_trace, kind=interp_method)
    spike_trace_interp = f_interp(x_interp)
    if same_spike_trace:
        spike_trace_filt_interp = spike_trace_interp
    else:
        f_interp_filt = interp1d(x_ori, spike_trace_filtered, kind=interp_method)
        spike_trace_filt_interp = f_interp_filt(x_interp)
    spike_trace_realigned = np.zeros(n_pnts * resample_factor)
    # Find position of the extremum
    middle_pos, half_extremum_win_r = int(x_interp.size/2), resample_factor * half_extremum_win
    extremum_pos = middle_pos - half_extremum_win_r + \
                   np.argmax(np.abs(spike_trace_filt_interp[middle_pos-half_extremum_win_r:middle_pos+half_extremum_win_r]))
    x_offset = middle_pos-extremum_pos
    if x_offset > 0:
        spike_trace_realigned[x_offset:] = spike_trace_interp[:-x_offset]
    elif x_offset < 0:
        spike_trace_realigned[:x_offset] = spike_trace_interp[-x_offset:]
    else:
        spike_trace_realigned = spike_trace_interp
    return spike_trace_realigned


class NeoEpoch(NeoMother):
    """ NeoEpoch instances are created from a NeoAll instance. The spiketrains contained in the NeoAll instance are
    divided into epochs from an event file. Spikes occuring around each event are kept ([``time_pre, ``time_post``]).
    NeoEpoch contains a list of Neo Segments. Each segment represent one epoch. Each segment contains as many Neo
    SpikeTrain as there are units. Each segment also contain a Neo AnalogSignal instance representing the LFP signal
    associated with the spiketrains.

        Attributes :
        ------------
        spikesorting_dirpath : str
            Path of the directory containng the spyking circus result file
        spikesorting_filename : str
            Name of the spyking circus result file
        signal_dirpath : str
            Path of the directory containing the LFP signals files (must be in EDF format)
        n_units : int
            Number of units
        unit_names : list | array
            Name of each unit
        unit_shapes : array
            Dhape of each unit
        srate : int
            Sampling frequency
        channel_names : array
            Name of the channels of the LFP files
        unit_preferred_chan : array
            Preferred channel for each unit
        electrode_names : array
            Name of the electrode of the LFP files
        electrode_names_unique : array
            List of the different electrode names
        event_filepath : str
            Path of the event file around which the epochs are created
        n_events : int
            Number of events
        ev_time : array
            Contains the time of each event (s)
        ev_duration : array
            Contains the duration (s) of each event
        time_pre : float (default : 1)
            Time taken before the events (s) when creating the epochs
        time_post : float (default : 1)
            Time taken after the events (s) when creating the epochs
        segments : list of Neo.Segment
            Each segment represents one epochs, thus each segment contains ``n_units`` spiketrains.
            Each segment also contain a Neo AnalogSignal instance representing the LFP signal associated with the
            spiketrains
        save_fig : bool
            If True, save the figures
        fig_dirpath : str
            Path where the figure are saved
        fig_dirpath_raster : str
            Path where the raster plots figure are saved
    """

    def __init__(self, neoAll, event_filepath, event_csv_sep=';', time_pre=1, time_post=1):
        NeoMother.__init__(self)
        self.spikesorting_dirpath = neoAll.spikesorting_dirpath
        self.spikesorting_filename = neoAll.spikesorting_filename
        self.signal_dirpath = neoAll.signal_dirpath
        self.n_units = neoAll.n_units
        self.unit_names = neoAll.unit_names
        self.unit_shapes = neoAll.unit_shapes
        self.srate = neoAll.srate
        self.channel_names = neoAll.channel_names
        self.unit_preferred_chan = neoAll.unit_preferred_chan
        self.electrode_names = neoAll.electrode_names
        self.electrode_names_unique = neoAll.electrode_names_unique
        self.event_filepath = event_filepath
        self.n_events = []
        self.ev_time = []
        self.ev_duration = []
        self.time_pre = time_pre
        self.time_post = time_post
        self.segments = []
        self.save_fig = neoAll.save_fig
        self.fig_dirpath = neoAll.fig_dirpath
        self.fig_dirpath_raster = neoAll.fig_dirpath_raster
        self.fig_dirpath_crosscorr = spikeAnalysis_utils.createuniquedir(os.path.join(self.fig_dirpath, 'Epoch_CrossCorr'))

        # Set seaborn style for matplotlib
        sns.set()
        sns.set_context('paper')

        # Read event file (csv file)
        with open(event_filepath, 'r') as f:
            reader = csv.reader(f, delimiter=event_csv_sep)
            next(reader, None)  # skip the headers
            event_data = [row for row in reader]
        ev_time = [float(row[EVENT_TIME_COL]) for row in event_data]
        ev_type = [row[EVENT_TYPE_COL] for row in event_data]
        ev_file = [row[EVENT_FILE_COL] for row in event_data]
        ev_time_file = [float(row[EVENT_TIME_FILE_COL]) for row in event_data]
        ev_duration = [float(row[EVENT_DURATION_COL]) for row in event_data]
        ev_chansel = [np.fromstring(row[EVENT_CHANNEL_COL], sep=',', dtype=int) for row in event_data]
        self.ev_time = np.array(ev_time)
        self.ev_duration = np.array(ev_duration)

        n_pnts_sig = -1
        # Create one segment per event
        for i_ev, time_i in enumerate(ev_time):
            seg_i = neo.Segment(name='{} - {:.1f}s'.format(ev_type[i_ev], ev_time[i_ev]), index=i_ev,
                                file_origin=ev_file[i_ev])
            # Add spiketrains on every units, select time around the event
            for i_unit in range(0, self.n_units):
                spktrain_unit = neoAll.segments[i_unit].spiketrains[0]
                spktrain_uni_sel, _ = neo_utils.spiketrainselectaroundevent(spktrain_unit, time_i, time_pre, time_post)
                seg_i.spiketrains.append(spktrain_uni_sel)
            # Read signal around event
            time_file_i = ev_time_file[i_ev]
            sig_data, _, info = eeg_utils.readfileedf(os.path.join(self.signal_dirpath, ev_file[i_ev]),
                                                      t_start=time_file_i-time_pre, t_end=time_file_i+time_post)
            # Make sure that every signal has the exact same number of points as the first one
            if i_ev == 0:
                n_pnts_sig = sig_data.shape[1]
            else:
                if sig_data.shape[1] > n_pnts_sig:
                    sig_data = sig_data[:, :n_pnts_sig]
                elif sig_data.shape[1] < n_pnts_sig:
                    sig_data = np.concatenate([sig_data, np.atleast_2d(sig_data[:, -1]).T *
                                               np.ones((sig_data.shape[0], n_pnts_sig-sig_data.shape[1]))], axis=1)
            sig = neo.AnalogSignal(sig_data.T, units='mV', sampling_rate=info['sfreq'] * Hz, t_start=-time_pre*s,
                                   name='{} signal - event {}'.format(ev_type[i_ev], i_ev), file_origin=ev_file[i_ev])
            seg_i.analogsignals.append(sig)
            self.segments.append(seg_i)
        self.n_events = len(ev_time)

    def __str__(self):
        desc_str = 'NeoEpoch instance. {} units and {} epochs\n'.format(self.n_units, self.n_events)
        desc_str += 'One Neo Segment per epoch, each segment has one Neo SpikeTrain for each unit, and one Neo AnalogSignal\n'
        return desc_str

    def plot_rasterplot(self, unit_pos=-1, bin_duration=0.01, sigma_gauss_kernel=20*ms, plot_unit_shape=1,
                        plot_event_signal=1):
        """ Plot the PSTH (peri/post stimulus time histogram) for selected units. A gaussian kernel is used for
        estimating the instantaneous firing rate. Plot the figure if ``save_fig`` is True, in ``fig_dirpath_raster``

        Parameters
        ----------
        unit_pos : int | array |list (default: all)
            Unit position.
        bin_duration : float (default: 0.01 s)
            Bin duration when plotting the rasterplot
        sigma_gauss_kernel : quantity (default: 20*ms)
            Duration of the gaussian kernel used for estimating the instantaneous firing rate.
            See elephant.statistics.instantaneous_rate
        plot_unit_shape : bool (default: True)
            If True, plot the shape of the selected units in the bottom-right corner
        plot_event_signal : bool (default: True)
            If True, plot the mean of the events signals (i.e. mean over the different events)

        """
        if type(sigma_gauss_kernel) == int or type(sigma_gauss_kernel) == float:
            sigma_gauss_kernel = sigma_gauss_kernel*s
        if np.isscalar(unit_pos) and unit_pos == -1:
            unit_pos = np.arange(0, self.n_units, 1)
        elif np.isscalar(unit_pos):
            unit_pos = np.array([unit_pos])
        smoothing_kernel = elephant.kernels.GaussianKernel(sigma=sigma_gauss_kernel)

        for upos in unit_pos:
            fig = plt.figure()
            spktrain_list = [seg.spiketrains[upos] for seg in self.segments]
            spkarray_list = [np.array(seg.spiketrains[upos]) for seg in self.segments]
            n_spk_per_event = [len(spkarray) for spkarray in spkarray_list]
            spktrain_cat = neo.SpikeTrain(times=np.concatenate(spkarray_list)*s, t_start=-self.time_pre,
                                          t_stop=self.time_post)
            spkrate = 1/len(spktrain_list)*np.array(elephant.statistics.instantaneous_rate(
                spktrain_cat, 1*ms, kernel=smoothing_kernel)).ravel()
            ax0 = plt.subplot2grid((4, 5), (0, 0), colspan=4, rowspan=3)
            for i, spkarray in enumerate(spkarray_list):
                plt.plot(spkarray, (i+1)*np.ones_like(spkarray), 'k.', markersize=4)
            ax0.set_ylim((0, len(spkarray_list)+2))
            pref_channel_name = self.getunitpreferredchannelname(self.unit_names[upos])
            ax0.set(title='Raster plot - unit : {} ({})'.format(self.unit_names[upos], pref_channel_name),
                    ylabel='ID number')
            ax0.autoscale(axis='x', tight=True)
            ax1 = plt.subplot2grid((4, 5), (3, 0), colspan=4)
            ax2 = ax1.twinx()
            ax1.hist(np.concatenate(spkarray_list), bins=int(np.round((self.time_post + self.time_pre) / bin_duration)))
            ax2.plot(np.linspace(-self.time_pre, self.time_post, len(spkrate)), spkrate, c='y', alpha=0.8)
            ax2.grid(False)
            ax1.set(xlabel='time (s)', ylabel='count')
            ax2.set(ylabel='Spike Rate (Hz)')
            ax2.set_ylim(0)
            ax2.autoscale(axis='x', tight=True)
            ax1.autoscale(axis='x', tight=True)
            ax3 = plt.subplot2grid((4, 5), (0, 4), rowspan=3)
            ax3.barh(range(1, len(spkarray_list)+1), n_spk_per_event)
            ax3.invert_xaxis()
            ax3.yaxis.tick_right()
            ax3.yaxis.set_label_position("right")
            ax3.set(xlabel='count', ylabel='Number of spikes per ID')
            ax3.set_ylim((0, len(spkarray_list)+2))
            if plot_unit_shape:
                ax4 = plt.subplot2grid((4, 5), (3, 4))
                self.subfun_plot_unit_shape(upos, ax=ax4, raster_plot_call=1)

            # plot mean signal
            if plot_event_signal:
                _, _, _, tetrode_channel_pos = self.channelpos2tetrode(self.unit_preferred_chan[upos])
                n_pnts = self.segments[0].analogsignals[0].shape[0]
                event_mean_signals = np.zeros((n_pnts, 4, len(spkarray_list)))
                for k, seg in enumerate(self.segments):
                    sig = np.array(seg.analogsignals[0])
                    event_mean_signals[:, :, k] = sig[:, tetrode_channel_pos]
                    # Mean on the event
                event_mean_signals = event_mean_signals.mean(2)
                mean_event_sig = event_mean_signals.mean(1)
                y_max, y_min = round(0.9*len(spktrain_list)), round(0.1*len(spktrain_list))
                x_max, x_min = mean_event_sig.max(), mean_event_sig.min()
                mean_event_sig_scaled = (y_max-y_min)/(x_max-x_min)*mean_event_sig + (y_min*x_max - y_max*x_min)/(x_max-x_min)
                t_vect = np.linspace(-self.time_pre, self.time_post, n_pnts)
                ax0.plot(t_vect, mean_event_sig_scaled, alpha=0.5)

            if self.save_fig:
                fig.savefig(os.path.join(self.fig_dirpath_raster, 'rasterplot_{}.{}'.format(self.unit_names[upos],
                                         RES_FIG_FORMAT)), dpi=RES_FIG_DPI)

    def plot_crosscorrelogram_around_events(self, unit_pos_a, unit_pos_b, bin_time=1 * ms, max_lag_time=80 * ms,
                                            n_repets=100, plot_superpose=1):
        """ Plot the cross-correlogram around the events of neoEpoch. Compute the shift-predictor to give a significance
         level. The shift-predictor is computed by shuffling the trials position ``n_repets`` times. The interval plotted
         is equal to the mean +/- 2 times the standard deviation.

        Parameters
        ----------
        unit_pos_a : int | array | list
            Position of first unit(s)
        unit_pos_b : int | array | list
            Position of second unit(s)
        bin_time : time (quantity) (default: 1*ms)
            Bin duration
        max_lag_time : time (quantity) (default: 80*ms)
            Max lagging time when computing the cross-corelogram
        n_repets : int (default: 100)
            Number of repetitions when computing the sampling distribution
        plot_superpose : bool (default: True)
            If True, superimpose the plots

        """
        if type(bin_time) == int or type(bin_time) == float:
            bin_time = bin_time * ms
        if type(max_lag_time) == int or type(max_lag_time) == float:
            max_lag_time = max_lag_time * ms
        unit_pos_a, unit_pos_b = np.atleast_1d(unit_pos_a), np.atleast_1d(unit_pos_b)

        if not unit_pos_a.size == 1 or not unit_pos_b.size == 1:
            for upos_a in unit_pos_a:
                for upos_b in unit_pos_b:
                    self.plot_crosscorrelogram_around_events(int(upos_a), int(upos_b), bin_time=bin_time,
                                                             max_lag_time=max_lag_time, n_repets=n_repets,
                                                             plot_superpose=plot_superpose)
        else:
            unit_pos_a, unit_pos_b = int(unit_pos_a), int(unit_pos_b)
            cch = np.zeros((self.n_events, 1 + int(np.floor(2.0*max_lag_time / bin_time))))
            for i in range(self.n_events):
                binned_spktrain_i_a = elephant.conversion.BinnedSpikeTrain(self.segments[i].spiketrains[unit_pos_a], bin_time)
                binned_spktrain_i_b = elephant.conversion.BinnedSpikeTrain(self.segments[i].spiketrains[unit_pos_b], bin_time)
                cch_i, bins = elephant.spike_train_correlation.cch(binned_spktrain_i_a, binned_spktrain_i_b,
                                                                  (-max_lag_time, max_lag_time)*bin_time)
                cch[i, :] = np.array(cch_i).flatten()

            # Shift predictor theoretically tells you what is the shape of the crosscorrelogram based solely on firing rates
            # of the cells, in the absence of any physical relationship (in the form of synaptic connectivity or common
            # input)
            shift_predictor = np.zeros((n_repets, 1 + int(np.floor(2.0*max_lag_time / bin_time))))
            cch_shuffled = np.zeros((self.n_events, 1 + int(np.floor(2.0*max_lag_time / bin_time))))
            for j in tqdm(range(n_repets)):
                # Shuffle the epochs for unit a
                epochs_pos_a_shuffled = np.arange(self.n_events)
                np.random.shuffle(epochs_pos_a_shuffled)
                for i in range(self.n_events):
                    binned_spktrain_i_a = elephant.conversion.BinnedSpikeTrain\
                        (self.segments[epochs_pos_a_shuffled[i]].spiketrains[unit_pos_a], bin_time)
                    binned_spktrain_i_b = elephant.conversion.BinnedSpikeTrain\
                        (self.segments[i].spiketrains[unit_pos_b], bin_time)
                    cch_i, bins = elephant.spike_train_correlation.cch(binned_spktrain_i_a, binned_spktrain_i_b,
                                                                       (-max_lag_time, max_lag_time)*bin_time)
                    cch_shuffled[i, :] = np.array(cch_i).flatten()
                shift_predictor[j, :] = np.sum(cch_shuffled, 0)
            shift_predictor_mean, shift_predictor_std = shift_predictor.mean(0), shift_predictor.std(0)

            # Plot
            f = plt.figure()
            if plot_superpose:  # Single axis
                ax = f.add_subplot(111)
                ax.bar(bins - float(bin_time) / 2, cch.sum(0), width=1, linewidth=0, label='Original Data')
                ax.plot(bins - float(bin_time) / 2, shift_predictor_mean, color='k', label='Shuffled Distribution')
                ax.fill_between(bins - float(bin_time) / 2, shift_predictor_mean - 2*shift_predictor_std,
                                shift_predictor_mean + 2*shift_predictor_std, alpha=0.4, color='y',
                                label='Mean +/- 2 * std - Mean Shuffled Distribution')
                ax.autoscale(axis='x', tight='True')
                ax.legend()
                ax.set(title='Cross correlogram - unit : {} VS {} '.format(self.unit_names[unit_pos_a], self.unit_names[unit_pos_b]),
                       xlabel='Lag (ms)', ylabel='counts')
            else:  # Multiple axis
                ax1 = f.add_subplot(131)
                ax1.bar(bins - float(bin_time) / 2, cch.sum(0), width=1, linewidth=0)
                ax1.set(title='Raw', xlabel='Lag (ms)', ylabel='counts')
                ax1.autoscale(axis='x', tight='True')
                ax2 = f.add_subplot(132, sharex=ax1, sharey=ax1)
                ax2.bar(bins - float(bin_time) / 2, shift_predictor.mean(0), width=1, linewidth=0)
                ax2.autoscale(axis='x', tight='True')
                ax2.set(title='Shift-Predictor - unit : {} VS {} '.format(self.unit_names[unit_pos_a], self.unit_names[unit_pos_b]),
                        xlabel='Lag (ms)', ylabel='counts')
                ax3 = f.add_subplot(133)
                ax3.bar(bins - float(bin_time) / 2, cch.sum(0) - shift_predictor.mean(0), width=1, linewidth=0)
                ax3.autoscale(axis='x', tight='True')
                ax3.set(title='Corrected', xlabel='Lag (ms)', ylabel='counts')
            if self.save_fig:
                f.savefig(os.path.join(self.fig_dirpath_crosscorr, 'epoch_cross_corr{}.{}'.format(self.unit_names[upos],
                                       RES_FIG_FORMAT)), dpi=RES_FIG_DPI)

    def save(self, dirpath, filename=[]):
        """ Save the instance as a pickle (.p) file
        The filename is appended with the currendt date and time

        Parameters
        ----------
        dirpath : str
            Directory path
        filename : str | None
            Output filename. If none is 'neoall_{currentdate}.p'

        """
        if not os.path.isdir(dirpath):
            print('Creating save directory : {}'.format(dirpath))
            os.mkdir(dirpath)
        if not filename:
            filename = 'neoepoch_{}.p'.format(
                datetime.strftime(datetime.now(), '%d%m%y_%H%M'))
        with open(os.path.join(dirpath, filename), 'wb') as f:
            _pickle.dump(self, f)


if __name__ == '__main__':

    # spykingcircus_dirpath = r'C:\Users\deudon\Desktop\SpikeSorting\_Data\002RM_day4_pointes\spykingcircus_results'
    # results_filename = r'20150325-103311-001_0'
    # event_filepath = r'C:\Users\deudon\Desktop\SpikeSorting\_Data\002RM_day4_pointes\Events\eventFile.csv'
    # signal_dirpath = r'C:\Users\deudon\Desktop\SpikeSorting\_Data\002RM_day4_pointes\signal\monopolaire_5kHz_d4_post_crise'
    # probe_filepath = r'C:\Users\deudon\Desktop\SpikeSorting\_Data\002RM_day4_pointes\spykingcircus_paramfiles\002_RM.prb'
    # artefact_csv_filepath = r'C:\Users\deudon\Desktop\SpikeSorting\_Data\002RM_day4_pointes\artefact_free_periods.csv'

    spykingcircus_dirpath = r'C:\Users\deudon\Desktop\SpikeSorting\_Data\VL14\DAY2_Elodie\spykingcircus_results'
    results_filename = r'20170113-104634-001_0'
    event_filepath = r''
    signal_dirpath = r'C:\Users\deudon\Desktop\SpikeSorting\_Data\VL14\DAY2_Elodie\data\edf_30kHz'
    probe_filepath = r'C:\Users\deudon\Desktop\SpikeSorting\_Data\VL14\DAY2_Elodie\014_VL.prb'
    artefact_csv_filepath = r'C:\Users\deudon\Desktop\SpikeSorting\_Data\VL14\DAY2_Elodie\artefact_day2_epifar.csv'

    csv_delimiter = ';'
    t_pre, t_post = 1, 1
    t_max, srate = -1, 30000
    epoch_t_start = np.array([-0.500, -0.250, -0.050, 0.050, 0.250])
    epoch_duration = np.array([0.250, 0.200, 0.100, 0.200, 0.250])
    epoch_names = ['Pre-IED baseline', 'Pre-IED', 'IED', 'Slow wave', 'Post-IED']

    neoAll = NeoAll(spykingcircus_dirpath, results_filename, probe_filepath=probe_filepath, signal_dirpath=signal_dirpath,
                    event_filepath=[], fig_dirpath=[], save_fig=0)

    # neoAll.plot_raw_unit_shape(0, signal_dirpath, artefact_csv_filepath, realign=True)
    # neoEpoch = NeoEpoch(neoAll, event_filepath)
    # neoAll.plot_raw_unit_shape(0, signal_dirpath, artefact_csv_filepath, fn_hz=[300, 3000], realign=True)
    # neoAll.create_epochs_around_events('Pointe', time_offset=epoch_t_start, epoch_duration=epoch_duration,
    #                                    epoch_names=epoch_names)
    # neoAll_base = NeoAll(spykingcircus_dirpath, results_filename, probe_filepath, signal_dirpath=signal_dirpath,
    #                      event_filepath=event_filepath, fig_dirpath=figure_dirpath, save_fig=1)
    #
    # neoAll_base.create_epochs_around_events('Pointe', time_offset=epoch_t_start, epoch_duration=epoch_duration,
    #                                         epoch_names=epoch_names)
    # t_rej_pre, t_rej_post = 2*s, 2*s
    # neoAll_base.reject_spikes_around_events('Pointe', t_rej_pre, t_rej_post)
    # event_times = neoAll.segments[0].events[0].times
    # duration_overlap = ((event_times[0:-1] + t_rej_post) - (event_times[1:] - t_rej_pre))
    # duration_overlap[duration_overlap < 0*s] = 0*s
    # duration_rejected = len(event_times) * (t_rej_pre + t_rej_post) - duration_overlap.sum()
    #
    # n_segments = len(neoAll.segments)
    # mean_spkrate, mean_spkrate_base = np.zeros(n_segments), np.zeros(n_segments)
    # for i in range(0, n_segments):
    #     spktrain, spktrain_base = neoAll.segments[i].spiketrains[0], neoAll_base.segments[i].spiketrains[0]
    #     mean_spkrate[i] = len(spktrain) / (spktrain.t_stop - spktrain.t_start)
    #     mean_spkrate_base[i] = len(spktrain_base) / (spktrain.t_stop - spktrain.t_start - duration_rejected)
    # f = plt.figure()
    # ax = f.add_subplot(111)

    # barw = 0.4
    # ax.bar(np.arange(0, n_segments) - barw/2, mean_spkrate, width=barw, linewidth=0)
    # ax.bar(np.arange(0, n_segments) + barw/2, mean_spkrate_base, width=barw, linewidth=0)
    # ax.autoscale(axis='x', tight=True)
    # ax.xaxis.set(ticks=range(0, n_segments), ticklabels=neoAll.unit_names)
    # ax.set(xlabel='Unit', ylabel='Mean spike rate (Hz)', title='Mean spiking rate for each unit across whole duration')
    # ax.legend(['All spikes', 'IDs spikes removed'])

    # neoEpoch.plot_crosscorrelogram_around_events([0, 1, 2, 3, 4], 0)

    # neoAll.plot_crosscorrelogram([16, 17, 18, 19, 20], [16, 17, 18, 19, 20], do_stat=0, merge_plots=1, n_surrogates=100)