import neo
import numpy as np


def spiketraintimesel(spiketrain, t_start, t_stop):
    """ Time selection on a Neo spiketrain. Select the spikes occuring between t_start and t_stop.

    Parameters
    ----------
    spiketrain : Neo SpikeTrain
        Input Neo SpikeTrain
    t_start : float
        Starting time of selection (s)
    t_stop :  float
        Ending time of selection (s)

    Returns
    -------
    spiketrainsel : Neo SpikeTrain
        Output Neo SpikeTrain containing only the selected spikes
    spiketimesel : array
        Time of the selected spikes
    """
    if not np.isscalar(t_start):
        if len(t_start) != len(t_stop):
            print('Argments t_start and t_stop must have the same size')
            return [], []
        else:
            spiketrainsel_list, spiketimesel_list = [], []
            for i in range(len(t_start)):
                spiketrainsel, spiketimesel = spiketraintimesel(spiketrain, t_start[i], t_stop[i])
                spiketrainsel_list.append(spiketrainsel)
                spiketimesel_list.append(spiketimesel)
            return spiketrainsel_list, spiketimesel_list
    else:
        spiketimes = np.array(spiketrain).ravel()
        spiketimesel = spiketimes[np.logical_and(spiketimes > t_start, spiketimes < t_stop)]
        spiketrainsel = neo.core.SpikeTrain(times=spiketimesel, units=spiketrain.units, t_start=spiketrain.t_start,
                                            t_stop=spiketrain.t_stop, sampling_rate=spiketrain.sampling_rate,
                                            file_origin=spiketrain.file_origin, name=spiketrain.name)
        return spiketrainsel, spiketimesel


def spiketraintimerejection(spiketrain, t_start, t_stop):
    """ Create a new spiketrain removing the spikes occuring between t_start and t_stop

    Parameters
    ----------
    spiketrain : Neo SpikeTrain
        Input Neo SpikeTrain
    t_start : float
        Starting time of rejection (s)
    t_stop :  float
        Ending time of rejection (s)

    Returns
    -------
    spiketrainsel : Neo SpikeTrain
        Output Neo SpikeTrain
    spiketimesel : array
        Time of the selected spikes (all spikes minus the rejected ones)
    spikerejected_ind : array
        Indices of the rejected spikes
    """
    t_start, t_stop = np.array(t_start), np.array(t_stop)
    if not t_start.size == 1:
        if t_start.size != t_stop.size:
            print('Argments t_start and t_stop must have the same size')
            return [], []
        else:
            spikerejected_ind = np.zeros(len(spiketrain), dtype=bool)
            for i in range(len(t_start)):
                _, _, spikerejected_ind_i = spiketraintimerejection(spiketrain, t_start[i], t_stop[i])
                spikerejected_ind = spikerejected_ind | spikerejected_ind_i
            spiketimesel = np.array(spiketrain).ravel()[~spikerejected_ind]
            spiketrainsel = neo.core.SpikeTrain(times=spiketimesel, units=spiketrain.units, t_start=spiketrain.t_start,
                                                t_stop=spiketrain.t_stop, sampling_rate=spiketrain.sampling_rate,
                                                file_origin=spiketrain.file_origin, name=spiketrain.name)
            return spiketrainsel, spiketimesel, spikerejected_ind
    else:
        spiketimes = np.array(spiketrain).ravel()
        spikerejected_ind = np.logical_and(spiketimes >= t_start, spiketimes <= t_stop)
        spiketimesel = spiketimes[np.logical_or(spiketimes < t_start, spiketimes > t_stop)]
        spiketrainsel = neo.core.SpikeTrain(times=spiketimesel, units=spiketrain.units, t_start=spiketrain.t_start,
                                            t_stop=spiketrain.t_stop, sampling_rate=spiketrain.sampling_rate,
                                            file_origin=spiketrain.file_origin, name=spiketrain.name)
        return spiketrainsel, spiketimesel, spikerejected_ind


def spiketrainselectaroundevent(spiketrain, t_event, t_pre, t_post, shifttime=1):
    """  Select spikes on a Neo spiketrain occuring before and after an event. Select spikes occuring in
    the window [t_event-t_pre < t_sel < t_event+t_post].
    If shifttime = 1, the spike times are shifted so that t_event is set to 0

    Parameters
    ----------
    spiketrain : Neo SpikeTrain
        Input Neo SpikeTrain
    t_event : float
        Time of the event (s)
    t_pre : float
        Time kept before the event (s)
    t_post : float
        Time kept after the event (s)
    shifttime : bool
        If True, the spike times are shifted so that t=0 at the time of the event, if False, the spike times are not
        changed

    Returns
    -------
    spiketrainsel : Neo SpikeTrain
        Selected SpikeTrain around the event
    spiketimesel : array
        Time of the selected spikes, around the event

    """
    if not shifttime:
        spiketrainsel, spiketimesel = spiketraintimesel(spiketrain, t_event-t_pre, t_event+t_post)
    else:
        _, spiketimesel = spiketraintimesel(spiketrain, t_event-t_pre, t_event+t_post)
        spiketrainsel = neo.core.SpikeTrain(times=spiketimesel-t_event, units=spiketrain.units, t_start=-t_pre,
                                            t_stop=t_post, sampling_rate=spiketrain.sampling_rate,
                                            file_origin=spiketrain.file_origin, name=spiketrain.name)
    return spiketrainsel, spiketimesel


def spiketrainrejectaroundevent(spiketrain, t_event, t_pre, t_post):
    """ Reject spikes around an event, from a Neo SpikeTrain. Reject spikes occuring in
    the window [t_event-t_pre < t_reject < t_event+t_post].

    Parameters
    ----------
    spiketrain : Neo SpikeTrain
        Input Neo SpikeTrain
    t_event : float
        Time of the event (s)
    t_pre : float
        Time kept before the event (s)
    t_post : float
        Time kept after the event (s)

    Returns
    -------
    spiketrainsel : Neo SpikeTrain
        SpikeTrain without the spikes rejected
    spiketimesel : array
        Time of the selected spikes (all spikes minus the rejected ones)
    """
    spiketrainsel, spiketimesel, _ = spiketraintimerejection(spiketrain, t_event - t_pre, t_event + t_post)
    return spiketrainsel, spiketimesel


def getmeanspikerateonepoch(seg, epoch_name):
    """ Compute the mean firing rate on the epoch specified by its name epoch_name

    Parameters
    ----------
    seg : Neo Segment
        Input Neo Segment
    epoch_name : str
        Name of the epoch

    Returns
    -------
    spkrate_per_epoch : float
        Mean spiking rate on the epoch in Hz (n_spikes / epoch_duration)
    """
    epoch_names = [epoch.name for epoch in seg.epochs]
    if epoch_name not in epoch_names:
        print('Cannot find epoch named {} in segment {}'.format(epoch_name, seg.name))
        return
    if not seg.spiketrains:
        print('No spiketrain for segment {}'.format(seg.name))
        return
    elif len(seg.spiketrains) > 1:
        print('More than one spiketrain in segment {}. Only consider the first one.'.format(seg.name))
    epoch_pos = np.where(np.array(epoch_names) == epoch_name)[0][0]
    epoch = seg.epochs[epoch_pos]
    epoch_t_starts, epoch_durations = epoch.times, epoch.durations
    _, spktimes = spiketraintimesel(seg.spiketrains[0], np.array(epoch_t_starts), np.array(epoch_t_starts +
                                                                                           epoch_durations))
    n_spk_per_epoch = np.array([len(times) for times in spktimes])
    spkrate_per_epoch = n_spk_per_epoch / epoch_durations
    return spkrate_per_epoch


