import mne
import numpy as np


def io_eeg_to_mne(filepath, read_data=True):
    """ Import an EEG file to MNE as a Raw instance.
    Supported formats : 'edf', 'eeg', 'trc'

    Parameters
    ----------
    filepath : str
        EEG filepath
    read_data : bool (default: True)
        If True, read the data

    Returns
    -------
    mne_raw : MNE RAW instance
        Output MNE structure
    ch_names : list
        Channel names

    """
    mne_raw = []
    possible_ext = ['.edf', '.fif', '.trc', '.ns5', '.nsx']
    file_ext = re.search('\.\w+$', filepath)
    if file_ext:
        file_ext = file_ext[0].lower()
    else:
        raise ValueError('Could not detect file extension of file {}'.format(filepath))
    if file_ext not in possible_ext:
        raise ValueError('The file {} has not a supported extension. Extensions must be in {}'.format(filepath,
                                                                                                      possible_ext))
    if file_ext in ['.edf']:
        try:
            mne_raw = mne.io.read_raw_edf(filepath, preload=read_data)
        except:
            mne_raw = mne.io.read_raw_edf(filepath, preload=True)
        ch_names = mne_raw.ch_names
    elif file_ext == '.fif':
        try:
            mne_raw = mne.io.read_raw_fif(filepath, preload=read_data)
        except:
            mne_raw = mne.io.read_raw_fif(filepath, preload=True)
        ch_names = mne_raw.ch_names
    elif file_ext == '.trc':
        trc_reader = neo.io.MicromedIO(filename=filepath)
        header = trc_reader.header
        ch_names = [header['signal_channels'][i][0] for i in range(trc_reader.signal_channels_count())]
        if read_data:
            bl = trc_reader.read(lazy=False)[0]
            seg = bl.segments[0]
            n_pnts, n_chan = len(seg.analogsignals[0]), len(seg.analogsignals)
            data = np.zeros((n_chan, n_pnts), dtype=float)
            for i, asig in enumerate(seg.analogsignals):
                # We need the ravel() here because Neo < 0.5 gave 1D, Neo 0.5 gives 2D (but still a single channel).
                data[i, :] = asig.magnitude.ravel()
            sfreq = int(seg.analogsignals[0].sampling_rate.magnitude)
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq)
            mne_raw = mne.io.RawArray(data, info)
    elif file_ext in ['.ns5', 'nsx']:
        nsx_reader = neo.io.BlackrockIO(filename=filepath)
        header = nsx_reader.header
        ch_names = [header['signal_channels'][i][0] for i in range(nsx_reader.signal_channels_count())]
        if read_data:
            bl = nsx_reader.read(lazy=False)[0]
            seg = bl.segments[0]
            n_pnts, n_chan = len(seg.analogsignals[0]), len(seg.analogsignals)
            data = np.zeros((n_chan, n_pnts), dtype=float)
            for i, asig in enumerate(seg.analogsignals):
                # We need the ravel() here because Neo < 0.5 gave 1D, Neo 0.5 gives
                # 2D (but still a single channel).
                data[i, :] = asig.magnitude.ravel()
            sfreq = int(seg.analogsignals[0].sampling_rate.magnitude)
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq)
            mne_raw = mne.io.RawArray(data, info)

    return mne_raw, ch_names


def readfileedf(edfilepath, t_start=[], t_end=[], channel_sel=[]):
    """ Read an edf file using the mne package

    Parameters
    ----------
    edfilepath : str
        Filepath
    t_start : float | none
        Starting time of the selection
    t_end : float | none
        Ending time of the selection
    channel_sel : list | array
        Index of selected channels - if none select all channels

    Returns
    -------
    data : array
        data matrix
    times : array
        time vector
    raw.info :
      mne-type dictionnary containing informations about the signal
    """
    raw = mne.io.read_raw_edf(edfilepath)
    if not t_start or t_start < raw.times[0]:
        t_start = raw.times[0]
    if not t_end or t_end > raw.times[-1]:
        t_end = raw.times[-1]
    if not channel_sel:
        channel_sel = np.arange(0, raw.info['nchan'], 1)
    ind_start, ind_end = raw.time_as_index([t_start, t_end])
    data, times = raw[channel_sel, ind_start:ind_end]

    return data, times, raw.info


def getchannelnamesfromedffile(edffilepath):
    """ Return the names of the channels from an EDF file

    Parameters
    ----------
    edffilepath : str
        Path of the edf file

    Returns
    -------
    ch_names : array
        Channel names
    """
    raw = mne.io.read_raw_edf(edffilepath)
    return raw.info['ch_names']


def smooth1d(x, w, mode='valid'):
    """ Smooth a 1D-array using a convolution between the input array ``x`` and the window ``w``. The window is
    normalized before the convolution.

    Parameters
    ----------
    x : array
        Input array to be smoothed
    w : array
        Window used to smooth the array
    mode : str (default: 'valid')
        Convolution mode

    Returns
    -------
    y : array
        Smooted array

    """
    x = np.array(x).ravel()
    y = np.convolve(x, w / w.sum(), mode)
    return y


# Cannot make the pyedflib library work
# def writemnerawtoedf(mne_raw, edf_filename, patient_id=''):
#     """ From a MNE raw signal, write an EDF file using the pyedflib package
#
#     Parameters
#     ----------
#     mne_raw : MNE BaseRaw instance
#         Raw signal from MNE
#     edf_filename : str
#         Output filename
#     patient_id : str (default: '')
#         Patient ID
#
#     Returns
#     -------
#
#     """
#     n_chan = len(mne_raw.ch_names)
#     edf_writer = pyedflib.EdfWriter(file_name=edf_filename, n_channels=n_chan)
#     edf_writer.setSamplefrequency(mne_raw.info['sfreq'])
#     edf_writer.setPatientName(patient_id)
#     edf_writer.set_transducer = ''
#     for i, channame_i in enumerate(mne_raw.ch_names):
#         channel_info_i = {'label': channame_i, 'dimension': '', 'sample_rate': mne_raw.info['sfreq'],
#                           'physical_max': 0, 'physical_min': 0, 'digital_max': 0, 'digital_min': 0,
#                           'transducer': '', 'prefilter': ''}
#         edf_writer.setSignalHeader(edfsignal=i, channel_info=channel_info_i)
#     data, _ = mne_raw.get_data()
#     edf_writer.writeSamples(data)




