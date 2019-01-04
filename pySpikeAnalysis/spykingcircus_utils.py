import h5py
import os
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


def get_unit_preferred_electrode(spykingcircus_dirpath, results_filename):
    """ Get the preferred electrode for all units. Reads the clusters Spyking Circus file.

    Parameters
    ----------
    spykingcircus_dirpath : str
        Path of the Spyking Circus results directory
    results_filename : str

    Returns
    -------
    pref_el : array
        Preferred electrode for all units

    """
    # Load clusters file :
    f = h5py.File(os.path.join(spykingcircus_dirpath, results_filename+'.clusters.hdf5'), 'r')
    pref_el = np.array(f['electrodes']).ravel().astype(int)
    return pref_el


def get_channel_groups_from_spykingcircus_probefile(probe_filepath):
    """ Get the channel group from Spyking Circus probe file

    Parameters
    ----------
    probe_filepath : str
        Path of the probe file

    Returns
    -------
    chan_grp_list : list
        Channel group list
    """
    f = open(probe_filepath, 'r')
    line = f.readline()
    chan_grp_list = list()
    while line != '' and line.find('channel_groups') == -1:
        line = f.readline()
    if line == '':
        print('Reached end of file without finding channel_groups')
        return None
    while line != '':
        while line != '' and re.search('\'channels\':', line) is None:
            line = f.readline()
        match = re.search(r'\[([\d, ]+)\]', line)
        if match is not None:
            channel_pos = np.fromstring(match.group(1), sep=',', dtype='int')
            chan_grp_list.append(channel_pos)
        line = f.readline()
    return chan_grp_list


def get_templates_from_spykingcircus_file(spykingcircus_result_dirpath, filename, unit_pos, gain=4):
    """ Get the templates (shapes) for a specific unit. Designed for tetrode recordings, this return the template of
    the unit on each wire of the tetrode. It requires the .basis.hdf5, .templates.hdf5 and the .clusters.hdf5 files
    from Spyking Circus.

    Parameters
    ----------
    spykingcircus_result_dirpath : str
        Path where the Spyking-Circus files are stored
    filename : str
        Base name of the Spyking-Circus results, without the extension
    unit_pos : int
        Number of the unit
    gain : float (default: 4)
        Gain of the acquisition device - For Blackrock, set the gain to 4

    Returns
    -------
    r_temp_i_tetrode : array
        Unit's template on each wire of the tetrode
    """
    f_whitening = h5py.File(os.path.join(spykingcircus_result_dirpath, filename+'.basis.hdf5'), 'r')
    f_template = h5py.File(os.path.join(spykingcircus_result_dirpath, filename+'.templates.hdf5'), 'r')
    f_clusters = h5py.File(os.path.join(spykingcircus_result_dirpath, filename+'.clusters.hdf5'), 'r')
    # Get preferred micro-wire and get 4 tetrode's micro wire position
    pref_micro_wires = np.array(f_clusters['electrodes']).flatten()
    pref_micro_wire_i = pref_micro_wires[unit_pos]
    tetrode_num = np.ceil((1+pref_micro_wire_i)/4.0)
    tetrode_micro_wire_pos = np.arange(4*(tetrode_num-1), 4*tetrode_num, 1).astype(int)
    # Get the invert of the whitening matrix
    iw = np.linalg.inv(f_whitening['spatial'])
    # Get templates
    template_size = f_template['temp_shape']
    n_times, n_el = int(template_size[0]), int(template_size[1])
    temp_x, temp_y, temp_data = f_template['temp_x'], f_template['temp_y'], f_template['temp_data']
    temp_x, temp_y, temp_data = np.array(temp_x).ravel(), np.array(temp_y).ravel(), np.array(temp_data).ravel()
    templates = csr_matrix((temp_data, (temp_x, temp_y)), shape=(n_times*n_el, int(template_size[2])))
    temp_i = templates[:, unit_pos].todense().reshape((n_times, n_el))
    # Multiply it by the inverted whitening matrix to get back to physical values
    r_temp_i = (np.array(np.matmul(iw, temp_i)) / gain).T
    r_temp_i_tetrode = r_temp_i[:, tetrode_micro_wire_pos]
    return r_temp_i_tetrode


# def plot_unit_shape(spykingcircus_result_dirpath, filename,  unit_pos, srate, gain=4, plot_mean=0):
#     """
#
#     Parameters
#     ----------
#     spykingcircus_result_dirpath :
#     filename :
#     unit_pos :
#     srate :
#     gain :
#     plot_mean :
#
#     Returns
#     -------
#
#     """
#     unit_shape = get_templates_from_spykingcircus_file(spykingcircus_result_dirpath, filename, unit_pos, gain)
#     n_t = unit_shape.shape[0]
#     temp_duration_ms = n_t / srate * 1000
#     t_vect = np.linspace(-temp_duration_ms / 2, temp_duration_ms / 2, n_t)
#     # Plot
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     for i in range(0, 4):
#         plt.plot(t_vect, unit_shape[:, i], lw=1)
#     if plot_mean:
#         plt.plot(t_vect, unit_shape.mean(1), 'k', alpha=0.7, lw=3)
#     ax.autoscale(axis='x', tight=True)
#     plt.plot([0, 0], ax.get_ylim(), 'k--', lw=1, alpha=0.6, zorder=1)
#     ax.autoscale(axis='y', tight=True)
#     plt.plot(ax.get_xlim(), [0, 0], 'k--', lw=1, alpha=0.6, zorder=1)
#     ax.set(title='Unit pos : {}'.format(unit_pos), xlabel='Time (ms)', ylabel='Amplitude (uV)')
