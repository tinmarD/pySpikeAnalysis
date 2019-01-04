"""
=====================
Neo Epoch - ex_0
=====================

This example shows how to create NeoEpoch

"""

from neoStructures import NeoAll, NeoEpoch
import numpy as np

###############################################
# Replace the next variables with your paths

spykingcircus_dirpath = r'C:\Users\deudon\Desktop\SpikeSorting\_Data\002RM_day4_pointes\002RM_day4_pointes\spykingcircus_results'
results_filename = r'20150325-103311-001_0'
event_filepath = r'C:\Users\deudon\Desktop\SpikeSorting\_Data\002RM_day4_pointes\002RM_day4_pointes\Events\eventFile.csv'
signal_dirpath = r'C:\Users\deudon\Desktop\SpikeSorting\_Data\002RM_day4_pointes\002RM_day4_pointes\signal\monopolaire_5kHz_d4_post_crise'
probe_filepath = r'C:\Users\deudon\Desktop\SpikeSorting\_Data\002RM_day4_pointes\002RM_day4_pointes\spykingcircus_paramfiles\002_RM.prb'
figure_dirpath = r'C:\Users\deudon\Desktop\SpikeSorting\_Data\002RM_day4_pointes\002RM_day4_pointes\fig'
artefact_csv_filepath = r'C:\Users\deudon\Desktop\SpikeSorting\_Data\002RM\Artefact_EDF_reconstruction_d1.csv'

#####################
# Epoching parameters
csv_delimiter = ';'
t_pre, t_post = 1, 1
t_max, srate = -1, 30000
epoch_t_start = np.array([-0.500, -0.250, -0.050, 0.050, 0.250])
epoch_duration = np.array([0.250, 0.200, 0.100, 0.200, 0.250])
epoch_names = ['Pre-IED baseline', 'Pre-IED', 'IED', 'Slow wave', 'Post-IED']

#################################
# First create a NeoAll instance
neo_all = NeoAll(spykingcircus_dirpath, results_filename, probe_filepath, signal_dirpath=signal_dirpath,
                event_filepath=event_filepath, fig_dirpath=figure_dirpath, save_fig=1)

##################################
# And create the NeoEpoch instance
neo_epoch = NeoEpoch(neo_all, event_filepath=event_filepath, time_pre=0.3, time_post=0.3)
print(neo_epoch)

#######################################
# You can save it with the save method :

neo_epoch.save(r'C:\Users\deudon\Desktop\SpikeSorting\_Scripts\_Python\pySpikeAnalysis\pySpikeAnalysis\sample_data')
