"""
========================================================
    Neo All - example 3 - Density plot, unit template
========================================================

This example shows how to plot density plots of units. The original EEG file must be given, as well a csv file
containing the time-periods that were kept for the spike-sorting (it is usual to remove artefact periods before
doing the spike-sorting, this file allows to have access to the original time of spikes, thus allowing to plot
the raw unit shape and to control the effect of filtering on the unit shape).

"""

from neoStructures import *
import pandas as pd
import matplotlib.pyplot as plt
from os.path import isdir, join

#######################
# Import the data and create the NeoAll instance
data_dir = join('pySpikeAnalysis', 'sample_data') if isdir('pySpikeAnalysis') else join('..', '..', 'pySpikeAnalysis', 'sample_data')
data_dir_sig = join('pySpikeAnalysis', 'sample_data') if isdir('pySpikeAnalysis') else join('..', '..', 'pySpikeAnalysis', 'sample_data_whole')
spykingcircus_dir = r'SpykingCircus_results'
probe_filename = r'000_AA.prb'
# signal_dir = join(data_dir_sig, r'EDF')
signal_dir = ''
results_filename = r'spykingcircusres'
artefact_csv_filepath = join(data_dir, r'artefact_free_periods.csv')

####################
# In comparison with the previous examples, we have added the `signal_dirpath` argument which points to the
# directory containing the data signal used for the spike-sorting.
neoAll = NeoAll(join(data_dir, spykingcircus_dir), results_filename, join(data_dir, probe_filename),
                signal_dirpath=signal_dir, save_fig=0)


##############################
# See information about NeoAll, we now have access to the number of channels and the number of electrodes.
print(neoAll)

##############################
# Providing the artefact_csv_filepath, which must be a CSV file with 3 columns in the order 'Filename', 't_start' and
# 't_end', with a 1 row header...
df = pd.read_table(artefact_csv_filepath)
print(df.head())

###############################
# the method :func:`neoStructure.NeoAll.plot_raw_unit_shape` allows to have access to the raw (unfiltered)
# spike trace and to investigate the effects of filtering on the unit shape.

# neoAll.plot_raw_unit_shape(34, signal_dir, artefact_csv_filepath, fn_hz=[300, 3000], plot_density_plot=False)

################################
# Density plot can be shown :

# neoAll.plot_raw_unit_shape(34, signal_dir, artefact_csv_filepath, fn_hz=[300, 3000], plot_mean_shape=False)

#############################
# It is possible to re-align the spike traces based on the extrema by setting realign to True

# neoAll.plot_raw_unit_shape(34, signal_dir, artefact_csv_filepath, fn_hz=[300, 3000], realign=True, plot_density_plot=False)

############################
# Realigned density plot :

# neoAll.plot_raw_unit_shape(34, signal_dir, artefact_csv_filepath, fn_hz=[300, 3000], realign=True, plot_mean_shape=False)

#############################
# Comparison can be made with the shape obtained from Spyking-Circus results :

# neoAll.plot_unit_shape(34)

