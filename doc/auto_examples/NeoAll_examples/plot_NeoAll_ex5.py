"""
======================================================
    Neo All - example 5 - Channel Groups / Index
======================================================

This example show how to use the channel indexes in the NeoAll class. In NeoAll, channel indexes are defined for each
tetrode and each electrode **if the signal directory path is given at init**. Otherwise channel indexes are created
based on the probe file.

"""

from neoStructures import *
import pandas as pd
import matplotlib.pyplot as plt
from os.path import isdir, join

#######################
# Import the data and create the NeoAll instance
data_dir = join('pySpikeAnalysis', 'sample_data') if isdir('pySpikeAnalysis') else join('..', '..', 'pySpikeAnalysis', 'sample_data')
spykingcircus_dir = r'SpykingCircus_results'
probe_filename = r'000_AA.prb'
results_filename = r'spykingcircusres'
artefact_csv_filepath = join(data_dir, r'artefact_free_periods.csv')

neoAll = NeoAll(join(data_dir, spykingcircus_dir), results_filename, join(data_dir, probe_filename), save_fig=0)

##############################
# See information about NeoAll
print(neoAll)

###############################
# Let's look at the different channel indexes :
print(neoAll.channel_indexes)


################################
# We can see that the first groups or ChannelIndex correpond to individual tetrodes and that the last groups
# correpond to the entire electrode (i.e. group of 2 or 3 tetrodes)

#################################
# We can use the channel indexes for plotting the firing rate evolution of multiple units in the same figure. Channel
# indexes define groups of channel. By default, a channel index is defined for each tetrode.
# Let's plot the firing rate evolution of all units of the tetrode 2 of electrode cp (ChannelIndex number 4,
# thus index 3) :
neoAll.plot_spikerate_evolution_group(grp_index=3)
