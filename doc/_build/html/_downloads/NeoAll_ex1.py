"""
=====================
Neo All - ex_1
=====================

This example shows how to use the ERP functions

"""

from neoStructures import *
import matplotlib.pyplot as plt
from os.path import isdir, join

#######################
# Import the data
data_dir = 'sample_data' if isdir('sample_data') else join('..', '..', 'sample_data')
spykingcircus_dir = r'SpykingCircus_results'
probe_filename = r'000_AA.prb'
signal_dir = r'EDF'
results_filename = r'spykingcircusres'

neoAll = NeoAll(join(data_dir, spykingcircus_dir), results_filename, join(data_dir, probe_filename),
                signal_dirpath=join(data_dir, signal_dir), save_fig=0)

##############################
# See information about NeoAll
print(neoAll)

#######################
# Plot the shape (template) of unit 2
neoAll.plot_unit_shape(2)